"""Manual agentic loop driving Groq (Llama 3.3) over our tool surface.

We use the manual loop (not a higher-level agent framework) because it gives us:
- Per-turn trace events we can stream to a CLI or UI
- Full control over the conversation history
- Typed exception handling at each turn (groq.* exception classes)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import groq

from ..config import Settings, get_settings
from .prompts import SYSTEM_PROMPT
from .tools import TOOL_SCHEMAS, ToolContext, build_context, dispatch

logger = logging.getLogger(__name__)


# Llama on Groq occasionally emits its native function-call format instead of
# clean structured tool_calls — Groq returns a 400 with code 'tool_use_failed'
# and the raw text in `failed_generation`. We recover by parsing that text.
# Patterns seen in the wild:
#   <function=NAME{json}</function>
#   <function=NAME>{json}</function>      ← with the > separator
#   <function_call>{"name":"NAME","arguments":{...}}</function_call>
_LLAMA_FN_RES = [
    re.compile(r"<function\s*=\s*([\w_.\-]+)\s*>?\s*(\{.*\})\s*</function>", re.DOTALL),
    re.compile(r"<tool_use>\s*<name>([\w_.\-]+)</name>\s*<arguments>(\{.*\})</arguments>\s*</tool_use>", re.DOTALL),
]


def _coerce_args(args: Any) -> dict:
    """Args might come in as a JSON string or a dict — normalize to dict."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _recover_tool_call(failed_generation: str) -> dict | None:
    """Try to parse a malformed tool call from Llama's text-format output.

    Returns {"name": str, "arguments": dict} on success, or None.
    """
    if not failed_generation:
        return None
    text = failed_generation.strip()

    # Format 1 & 2: regex-based extractors for Llama's text-tag formats
    for pattern in _LLAMA_FN_RES:
        m = pattern.search(text)
        if not m:
            continue
        try:
            args = json.loads(m.group(2))
        except json.JSONDecodeError:
            continue
        return {"name": m.group(1), "arguments": _coerce_args(args)}

    # Format 3: raw JSON {"name":..., "arguments":...} or {"function":...}
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or (obj.get("function") or {}).get("name")
    args = obj.get("arguments") or obj.get("parameters") or (obj.get("function") or {}).get("arguments") or {}
    if name:
        return {"name": name, "arguments": _coerce_args(args)}
    return None


# --- Trace events for UI streaming ---

@dataclass
class TraceEvent:
    """A single visible event during an agent turn."""

    kind: str            # 'text' | 'tool_call' | 'tool_result' | 'usage' | 'error'
    payload: Any         # shape depends on kind


@dataclass
class AgentTurnResult:
    """Final result of running one user query through the loop."""

    final_text: str
    events: list[TraceEvent] = field(default_factory=list)
    turns: int = 0
    usage: dict[str, int] = field(default_factory=dict)


class Agent:
    """The UAE copilot agent backed by Groq.

    Single-turn interface: `run(user_message)` returns the final answer plus a
    trace. For multi-turn chat, instantiate once and call `run` repeatedly —
    the conversation history is kept on the instance.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        on_event: Callable[[TraceEvent], None] | None = None,
    ):
        self.settings = settings or get_settings()
        if not self.settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
                "and add it to .env."
            )
        self.client = groq.Groq(api_key=self.settings.groq_api_key)
        self.ctx: ToolContext = build_context(self.settings)
        # System message persists for the lifetime of the conversation
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.on_event = on_event or (lambda _e: None)

    # --- Public API ---

    def reset(self) -> None:
        """Clear conversation history (keeps the system prompt)."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def run(self, user_message: str) -> AgentTurnResult:
        """Run one user turn end-to-end, looping over tool calls until the model
        produces a final answer."""
        self.messages.append({"role": "user", "content": user_message})
        events: list[TraceEvent] = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        final_text = ""
        turns = 0

        for turn in range(self.settings.max_agent_turns):
            turns = turn + 1
            try:
                response = self._call_model()
            except groq.BadRequestError as e:
                # Recover from Llama's occasional malformed tool-call format
                body = getattr(e, "body", None) or {}
                err_info = body.get("error", {}) if isinstance(body, dict) else {}
                if err_info.get("code") == "tool_use_failed":
                    recovered = _recover_tool_call(err_info.get("failed_generation", ""))
                    if recovered:
                        logger.warning(
                            "Recovered malformed tool call: %s(%s)",
                            recovered["name"], recovered["arguments"],
                        )
                        self._inject_recovered_call(
                            recovered, turn_index=turn, events=events
                        )
                        continue  # loop back so the model sees the tool result
                err = TraceEvent("error", {"status": e.status_code, "message": str(e)})
                events.append(err)
                self.on_event(err)
                raise
            except groq.APIStatusError as e:
                err = TraceEvent("error", {"status": e.status_code, "message": str(e)})
                events.append(err)
                self.on_event(err)
                raise

            # Track usage
            if response.usage:
                total_usage["prompt_tokens"] += response.usage.prompt_tokens or 0
                total_usage["completion_tokens"] += response.usage.completion_tokens or 0
                total_usage["total_tokens"] += response.usage.total_tokens or 0
                usage_event = TraceEvent("usage", dict(total_usage))
                events.append(usage_event)
                self.on_event(usage_event)

            choice = response.choices[0]
            msg = choice.message

            # Append the assistant message verbatim — Groq returns a SDK object;
            # we re-shape to a plain dict so the conversation history is JSON-serializable.
            assistant_dict: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
            }
            if msg.tool_calls:
                assistant_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            self.messages.append(assistant_dict)

            # Emit any text the model produced this turn
            if msg.content:
                ev = TraceEvent("text", msg.content)
                events.append(ev)
                self.on_event(ev)

            # If no tool calls, we're done
            if not msg.tool_calls:
                final_text = (msg.content or "").strip()
                # Honour OpenAI-style finish reasons for visibility
                if choice.finish_reason == "length":
                    final_text += "\n\n[Response truncated — hit max_tokens limit.]"
                break

            # Execute every tool call and append one `tool` message per result
            for tc in msg.tool_calls:
                name = tc.function.name
                # Llama sometimes returns malformed JSON for tool arguments — handle it
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError as e:
                    args = {}
                    error_payload = json.dumps(
                        {"error": f"Could not parse tool arguments as JSON: {e}. Raw: {tc.function.arguments!r}"}
                    )
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": error_payload,
                        }
                    )
                    bad_call = TraceEvent("tool_call", {"name": name, "input": args, "parse_error": True})
                    events.append(bad_call)
                    self.on_event(bad_call)
                    bad_result = TraceEvent(
                        "tool_result",
                        {"name": name, "tool_use_id": tc.id, "result": error_payload},
                    )
                    events.append(bad_result)
                    self.on_event(bad_result)
                    continue

                call_event = TraceEvent("tool_call", {"name": name, "input": args})
                events.append(call_event)
                self.on_event(call_event)

                result_str = dispatch(self.ctx, name, args)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    }
                )
                result_event = TraceEvent(
                    "tool_result",
                    {"name": name, "tool_use_id": tc.id, "result": result_str},
                )
                events.append(result_event)
                self.on_event(result_event)
            # Loop back to let the model react to the tool results
        else:
            final_text = (
                f"[Agent loop exceeded {self.settings.max_agent_turns} turns without "
                "producing a final answer. This usually means the agent is stuck in a "
                "tool-calling loop — try rephrasing the question.]"
            )

        return AgentTurnResult(
            final_text=final_text,
            events=events,
            turns=turns,
            usage=total_usage,
        )

    # --- Internals ---

    def _call_model(self):
        """One chat-completions call to Groq."""
        return self.client.chat.completions.create(
            model=self.settings.model,
            messages=self.messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
        )

    def _inject_recovered_call(
        self,
        recovered: dict,
        turn_index: int,
        events: list[TraceEvent],
    ) -> None:
        """Materialize a recovered tool call into the conversation as if the
        model had produced a proper structured tool_call."""
        synthetic_id = f"call_recovered_{turn_index}"
        name = recovered["name"]
        args = recovered["arguments"]

        # Append a synthetic assistant turn with a proper tool_call
        self.messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": synthetic_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args),
                        },
                    }
                ],
            }
        )
        call_event = TraceEvent(
            "tool_call", {"name": name, "input": args, "recovered": True}
        )
        events.append(call_event)
        self.on_event(call_event)

        # Execute the tool and append its result
        result_str = dispatch(self.ctx, name, args)
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": synthetic_id,
                "content": result_str,
            }
        )
        result_event = TraceEvent(
            "tool_result",
            {"name": name, "tool_use_id": synthetic_id, "result": result_str},
        )
        events.append(result_event)
        self.on_event(result_event)
