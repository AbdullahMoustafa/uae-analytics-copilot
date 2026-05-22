"""Manual agentic loop driving Ollama (local) over our tool surface.

We use the OpenAI SDK pointed at Ollama's OpenAI-compatible endpoint at
http://localhost:11434/v1 — no API key required.

We use the manual loop (not a higher-level agent framework) because it gives us:
- Per-turn trace events we can stream to a CLI or UI
- Full control over the conversation history
- Typed exception handling at each turn
- A recovery path for malformed text-format tool calls
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import openai

from ..config import Settings, get_settings
from .prompts import SYSTEM_PROMPT
from .tools import TOOL_SCHEMAS, ToolContext, build_context, dispatch

logger = logging.getLogger(__name__)


# Local LLMs sometimes emit native function-call text instead of structured
# tool_calls. We recover by parsing that text. Patterns seen in the wild:
#   <function=NAME{json}</function>
#   <function=NAME>{json}</function>     ← `>` between name and json
#   <function=NAME{json}></function>     ← `>` between json and closing tag
#   <function=NAME>{json}></function>    ← both
#   <function_call>{"name":"NAME","arguments":{...}}</function_call>
#   <tool_use><name>NAME</name><arguments>{...}</arguments></tool_use>
_TEXT_FN_RES = [
    re.compile(
        r"<function\s*=\s*([\w_.\-:]+)\s*>?\s*(\{.*\})\s*>?\s*</function>",
        re.DOTALL,
    ),
    re.compile(
        r"<tool_use>\s*<name>([\w_.\-:]+)</name>\s*<arguments>(\{.*\})</arguments>\s*</tool_use>",
        re.DOTALL,
    ),
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


def _recover_tool_call(text: str) -> dict | None:
    """Try to parse a malformed tool call from text-format model output.

    Returns {"name": str, "arguments": dict} on success, or None.
    """
    if not text:
        return None
    text = text.strip()

    for pattern in _TEXT_FN_RES:
        m = pattern.search(text)
        if not m:
            continue
        try:
            args = json.loads(m.group(2))
        except json.JSONDecodeError:
            continue
        return {"name": m.group(1), "arguments": _coerce_args(args)}

    # Raw JSON {"name":..., "arguments":...} or {"function":...}
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or (obj.get("function") or {}).get("name")
    args = (
        obj.get("arguments")
        or obj.get("parameters")
        or (obj.get("function") or {}).get("arguments")
        or {}
    )
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
    """The UAE copilot agent backed by a local Ollama model.

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
        # Ollama doesn't require auth — but the OpenAI SDK insists on something
        self.client = openai.OpenAI(
            base_url=self.settings.ollama_base_url,
            api_key="ollama",
        )
        self.ctx: ToolContext = build_context(self.settings)
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
            except openai.APIConnectionError as e:
                msg = (
                    f"Could not reach Ollama at {self.settings.ollama_host}. "
                    "Is Ollama running? Try `ollama serve` in a separate terminal."
                )
                err = TraceEvent("error", {"status": "connection", "message": msg})
                events.append(err)
                self.on_event(err)
                raise RuntimeError(msg) from e
            except openai.NotFoundError as e:
                msg = (
                    f"Model '{self.settings.model}' is not available on Ollama. "
                    f"Pull it first: `ollama pull {self.settings.model}`"
                )
                err = TraceEvent("error", {"status": 404, "message": msg})
                events.append(err)
                self.on_event(err)
                raise RuntimeError(msg) from e
            except openai.BadRequestError as e:
                # Some Ollama models emit text-format tool calls — try to recover
                recovered = self._try_recover_from_bad_request(e)
                if recovered:
                    logger.warning(
                        "Recovered malformed tool call: %s(%s)",
                        recovered["name"], recovered["arguments"],
                    )
                    self._inject_recovered_call(recovered, turn_index=turn, events=events)
                    continue
                err = TraceEvent("error", {"status": e.status_code, "message": str(e)})
                events.append(err)
                self.on_event(err)
                raise
            except openai.APIStatusError as e:
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

            # If the model emitted text-formatted tool calls without using the
            # structured tool_calls field, recover from the content too.
            if not msg.tool_calls and msg.content:
                inline = _recover_tool_call(msg.content)
                if inline:
                    logger.warning(
                        "Recovered inline tool call from content: %s(%s)",
                        inline["name"], inline["arguments"],
                    )
                    self._inject_recovered_call(inline, turn_index=turn, events=events)
                    continue

            # Append the assistant message as a plain dict (JSON-serializable)
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

            if msg.content:
                ev = TraceEvent("text", msg.content)
                events.append(ev)
                self.on_event(ev)

            if not msg.tool_calls:
                final_text = (msg.content or "").strip()
                if choice.finish_reason == "length":
                    final_text += "\n\n[Response truncated — hit max_tokens limit.]"
                break

            # Execute every tool call and append one `tool` message per result
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError as e:
                    args = {}
                    error_payload = json.dumps(
                        {"error": f"Could not parse tool arguments as JSON: {e}. Raw: {tc.function.arguments!r}"}
                    )
                    self.messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": error_payload}
                    )
                    bad_call = TraceEvent(
                        "tool_call", {"name": name, "input": args, "parse_error": True}
                    )
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
                    {"role": "tool", "tool_call_id": tc.id, "content": result_str}
                )
                result_event = TraceEvent(
                    "tool_result",
                    {"name": name, "tool_use_id": tc.id, "result": result_str},
                )
                events.append(result_event)
                self.on_event(result_event)
        else:
            final_text = (
                f"[Agent loop exceeded {self.settings.max_agent_turns} turns without "
                "producing a final answer. This usually means the agent is stuck in a "
                "tool-calling loop — try rephrasing the question or using a stronger model.]"
            )

        return AgentTurnResult(
            final_text=final_text,
            events=events,
            turns=turns,
            usage=total_usage,
        )

    # --- Internals ---

    def _call_model(self):
        """One chat-completions call to the local Ollama server."""
        return self.client.chat.completions.create(
            model=self.settings.model,
            messages=self.messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            # Ollama-specific: bump context window beyond its 2K default
            extra_body={"options": {"num_ctx": self.settings.num_ctx}},
        )

    def _try_recover_from_bad_request(self, exc: openai.BadRequestError) -> dict | None:
        """Some Ollama backends surface tool-format errors as a 400. Try to
        salvage the malformed call from the error body."""
        body = getattr(exc, "body", None) or {}
        err_info = body.get("error", {}) if isinstance(body, dict) else {}
        text = err_info.get("failed_generation") or err_info.get("message") or str(exc)
        return _recover_tool_call(text)

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

        result_str = dispatch(self.ctx, name, args)
        self.messages.append(
            {"role": "tool", "tool_call_id": synthetic_id, "content": result_str}
        )
        result_event = TraceEvent(
            "tool_result",
            {"name": name, "tool_use_id": synthetic_id, "result": result_str},
        )
        events.append(result_event)
        self.on_event(result_event)
