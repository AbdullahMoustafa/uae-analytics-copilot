"""Streamlit chat UI for the UAE Copilot.

Launch with:  streamlit run src/uae_copilot/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# When run via `streamlit run`, the project root may not be on sys.path
_here = Path(__file__).resolve()
_root = _here.parent.parent.parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from uae_copilot.agent.agent import Agent, TraceEvent  # noqa: E402
from uae_copilot.config import configure_logging, get_settings  # noqa: E402

st.set_page_config(page_title="UAE Analytics Copilot", page_icon="📊", layout="wide")


@st.cache_resource
def get_agent() -> Agent:
    configure_logging()
    return Agent()


def _render_tool_call(name: str, args: dict, result_str: str) -> None:
    with st.expander(f"🔧  {name}({_brief_args(args)})", expanded=False):
        st.markdown("**Arguments**")
        st.json(args)
        st.markdown("**Result**")
        try:
            st.json(json.loads(result_str))
        except Exception:
            st.code(result_str)


def _brief_args(args: dict) -> str:
    parts = []
    for k, v in (args or {}).items():
        if isinstance(v, str) and len(v) > 40:
            v = v[:37] + "..."
        parts.append(f"{k}={v!r}")
    return ", ".join(parts)


def main() -> None:
    settings = get_settings()

    with st.sidebar:
        st.title("📊 UAE Copilot")
        st.caption("Agentic RAG over UAE economic indicators")
        st.markdown("---")
        st.markdown("**Provider:** `Groq`")
        st.markdown(f"**Model:** `{settings.model}`")
        st.markdown(f"**Country:** `{settings.country_iso3}`")
        st.markdown("---")
        st.markdown("**Try asking:**")
        st.markdown(
            "- What does 'GDP (current US$)' mean?\n"
            "- Show UAE inflation 2010–2024\n"
            "- Why does UAE GDP differ between World Bank and IMF?\n"
            "- What energy indicators do you have?\n"
            "- Compute YoY GDP growth since 2015"
        )
        st.markdown("---")
        if st.button("🔄 Reset conversation"):
            st.session_state.clear()
            st.rerun()

    st.title("UAE Analytics Knowledge Copilot")

    if not settings.groq_api_key:
        st.error(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
            "and add it to your .env file."
        )
        return
    if not settings.warehouse_path.exists():
        st.warning(
            "Data not loaded yet. Run `uae-copilot ingest && uae-copilot index` in your terminal first."
        )
        return

    # State
    if "messages" not in st.session_state:
        st.session_state.messages = []  # display-only history: [{role, text, trace}]
    if "agent" not in st.session_state:
        st.session_state.agent = get_agent()

    # Replay history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])
            for ev in msg.get("trace", []):
                if ev["kind"] == "tool_pair":
                    _render_tool_call(ev["name"], ev["args"], ev["result"])

    # New input
    user_input = st.chat_input("Ask about UAE economic indicators...")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "text": user_input, "trace": []})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        status = st.status("Thinking...", expanded=True)
        trace: list[dict] = []
        pending_call: dict | None = None

        def on_event(ev: TraceEvent) -> None:
            nonlocal pending_call
            if ev.kind == "tool_call":
                pending_call = {"name": ev.payload["name"], "args": ev.payload["input"]}
                status.update(label=f"Calling {ev.payload['name']}...", state="running")
            elif ev.kind == "tool_result" and pending_call is not None:
                trace.append(
                    {
                        "kind": "tool_pair",
                        "name": pending_call["name"],
                        "args": pending_call["args"],
                        "result": ev.payload["result"],
                    }
                )
                pending_call = None

        try:
            result = st.session_state.agent.run(user_input)
        except Exception as e:
            status.update(label="Error", state="error", expanded=True)
            st.error(f"{type(e).__name__}: {e}")
            return

        # Re-emit trace via the callback we registered when the agent was built.
        # Streamlit reruns the script per interaction, so the agent's `on_event`
        # may not match this run's closure. Walk the events ourselves:
        for ev in result.events:
            if ev.kind == "tool_call":
                pending_call = {"name": ev.payload["name"], "args": ev.payload["input"]}
            elif ev.kind == "tool_result" and pending_call is not None:
                trace.append(
                    {
                        "kind": "tool_pair",
                        "name": pending_call["name"],
                        "args": pending_call["args"],
                        "result": ev.payload["result"],
                    }
                )
                pending_call = None

        status.update(label=f"Done in {result.turns} turn(s)", state="complete", expanded=False)
        st.markdown(result.final_text or "_[no response]_")
        for ev in trace:
            _render_tool_call(ev["name"], ev["args"], ev["result"])

        u = result.usage
        st.caption(
            f"turns={result.turns} • prompt={u.get('prompt_tokens', 0)} "
            f"• completion={u.get('completion_tokens', 0)} • total={u.get('total_tokens', 0)}"
        )

        st.session_state.messages.append(
            {"role": "assistant", "text": result.final_text, "trace": trace}
        )


if __name__ == "__main__":
    main()
