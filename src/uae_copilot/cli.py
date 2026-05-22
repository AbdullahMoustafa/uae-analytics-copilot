"""Typer CLI for the UAE Copilot.

Provides three subcommands:
  - ingest   : pull raw data from World Bank + IMF
  - index    : build the vector store and warehouse from raw data
  - chat     : start an interactive chat session with the agent
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table

from .agent.agent import Agent, TraceEvent
from .config import configure_logging, get_settings
from .ingest.imf import IMFIngestor
from .ingest.world_bank import WorldBankIngestor
from .knowledge.catalog import build_catalog
from .knowledge.definitions import build_definition_documents
from .knowledge.lineage import write_lineage
from .retrieval.vector_store import DefinitionStore
from .storage.duckdb_store import Warehouse

app = typer.Typer(add_completion=False, help="UAE Analytics Knowledge Copilot")
console = Console()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@app.command()
def ingest(
    sources: list[str] = typer.Option(
        ["worldbank", "imf"],
        "--source",
        "-s",
        help="Sources to ingest. Repeat flag for multiple.",
    ),
) -> None:
    """Pull raw indicator definitions, observations, and topics from open data sources."""
    configure_logging()
    settings = get_settings()
    settings.ensure_dirs()

    console.print(Panel.fit("[bold cyan]UAE Copilot — Ingest[/bold cyan]"))
    console.print(f"Sources: [yellow]{', '.join(sources)}[/yellow]")
    console.print(f"Output:  [yellow]{settings.raw_dir}[/yellow]\n")

    available = {"worldbank": WorldBankIngestor, "imf": IMFIngestor}
    for src in sources:
        if src not in available:
            console.print(f"[red]Unknown source:[/red] {src}")
            continue
        console.print(f"[bold]> {src}[/bold]")
        cls = available[src]
        with cls(
            raw_dir=settings.raw_dir,
            country_iso3=settings.country_iso3,
            country_iso2=settings.country_iso2,
        ) as ingestor:
            indicators = ingestor.fetch_indicators()
            ingestor.fetch_observations(indicators)
            ingestor.fetch_topics()
        console.print(f"  [green]✓[/green] {src} done\n")

    console.print(
        "[bold green]Done.[/bold green] Next: [cyan]python scripts/build_index.py[/cyan] "
        "(or [cyan]uae-copilot index[/cyan])"
    )


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------

@app.command()
def index() -> None:
    """Build the vector store, DuckDB warehouse, lineage graph, and topic catalog."""
    configure_logging()
    settings = get_settings()
    settings.ensure_dirs()

    console.print(Panel.fit("[bold cyan]UAE Copilot — Build Index[/bold cyan]"))

    # 1. Warehouse
    console.print("[bold]> DuckDB warehouse[/bold]")
    wh = Warehouse(settings.warehouse_path)
    wh.initialize()
    n_ind = wh.load_indicators_from_raw(settings.raw_dir)
    n_obs = wh.load_observations_from_raw(settings.raw_dir)
    console.print(f"  [green]✓[/green] {n_ind} indicators, {n_obs} observations\n")

    # 2. Lineage graph
    console.print("[bold]> Lineage graph[/bold]")
    write_lineage(settings.processed_dir)
    console.print(f"  [green]✓[/green] {settings.processed_dir / 'lineage.json'}\n")

    # 3. Topic catalog
    console.print("[bold]> Topic catalog[/bold]")
    build_catalog(settings.raw_dir, settings.processed_dir)
    console.print(f"  [green]✓[/green] {settings.processed_dir / 'catalog.json'}\n")

    # 4. Vector index
    console.print("[bold]> Vector index (definition embeddings)[/bold]")
    docs = build_definition_documents(settings.raw_dir)
    if not docs:
        console.print("[yellow]No definitions to index. Did you run `ingest` first?[/yellow]")
        raise typer.Exit(code=1)

    store = DefinitionStore(
        chroma_dir=settings.chroma_dir,
        collection_name=settings.chroma_collection,
        embed_model=settings.embed_model,
    )
    store.reset()
    store.upsert(docs)
    console.print(f"  [green]✓[/green] {store.count()} documents in collection '{settings.chroma_collection}'\n")

    console.print(
        "[bold green]Index ready.[/bold green] Start chatting with [cyan]uae-copilot chat[/cyan]"
    )


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------

@app.command()
def chat(
    show_trace: bool = typer.Option(
        True, "--trace/--no-trace", help="Show tool calls and intermediate steps."
    ),
) -> None:
    """Interactive chat with the UAE Copilot agent."""
    configure_logging()
    settings = get_settings()

    # Check Ollama is reachable
    import httpx
    try:
        r = httpx.get(f"{settings.ollama_host}/api/tags", timeout=3.0)
        r.raise_for_status()
        installed_models = {m["name"] for m in r.json().get("models", [])}
    except Exception as e:
        console.print(
            f"[bold red]Cannot reach Ollama at {settings.ollama_host}[/bold red]\n"
            f"  {type(e).__name__}: {e}\n\n"
            "Install Ollama from [link]https://ollama.com/download[/link], then in a separate terminal run:\n"
            "  [cyan]ollama serve[/cyan]"
        )
        raise typer.Exit(code=1)

    if settings.model not in installed_models:
        console.print(
            f"[yellow]Model [bold]{settings.model}[/bold] is not pulled yet.[/yellow]\n"
            f"Run: [cyan]ollama pull {settings.model}[/cyan]"
        )
        if installed_models:
            console.print(f"Installed models: {', '.join(sorted(installed_models))}")
        raise typer.Exit(code=1)

    if not settings.warehouse_path.exists() or not settings.chroma_dir.exists():
        console.print(
            "[yellow]The warehouse or vector index is missing. Run:[/yellow]\n"
            "  [cyan]uae-copilot ingest && uae-copilot index[/cyan]"
        )
        raise typer.Exit(code=1)

    console.print(
        Panel.fit(
            "[bold cyan]UAE Analytics Knowledge Copilot[/bold cyan]\n"
            f"Provider: [yellow]Ollama (local)[/yellow]  •  Model: [yellow]{settings.model}[/yellow]\n"
            "Type your question. /reset to clear history, /quit to exit.",
            border_style="cyan",
        )
    )

    def on_event(ev: TraceEvent) -> None:
        if not show_trace:
            return
        if ev.kind == "tool_call":
            console.print(
                f"  [dim cyan]↪ tool[/dim cyan] [bold]{ev.payload['name']}[/bold]"
                f"({_brief_args(ev.payload['input'])})"
            )
        elif ev.kind == "tool_result":
            try:
                parsed = json.loads(ev.payload["result"])
                if "error" in parsed:
                    console.print(f"  [dim red]   error: {parsed['error']}[/dim red]")
                else:
                    summary = _summarize_result(ev.payload["name"], parsed)
                    if summary:
                        console.print(f"  [dim green]   ↳ {summary}[/dim green]")
            except Exception:
                pass

    agent = Agent(settings=settings, on_event=on_event)

    while True:
        try:
            user_input = Prompt.ask("\n[bold magenta]you[/bold magenta]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]bye[/dim]")
            return

        cmd = user_input.strip().lower()
        if cmd in {"/quit", "/exit", ":q"}:
            console.print("[dim]bye[/dim]")
            return
        if cmd == "/reset":
            agent.reset()
            console.print("[dim]history cleared[/dim]")
            continue
        if not user_input.strip():
            continue

        try:
            result = agent.run(user_input)
        except Exception as e:
            console.print(f"[bold red]Agent error:[/bold red] {type(e).__name__}: {e}")
            continue

        console.print()
        console.print(Panel(Markdown(result.final_text or "[no response]"), title="copilot", border_style="cyan"))
        if show_trace:
            u = result.usage
            console.print(
                f"[dim]turns={result.turns}  "
                f"prompt={u.get('prompt_tokens', 0)}  "
                f"completion={u.get('completion_tokens', 0)}  "
                f"total={u.get('total_tokens', 0)}[/dim]"
            )


# ---------------------------------------------------------------------------
# topics  (helper for inspection without launching the agent)
# ---------------------------------------------------------------------------

@app.command()
def topics() -> None:
    """List topics in the catalog."""
    configure_logging()
    settings = get_settings()
    from .knowledge.catalog import load_catalog

    catalog = load_catalog(settings.processed_dir)
    table = Table(title="Topic catalog")
    table.add_column("Topic", style="cyan")
    table.add_column("Indicators", justify="right", style="green")
    table.add_column("Sources")
    for t in catalog.get("topics", []):
        table.add_row(t["name"], str(t.get("indicator_count", 0)), ", ".join(t.get("sources", [])))
    console.print(table)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _brief_args(args: dict) -> str:
    """Compact one-line view of tool arguments for the trace."""
    parts = []
    for k, v in (args or {}).items():
        if isinstance(v, str) and len(v) > 60:
            v = v[:57] + "..."
        parts.append(f"{k}={v!r}")
    return ", ".join(parts)


def _summarize_result(tool_name: str, payload: dict) -> str:
    """Render a brief, human-friendly summary of a tool result for the trace."""
    if tool_name == "search_definitions":
        n = len(payload.get("results", []))
        return f"{n} hits"
    if tool_name == "get_indicator":
        meta = payload.get("indicator", {})
        n = payload.get("observation_count", 0)
        yr = payload.get("year_range") or [None, None]
        return f"{meta.get('name', '?')} — {n} obs, {yr[0]}..{yr[1]}"
    if tool_name == "compare_sources":
        return f"concept={payload.get('concept', {}).get('name', '?')}, {payload.get('observation_count', 0)} aligned years"
    if tool_name == "list_topics":
        return f"{payload.get('topic_count', 0)} topics"
    if tool_name == "list_indicators_in_topic":
        return f"{payload.get('indicator_count', 0)} indicators in {payload.get('topic', '?')}"
    if tool_name == "get_lineage":
        return "concept mapped" if payload.get("lineage_found") else "no lineage"
    if tool_name == "run_sql":
        return f"{payload.get('row_count', 0)} rows{' (truncated)' if payload.get('truncated') else ''}"
    return ""


if __name__ == "__main__":
    app()
