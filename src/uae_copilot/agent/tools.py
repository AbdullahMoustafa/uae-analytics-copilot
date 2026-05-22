"""Tool definitions and dispatcher for the agent loop.

Tools are pure functions over the warehouse, vector store, and knowledge graph.
The dispatcher routes a Claude tool_use block to the right Python function and
returns a JSON-serializable result.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from ..config import Settings
from ..knowledge.catalog import load_catalog
from ..knowledge.lineage import load_lineage, lookup_by_source_code
from ..retrieval.vector_store import DefinitionStore
from ..storage.duckdb_store import Warehouse

logger = logging.getLogger(__name__)


# --- Tool schemas (OpenAI function-calling format, also accepted by Ollama) ---
# Passed to the model via the `tools` parameter on chat.completions.create().

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_definitions",
            "description": (
                "Semantic search over the UAE indicator definition dictionary. "
                "Use this FIRST for any conceptual question ('what does X mean', 'find indicators about Y'). "
                "Returns the top-k matching indicators with name, code, source, topic, unit, and methodology text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language description of the concept or indicator you're looking for.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return. Default 5, max 15.",
                    },
                    "source": {
                        "type": "string",
                        "description": "Optional source filter: 'worldbank' or 'imf'.",
                        "enum": ["worldbank", "imf"],
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_indicator",
            "description": (
                "Fetch the UAE time series for one specific indicator. "
                "Use after you've identified the indicator's source and code (via search_definitions or list_indicators_in_topic). "
                "Returns metadata + year-by-year values."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Data source.",
                        "enum": ["worldbank", "imf"],
                    },
                    "source_code": {
                        "type": "string",
                        "description": "Native source code, e.g. 'NY.GDP.MKTP.CD' (WB) or 'NGDPD' (IMF).",
                    },
                    "start_year": {
                        "type": "integer",
                        "description": "Optional: only return observations from this year onward.",
                    },
                    "end_year": {
                        "type": "integer",
                        "description": "Optional: only return observations up to this year.",
                    },
                },
                "required": ["source", "source_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_sources",
            "description": (
                "Compare values for the SAME concept across multiple sources (e.g. WB vs IMF GDP). "
                "Use this when the user asks 'why does X differ' or wants reconciliation. "
                "Pass either a concept_id (from the lineage graph) or a free-text concept name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "concept": {
                        "type": "string",
                        "description": (
                            "Concept identifier (e.g. 'gdp_current_usd') or natural-language name "
                            "(e.g. 'GDP in current US dollars'). The tool resolves by exact concept_id first, then by name match."
                        ),
                    },
                },
                "required": ["concept"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_topics",
            "description": (
                "List all topics in the catalog with their indicator counts. "
                "Think of these as 'available dashboards' the user can browse."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_indicators_in_topic",
            "description": "Drill into one topic and list every indicator it contains across all sources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic name as returned by list_topics (case-insensitive match).",
                    }
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_lineage",
            "description": (
                "Return the lineage entry for a specific indicator: which other sources publish "
                "the same concept and the curated notes on why they differ."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "enum": ["worldbank", "imf"]},
                    "source_code": {"type": "string"},
                },
                "required": ["source", "source_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": (
                "Run a read-only SQL query against the warehouse for ad-hoc analysis. "
                "Tables: indicators(source, source_code, name, description, unit, topic, source_organization, periodicity); "
                "observations(source, source_code, country_iso3, year, value). "
                "Only SELECT and WITH ... SELECT statements are allowed. Returns up to 200 rows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A single SELECT (or WITH ... SELECT) statement. No semicolons mid-query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


# --- Dispatcher ---

@dataclass
class ToolContext:
    """Shared dependencies the dispatcher passes to every tool."""

    settings: Settings
    warehouse: Warehouse
    vector_store: DefinitionStore
    lineage: list[dict]
    catalog: dict


def build_context(settings: Settings) -> ToolContext:
    """Wire up all the tool dependencies from settings."""
    warehouse = Warehouse(settings.warehouse_path)
    vector_store = DefinitionStore(
        chroma_dir=settings.chroma_dir,
        collection_name=settings.chroma_collection,
        embed_model=settings.embed_model,
    )
    lineage = load_lineage(settings.processed_dir)
    catalog = load_catalog(settings.processed_dir)
    return ToolContext(
        settings=settings,
        warehouse=warehouse,
        vector_store=vector_store,
        lineage=lineage,
        catalog=catalog,
    )


def dispatch(ctx: ToolContext, name: str, input_: dict) -> str:
    """Execute a tool call and return a JSON string for the tool_result block.

    We always return a string (Claude tool results take a string `content`).
    Errors are returned as JSON with an "error" field rather than raised — Claude
    can then adjust its plan.
    """
    try:
        if name == "search_definitions":
            return json.dumps(_search_definitions(ctx, **input_), ensure_ascii=False)
        if name == "get_indicator":
            return json.dumps(_get_indicator(ctx, **input_), ensure_ascii=False)
        if name == "compare_sources":
            return json.dumps(_compare_sources(ctx, **input_), ensure_ascii=False)
        if name == "list_topics":
            return json.dumps(_list_topics(ctx), ensure_ascii=False)
        if name == "list_indicators_in_topic":
            return json.dumps(_list_indicators_in_topic(ctx, **input_), ensure_ascii=False)
        if name == "get_lineage":
            return json.dumps(_get_lineage(ctx, **input_), ensure_ascii=False)
        if name == "run_sql":
            return json.dumps(_run_sql(ctx, **input_), ensure_ascii=False)
        return json.dumps({"error": f"Unknown tool: {name}"})
    except TypeError as e:
        # Wrong/missing argument names
        return json.dumps({"error": f"Invalid arguments to {name}: {e}"})
    except Exception as e:
        logger.exception("Tool %s failed", name)
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


# --- Individual tool implementations ---

def _search_definitions(
    ctx: ToolContext, query: str, k: int = 5, source: str | None = None
) -> dict:
    k = max(1, min(int(k), 15))
    where = {"source": source} if source else None
    hits = ctx.vector_store.search(query=query, k=k, where=where)
    return {
        "query": query,
        "results": [
            {
                "id": h["id"],
                "source": h["metadata"].get("source"),
                "source_code": h["metadata"].get("source_code"),
                "name": h["metadata"].get("name"),
                "topic": h["metadata"].get("topic"),
                "unit": h["metadata"].get("unit"),
                "source_organization": h["metadata"].get("source_organization"),
                "definition": h["text"],
                "similarity_distance": h["distance"],
            }
            for h in hits
        ],
    }


def _get_indicator(
    ctx: ToolContext,
    source: str,
    source_code: str,
    start_year: int | str | None = None,
    end_year: int | str | None = None,
) -> dict:
    meta = ctx.warehouse.get_indicator_meta(source, source_code)
    if not meta:
        return {"error": f"Indicator not found: {source}:{source_code}"}

    # Coerce — Llama sometimes sends ints as strings
    start_year_i = int(start_year) if start_year is not None and str(start_year).strip() else None
    end_year_i = int(end_year) if end_year is not None and str(end_year).strip() else None

    series = ctx.warehouse.get_indicator_timeseries(
        source, source_code, ctx.settings.country_iso3
    )
    if start_year_i is not None:
        series = [s for s in series if s["year"] >= start_year_i]
    if end_year_i is not None:
        series = [s for s in series if s["year"] <= end_year_i]

    # Strip leading/trailing nulls so the agent sees the actual observation range
    while series and series[0]["value"] is None:
        series.pop(0)
    while series and series[-1]["value"] is None:
        series.pop()

    return {
        "indicator": meta,
        "country_iso3": ctx.settings.country_iso3,
        "observation_count": len(series),
        "year_range": [series[0]["year"], series[-1]["year"]] if series else None,
        "series": series,
    }


def _compare_sources(ctx: ToolContext, concept: str) -> dict:
    # Resolve by concept_id first, then by case-insensitive name contains
    concept_l = concept.lower().strip()
    match = next(
        (c for c in ctx.lineage if c["concept_id"] == concept_l),
        None,
    )
    if match is None:
        match = next(
            (c for c in ctx.lineage if concept_l in c["name"].lower()),
            None,
        )
    if match is None:
        # Last resort: substring against description
        match = next(
            (c for c in ctx.lineage if concept_l in c["description"].lower()),
            None,
        )
    if match is None:
        return {
            "error": f"No concept matched '{concept}'. Try one of: "
            + ", ".join(c["concept_id"] for c in ctx.lineage),
        }

    # Pull each source's series
    series_by_source: dict[str, list[dict]] = {}
    for src, code in match["sources"].items():
        series_by_source[src] = ctx.warehouse.get_indicator_timeseries(
            src, code, ctx.settings.country_iso3
        )

    # Build year-aligned comparison
    all_years = sorted(
        {p["year"] for s in series_by_source.values() for p in s if p["value"] is not None}
    )
    aligned = []
    for year in all_years:
        row: dict[str, Any] = {"year": year}
        for src, s in series_by_source.items():
            v = next((p["value"] for p in s if p["year"] == year), None)
            row[src] = v
        aligned.append(row)

    return {
        "concept": {
            "concept_id": match["concept_id"],
            "name": match["name"],
            "description": match["description"],
            "lineage_notes": match["notes"],
        },
        "sources_mapped": match["sources"],
        "aligned_observations": aligned,
        "observation_count": len(aligned),
    }


def _list_topics(ctx: ToolContext) -> dict:
    """Return a compact catalog: name + indicator count + sources only.

    Descriptions are intentionally dropped here — they add ~8K tokens which
    blows past the model's context window on local Ollama runs. The agent can drill in with
    list_indicators_in_topic to see what each topic contains.
    Topics with no ingested indicators are filtered out.
    """
    topics = [
        {
            "name": t["name"],
            "indicator_count": t.get("indicator_count", 0),
            "sources": t.get("sources", []),
        }
        for t in ctx.catalog.get("topics", [])
        if t.get("indicator_count", 0) > 0
    ]
    return {"topic_count": len(topics), "topics": topics}


def _list_indicators_in_topic(ctx: ToolContext, topic: str) -> dict:
    topic_l = topic.lower().strip()
    match = next(
        (t for t in ctx.catalog.get("topics", []) if t["name"].lower() == topic_l),
        None,
    )
    if match is None:
        # Try substring match
        match = next(
            (t for t in ctx.catalog.get("topics", []) if topic_l in t["name"].lower()),
            None,
        )
    if match is None:
        available = [t["name"] for t in ctx.catalog.get("topics", [])]
        return {"error": f"Topic '{topic}' not found", "available_topics": available}
    return {
        "topic": match["name"],
        "description": match.get("description", ""),
        "indicators": match.get("indicators", []),
        "indicator_count": match.get("indicator_count", 0),
    }


def _get_lineage(ctx: ToolContext, source: str, source_code: str) -> dict:
    concept = lookup_by_source_code(ctx.lineage, source, source_code)
    if concept is None:
        return {
            "source": source,
            "source_code": source_code,
            "lineage_found": False,
            "message": (
                "This indicator is not currently mapped to a cross-source concept. "
                "It exists only in the source you queried."
            ),
        }
    return {
        "source": source,
        "source_code": source_code,
        "lineage_found": True,
        "concept": concept,
    }


def _run_sql(ctx: ToolContext, query: str) -> dict:
    result = ctx.warehouse.run_select(query)
    return {
        "query": query,
        "columns": result["columns"],
        "rows": result["rows"],
        "row_count": result["row_count"],
        "truncated": result["truncated"],
    }
