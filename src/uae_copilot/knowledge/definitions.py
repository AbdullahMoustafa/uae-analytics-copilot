"""Indicator definition dictionary — the searchable corpus for RAG.

Each definition becomes one chunk in the vector store. The agent's
`search_definitions` tool reads from here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def build_definition_documents(raw_dir: Path) -> list[dict]:
    """Read raw indicators from every source and produce a flat list of definition docs.

    Each doc has:
      - id: stable UID (e.g. "worldbank:NY.GDP.MKTP.CD")
      - text: the rich, searchable text (name + description + topic + organization)
      - metadata: structured fields for filtering and citation
    """
    docs: list[dict] = []
    sources = [d for d in raw_dir.iterdir() if d.is_dir()] if raw_dir.exists() else []

    for source_dir in sources:
        indicators_file = source_dir / "indicators.json"
        if not indicators_file.exists():
            continue
        try:
            indicators = json.loads(indicators_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            logger.warning("Could not parse %s: %s", indicators_file, e)
            continue

        for ind in indicators:
            doc_text = _format_definition(ind)
            docs.append(
                {
                    "id": f"{ind['source']}:{ind['source_code']}",
                    "text": doc_text,
                    "metadata": {
                        "source": ind["source"],
                        "source_code": ind["source_code"],
                        "name": ind["name"],
                        "topic": ind.get("topic", ""),
                        "unit": ind.get("unit", ""),
                        "source_organization": ind.get("source_organization", ""),
                        "periodicity": ind.get("periodicity", ""),
                    },
                }
            )

    logger.info("Built %d definition documents from %d sources", len(docs), len(sources))
    return docs


def _format_definition(indicator: dict) -> str:
    """Format an indicator dict as the searchable text body for embedding."""
    parts = [
        f"Indicator: {indicator['name']}",
        f"Code: {indicator['source_code']}",
        f"Source: {indicator.get('source_organization') or indicator['source']}",
    ]
    if indicator.get("topic"):
        parts.append(f"Topic: {indicator['topic']}")
    if indicator.get("unit"):
        parts.append(f"Unit: {indicator['unit']}")
    if indicator.get("periodicity"):
        parts.append(f"Periodicity: {indicator['periodicity']}")
    if indicator.get("description"):
        parts.append("")  # blank line before the prose
        parts.append(indicator["description"])
    return "\n".join(parts)
