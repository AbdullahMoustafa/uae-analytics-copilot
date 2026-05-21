"""Topic catalog — what we treat as 'dashboards' for the agent's browsing tools."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def build_catalog(raw_dir: Path, processed_dir: Path) -> Path:
    """Merge topics across sources and group indicators under them.

    Produces a 'dashboard catalog' the agent can browse via `list_topics`
    and `list_indicators_in_topic`.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    topics_by_name: dict[str, dict] = {}
    indicators_by_topic: dict[str, list[dict]] = defaultdict(list)

    if not raw_dir.exists():
        logger.warning("Raw dir %s does not exist; catalog will be empty", raw_dir)
        out = processed_dir / "catalog.json"
        out.write_text(json.dumps({"topics": []}, indent=2), encoding="utf-8")
        return out

    for source_dir in raw_dir.iterdir():
        if not source_dir.is_dir():
            continue

        # Collect topics
        topics_file = source_dir / "topics.json"
        if topics_file.exists():
            try:
                for t in json.loads(topics_file.read_text(encoding="utf-8")):
                    key = t["name"].strip()
                    if not key:
                        continue
                    if key not in topics_by_name:
                        topics_by_name[key] = {
                            "name": key,
                            "description": t.get("description", ""),
                            "sources": [],
                        }
                    if t["source"] not in topics_by_name[key]["sources"]:
                        topics_by_name[key]["sources"].append(t["source"])
            except json.JSONDecodeError as e:
                logger.warning("Bad JSON in %s: %s", topics_file, e)

        # Collect indicators and group them under their topic
        indicators_file = source_dir / "indicators.json"
        if indicators_file.exists():
            try:
                for ind in json.loads(indicators_file.read_text(encoding="utf-8")):
                    topic = (ind.get("topic") or "Uncategorized").strip()
                    indicators_by_topic[topic].append(
                        {
                            "source": ind["source"],
                            "source_code": ind["source_code"],
                            "name": ind["name"],
                            "unit": ind.get("unit", ""),
                        }
                    )
                    # Ensure the topic itself shows up in the catalog
                    if topic not in topics_by_name:
                        topics_by_name[topic] = {
                            "name": topic,
                            "description": "",
                            "sources": [ind["source"]],
                        }
            except json.JSONDecodeError as e:
                logger.warning("Bad JSON in %s: %s", indicators_file, e)

    # Materialize the catalog
    catalog = {
        "topics": [
            {
                **info,
                "indicators": indicators_by_topic.get(name, []),
                "indicator_count": len(indicators_by_topic.get(name, [])),
            }
            for name, info in sorted(topics_by_name.items())
        ]
    }

    out = processed_dir / "catalog.json"
    out.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "Wrote catalog with %d topics covering %d indicators to %s",
        len(catalog["topics"]),
        sum(t["indicator_count"] for t in catalog["topics"]),
        out,
    )
    return out


def load_catalog(processed_dir: Path) -> dict:
    """Load the persisted catalog."""
    path = processed_dir / "catalog.json"
    if not path.exists():
        return {"topics": []}
    return json.loads(path.read_text(encoding="utf-8"))
