"""Smoke tests — these run without any API calls or network.

They verify the wiring (imports, schemas, dispatcher round-trip on empty data).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))


def test_imports():
    """All modules import cleanly."""
    from uae_copilot import agent, config, ingest, knowledge, retrieval, storage  # noqa: F401
    from uae_copilot.agent import agent as agent_mod, prompts, tools  # noqa: F401
    from uae_copilot.ingest import imf, world_bank  # noqa: F401
    from uae_copilot.knowledge import catalog, definitions, lineage  # noqa: F401
    from uae_copilot.retrieval import embeddings, vector_store  # noqa: F401
    from uae_copilot.storage import duckdb_store  # noqa: F401


def test_tool_schemas_valid():
    """Every tool schema has the fields the Groq / OpenAI function-calling API requires."""
    from uae_copilot.agent.tools import TOOL_SCHEMAS

    assert len(TOOL_SCHEMAS) >= 5
    for t in TOOL_SCHEMAS:
        assert t.get("type") == "function"
        fn = t.get("function", {})
        assert fn.get("name")
        assert fn.get("description")
        params = fn.get("parameters", {})
        assert params.get("type") == "object"
        assert "properties" in params


def test_lineage_writes_and_loads(tmp_path: Path):
    """Lineage graph round-trips through disk."""
    from uae_copilot.knowledge.lineage import load_lineage, lookup_by_source_code, write_lineage

    out = write_lineage(tmp_path)
    assert out.exists()
    loaded = load_lineage(tmp_path)
    assert len(loaded) > 0
    # GDP concept is canonical — should exist with both WB and IMF entries
    concept = lookup_by_source_code(loaded, "worldbank", "NY.GDP.MKTP.CD")
    assert concept is not None
    assert concept["concept_id"] == "gdp_current_usd"
    assert "imf" in concept["sources"]


def test_warehouse_initialize_and_query(tmp_path: Path):
    """Warehouse initializes, accepts a SELECT, rejects mutations."""
    from uae_copilot.storage.duckdb_store import Warehouse

    wh = Warehouse(tmp_path / "test.duckdb")
    wh.initialize()
    assert "indicators" in wh.list_tables()
    assert "observations" in wh.list_tables()

    result = wh.run_select("SELECT 1 AS one, 2 AS two")
    assert result["columns"] == ["one", "two"]
    assert result["rows"] == [[1, 2]]

    import pytest

    with pytest.raises(ValueError):
        wh.run_select("DROP TABLE indicators")
    with pytest.raises(ValueError):
        wh.run_select("INSERT INTO indicators VALUES ('x','y','z','','','','','')")
    with pytest.raises(ValueError):
        wh.run_select("SELECT 1; SELECT 2")


def test_tool_dispatcher_handles_unknown_tool():
    """Dispatcher returns a JSON error for unknown tools instead of crashing."""
    from uae_copilot.agent.tools import dispatch
    from unittest.mock import MagicMock

    fake_ctx = MagicMock()
    result = dispatch(fake_ctx, "no_such_tool", {})
    payload = json.loads(result)
    assert "error" in payload
    assert "no_such_tool" in payload["error"]
