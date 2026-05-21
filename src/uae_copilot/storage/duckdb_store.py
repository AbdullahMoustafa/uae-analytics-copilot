"""DuckDB warehouse for indicator metadata and observations.

DuckDB is the right tool here: zero-config, single-file, columnar, fast on
analytical workloads, and the agent can write arbitrary SELECT queries against
it via the `run_sql` tool.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import duckdb

logger = logging.getLogger(__name__)


# Schemas are intentionally simple — one row per (source, source_code) for
# indicators and one row per (source, source_code, country, year) for observations.
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS indicators (
    source              VARCHAR NOT NULL,
    source_code         VARCHAR NOT NULL,
    name                VARCHAR NOT NULL,
    description         VARCHAR,
    unit                VARCHAR,
    topic               VARCHAR,
    source_organization VARCHAR,
    periodicity         VARCHAR,
    PRIMARY KEY (source, source_code)
);

CREATE TABLE IF NOT EXISTS observations (
    source        VARCHAR NOT NULL,
    source_code   VARCHAR NOT NULL,
    country_iso3  VARCHAR NOT NULL,
    year          INTEGER NOT NULL,
    value         DOUBLE,
    PRIMARY KEY (source, source_code, country_iso3, year)
);

CREATE INDEX IF NOT EXISTS idx_obs_source_year ON observations(source, year);
CREATE INDEX IF NOT EXISTS idx_obs_year ON observations(year);
"""


class Warehouse:
    """A thin DuckDB wrapper for ingest and query."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[duckdb.DuckDBPyConnection]:
        conn = duckdb.connect(str(self.path))
        try:
            yield conn
        finally:
            conn.close()

    def initialize(self) -> None:
        """Create tables if they don't exist."""
        with self.connect() as conn:
            conn.execute(SCHEMA_SQL)
        logger.info("Initialized warehouse at %s", self.path)

    # --- Ingest helpers ---

    def load_indicators_from_raw(self, raw_dir: Path) -> int:
        """Bulk-load indicators from each source's raw indicators.json."""
        total = 0
        with self.connect() as conn:
            for source_dir in sorted(raw_dir.iterdir()) if raw_dir.exists() else []:
                if not source_dir.is_dir():
                    continue
                f = source_dir / "indicators.json"
                if not f.exists():
                    continue
                records = json.loads(f.read_text(encoding="utf-8"))
                if not records:
                    continue
                # Use parametrized executemany — DuckDB's read_json can be picky about
                # schemas across sources, and we control the shape here.
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO indicators
                    (source, source_code, name, description, unit, topic, source_organization, periodicity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            r["source"],
                            r["source_code"],
                            r["name"],
                            r.get("description", ""),
                            r.get("unit", ""),
                            r.get("topic", ""),
                            r.get("source_organization", ""),
                            r.get("periodicity", ""),
                        )
                        for r in records
                    ],
                )
                total += len(records)
        logger.info("Loaded %d indicators into warehouse", total)
        return total

    def load_observations_from_raw(self, raw_dir: Path) -> int:
        """Bulk-load observations from each source's raw observations.json."""
        total = 0
        with self.connect() as conn:
            for source_dir in sorted(raw_dir.iterdir()) if raw_dir.exists() else []:
                if not source_dir.is_dir():
                    continue
                f = source_dir / "observations.json"
                if not f.exists():
                    continue
                records = json.loads(f.read_text(encoding="utf-8"))
                if not records:
                    continue
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO observations
                    (source, source_code, country_iso3, year, value)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            r["source"],
                            r["source_code"],
                            r["country_iso3"],
                            int(r["year"]),
                            r.get("value"),
                        )
                        for r in records
                    ],
                )
                total += len(records)
        logger.info("Loaded %d observations into warehouse", total)
        return total

    # --- Read helpers used by the agent's tools ---

    def get_indicator_timeseries(
        self, source: str, source_code: str, country_iso3: str
    ) -> list[dict]:
        """Return [{year, value}] sorted ascending."""
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT year, value
                FROM observations
                WHERE source = ? AND source_code = ? AND country_iso3 = ?
                ORDER BY year
                """,
                [source, source_code, country_iso3],
            ).fetchall()
        return [{"year": r[0], "value": r[1]} for r in rows]

    def get_indicator_meta(self, source: str, source_code: str) -> dict | None:
        """Look up the metadata row for one indicator."""
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT source, source_code, name, description, unit, topic,
                       source_organization, periodicity
                FROM indicators
                WHERE source = ? AND source_code = ?
                """,
                [source, source_code],
            ).fetchone()
        if not row:
            return None
        keys = [
            "source",
            "source_code",
            "name",
            "description",
            "unit",
            "topic",
            "source_organization",
            "periodicity",
        ]
        return dict(zip(keys, row, strict=True))

    def run_select(self, sql: str, max_rows: int = 200) -> dict[str, Any]:
        """Run a read-only SELECT and return rows + column names.

        Safety: we reject anything that isn't a single SELECT statement so the agent
        can't mutate the warehouse through this tool.
        """
        cleaned = sql.strip().rstrip(";")
        if not cleaned:
            raise ValueError("Empty SQL")
        # Allow only single statements
        if ";" in cleaned:
            raise ValueError("Multiple statements not allowed; submit a single SELECT.")
        lowered = cleaned.lower().lstrip()
        if not (lowered.startswith("select") or lowered.startswith("with")):
            raise ValueError("Only SELECT (or WITH ... SELECT) statements are allowed.")
        # Belt and braces: block obvious mutation keywords
        for forbidden in ("insert ", "update ", "delete ", "drop ", "alter ", "create ", "attach "):
            if forbidden in lowered + " ":
                raise ValueError(f"Statement contains forbidden keyword: {forbidden.strip()}")

        with self.connect() as conn:
            cur = conn.execute(cleaned)
            cols = [d[0] for d in (cur.description or [])]
            rows = cur.fetchmany(max_rows)
        return {
            "columns": cols,
            "rows": [list(r) for r in rows],
            "row_count": len(rows),
            "truncated": len(rows) == max_rows,
        }

    def list_tables(self) -> list[str]:
        """Helper for the agent's introspection."""
        with self.connect() as conn:
            return [r[0] for r in conn.execute("SHOW TABLES").fetchall()]

    def describe_table(self, table: str) -> list[dict]:
        """Return column name/type for one table."""
        with self.connect() as conn:
            rows = conn.execute(f"DESCRIBE {table}").fetchall()
        return [{"column": r[0], "type": r[1]} for r in rows]
