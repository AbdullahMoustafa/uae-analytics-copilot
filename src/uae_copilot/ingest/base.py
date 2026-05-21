"""Base ingestion abstractions shared across data sources."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Indicator:
    """A normalized indicator definition shared across sources."""

    source: str                       # "worldbank" | "imf" | "un"
    source_code: str                  # native ID, e.g. "NY.GDP.MKTP.CD"
    name: str                         # human-readable name
    description: str                  # methodology / definition note
    unit: str = ""                    # e.g. "current US$"
    topic: str = ""                   # e.g. "Economy & Growth"
    source_organization: str = ""     # publishing agency
    periodicity: str = ""             # "annual" | "quarterly" | ...
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def uid(self) -> str:
        """Stable identifier across the system."""
        return f"{self.source}:{self.source_code}"


@dataclass(slots=True)
class Observation:
    """A single (indicator, country, year) data point."""

    source: str
    source_code: str
    country_iso3: str
    year: int
    value: float | None


@dataclass(slots=True)
class Topic:
    """A topical grouping of indicators (what we treat as a 'dashboard')."""

    source: str
    topic_id: str
    name: str
    description: str = ""


class BaseIngestor(ABC):
    """A source-specific ingestor."""

    source_name: str = "unknown"

    def __init__(self, raw_dir: Path, country_iso3: str = "ARE", country_iso2: str = "AE"):
        self.raw_dir = raw_dir
        self.country_iso3 = country_iso3
        self.country_iso2 = country_iso2
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=60.0),
            headers={"User-Agent": "uae-copilot/0.1 (analytics knowledge agent)"},
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _get_json(self, url: str, params: dict | None = None) -> Any:
        """GET with retry/backoff. Returns parsed JSON."""
        logger.debug("GET %s params=%s", url, params)
        resp = self._client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    @abstractmethod
    def fetch_indicators(self) -> list[Indicator]:
        """Return all indicators we want to ingest from this source."""

    @abstractmethod
    def fetch_observations(self, indicators: list[Indicator]) -> list[Observation]:
        """Return time-series observations for the given indicators."""

    @abstractmethod
    def fetch_topics(self) -> list[Topic]:
        """Return topical groupings (dashboards) from this source."""
