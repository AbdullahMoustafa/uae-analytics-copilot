"""IMF DataMapper API ingestor.

IMF publishes overlapping economic indicators to the World Bank but with its own
methodology and (often slightly different) values. This is the key second source
that makes the `compare_sources` agent tool useful.

API docs: https://www.imf.org/external/datamapper/api/help
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .base import BaseIngestor, Indicator, Observation, Topic

logger = logging.getLogger(__name__)

BASE_URL = "https://www.imf.org/external/datamapper/api/v1"

# A focused IMF indicator set chosen to overlap WB headline indicators.
# These power the "WB vs IMF" lineage comparisons the agent surfaces.
CURATED_INDICATORS: list[str] = [
    "NGDPD",          # GDP, current prices (US$ billions)
    "NGDP_RPCH",      # Real GDP growth (%)
    "NGDPDPC",        # GDP per capita, current prices
    "PCPIPCH",        # Inflation, average consumer prices (%)
    "PCPIEPCH",       # Inflation, end of period
    "LUR",            # Unemployment rate (%)
    "GGXWDG_NGDP",    # General government gross debt (% GDP)
    "GGXCNL_NGDP",    # General government net lending/borrowing (% GDP)
    "BCA_NGDPD",      # Current account balance (% GDP)
    "LP",             # Population (millions)
]


class IMFIngestor(BaseIngestor):
    source_name = "imf"

    def fetch_indicators(self) -> list[Indicator]:
        """Fetch IMF indicator metadata.

        IMF returns the full catalog in one call; we filter to our curated set.
        """
        logger.info("Fetching IMF indicator catalog")
        try:
            data = self._get_json(f"{BASE_URL}/indicators")
        except Exception as e:
            logger.error("Failed to fetch IMF indicator catalog: %s", e)
            return []

        catalog = data.get("indicators", {})
        indicators: list[Indicator] = []

        for code in CURATED_INDICATORS:
            meta = catalog.get(code)
            if not meta:
                logger.warning("IMF indicator %s not in catalog", code)
                continue

            indicators.append(
                Indicator(
                    source=self.source_name,
                    source_code=code,
                    name=meta.get("label", code),
                    description=(meta.get("description") or "").strip(),
                    unit=meta.get("unit", "") or "",
                    topic=self._infer_topic(code),
                    source_organization=meta.get("source", "International Monetary Fund"),
                    periodicity="annual",
                    extras={"dataset": meta.get("dataset", "")},
                )
            )

        self._save_raw("indicators.json", [self._to_dict(i) for i in indicators])
        logger.info("Saved %d IMF indicators", len(indicators))
        return indicators

    def fetch_observations(self, indicators: list[Indicator]) -> list[Observation]:
        """Fetch UAE time series for each IMF indicator."""
        observations: list[Observation] = []
        logger.info("Fetching IMF observations for %d indicators", len(indicators))

        for ind in indicators:
            try:
                data = self._get_json(f"{BASE_URL}/{ind.source_code}/{self.country_iso3}")
            except Exception as e:
                logger.warning("Skipping IMF %s observations (%s)", ind.source_code, e)
                continue

            series = data.get("values", {}).get(ind.source_code, {}).get(self.country_iso3, {})
            for year_str, value in series.items():
                try:
                    year = int(year_str)
                except (ValueError, TypeError):
                    continue
                observations.append(
                    Observation(
                        source=self.source_name,
                        source_code=ind.source_code,
                        country_iso3=self.country_iso3,
                        year=year,
                        value=float(value) if value is not None else None,
                    )
                )

        self._save_raw(
            "observations.json",
            [
                {
                    "source": o.source,
                    "source_code": o.source_code,
                    "country_iso3": o.country_iso3,
                    "year": o.year,
                    "value": o.value,
                }
                for o in observations
            ],
        )
        logger.info("Saved %d IMF observations", len(observations))
        return observations

    def fetch_topics(self) -> list[Topic]:
        """IMF doesn't expose a topic taxonomy — we synthesize one from our indicator groupings."""
        topics_seen: set[str] = set()
        topics: list[Topic] = []
        for code in CURATED_INDICATORS:
            t_name = self._infer_topic(code)
            if t_name in topics_seen:
                continue
            topics_seen.add(t_name)
            topics.append(
                Topic(
                    source=self.source_name,
                    topic_id=t_name.lower().replace(" ", "_"),
                    name=t_name,
                    description=f"IMF indicators classified under {t_name}.",
                )
            )

        self._save_raw(
            "topics.json",
            [{"source": t.source, "topic_id": t.topic_id, "name": t.name, "description": t.description} for t in topics],
        )
        return topics

    @staticmethod
    def _infer_topic(code: str) -> str:
        """Map IMF codes to readable topic names."""
        if code.startswith("NGDP") or code in {"BCA_NGDPD"}:
            return "Economy & Growth"
        if code.startswith("PCPI"):
            return "Inflation & Prices"
        if code == "LUR":
            return "Labor & Employment"
        if code.startswith("GG"):
            return "Government Finance"
        if code == "LP":
            return "Population & Demographics"
        return "Other"

    def _save_raw(self, filename: str, payload: list) -> None:
        out_dir = self.raw_dir / self.source_name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / filename).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @staticmethod
    def _to_dict(i: Indicator) -> dict:
        return {
            "source": i.source,
            "source_code": i.source_code,
            "name": i.name,
            "description": i.description,
            "unit": i.unit,
            "topic": i.topic,
            "source_organization": i.source_organization,
            "periodicity": i.periodicity,
        }
