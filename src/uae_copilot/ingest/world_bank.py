"""World Bank Open Data ingestor.

Pulls a curated subset of indicators most relevant to UAE economic analysis.
The full WB catalog is ~1,500 indicators — we narrow to ~100 high-signal ones
across major topics to keep the agent's surface area focused and the index
fast to build.

API docs: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .base import BaseIngestor, Indicator, Observation, Topic

logger = logging.getLogger(__name__)

BASE_URL = "https://api.worldbank.org/v2"

# Curated indicator codes by topic. These were chosen for UAE relevance,
# data coverage (most have 30+ years of observations), and cross-source
# overlap with IMF/UN so the `compare_sources` tool has signal.
CURATED_INDICATORS: dict[str, list[str]] = {
    "Economy & Growth": [
        "NY.GDP.MKTP.CD",      # GDP (current US$)
        "NY.GDP.MKTP.KD.ZG",   # GDP growth (annual %)
        "NY.GDP.PCAP.CD",      # GDP per capita (current US$)
        "NY.GDP.PCAP.KD.ZG",   # GDP per capita growth
        "NV.IND.TOTL.ZS",      # Industry value added (% GDP)
        "NV.SRV.TOTL.ZS",      # Services value added (% GDP)
        "NV.AGR.TOTL.ZS",      # Agriculture value added (% GDP)
        "NE.EXP.GNFS.ZS",      # Exports of goods & services (% GDP)
        "NE.IMP.GNFS.ZS",      # Imports of goods & services (% GDP)
        "BX.KLT.DINV.WD.GD.ZS", # FDI net inflows (% GDP)
    ],
    "Inflation & Prices": [
        "FP.CPI.TOTL.ZG",      # Inflation, CPI (annual %)
        "NY.GDP.DEFL.KD.ZG",   # Inflation, GDP deflator (annual %)
    ],
    "Population & Demographics": [
        "SP.POP.TOTL",         # Population total
        "SP.POP.GROW",         # Population growth (annual %)
        "SP.URB.TOTL.IN.ZS",   # Urban population (% of total)
        "SP.DYN.LE00.IN",      # Life expectancy at birth
        "SP.DYN.TFRT.IN",      # Fertility rate
    ],
    "Labor & Employment": [
        "SL.UEM.TOTL.ZS",      # Unemployment (% labor force)
        "SL.TLF.CACT.ZS",      # Labor force participation rate
        "SL.EMP.TOTL.SP.ZS",   # Employment to population ratio
    ],
    "Trade": [
        "TX.VAL.MRCH.CD.WT",   # Merchandise exports (current US$)
        "TM.VAL.MRCH.CD.WT",   # Merchandise imports (current US$)
        "BN.CAB.XOKA.GD.ZS",   # Current account balance (% GDP)
    ],
    "Energy & Environment": [
        "EG.USE.PCAP.KG.OE",   # Energy use per capita
        "EG.USE.ELEC.KH.PC",   # Electric power consumption per capita
        "EN.ATM.CO2E.PC",      # CO2 emissions per capita
        "EG.FEC.RNEW.ZS",      # Renewable energy consumption (%)
    ],
    "Health": [
        "SH.XPD.CHEX.GD.ZS",   # Current health expenditure (% GDP)
        "SH.MED.PHYS.ZS",      # Physicians per 1,000 people
    ],
    "Education": [
        "SE.XPD.TOTL.GD.ZS",   # Government expenditure on education (% GDP)
        "SE.ADT.LITR.ZS",      # Literacy rate, adult total (%)
        "SE.TER.ENRR",         # School enrollment, tertiary (% gross)
    ],
    "Financial Sector": [
        "FS.AST.DOMS.GD.ZS",   # Domestic credit by financial sector (% GDP)
        "FR.INR.LEND",         # Lending interest rate (%)
        "FR.INR.DPST",         # Deposit interest rate (%)
    ],
    "Government Finance": [
        "GC.DOD.TOTL.GD.ZS",   # Central govt debt (% GDP)
        "GC.TAX.TOTL.GD.ZS",   # Tax revenue (% GDP)
    ],
}


class WorldBankIngestor(BaseIngestor):
    source_name = "worldbank"

    def fetch_indicators(self) -> list[Indicator]:
        """Fetch metadata for our curated indicator list."""
        indicators: list[Indicator] = []
        all_codes = [(topic, code) for topic, codes in CURATED_INDICATORS.items() for code in codes]
        logger.info("Fetching %d indicator definitions from World Bank", len(all_codes))

        for topic, code in all_codes:
            try:
                data = self._get_json(f"{BASE_URL}/indicator/{code}", params={"format": "json"})
            except Exception as e:
                logger.warning("Skipping %s (%s)", code, e)
                continue

            if not isinstance(data, list) or len(data) < 2 or not data[1]:
                logger.warning("No metadata for %s", code)
                continue

            meta = data[1][0]
            indicators.append(
                Indicator(
                    source=self.source_name,
                    source_code=code,
                    name=meta.get("name", code),
                    description=(meta.get("sourceNote") or "").strip(),
                    unit=meta.get("unit", "") or "",
                    topic=topic,
                    source_organization=meta.get("sourceOrganization", "") or "World Bank",
                    periodicity="annual",
                    extras={"wb_source": meta.get("source", {})},
                )
            )

        self._save_raw("indicators.json", [self._indicator_to_dict(i) for i in indicators])
        logger.info("Saved %d World Bank indicators", len(indicators))
        return indicators

    def fetch_observations(self, indicators: list[Indicator]) -> list[Observation]:
        """Fetch UAE time series for each indicator (all available years)."""
        observations: list[Observation] = []
        logger.info("Fetching observations for %d indicators (country=%s)", len(indicators), self.country_iso3)

        for ind in indicators:
            try:
                # WB caps per_page at 500; UAE histories are < 70 years, one page is enough.
                data = self._get_json(
                    f"{BASE_URL}/country/{self.country_iso3}/indicator/{ind.source_code}",
                    params={"format": "json", "per_page": 500},
                )
            except Exception as e:
                logger.warning("Skipping observations for %s (%s)", ind.source_code, e)
                continue

            if not isinstance(data, list) or len(data) < 2 or not data[1]:
                continue

            for obs in data[1]:
                try:
                    year = int(obs["date"])
                except (KeyError, ValueError, TypeError):
                    continue
                observations.append(
                    Observation(
                        source=self.source_name,
                        source_code=ind.source_code,
                        country_iso3=self.country_iso3,
                        year=year,
                        value=obs.get("value"),
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
        logger.info("Saved %d World Bank observations", len(observations))
        return observations

    def fetch_topics(self) -> list[Topic]:
        """Use the World Bank's published topic catalog as our 'dashboard' list."""
        try:
            data = self._get_json(f"{BASE_URL}/topic", params={"format": "json", "per_page": 100})
        except Exception as e:
            logger.warning("Could not fetch WB topics: %s", e)
            return []

        if not isinstance(data, list) or len(data) < 2:
            return []

        topics = [
            Topic(
                source=self.source_name,
                topic_id=str(t.get("id", "")),
                name=t.get("value", "").strip(),
                description=(t.get("sourceNote") or "").strip(),
            )
            for t in data[1]
            if t.get("value")
        ]
        self._save_raw(
            "topics.json",
            [{"source": t.source, "topic_id": t.topic_id, "name": t.name, "description": t.description} for t in topics],
        )
        logger.info("Saved %d World Bank topics", len(topics))
        return topics

    def _save_raw(self, filename: str, payload: list) -> None:
        out_dir = self.raw_dir / self.source_name
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / filename
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _indicator_to_dict(i: Indicator) -> dict:
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
