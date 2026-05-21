"""Cross-source lineage: which indicators across sources represent the same concept.

This is the artifact that powers the agent's "why does the number differ?" answers.
The mapping is curated rather than learned — for a real analytics team this is
exactly how a metric-consistency layer gets built (a few hours of analyst work,
then the agent reasons over it for years).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ConceptMapping:
    """A single 'concept' (e.g. 'GDP, current USD') mapped to its source codes."""

    concept_id: str            # internal key, e.g. "gdp_current_usd"
    name: str                  # display name
    description: str           # what the concept actually represents
    sources: dict[str, str] = field(default_factory=dict)  # source -> source_code
    notes: str = ""            # methodology divergence notes


# Curated cross-source concept map. Add a row here when you ingest a new indicator
# in two or more sources and want the agent to be able to reconcile them.
CONCEPT_MAPPINGS: list[ConceptMapping] = [
    ConceptMapping(
        concept_id="gdp_current_usd",
        name="GDP, current US$",
        description=(
            "Gross Domestic Product at current market prices, expressed in US dollars. "
            "Measures the total value of goods and services produced in the economy in a given year."
        ),
        sources={
            "worldbank": "NY.GDP.MKTP.CD",
            "imf": "NGDPD",
        },
        notes=(
            "WB reports in US$ at official exchange rates; IMF NGDPD is also USD but is reported "
            "in billions (WB is in absolute units). Year-by-year values diverge by 1-5% typically, "
            "driven by (a) different vintages of national accounts data, (b) revision timing, "
            "and (c) IMF's WEO projections vs WB's measured series for recent years."
        ),
    ),
    ConceptMapping(
        concept_id="real_gdp_growth",
        name="Real GDP growth (annual %)",
        description="Year-over-year growth rate of real (inflation-adjusted) GDP.",
        sources={
            "worldbank": "NY.GDP.MKTP.KD.ZG",
            "imf": "NGDP_RPCH",
        },
        notes=(
            "Both sources compute YoY % change of constant-price GDP, but base years and "
            "deflator methodologies differ. IMF figures may include their forward projections."
        ),
    ),
    ConceptMapping(
        concept_id="gdp_per_capita_usd",
        name="GDP per capita, current US$",
        description="Nominal GDP divided by mid-year population, in current US dollars.",
        sources={
            "worldbank": "NY.GDP.PCAP.CD",
            "imf": "NGDPDPC",
        },
        notes="Population denominator may differ between sources.",
    ),
    ConceptMapping(
        concept_id="cpi_inflation",
        name="CPI inflation (annual %)",
        description="Annual percentage change in the Consumer Price Index.",
        sources={
            "worldbank": "FP.CPI.TOTL.ZG",
            "imf": "PCPIPCH",
        },
        notes=(
            "WB uses average-of-period CPI; IMF PCPIPCH is also period-average. "
            "PCPIEPCH (end of period) is a separate IMF series and not mapped here."
        ),
    ),
    ConceptMapping(
        concept_id="unemployment_rate",
        name="Unemployment rate (% labor force)",
        description="Share of the labor force that is unemployed but actively seeking work.",
        sources={
            "worldbank": "SL.UEM.TOTL.ZS",
            "imf": "LUR",
        },
        notes=(
            "WB sources from ILO modelled estimates; IMF figures come from country authorities. "
            "Material gaps possible for years where local labor force surveys are sparse."
        ),
    ),
    ConceptMapping(
        concept_id="current_account_pct_gdp",
        name="Current account balance (% GDP)",
        description="Sum of net exports, net primary income, and net secondary income, as a share of GDP.",
        sources={
            "worldbank": "BN.CAB.XOKA.GD.ZS",
            "imf": "BCA_NGDPD",
        },
        notes="Both based on BPM6, but timing of revisions to balance-of-payments data differs.",
    ),
    ConceptMapping(
        concept_id="population_total",
        name="Population, total",
        description="De facto total population (all residents regardless of legal status).",
        sources={
            "worldbank": "SP.POP.TOTL",
            "imf": "LP",
        },
        notes=(
            "WB sources UN World Population Prospects; IMF takes country authorities' figures. "
            "Mid-year estimates may differ by a few hundred thousand for the UAE."
        ),
    ),
    ConceptMapping(
        concept_id="govt_debt_pct_gdp",
        name="General government gross debt (% GDP)",
        description="Total gross debt of the general government as a share of GDP.",
        sources={
            "worldbank": "GC.DOD.TOTL.GD.ZS",  # WB equivalent is central govt; not 1:1
            "imf": "GGXWDG_NGDP",
        },
        notes=(
            "IMPORTANT MISMATCH: WB GC.DOD.TOTL.GD.ZS is *central government* debt; "
            "IMF GGXWDG_NGDP is *general government* (broader: includes states, local, social security). "
            "Use IMF for the headline 'public debt' number; WB's value will be systematically lower."
        ),
    ),
]


def write_lineage(processed_dir: Path) -> Path:
    """Persist the concept mapping as JSON for the agent's lineage tool to load."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    out = processed_dir / "lineage.json"
    payload = [
        {
            "concept_id": c.concept_id,
            "name": c.name,
            "description": c.description,
            "sources": c.sources,
            "notes": c.notes,
        }
        for c in CONCEPT_MAPPINGS
    ]
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote lineage graph with %d concepts to %s", len(payload), out)
    return out


def load_lineage(processed_dir: Path) -> list[dict]:
    """Load the persisted concept mapping."""
    path = processed_dir / "lineage.json"
    if not path.exists():
        logger.warning("Lineage file not found at %s; returning empty list", path)
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def lookup_by_source_code(lineage: list[dict], source: str, source_code: str) -> dict | None:
    """Return the concept that contains (source, source_code), if any."""
    for concept in lineage:
        if concept["sources"].get(source) == source_code:
            return concept
    return None
