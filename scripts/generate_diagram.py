"""Render the project pipeline as a portrait-orientation PNG suitable for LinkedIn.

Output: pipeline.png in the project root.
Usage:  python scripts/generate_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


STAGES = [
    (
        "1.  DATA SOURCES",
        "#1E40AF",
        [
            "World Bank Open Data API",
            "IMF DataMapper",
            "Internal warehouse: dbt schemas / LookML / Cube",
        ],
    ),
    (
        "2.  INGESTION",
        "#2563EB",
        [
            "Pull indicators, observations, and topics",
            "Retry with exponential backoff",
            "Save raw JSON snapshots per source",
        ],
    ),
    (
        "3.  KNOWLEDGE BUILD",
        "#3B82F6",
        [
            "Metric dictionary (definitions + methodology)",
            "Cross-source lineage graph",
            "Topic catalog (the 'dashboard' list)",
        ],
    ),
    (
        "4.  STORAGE",
        "#0EA5E9",
        [
            "DuckDB warehouse — indicators + observations",
            "ChromaDB vector store — definition embeddings",
        ],
    ),
    (
        "5.  AGENT LAYER",
        "#0891B2",
        [
            "LLM:  Ollama (local)  ·  Claude  ·  Groq",
            "7 tools: search · get · compare · SQL · lineage",
            "Reasoning loop + citation tracking + recovery",
        ],
    ),
    (
        "6.  DASHBOARD INTEGRATION",
        "#0D9488",
        [
            "Streamlit chat UI",
            "REST API (FastAPI wrapper)",
            "Embedded chatbot in Power BI / Tableau / Looker",
        ],
    ),
]


def render(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 13), dpi=150)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 150)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Header title
    ax.text(
        50, 145, "UAE Analytics Copilot",
        ha="center", va="center",
        fontsize=22, fontweight="bold", color="#0F172A",
    )
    ax.text(
        50, 140, "Pipeline:  data selection  →  dashboard integration",
        ha="center", va="center",
        fontsize=11, color="#475569",
    )

    box_w = 90
    box_h = 17
    x_c = 50
    y_top = 132
    gap = 3.5

    for i, (title, color, body) in enumerate(STAGES):
        y_box_top = y_top - i * (box_h + gap)
        y_box_bot = y_box_top - box_h

        # White rounded card
        card = FancyBboxPatch(
            (x_c - box_w / 2, y_box_bot),
            box_w, box_h,
            boxstyle="round,pad=0,rounding_size=1.5",
            edgecolor="#E2E8F0",
            facecolor="white",
            linewidth=1.5,
        )
        ax.add_patch(card)

        # Colored left accent bar
        accent = Rectangle(
            (x_c - box_w / 2 + 0.4, y_box_bot + 1),
            1.2, box_h - 2,
            facecolor=color, edgecolor="none",
        )
        ax.add_patch(accent)

        # Title (colored, bold)
        ax.text(
            x_c - box_w / 2 + 4, y_box_top - 3,
            title,
            ha="left", va="top",
            fontsize=12.5, fontweight="bold", color=color,
        )

        # Body bullets
        for j, bullet in enumerate(body):
            ax.text(
                x_c - box_w / 2 + 4.5, y_box_top - 7.2 - j * 3.1,
                f"•  {bullet}",
                ha="left", va="top",
                fontsize=9.8, color="#334155",
            )

        # Arrow to next stage
        if i < len(STAGES) - 1:
            arrow = FancyArrowPatch(
                (x_c, y_box_bot),
                (x_c, y_box_bot - gap + 0.4),
                arrowstyle="-|>,head_width=4,head_length=5",
                mutation_scale=12,
                color="#94A3B8",
                linewidth=2.5,
            )
            ax.add_patch(arrow)

    # Footer
    ax.text(
        50, 3, "github.com/AbdullahMoustafa/uae-analytics-copilot",
        ha="center", va="center",
        fontsize=9, color="#64748B",
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "pipeline.png"
    render(out)
