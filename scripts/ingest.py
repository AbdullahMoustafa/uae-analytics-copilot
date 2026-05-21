"""Entry point: pull raw data from all configured sources.

Forwards any CLI args to `uae-copilot ingest`. Example:
    python scripts/ingest.py
    python scripts/ingest.py --source worldbank
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable when running this script directly
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from uae_copilot.cli import app  # noqa: E402

if __name__ == "__main__":
    # Prepend the subcommand so Typer dispatches to `ingest`
    sys.argv = [sys.argv[0], "ingest", *sys.argv[1:]]
    app()
