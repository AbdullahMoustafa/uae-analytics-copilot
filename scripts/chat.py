"""Entry point: interactive chat with the UAE Copilot agent."""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from uae_copilot.cli import app  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "chat", *sys.argv[1:]]
    app()
