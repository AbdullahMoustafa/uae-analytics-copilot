"""Configuration for the UAE Copilot, loaded from env vars / .env."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    All fields can be overridden by environment variables with prefix `UAE_COPILOT_`.
    Ollama runs locally and requires no API key.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="UAE_COPILOT_",
        extra="ignore",
    )

    # --- Ollama (local) ---
    ollama_host: str = "http://localhost:11434"
    # qwen2.5:7b is the safe default — needs ~5 GB RAM and runs on most laptops.
    # Override to qwen2.5:14b in .env if you have ~10 GB free RAM for better
    # tool-call accuracy on multi-step queries.
    model: str = "qwen2.5:7b"
    temperature: float = 0.2  # low for analytical / tool-call accuracy
    max_tokens: int = 4096
    num_ctx: int = 8192       # Ollama's default is 2K — bump for our system prompt + tools
    max_agent_turns: int = 10  # safety cap on the agent loop

    # --- Paths ---
    data_dir: Path = Path("data")

    # --- Retrieval ---
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_collection: str = "uae_indicator_definitions"

    # --- UAE-specific ---
    country_iso3: str = "ARE"
    country_iso2: str = "AE"

    # --- Logging ---
    log_level: str = "INFO"

    # --- Derived paths ---
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def warehouse_path(self) -> Path:
        return self.data_dir / "warehouse.duckdb"

    @property
    def ollama_base_url(self) -> str:
        """Base URL for the OpenAI-compatible Ollama endpoint."""
        return self.ollama_host.rstrip("/") + "/v1"

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.processed_dir, self.chroma_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def configure_logging(level: str | None = None) -> None:
    """Set up root logger with a consistent format."""
    lvl = level or get_settings().log_level
    logging.basicConfig(
        level=getattr(logging, lvl.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)-30s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
