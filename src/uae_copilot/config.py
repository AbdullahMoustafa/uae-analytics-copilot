"""Configuration for the UAE Copilot, loaded from env vars / .env."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    All fields can be overridden by environment variables (prefix `UAE_COPILOT_`)
    except the Groq key, which uses the standard `GROQ_API_KEY`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="UAE_COPILOT_",
        extra="ignore",
    )

    # --- Groq API ---
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    # llama-3.3-70b-versatile has the strongest tool calling on Groq's free tier
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.2  # low temperature for analytical / tool-call accuracy
    max_tokens: int = 4096
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
    # Silence noisy 3rd-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
