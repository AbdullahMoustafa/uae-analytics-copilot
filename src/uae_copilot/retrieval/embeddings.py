"""Embedding provider.

Defaults to a small, fast, local sentence-transformers model so the project
runs out of the box with no extra API keys. Swap for a hosted embedding service
(Voyage AI, Cohere, OpenAI) if you want higher recall and don't mind the key.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """sentence-transformers backed embedder."""

    def __init__(self, model_name: str):
        # Import lazily — sentence-transformers is a heavy import (~3s cold start)
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", model_name)
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list[float] vectors."""
        if not texts:
            return []
        vecs = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vecs.tolist()


@lru_cache(maxsize=1)
def get_embedder(model_name: str) -> LocalEmbedder:
    """Module-level cached embedder so we don't reload the model per call."""
    return LocalEmbedder(model_name)
