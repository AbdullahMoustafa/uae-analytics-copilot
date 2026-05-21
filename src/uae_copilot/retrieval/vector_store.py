"""ChromaDB vector store for indicator definitions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from .embeddings import LocalEmbedder, get_embedder

logger = logging.getLogger(__name__)


class DefinitionStore:
    """Persistent vector store over indicator definition documents."""

    def __init__(self, chroma_dir: Path, collection_name: str, embed_model: str):
        chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection_name = collection_name
        self._embedder: LocalEmbedder = get_embedder(embed_model)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Wipe and recreate the collection. Used by the build_index script."""
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Reset collection '%s'", self._collection_name)

    def upsert(self, docs: list[dict]) -> None:
        """Upsert a batch of documents.

        Each doc must have: id, text, metadata.
        Embeddings are computed in-process.
        """
        if not docs:
            logger.info("No documents to upsert")
            return

        ids = [d["id"] for d in docs]
        texts = [d["text"] for d in docs]
        metadatas = [d["metadata"] for d in docs]

        logger.info("Embedding %d documents", len(docs))
        embeddings = self._embedder.embed(texts)

        # Chroma expects scalar metadata values; coerce empty strings/None as needed
        clean_metadatas = [
            {k: (v if v is not None else "") for k, v in m.items()}
            for m in metadatas
        ]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=clean_metadatas,
        )
        logger.info("Upserted %d docs into '%s'", len(docs), self._collection_name)

    def search(
        self,
        query: str,
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict]:
        """Top-k nearest neighbours to `query`."""
        if not query.strip():
            return []
        qvec = self._embedder.embed([query])[0]
        results = self._collection.query(
            query_embeddings=[qvec],
            n_results=k,
            where=where,
        )
        # Flatten Chroma's nested-list result shape
        out: list[dict] = []
        ids = (results.get("ids") or [[]])[0]
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]
        for i, doc_id in enumerate(ids):
            out.append(
                {
                    "id": doc_id,
                    "text": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": dists[i] if i < len(dists) else None,
                }
            )
        return out

    def count(self) -> int:
        return self._collection.count()
