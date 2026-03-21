"""Embedding model wrapper for the literature RAG pipeline.

Supports SPECTER2 (recommended for scientific text) and falls back to
all-MiniLM-L6-v2 if SPECTER2 is unavailable.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "allenai/specter2_base"
FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_BATCH_SIZE = 64


class LiteratureEmbedder:
    """Generate embeddings for scientific text chunks.

    Attempts to load *model_name* via ``sentence_transformers``.  If it fails
    (missing adapter, CUDA OOM, etc.) it falls back to the smaller general
    model automatically.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        self.dimension: int = 0
        self._model = self._load_model(model_name, device)

    def _load_model(self, name: str, device: str | None):  # noqa: ANN202
        from sentence_transformers import SentenceTransformer

        try:
            logger.info("Loading embedding model %s …", name)
            model = SentenceTransformer(name, device=device)
            self.dimension = model.get_sentence_embedding_dimension() or 384
            logger.info("Loaded %s  (dim=%d)", name, self.dimension)
            self.model_name = name
            return model
        except Exception as exc:
            if name == FALLBACK_MODEL:
                raise
            logger.warning("Failed to load %s (%s), trying fallback", name, exc)
            return self._load_model(FALLBACK_MODEL, device)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (N, dim) float32 array of embeddings."""
        return self._model.encode(
            list(texts),
            batch_size=_BATCH_SIZE,
            show_progress_bar=len(texts) > _BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query and return a plain list."""
        vec = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return vec[0].tolist()
