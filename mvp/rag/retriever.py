"""Hybrid retriever: SPECTER2 vector search + BM25 keyword search + Parent Document Retriever.

Architecture:
  1. Vector search  – ChromaDB cosine similarity on child-chunk embeddings
  2. Keyword search  – BM25 on child-chunk texts (in-memory, loaded lazily)
  3. Reciprocal Rank Fusion (RRF) merges both rankings
  4. Parent Lookup    – SQLite fetch of parent chunks for full context
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

from .bm25 import BM25Okapi
from .embeddings import LiteratureEmbedder
from .models import DocumentSource, RetrievalResult, SectionType
from .tracking import TrackingDB

logger = logging.getLogger(__name__)

COLLECTION_NAME = "literature_chunks"
RRF_K = 60  # standard RRF constant


def _rrf_fuse(
    rankings: Sequence[list[tuple[str, float]]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over multiple ranked lists.

    Each ranking is a list of (doc_id, score) pairs, ordered by relevance.
    Returns a merged list sorted by fused score.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _original_score) in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """Search literature via vector similarity + BM25 keywords, return parent context."""

    def __init__(
        self,
        vectordb_dir: str | Path,
        tracking_db_path: str | Path,
        embedder: LiteratureEmbedder | None = None,
        model_name: str = "allenai/specter2_base",
    ) -> None:
        import chromadb

        self._vectordb_dir = Path(vectordb_dir)
        self._client = chromadb.PersistentClient(path=str(self._vectordb_dir))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._tracking = TrackingDB(tracking_db_path)
        self._embedder = embedder
        self._model_name = model_name

        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._bm25_meta: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # BM25 index (lazy, built on first keyword search)
    # ------------------------------------------------------------------
    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return

        logger.info("Building BM25 index from ChromaDB collection …")
        count = self._collection.count()
        if count == 0:
            self._bm25 = BM25Okapi([[""]])
            return

        batch_size = 5000
        all_ids: list[str] = []
        all_tokens: list[list[str]] = []
        all_meta: dict[str, dict] = {}

        offset = 0
        while offset < count:
            result = self._collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            ids = result["ids"]
            docs = result["documents"] or [""] * len(ids)
            metas = result["metadatas"] or [{}] * len(ids)
            for cid, doc_text, meta in zip(ids, docs, metas):
                all_ids.append(cid)
                all_tokens.append(doc_text.lower().split())
                all_meta[cid] = meta  # type: ignore[arg-type]
            offset += batch_size

        self._bm25_ids = all_ids
        self._bm25_meta = all_meta
        self._bm25 = BM25Okapi(all_tokens) if all_tokens else BM25Okapi([[""]])
        logger.info("BM25 index built with %d documents", len(all_ids))

    def _ensure_embedder(self) -> LiteratureEmbedder:
        if self._embedder is None:
            self._embedder = LiteratureEmbedder(model_name=self._model_name)
        return self._embedder

    # ------------------------------------------------------------------
    # Individual search methods
    # ------------------------------------------------------------------
    def _vector_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        embedding = self._ensure_embedder().embed_query(query)
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["distances"],
        )
        ids = results["ids"][0] if results["ids"] else []
        distances = results["distances"][0] if results["distances"] else []
        ranked: list[tuple[str, float]] = []
        for cid, dist in zip(ids, distances):
            score = 1.0 - dist
            ranked.append((cid, score))
        return ranked

    def _keyword_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        self._ensure_bm25()
        assert self._bm25 is not None
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        ranked: list[tuple[str, float]] = []
        for idx, score in indexed:
            if score > 0 and idx < len(self._bm25_ids):
                ranked.append((self._bm25_ids[idx], float(score)))
        return ranked

    # ------------------------------------------------------------------
    # Hybrid search (public API)
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 10,
        vector_weight: float = 0.5,
        keyword_weight: float = 0.5,
        section_filter: SectionType | None = None,
    ) -> list[RetrievalResult]:
        """Run hybrid search and return results with parent context.

        Parameters
        ----------
        query : str
            Natural-language query or chemical term.
        top_k : int
            Number of results to return.
        vector_weight / keyword_weight : float
            Relative weights (only controls how many candidates each method contributes).
        section_filter : SectionType | None
            Optional filter to limit results to a specific section type.
        """
        fetch_k = top_k * 3

        vec_results = self._vector_search(query, fetch_k)
        kw_results = self._keyword_search(query, fetch_k)

        fused = _rrf_fuse([vec_results, kw_results])[:top_k]

        if not fused:
            return []

        child_ids = [cid for cid, _ in fused]
        scores_map = {cid: score for cid, score in fused}

        chroma_result = self._collection.get(
            ids=child_ids,
            include=["documents", "metadatas"],
        )
        child_docs = dict(zip(chroma_result["ids"], chroma_result["documents"] or []))
        child_metas = dict(zip(chroma_result["ids"], chroma_result["metadatas"] or []))

        parent_ids = list({
            m.get("parent_id", "") for m in child_metas.values() if m.get("parent_id")
        })
        parent_map: dict[str, dict] = {}
        if parent_ids:
            rows = self._tracking.get_parent_chunks_batch(parent_ids)
            parent_map = {r["parent_id"]: r for r in rows}

        results: list[RetrievalResult] = []
        for cid in child_ids:
            meta = child_metas.get(cid, {})
            if meta is None:
                meta = {}
            pid = meta.get("parent_id", "")
            parent_row = parent_map.get(pid, {})

            section_val = meta.get("section", "other")
            try:
                section = SectionType(section_val)
            except ValueError:
                section = SectionType.OTHER

            if section_filter and section != section_filter:
                continue

            source_val = meta.get("source", "pmc")
            try:
                source = DocumentSource(source_val)
            except ValueError:
                source = DocumentSource.PMC

            authors_raw = meta.get("authors", [])
            if isinstance(authors_raw, str):
                try:
                    authors_raw = json.loads(authors_raw)
                except (json.JSONDecodeError, TypeError):
                    authors_raw = [authors_raw] if authors_raw else []

            year_val = meta.get("year")
            if isinstance(year_val, str):
                try:
                    year_val = int(year_val) if year_val else None
                except ValueError:
                    year_val = None
            if year_val == 0:
                year_val = None

            results.append(RetrievalResult(
                child_id=cid,
                parent_id=pid,
                doc_id=meta.get("doc_id", ""),
                child_text=child_docs.get(cid, ""),
                parent_text=parent_row.get("text", ""),
                section=section,
                score=scores_map.get(cid, 0.0),
                source=source,
                title=meta.get("title", ""),
                authors=authors_raw if isinstance(authors_raw, list) else [],
                year=year_val,
                doi=meta.get("doi", ""),
            ))

        return results[:top_k]

    # ------------------------------------------------------------------
    # Convenience: vector-only and keyword-only
    # ------------------------------------------------------------------
    def vector_search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Semantic-only search (no BM25)."""
        vec_results = self._vector_search(query, top_k)
        if not vec_results:
            return []
        return self._enrich(vec_results[:top_k])

    def keyword_search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Keyword-only search (BM25, no vectors)."""
        kw_results = self._keyword_search(query, top_k)
        if not kw_results:
            return []
        return self._enrich(kw_results[:top_k])

    def _enrich(self, ranked: list[tuple[str, float]]) -> list[RetrievalResult]:
        """Convert raw ranked list into RetrievalResult objects with parent context."""
        child_ids = [cid for cid, _ in ranked]
        scores_map = dict(ranked)

        chroma_result = self._collection.get(
            ids=child_ids,
            include=["documents", "metadatas"],
        )
        child_docs = dict(zip(chroma_result["ids"], chroma_result["documents"] or []))
        child_metas = dict(zip(chroma_result["ids"], chroma_result["metadatas"] or []))

        parent_ids = list({
            m.get("parent_id", "") for m in child_metas.values() if m and m.get("parent_id")
        })
        parent_map: dict[str, dict] = {}
        if parent_ids:
            rows = self._tracking.get_parent_chunks_batch(parent_ids)
            parent_map = {r["parent_id"]: r for r in rows}

        results: list[RetrievalResult] = []
        for cid in child_ids:
            meta = child_metas.get(cid) or {}
            pid = meta.get("parent_id", "")
            parent_row = parent_map.get(pid, {})

            results.append(RetrievalResult(
                child_id=cid,
                parent_id=pid,
                doc_id=meta.get("doc_id", ""),
                child_text=child_docs.get(cid, ""),
                parent_text=parent_row.get("text", ""),
                score=scores_map.get(cid, 0.0),
                title=meta.get("title", ""),
                doi=meta.get("doi", ""),
            ))
        return results

    def close(self) -> None:
        self._tracking.close()
