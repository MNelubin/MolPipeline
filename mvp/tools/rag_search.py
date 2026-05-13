"""LangChain-compatible tools for the Literature RAG agent.

Three tools:
  - literature_vector_search  – semantic search via SPECTER2 embeddings
  - literature_keyword_search – BM25 keyword search for exact terms
  - citation_extractor        – extract a specific fact from a chunk with citation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..rag.models import RetrievalResult, SectionType
from ..rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)

_DEFAULT_VECTORDB = Path("data/vectordb/literature")
_DEFAULT_TRACKING = Path("data/literature_tracking.db")

_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever  # noqa: PLW0603
    if _retriever is None:
        _retriever = HybridRetriever(
            vectordb_dir=_DEFAULT_VECTORDB,
            tracking_db_path=_DEFAULT_TRACKING,
        )
    return _retriever


def set_retriever(retriever: HybridRetriever) -> None:
    """Allow dependency injection for testing or custom configuration."""
    global _retriever  # noqa: PLW0603
    _retriever = retriever


# ---------------------------------------------------------------------------
# Tool I/O models
# ---------------------------------------------------------------------------

class VectorSearchInput(BaseModel):
    query: str = Field(..., description="Natural-language query about a reaction, molecule, or synthesis method")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")
    section_filter: str | None = Field(
        None,
        description="Optional section filter: abstract, experimental, results, discussion, claims",
    )


class KeywordSearchInput(BaseModel):
    query: str = Field(..., description="Keywords to search for (reagent names, IUPAC names, CAS numbers, reaction names)")
    top_k: int = Field(5, ge=1, le=20)


class CitationExtractorInput(BaseModel):
    text: str = Field(..., description="Text chunk from a literature search result")
    question: str = Field(..., description="Specific question to answer from the text")
    doi: str = Field("", description="DOI of the source document")
    title: str = Field("", description="Title of the source document")


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def _format_result(r: RetrievalResult, idx: int) -> dict[str, Any]:
    """Convert a RetrievalResult to a serialisable dict for the agent."""
    return {
        "rank": idx + 1,
        "score": round(r.score, 4),
        "title": r.title,
        "doi": r.doi,
        "year": r.year,
        "section": r.section.value,
        "source": r.source.value,
        "child_text": r.child_text[:1000],
        "parent_text": r.parent_text[:3000] if r.parent_text else "",
        "authors": r.authors[:5],
    }


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def literature_vector_search(params: VectorSearchInput) -> list[dict[str, Any]]:
    """Semantic search over scientific articles and patents using SPECTER2 embeddings.

    Best for: conceptual queries like "how to perform Suzuki coupling with
    electron-poor substrates" or "asymmetric aldol reaction conditions".
    """
    retriever = _get_retriever()
    section = None
    if params.section_filter:
        try:
            section = SectionType(params.section_filter)
        except ValueError:
            pass

    results = retriever.search(
        query=params.query,
        top_k=params.top_k,
        section_filter=section,
    )
    logger.info("Vector search '%s' → %d results", params.query[:60], len(results))
    return [_format_result(r, i) for i, r in enumerate(results)]


def literature_keyword_search(params: KeywordSearchInput) -> list[dict[str, Any]]:
    """BM25 keyword search over scientific articles and patents.

    Best for: exact chemical names, reagent names (e.g. "Pd(PPh3)4"),
    IUPAC names, CAS numbers, specific reaction names ("Suzuki-Miyaura").
    """
    retriever = _get_retriever()
    results = retriever.keyword_search(query=params.query, top_k=params.top_k)
    logger.info("Keyword search '%s' → %d results", params.query[:60], len(results))
    return [_format_result(r, i) for i, r in enumerate(results)]


# ---------------------------------------------------------------------------
# Convenience wrappers for cascade fallback (used by experiment_planner_node)
# ---------------------------------------------------------------------------

def search_synthesis_procedures(
    reaction_smiles: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Search literature for synthesis procedures matching a reaction.

    Fallback #1 in the cascade: ORD -> RAG -> Research.
    Returns list of dicts with keys: text, title, doi, citation.
    """
    try:
        results = literature_vector_search(VectorSearchInput(
            query=f"synthesis procedure for reaction {reaction_smiles}",
            top_k=top_k,
            section_filter="experimental",
        ))
        procedures = []
        for r in results:
            text = r.get("parent_text") or r.get("child_text", "")
            if text and len(text) > 50:
                procedures.append({
                    "text": text[:2000],
                    "title": r.get("title", ""),
                    "doi": r.get("doi", ""),
                    "citation": f"[{r.get('title', 'Source')}] (DOI: {r.get('doi', 'N/A')})",
                    "score": r.get("score", 0),
                })
        return procedures
    except Exception as e:
        logger.warning("search_synthesis_procedures failed: %s", e)
        return []


def search_reaction_conditions(
    reaction_type: str,
    reagents: str = "",
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Search literature for reaction conditions and parameters.

    Returns list of dicts with keys: text, title, doi, citation.
    """
    query = f"{reaction_type} reaction conditions"
    if reagents:
        query += f" with {reagents}"
    try:
        results = literature_vector_search(VectorSearchInput(
            query=query,
            top_k=top_k,
        ))
        conditions = []
        for r in results:
            text = r.get("parent_text") or r.get("child_text", "")
            if text and len(text) > 30:
                conditions.append({
                    "text": text[:2000],
                    "title": r.get("title", ""),
                    "doi": r.get("doi", ""),
                    "citation": f"[{r.get('title', 'Source')}] (DOI: {r.get('doi', 'N/A')})",
                    "score": r.get("score", 0),
                })
        return conditions
    except Exception as e:
        logger.warning("search_reaction_conditions failed: %s", e)
        return []


def citation_extractor(params: CitationExtractorInput) -> dict[str, Any]:
    """Extract a specific fact from a text chunk and format it with a citation.

    Returns the answer with a properly formatted citation including DOI.
    If the text does not contain an answer, returns a "not found" response.
    """
    text_lower = params.text.lower()
    question_keywords = [w for w in params.question.lower().split() if len(w) > 3]

    relevance_count = sum(1 for kw in question_keywords if kw in text_lower)
    is_relevant = relevance_count >= max(1, len(question_keywords) // 3)

    citation = ""
    if params.doi:
        citation = f"[{params.title or 'Source'}] (DOI: {params.doi})"
    elif params.title:
        citation = f"[{params.title}]"

    if not is_relevant:
        return {
            "found": False,
            "answer": "",
            "citation": citation,
            "relevance_score": relevance_count / max(len(question_keywords), 1),
        }

    excerpt_start = 0
    best_density = 0
    window = 500
    for i in range(0, max(1, len(text_lower) - window), 50):
        segment = text_lower[i : i + window]
        density = sum(1 for kw in question_keywords if kw in segment)
        if density > best_density:
            best_density = density
            excerpt_start = i

    excerpt = params.text[excerpt_start : excerpt_start + window].strip()
    if excerpt_start > 0:
        excerpt = "…" + excerpt
    if excerpt_start + window < len(params.text):
        excerpt = excerpt + "…"

    return {
        "found": True,
        "answer": excerpt,
        "citation": citation,
        "relevance_score": relevance_count / max(len(question_keywords), 1),
    }
