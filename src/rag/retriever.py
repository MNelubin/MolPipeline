"""RAG retriever for searching reactions, procedures, and techniques."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.tools import tool

from src.config import (
    CHROMA_COLLECTION_PROCEDURES,
    CHROMA_COLLECTION_REACTIONS,
    CHROMA_COLLECTION_TECHNIQUES,
    RAG_TOP_K,
    VECTORDB_DIR,
)
from src.rag.embeddings import get_embedding_function

_embeddings = None
_stores: dict[str, Chroma] = {}


def _get_store(collection_name: str) -> Chroma:
    """Get or create a ChromaDB vector store."""
    global _embeddings
    if _embeddings is None:
        _embeddings = get_embedding_function()

    if collection_name not in _stores:
        persist_dir = str(Path(VECTORDB_DIR) / collection_name)
        _stores[collection_name] = Chroma(
            collection_name=collection_name,
            embedding_function=_embeddings,
            persist_directory=persist_dir,
        )
    return _stores[collection_name]


@tool
def reaction_search_rag(query: str, top_k: int = RAG_TOP_K) -> list[dict]:
    """Search for similar reactions in the ORD/USPTO database via RAG.

    Args:
        query: Search query — can be SMILES, reaction description, or conditions
        top_k: Number of results to return
    """
    store = _get_store(CHROMA_COLLECTION_REACTIONS)

    try:
        results = store.similarity_search_with_score(query, k=top_k)
    except Exception as e:
        return [{"error": f"RAG search failed: {str(e)}"}]

    output = []
    for doc, score in results:
        output.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "similarity_score": round(float(score), 4),
        })
    return output


@tool
def procedure_search_rag(query: str, top_k: int = RAG_TOP_K) -> list[dict]:
    """Search for laboratory procedures from ORD that match given reaction conditions.

    Args:
        query: Description of reaction type, conditions, and product properties
        top_k: Number of results to return
    """
    store = _get_store(CHROMA_COLLECTION_PROCEDURES)

    try:
        results = store.similarity_search_with_score(query, k=top_k)
    except Exception as e:
        return [{"error": f"RAG search failed: {str(e)}"}]

    output = []
    for doc, score in results:
        output.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "similarity_score": round(float(score), 4),
        })
    return output


@tool
def technique_lookup(technique_name: str) -> list[dict]:
    """Look up a laboratory technique description (filtration, recrystallization, etc.).

    Args:
        technique_name: Name of the technique to look up
    """
    store = _get_store(CHROMA_COLLECTION_TECHNIQUES)

    try:
        results = store.similarity_search_with_score(technique_name, k=3)
    except Exception as e:
        return [{"error": f"RAG search failed: {str(e)}"}]

    output = []
    for doc, score in results:
        output.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "similarity_score": round(float(score), 4),
        })
    return output
