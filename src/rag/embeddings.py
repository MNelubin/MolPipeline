"""Embedding configuration for RAG."""

from langchain_community.embeddings import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL


def get_embedding_function() -> HuggingFaceEmbeddings:
    """Get the embedding function for vector store."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
