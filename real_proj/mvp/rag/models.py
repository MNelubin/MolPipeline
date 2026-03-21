"""Data models for the literature RAG pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentSource(str, Enum):
    PMC = "pmc"
    BIGQUERY_PATENT = "bigquery_patent"
    USPTO = "uspto"
    S2ORC = "s2orc"
    MANUAL = "manual"


class SectionType(str, Enum):
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    EXPERIMENTAL = "experimental"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CLAIMS = "claims"
    DESCRIPTION = "description"
    OTHER = "other"


class IndexingStatus(str, Enum):
    INDEXED = "indexed"
    FAILED = "failed"
    SKIPPED = "skipped"


class LiteratureDocument(BaseModel):
    """A parsed scientific article or patent before chunking."""

    doc_id: str = Field(..., description="DOI or patent publication number")
    source: DocumentSource
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    journal: str = ""
    doi: str = ""
    pmcid: str = ""
    sections: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping section_name -> full text of that section",
    )
    raw_text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParentChunk(BaseModel):
    """Large chunk (~2048 tokens) that provides full context for retrieval."""

    parent_id: str
    doc_id: str
    section: SectionType = SectionType.OTHER
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChildChunk(BaseModel):
    """Small chunk (~512 tokens) indexed in the vector DB for precise search."""

    child_id: str
    parent_id: str
    doc_id: str
    section: SectionType = SectionType.OTHER
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """A single result returned by the hybrid retriever."""

    child_id: str
    parent_id: str
    doc_id: str
    child_text: str
    parent_text: str
    section: SectionType = SectionType.OTHER
    score: float = 0.0
    source: DocumentSource = DocumentSource.PMC
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
