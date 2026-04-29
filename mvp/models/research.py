"""Pydantic models for the Research Intent Agent."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ResearchQuery(BaseModel):
    """Parsed and enriched representation of a vague user request."""

    original_input: str
    interpreted_intent: str
    search_queries: list[str] = Field(default_factory=list)
    language: Literal["ru", "en"] = "en"


class WebSource(BaseModel):
    """A single web source discovered during research."""

    url: str
    title: str
    snippet: str = ""
    source_type: Literal["pubmed", "web", "wikipedia"] = "web"


class CandidateMolecule(BaseModel):
    """A molecule extracted from web sources and resolved via PubChem."""

    name: str
    canonical_smiles: str | None = None
    pubchem_cid: int | None = None
    relevance_reason: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_urls: list[str] = Field(default_factory=list)


class ResearchResult(BaseModel):
    """Final output of the ResearchAgent pipeline."""

    original_query: str
    interpreted_intent: str
    candidates: list[CandidateMolecule] = Field(default_factory=list)
    sources: list[WebSource] = Field(default_factory=list)
    summary: str = ""
    is_successful: bool = False
    error: str | None = None
