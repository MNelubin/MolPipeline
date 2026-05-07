"""Tests for standalone research workspace source citation metadata."""

from __future__ import annotations

from unittest.mock import patch

from ..models.research import WebSource
from ..research_workspace import run_research_workspace


def test_research_workspace_adds_stable_citation_metadata():
    sources = [
        WebSource(
            url="https://example.org/materials/pvc",
            title="PVC material overview",
            snippet="PVC profiles use stabilizers and fillers.",
            source_type="web",
        )
    ]

    with patch("mvp.research_workspace.formulate_search_queries") as mock_queries, \
         patch("mvp.research_workspace.search_all", return_value=sources), \
         patch("mvp.research_workspace.fetch_and_extract", return_value="PVC profiles use stabilizers and fillers."), \
         patch("mvp.research_workspace.extract_molecules_from_text", return_value=[]), \
         patch("mvp.research_workspace.resolve_candidates", return_value=[]), \
         patch("mvp.research_workspace._optional_rag_results", return_value=[]), \
         patch("mvp.research_workspace._analyze_research_evidence", return_value={}):
        mock_queries.return_value.search_queries = ["PVC window composition"]
        mock_queries.return_value.interpreted_intent = "PVC window composition"
        result = run_research_workspace("PVC window composition", max_sources=1)

    assert result["sources"][0]["citation_id"] == "S1"
    assert result["sources"][0]["domain"] == "example.org"
    assert result["sources"][0]["citation_markdown"] == "[S1](https://example.org/materials/pvc)"
    assert result["evidence"][0]["citation_id"] == "S1"
    assert result["citations"] == result["sources"]
