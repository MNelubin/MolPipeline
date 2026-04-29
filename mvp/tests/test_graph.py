"""Tests for graph construction and routing logic.

Tests the LangGraph graph structure and conditional routing functions
based on the current validate_and_guard merged architecture.
"""

from __future__ import annotations

import pytest

from ..graph import build_graph, _after_validate, _after_classify


# ═════════════════════════════════════════════════════════════════════════════
# Graph construction
# ═════════════════════════════════════════════════════════════════════════════


class TestBuildGraph:
    def test_graph_builds_without_error(self):
        graph = build_graph()
        assert graph is not None

    def test_graph_is_compiled(self):
        graph = build_graph()
        assert hasattr(graph, "invoke")

    def test_graph_has_nodes(self):
        graph = build_graph()
        assert hasattr(graph, "nodes") or hasattr(graph, "graph")


# ═════════════════════════════════════════════════════════════════════════════
# Routing: _after_classify
# ═════════════════════════════════════════════════════════════════════════════


class TestAfterClassify:
    def test_molecule_goes_to_validate(self):
        state = {"input_type": "molecule"}
        assert _after_classify(state) == "validate_and_guard"

    def test_research_goes_to_research(self):
        state = {"input_type": "research"}
        assert _after_classify(state) == "research"

    def test_invalid_goes_to_end(self):
        state = {"input_type": "invalid"}
        assert _after_classify(state) == "end"

    def test_missing_input_type_goes_to_end(self):
        state = {}
        assert _after_classify(state) == "end"


# ═════════════════════════════════════════════════════════════════════════════
# Routing: _after_validate
# ═════════════════════════════════════════════════════════════════════════════


class TestAfterValidate:
    def test_found_goes_to_molecule_info(self):
        state = {"validation": {"resolve_status": "found"}}
        assert _after_validate(state) == "molecule_info"

    def test_banned_goes_to_end(self):
        state = {"validation": {"resolve_status": "banned"}}
        assert _after_validate(state) == "end"

    def test_not_found_goes_to_research_fallback(self):
        state = {"validation": {"resolve_status": "not_found"}}
        assert _after_validate(state) == "research_fallback"

    def test_not_found_with_cycle_goes_to_end(self):
        state = {
            "validation": {"resolve_status": "not_found"},
            "cycle_counts": {"validate_research": 1},
        }
        assert _after_validate(state) == "end"

    def test_missing_state_goes_to_research_fallback(self):
        state = {}
        assert _after_validate(state) == "research_fallback"


# ═════════════════════════════════════════════════════════════════════════════
# Integration: pipeline state (requires PubChem + thread_id)
# ═════════════════════════════════════════════════════════════════════════════


class TestPipelineStateIntegrity:
    _CONFIG = {"configurable": {"thread_id": "test-graph-1"}}

    @pytest.mark.integration
    def test_aspirin_smiles_resolves_to_found(self):
        graph = build_graph()
        result = graph.invoke(
            {"query": "CC(=O)Oc1ccccc1C(=O)O"},
            config=self._CONFIG,
        )
        assert result["validation"]["resolve_status"] == "found"
