"""Tests for the retrosynthesis API helpers and endpoints."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from ..api import (
    ResearchAnalyzeRequest,
    RetroAnalyzeRequest,
    RetroSearchRequest,
    _retro_sources_snapshot,
    _run_retro_analyze,
    _run_retro_search,
    research_analyze,
    retro_analyze,
    retro_search,
    retro_sources,
)


class TestRetroSearchHelpers:
    def test_run_retro_search_returns_runtime_summary(self):
        runtime_result = {
            "routes": [{"reactants": "CC=O.O", "source": "aizynthfinder", "final_score": 0.8}],
            "best_route": {"reactants": "CC=O.O", "source": "aizynthfinder", "final_score": 0.8},
            "sources_used": ["ord", "aizynthfinder"],
            "total_found": 4,
            "total_unique": 3,
            "source_counts": {"ord": 2, "aizynthfinder": 2},
            "source_counts_deduped": {"ord": 1, "aizynthfinder": 2},
            "source_mode": "aizynthfinder",
            "source_errors": {},
        }
        with patch("mvp.api._resolve_to_smiles", return_value=("CCO", "smiles")), \
             patch("mvp.api.search_and_rank", return_value=runtime_result), \
             patch("mvp.api._attach_procedure_steps") as mock_attach:
            result = _run_retro_search("ethanol", top_n=3, source_mode="aizynthfinder")

        assert result["smiles"] == "CCO"
        assert result["resolution"] == "smiles"
        assert result["total_found"] == 4
        assert result["total_unique"] == 3
        assert result["source_counts_deduped"]["aizynthfinder"] == 2
        assert result["source_mode"] == "aizynthfinder"
        mock_attach.assert_called_once()

    def test_run_retro_analyze_returns_card_and_routes(self):
        resolved = {
            "validation": {"is_valid": True, "input_type": "name"},
            "smiles": "CCO",
            "pubchem_cid": 702,
        }
        guard_result = {"overall_status": "SAFE", "molecule_check": {}, "reaction_check": {}, "safety_data": {}}
        retro_result = {"routes": [{"reactants": "CC=O.O", "source": "ord"}], "source_mode": "ord", "source_errors": {}}
        molecule_state = {"molecule_info": {"name": "ethanol", "smiles": "CCO"}}

        with patch("mvp.api._attach_procedure_steps") as mock_attach, \
             patch("mvp.nodes.validate_and_guard_node._resolve_molecule", return_value=resolved), \
             patch("mvp.nodes.validate_and_guard_node._run_safety_checks", return_value=guard_result), \
             patch("mvp.nodes.molecule_info_node.molecule_info_node", return_value=molecule_state), \
             patch("mvp.api.search_and_rank", return_value=retro_result):
            result = _run_retro_analyze("ethanol", top_n=5, source_mode="ord", model="openai/gpt-4o")

        assert result["status"] == "ok"
        assert result["smiles"] == "CCO"
        assert result["source_mode"] == "ord"
        assert result["molecule_info"]["name"] == "ethanol"
        assert result["source_errors"] == {}
        mock_attach.assert_called_once_with(retro_result["routes"])

    def test_run_retro_analyze_surfaces_requested_source_failure(self):
        resolved = {
            "validation": {"is_valid": True, "input_type": "name"},
            "smiles": "CCO",
            "pubchem_cid": 702,
        }
        guard_result = {"overall_status": "SAFE", "molecule_check": {}, "reaction_check": {}, "safety_data": {}}
        retro_result = {"routes": [], "source_mode": "aizynthfinder", "source_errors": {"aizynthfinder": "planner down"}}
        molecule_state = {"molecule_info": {"name": "ethanol", "smiles": "CCO"}}

        with patch("mvp.api._attach_procedure_steps") as mock_attach, \
             patch("mvp.nodes.validate_and_guard_node._resolve_molecule", return_value=resolved), \
             patch("mvp.nodes.validate_and_guard_node._run_safety_checks", return_value=guard_result), \
             patch("mvp.nodes.molecule_info_node.molecule_info_node", return_value=molecule_state), \
             patch("mvp.api.search_and_rank", return_value=retro_result):
            result = _run_retro_analyze("ethanol", top_n=5, source_mode="aizynthfinder", model="openai/gpt-4o")

        assert result["error"] == "aizynthfinder failed: planner down"
        assert result["source_errors"] == {"aizynthfinder": "planner down"}
        mock_attach.assert_called_once_with([])

    def test_run_retro_analyze_skips_routes_when_blocked(self):
        resolved = {
            "validation": {"is_valid": True, "input_type": "name"},
            "smiles": "CCO",
            "pubchem_cid": 702,
        }
        guard_result = {
            "overall_status": "CRITICAL_STOP",
            "molecule_check": {"reason": "blocked"},
            "reaction_check": {},
            "safety_data": {},
        }
        molecule_state = {"molecule_info": {"name": "ethanol", "smiles": "CCO"}}

        with patch("mvp.nodes.validate_and_guard_node._resolve_molecule", return_value=resolved), \
             patch("mvp.nodes.validate_and_guard_node._run_safety_checks", return_value=guard_result), \
             patch("mvp.nodes.molecule_info_node.molecule_info_node", return_value=molecule_state), \
             patch("mvp.api.search_and_rank") as mock_search:
            result = _run_retro_analyze("ethanol", top_n=5, source_mode="ord")

        assert result["status"] == "blocked"
        assert result["retro_result"]["routes"] == []
        mock_search.assert_not_called()

    def test_retro_sources_snapshot_probes_aizynth(self):
        with patch("mvp.api._cfg.RETRO_ENABLE_AIZYNTH", True), \
            patch("mvp.api._cfg.AIZYNTH_BASE_URL", "http://aizynth:8052"), \
            patch("mvp.api._cfg.AIZYNTH_TIMEOUT_SEC", 15), \
            patch("mvp.api.get_retrocast_runtime_info", return_value={"available": False, "version": None, "adapters": [], "error": "missing"}), \
            patch("mvp.api.get_aizynth_resources", return_value={"stocks": ["zinc"], "expansion_models": ["uspto"]}):
            snapshot = _retro_sources_snapshot()

        aizynth = snapshot["sources"]["aizynthfinder"]
        assert aizynth["enabled"] is True
        assert aizynth["configured"] is True
        assert aizynth["reachable"] is True
        assert aizynth["details"]["stocks"] == ["zinc"]
        assert any(mode["id"] == "aizynthfinder" and mode["enabled"] for mode in snapshot["source_modes"])

    def test_retro_sources_snapshot_reports_retrocast_bridge(self):
        retrocast_info = {
            "available": True,
            "version": "0.5.3",
            "adapters": ["aizynth", "retrostar"],
            "error": None,
        }
        with patch("mvp.api.get_retrocast_runtime_info", return_value=retrocast_info):
            snapshot = _retro_sources_snapshot()

        retrocast = snapshot["sources"]["retrocast"]
        assert retrocast["configured"] is True
        assert retrocast["reachable"] is True
        assert retrocast["standalone_source"] is False
        assert retrocast["adapters"] == ["aizynth", "retrostar"]


class TestRetroSearchEndpoints:
    def test_retro_search_endpoint_returns_payload(self):
        runtime_result = {
            "smiles": "CCO",
            "resolution": "smiles",
            "source_mode": "all",
            "total_found": 2,
            "total_unique": 1,
            "sources_used": ["ord", "web"],
            "source_counts": {"ord": 1, "web": 1},
            "source_counts_deduped": {"web": 1},
            "routes": [{"reactants": "O.CCO", "source": "web", "final_score": 0.9}],
        }
        with patch("mvp.api._run_retro_search", return_value=runtime_result):
            result = asyncio.run(retro_search(RetroSearchRequest(query="ethanol", top_n=5, source_mode="all")))

        assert result.smiles == "CCO"
        assert result.source_mode == "all"
        assert result.sources_used == ["ord", "web"]
        assert result.returned == 1
        assert result.source_counts_deduped == {"web": 1}

    def test_retro_analyze_endpoint_returns_card_payload(self):
        analysis_result = {
            "status": "ok",
            "query": "ethanol",
            "smiles": "CCO",
            "resolution": "smiles",
            "source_mode": "ord",
            "molecule_info": {"name": "ethanol", "smiles": "CCO"},
            "guard_result": {"overall_status": "SAFE"},
            "retro_result": {"routes": [{"reactants": "CC=O.O", "source": "ord"}], "source_mode": "ord"},
            "error": None,
        }
        with patch("mvp.api._run_retro_analyze", return_value=analysis_result):
            result = asyncio.run(
                retro_analyze(RetroAnalyzeRequest(query="ethanol", top_n=5, source_mode="ord"))
            )

        assert result.status == "ok"
        assert result.source_mode == "ord"
        assert result.molecule_info["name"] == "ethanol"

    def test_retro_sources_endpoint_returns_snapshot(self):
        snapshot = {
            "ord_authoritative": False,
            "tree_include_experimental": True,
            "source_modes": [{"id": "auto", "label": "Auto", "enabled": True}],
            "sources": {
                "ord": {"enabled": True, "configured": True, "mode": "sqlite"},
                "aizynthfinder": {
                    "enabled": True,
                    "configured": True,
                    "reachable": True,
                    "mode": "service_tree_search",
                },
            },
        }
        with patch("mvp.api._retro_sources_snapshot", return_value=snapshot):
            result = asyncio.run(retro_sources())

        assert result.ord_authoritative is False
        assert result.tree_include_experimental is True
        assert result.source_modes[0]["id"] == "auto"
        assert result.sources["aizynthfinder"]["reachable"] is True


class TestResearchEndpoint:
    def test_research_analyze_endpoint_returns_workspace_payload(self):
        workspace_result = {
            "status": "ok",
            "query": "aspirin synthesis literature",
            "mode": "literature",
            "interpreted_intent": "Find aspirin synthesis literature",
            "search_queries": ["aspirin synthesis literature PubMed"],
            "summary": "Found literature sources.",
            "candidates": [{"name": "aspirin", "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O"}],
            "sources": [{"url": "https://pubmed.ncbi.nlm.nih.gov/1/", "title": "Aspirin"}],
            "evidence": [{"url": "https://pubmed.ncbi.nlm.nih.gov/1/", "excerpt": "aspirin synthesis"}],
            "rag_results": [],
            "source_errors": {},
        }
        with patch("mvp.api.run_research_workspace", return_value=workspace_result):
            result = asyncio.run(
                research_analyze(ResearchAnalyzeRequest(query="aspirin synthesis literature"))
            )

        assert result.status == "ok"
        assert result.mode == "literature"
        assert result.candidates[0]["name"] == "aspirin"
        assert result.evidence[0]["excerpt"] == "aspirin synthesis"
