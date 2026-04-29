"""Tests for the retrosynthesis API helpers and endpoints."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from ..api import (
    RetroSearchRequest,
    _retro_sources_snapshot,
    _run_retro_search,
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
        }
        with patch("mvp.api._resolve_to_smiles", return_value=("CCO", "smiles")), \
             patch("mvp.api.search_and_rank", return_value=runtime_result):
            result = _run_retro_search("ethanol", top_n=3)

        assert result["smiles"] == "CCO"
        assert result["resolution"] == "smiles"
        assert result["total_found"] == 4
        assert result["total_unique"] == 3
        assert result["source_counts_deduped"]["aizynthfinder"] == 2

    def test_retro_sources_snapshot_probes_aizynth(self):
        with patch("mvp.api._cfg.RETRO_ENABLE_AIZYNTH", True), \
             patch("mvp.api._cfg.AIZYNTH_BASE_URL", "http://aizynth:8052"), \
             patch("mvp.api._cfg.AIZYNTH_TIMEOUT_SEC", 15), \
             patch("mvp.api.get_aizynth_resources", return_value={"stocks": ["zinc"], "expansion_models": ["uspto"]}):
            snapshot = _retro_sources_snapshot()

        aizynth = snapshot["sources"]["aizynthfinder"]
        assert aizynth["enabled"] is True
        assert aizynth["configured"] is True
        assert aizynth["reachable"] is True
        assert aizynth["details"]["stocks"] == ["zinc"]


class TestRetroSearchEndpoints:
    def test_retro_search_endpoint_returns_payload(self):
        runtime_result = {
            "smiles": "CCO",
            "resolution": "smiles",
            "total_found": 2,
            "total_unique": 1,
            "sources_used": ["ord", "web"],
            "source_counts": {"ord": 1, "web": 1},
            "source_counts_deduped": {"web": 1},
            "routes": [{"reactants": "O.CCO", "source": "web", "final_score": 0.9}],
        }
        with patch("mvp.api._run_retro_search", return_value=runtime_result):
            result = asyncio.run(retro_search(RetroSearchRequest(query="ethanol", top_n=5)))

        assert result.smiles == "CCO"
        assert result.sources_used == ["ord", "web"]
        assert result.returned == 1
        assert result.source_counts_deduped == {"web": 1}

    def test_retro_sources_endpoint_returns_snapshot(self):
        snapshot = {
            "ord_authoritative": False,
            "tree_include_experimental": True,
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
        assert result.sources["aizynthfinder"]["reachable"] is True
