"""Tests for the runtime retrosynthesis collector in mvp.tools.retro_tools."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ..tools.retro_tools import (
    collect_candidate_routes,
    get_aizynthfinder_routes,
    get_enabled_sources_for_mode,
    search_and_rank,
)


class TestCollectCandidateRoutes:
    def test_ord_authoritative_short_circuits_other_sources(self):
        ord_routes = [{"reactants": "A.B", "source": "ord", "score": 0.9}]

        with patch("mvp.tools.retro_tools.get_ord_routes", return_value=ord_routes), \
             patch("mvp.tools.retro_tools.get_web_routes") as mock_web, \
             patch("mvp.tools.retro_tools.get_retro_model_routes") as mock_model:
            routes, sources, errors = collect_candidate_routes("CCO", ord_authoritative=True)

        assert routes == ord_routes
        assert sources == ["ord"]
        assert errors == {}
        mock_web.assert_not_called()
        mock_model.assert_not_called()

    def test_non_authoritative_mode_merges_multiple_sources(self):
        ord_routes = [{"reactants": "A.B", "source": "ord", "score": 0.9}]
        web_routes = [{"reactants": "C.D", "source": "web", "score": 0.5}]
        model_routes = [{"reactants": "E.F", "source": "retro_model", "score": 0.7}]

        with patch("mvp.tools.retro_tools.get_ord_routes", return_value=ord_routes), \
             patch("mvp.tools.retro_tools.get_web_routes", return_value=web_routes), \
             patch("mvp.tools.retro_tools.get_retro_model_routes", return_value=model_routes):
            routes, sources, errors = collect_candidate_routes("CCO", ord_authoritative=False)

        assert routes == ord_routes + web_routes + model_routes
        assert sources == ["ord", "web", "retro_model"]
        assert errors == {}

    def test_empty_smiles_returns_empty_collection(self):
        routes, sources, errors = collect_candidate_routes("")
        assert routes == []
        assert sources == []
        assert errors == {}

    def test_enabled_sources_limits_collection_to_requested_mode(self):
        ord_routes = [{"reactants": "A.B", "source": "ord", "score": 0.9}]

        with patch("mvp.tools.retro_tools.get_ord_routes", return_value=ord_routes), \
             patch("mvp.tools.retro_tools.get_web_routes") as mock_web, \
             patch("mvp.tools.retro_tools.get_retro_model_routes") as mock_model:
            routes, sources, errors = collect_candidate_routes(
                "CCO",
                enabled_sources={"ord"},
                ord_authoritative=False,
            )

        assert routes == ord_routes
        assert sources == ["ord"]
        assert errors == {}
        mock_web.assert_not_called()
        mock_model.assert_not_called()

    def test_collects_source_errors_for_failed_requested_source(self):
        with patch("mvp.tools.retro_tools.get_ord_routes", return_value=[]), \
             patch("mvp.tools.retro_tools.get_aizynthfinder_routes", side_effect=RuntimeError("planner down")):
            routes, sources, errors = collect_candidate_routes(
                "CCO",
                enabled_sources={"aizynthfinder"},
                ord_authoritative=False,
                include_experimental=True,
            )

        assert routes == []
        assert sources == []
        assert errors == {"aizynthfinder": "planner down"}


class TestSourceModes:
    def test_auto_mode_returns_default_behavior(self):
        assert get_enabled_sources_for_mode("auto") is None

    def test_all_mode_expands_to_all_primary_sources(self):
        assert get_enabled_sources_for_mode("all") == {"ord", "web", "retro_model", "aizynthfinder"}

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            get_enabled_sources_for_mode("missing")


class TestAiZynthRuntimeAdapter:
    def test_get_aizynthfinder_routes_calls_service_client(self):
        normalized = [
            {
                "reactants": "CC=O.O",
                "reaction_smiles": "CC=O.O>>CCO",
                "source": "aizynthfinder",
                "score": 0.75,
                "plausibility": 0.75,
                "provenance": {"provider": "aizynthfinder"},
            }
        ]

        with patch("mvp.tools.retro_tools.RETRO_ENABLE_AIZYNTH", True), \
             patch("mvp.tools.retro_tools.AIZYNTH_BASE_URL", "http://aizynth:8052"), \
             patch("mvp.services.aizynth_client.run_aizynth_retrosynthesis", return_value={"routes": []}) as mock_run, \
             patch("mvp.services.aizynth_client.normalize_aizynth_routes", return_value=normalized):
            routes = get_aizynthfinder_routes("CCO", top_n=5)

        mock_run.assert_called_once()
        assert len(routes) == 1
        assert routes[0]["source"] == "aizynthfinder"
        assert routes[0]["provenance"]["provider"] == "aizynthfinder"


class TestSearchAndRankRuntime:
    def test_search_and_rank_uses_collector_output(self):
        collected = [
            {"reactants": "CCO.O", "source": "ord", "score": 0.5},
            {"reactants": "O.CCO", "source": "web", "score": 0.9},
        ]

        def _score_passthrough(route):
            route.setdefault("final_score", route["score"])
            return route

        with patch("mvp.tools.retro_tools.collect_candidate_routes", return_value=(collected, ["ord", "web"], {})), \
             patch("mvp.tools.retro_tools.score_route", side_effect=_score_passthrough):
            result = search_and_rank("CCO", top_n=5)

        assert result["sources_used"] == ["ord", "web"]
        assert result["total_found"] == 2
        assert result["total_unique"] == 1
        assert result["source_counts"] == {"ord": 1, "web": 1}
        assert result["source_counts_deduped"] == {"web": 1}
        assert len(result["routes"]) == 1
        assert result["routes"][0]["source"] == "web"
        assert result["source_mode"] == "auto"
        assert result["source_errors"] == {}

    def test_search_and_rank_passes_source_mode_to_collector(self):
        collected = [{"reactants": "CCO.O", "source": "aizynthfinder", "score": 0.9}]

        def _score_passthrough(route):
            route.setdefault("final_score", route["score"])
            return route

        with patch("mvp.tools.retro_tools.collect_candidate_routes", return_value=(collected, ["aizynthfinder"], {})) as mock_collect, \
             patch("mvp.tools.retro_tools.score_route", side_effect=_score_passthrough):
            result = search_and_rank("CCO", top_n=5, source_mode="aizynthfinder")

        mock_collect.assert_called_once()
        assert mock_collect.call_args.kwargs["enabled_sources"] == {"aizynthfinder"}
        assert result["source_mode"] == "aizynthfinder"

    def test_search_and_rank_returns_source_errors_when_no_routes(self):
        with patch("mvp.tools.retro_tools.collect_candidate_routes", return_value=([], [], {"aizynthfinder": "planner down"})):
            result = search_and_rank("CCO", top_n=5, source_mode="aizynthfinder")

        assert result["routes"] == []
        assert result["source_errors"] == {"aizynthfinder": "planner down"}
