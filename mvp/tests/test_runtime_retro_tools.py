"""Tests for the runtime retrosynthesis collector in mvp.tools.retro_tools."""

from __future__ import annotations

from unittest.mock import patch

from ..tools.retro_tools import collect_candidate_routes, get_aizynthfinder_routes, search_and_rank


class TestCollectCandidateRoutes:
    def test_ord_authoritative_short_circuits_other_sources(self):
        ord_routes = [{"reactants": "A.B", "source": "ord", "score": 0.9}]

        with patch("mvp.tools.retro_tools.get_ord_routes", return_value=ord_routes), \
             patch("mvp.tools.retro_tools.get_web_routes") as mock_web, \
             patch("mvp.tools.retro_tools.get_retro_model_routes") as mock_model:
            routes, sources = collect_candidate_routes("CCO", ord_authoritative=True)

        assert routes == ord_routes
        assert sources == ["ord"]
        mock_web.assert_not_called()
        mock_model.assert_not_called()

    def test_non_authoritative_mode_merges_multiple_sources(self):
        ord_routes = [{"reactants": "A.B", "source": "ord", "score": 0.9}]
        web_routes = [{"reactants": "C.D", "source": "web", "score": 0.5}]
        model_routes = [{"reactants": "E.F", "source": "retro_model", "score": 0.7}]

        with patch("mvp.tools.retro_tools.get_ord_routes", return_value=ord_routes), \
             patch("mvp.tools.retro_tools.get_web_routes", return_value=web_routes), \
             patch("mvp.tools.retro_tools.get_retro_model_routes", return_value=model_routes):
            routes, sources = collect_candidate_routes("CCO", ord_authoritative=False)

        assert routes == ord_routes + web_routes + model_routes
        assert sources == ["ord", "web", "retro_model"]

    def test_empty_smiles_returns_empty_collection(self):
        routes, sources = collect_candidate_routes("")
        assert routes == []
        assert sources == []


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

        with patch("mvp.tools.retro_tools.collect_candidate_routes", return_value=(collected, ["ord", "web"])), \
             patch("mvp.tools.retro_tools.score_route", side_effect=_score_passthrough):
            result = search_and_rank("CCO", top_n=5)

        assert result["sources_used"] == ["ord", "web"]
        assert result["total_found"] == 2
        assert len(result["routes"]) == 1
        assert result["routes"][0]["source"] == "web"
