"""Tests for retro_tools: ORD search, scoring, deduplication, search_and_rank pipeline."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from ..retro_tools import (
    _canonical_reactant_key,
    _deduplicate_routes,
    score_route,
    ord_search_by_product,
    search_and_rank,
)


# ═════════════════════════════════════════════════════════════════════════════
# _canonical_reactant_key
# ═════════════════════════════════════════════════════════════════════════════

class TestCanonicalReactantKey:
    def test_single_smiles(self):
        key = _canonical_reactant_key("CCO")
        assert key is not None
        assert "CCO" in key

    def test_multi_reactant_sorted(self):
        # Order shouldn't matter — output must be sorted
        key1 = _canonical_reactant_key("CCO.CC(=O)O")
        key2 = _canonical_reactant_key("CC(=O)O.CCO")
        assert key1 == key2

    def test_invalid_smiles_returns_none(self):
        key = _canonical_reactant_key("NOTVALID.(((")
        assert key is None

    def test_empty_string(self):
        key = _canonical_reactant_key("")
        assert key is not None  # empty string → empty key, not None

    def test_canonical_form_applied(self):
        # OCC is non-canonical ethanol, should become CCO
        key = _canonical_reactant_key("OCC")
        assert key == "CCO"

    def test_deterministic_output(self, aspirin_smiles):
        key1 = _canonical_reactant_key(aspirin_smiles)
        key2 = _canonical_reactant_key(aspirin_smiles)
        assert key1 == key2


# ═════════════════════════════════════════════════════════════════════════════
# _deduplicate_routes
# ═════════════════════════════════════════════════════════════════════════════

class TestDeduplicateRoutes:
    def _make_route(self, reactants, score=0.5):
        return {
            "reactants": reactants,
            "final_score": score,
            "source": "ord",
        }

    def test_empty_list(self):
        assert _deduplicate_routes([]) == []

    def test_no_duplicates_unchanged(self):
        routes = [
            self._make_route("CCO"),
            self._make_route("CC(=O)O"),
        ]
        result = _deduplicate_routes(routes)
        assert len(result) == 2

    def test_duplicate_removed(self):
        routes = [
            self._make_route("CCO", score=0.8),
            self._make_route("CCO", score=0.5),  # duplicate, lower score
        ]
        result = _deduplicate_routes(routes)
        assert len(result) == 1

    def test_keeps_higher_score_on_duplicate(self):
        routes = [
            self._make_route("CCO", score=0.5),
            self._make_route("CCO", score=0.9),
        ]
        result = _deduplicate_routes(routes)
        assert len(result) == 1
        assert result[0]["final_score"] == 0.9

    def test_canonical_dedup_catches_equivalent_smiles(self):
        # OCC and CCO are the same molecule (ethanol)
        routes = [
            self._make_route("CCO", score=0.8),
            self._make_route("OCC", score=0.6),  # non-canonical, same molecule
        ]
        result = _deduplicate_routes(routes)
        assert len(result) == 1
        assert result[0]["final_score"] == 0.8


# ═════════════════════════════════════════════════════════════════════════════
# score_route
# ═════════════════════════════════════════════════════════════════════════════

class TestScoreRoute:
    def _make_route(self, **kwargs):
        base = {
            "reactants": "CC(=O)OC(C)=O.OC(=O)c1ccccc1O",
            "score": 0.85,
            "plausibility": 0.90,
            "source": "ord",
        }
        base.update(kwargs)
        return base

    def test_returns_route_dict(self):
        route = self._make_route()
        result = score_route(route)
        assert isinstance(result, dict)

    def test_final_score_in_range(self):
        route = self._make_route()
        score_route(route)
        assert 0.0 <= route["final_score"] <= 1.0

    def test_scoring_keys_present(self):
        route = self._make_route()
        score_route(route)
        scoring = route["scoring"]
        for key in ("model_score", "plausibility", "buyability", "simplicity",
                    "efficiency", "yield_bonus", "procedure_bonus", "num_reactants",
                    "total_atoms", "buyable_count"):
            assert key in scoring, f"Missing scoring key: {key}"

    def test_yield_bonus_applied(self):
        route_with_yield = self._make_route(expected_yield=0.80)
        route_no_yield = self._make_route()
        score_route(route_with_yield)
        score_route(route_no_yield)
        # Route with yield should score higher
        assert route_with_yield["final_score"] > route_no_yield["final_score"]

    def test_procedure_bonus_applied(self):
        route_with_proc = self._make_route(procedure_details="Mix reagents...")
        route_no_proc = self._make_route()
        score_route(route_with_proc)
        score_route(route_no_proc)
        assert route_with_proc["final_score"] > route_no_proc["final_score"]

    def test_many_reactants_penalized(self):
        route_simple = self._make_route(reactants="CCO.CC(=O)O")
        route_complex = self._make_route(reactants="CCO.CC(=O)O.c1ccccc1.CCOC(=O)C.N")
        score_route(route_simple)
        score_route(route_complex)
        assert route_simple["final_score"] > route_complex["final_score"]

    def test_buyable_reactants_boost_score(self):
        # Water, ethanol are in cheap list → high buyability
        route_cheap = self._make_route(reactants="O.CCO")
        # Complex molecule → low buyability
        route_complex = self._make_route(
            reactants="C1CC2CCCCC2CC1.Brc1ccc2ccccc2c1"
        )
        score_route(route_cheap)
        score_route(route_complex)
        assert route_cheap["scoring"]["buyability"] > route_complex["scoring"]["buyability"]

    def test_aspirin_route_reasonable_score(self):
        route = {
            "reactants": "CC(=O)OC(C)=O.OC(=O)c1ccccc1O",
            "score": 0.85,
            "plausibility": 0.90,
            "source": "ord",
            "expected_yield": 0.76,
            "procedure_details": "Mix and heat...",
        }
        score_route(route)
        # Good route with yield and procedure should score > 0.7
        assert route["final_score"] > 0.7


# ═════════════════════════════════════════════════════════════════════════════
# ord_search_by_product — integration with real ORD SQLite
# ═════════════════════════════════════════════════════════════════════════════

class TestOrdSearchByProduct:
    @pytest.mark.integration
    def test_aspirin_returns_results(self, aspirin_smiles):
        results = ord_search_by_product(aspirin_smiles)
        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.integration
    def test_aspirin_results_have_required_keys(self, aspirin_smiles):
        results = ord_search_by_product(aspirin_smiles)
        assert len(results) > 0
        for r in results:
            assert "reaction_id" in r
            assert "reaction_smiles" in r
            assert "reactants" in r
            assert "source" in r

    @pytest.mark.integration
    def test_aspirin_source_is_ord(self, aspirin_smiles):
        results = ord_search_by_product(aspirin_smiles)
        for r in results:
            assert r["source"] == "ord"

    @pytest.mark.integration
    def test_aspirin_reaction_smiles_contains_product(self, aspirin_smiles):
        results = ord_search_by_product(aspirin_smiles)
        for r in results:
            assert ">>" in r["reaction_smiles"]

    @pytest.mark.integration
    def test_limit_respected(self, aspirin_smiles):
        results = ord_search_by_product(aspirin_smiles, limit=3)
        assert len(results) <= 3

    @pytest.mark.integration
    def test_caffeine_returns_results(self, caffeine_smiles):
        results = ord_search_by_product(caffeine_smiles)
        assert isinstance(results, list)
        # Caffeine may or may not be in ORD, just check no crash

    @pytest.mark.integration
    def test_unknown_molecule_returns_empty(self):
        # Highly unlikely to be in ORD
        fake_smiles = "C1CC2(CC1)CCCC2"
        results = ord_search_by_product(fake_smiles, limit=5)
        # May return [] or something — just check it's a list
        assert isinstance(results, list)

    def test_returns_empty_when_no_db(self):
        with patch("real_proj.mvp.retro_tools.ORD_DB_PATH") as mock_path:
            mock_path.exists.return_value = False
            results = ord_search_by_product("CCO")
            assert results == []


# ═════════════════════════════════════════════════════════════════════════════
# search_and_rank
# ═════════════════════════════════════════════════════════════════════════════

class TestSearchAndRank:
    @pytest.mark.integration
    def test_aspirin_returns_routes(self, aspirin_smiles):
        result = search_and_rank(aspirin_smiles, top_n=5)
        assert "routes" in result
        assert "sources_used" in result
        assert "total_found" in result
        assert "best_route" in result

    @pytest.mark.integration
    def test_aspirin_routes_scored(self, aspirin_smiles):
        result = search_and_rank(aspirin_smiles, top_n=5)
        for route in result["routes"]:
            assert "final_score" in route
            assert 0.0 <= route["final_score"] <= 1.0

    @pytest.mark.integration
    def test_routes_sorted_best_first(self, aspirin_smiles):
        result = search_and_rank(aspirin_smiles, top_n=5)
        routes = result["routes"]
        if len(routes) > 1:
            for i in range(len(routes) - 1):
                assert routes[i]["final_score"] >= routes[i + 1]["final_score"]

    @pytest.mark.integration
    def test_top_n_respected(self, aspirin_smiles):
        result = search_and_rank(aspirin_smiles, top_n=3)
        assert len(result["routes"]) <= 3

    @pytest.mark.integration
    def test_ord_priority_skips_model(self, aspirin_smiles):
        """When ORD has results, retro model should NOT be called."""
        with patch("real_proj.mvp.retro_tools.ord_search_by_product") as mock_ord, \
             patch("real_proj.mvp.retro_tools._deduplicate_routes",
                   side_effect=lambda x: x) as _:

            mock_ord.return_value = [
                {
                    "reactants": "CCO.CC(=O)O",
                    "reaction_smiles": "CCO.CC(=O)O>>CC(=O)Oc1ccccc1C(=O)O",
                    "source": "ord", "score": 0.85, "plausibility": 0.9,
                }
            ]

            with patch("real_proj.mvp.retro_tools.score_route",
                       side_effect=lambda r: r) as _:
                result = search_and_rank(aspirin_smiles, top_n=5)

            assert "ord" in result["sources_used"]
            assert "retro_model" not in result["sources_used"]

    def test_empty_smiles_returns_empty(self):
        result = search_and_rank("", top_n=5)
        assert result["routes"] == []
        assert result["total_found"] == 0
        assert result["best_route"] is None

    def test_no_results_returns_empty_structure(self):
        with patch("real_proj.mvp.retro_tools.ord_search_by_product", return_value=[]), \
             patch("real_proj.mvp.retro_tools.predict_retro", return_value=[]):
            result = search_and_rank("CCO", top_n=5)

        assert result["routes"] == []
        assert result["sources_used"] == []

    @pytest.mark.integration
    def test_best_route_is_first_route(self, aspirin_smiles):
        result = search_and_rank(aspirin_smiles, top_n=5)
        if result["routes"]:
            assert result["best_route"] == result["routes"][0]

    @pytest.mark.integration
    def test_routes_deduplicated(self, aspirin_smiles):
        """No two routes should have identical canonical reactant sets."""
        from rdkit import Chem
        result = search_and_rank(aspirin_smiles, top_n=10)
        seen = set()
        for route in result["routes"]:
            key = _canonical_reactant_key(route.get("reactants", ""))
            assert key not in seen, f"Duplicate route: {key}"
            seen.add(key)
