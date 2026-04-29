"""Tests for tree_expansion: recursive retrosynthesis tree building.

Updated for current implementation:
  _find_top_routes (was _find_best_route in old code)
  _build_node positional args: (smiles, depth, max_depth, visited, start_time, timeout_sec)
"""

from __future__ import annotations

import time
from unittest.mock import patch, MagicMock

import pytest

from ..tree_expansion import (
    _canonicalize,
    _resolve_name,
    _find_top_routes,
    _build_node,
    expand_tree,
    _collect_stats,
    _walk,
    _empty_stats,
)


# ═════════════════════════════════════════════════════════════════════════════
# _canonicalize
# ═════════════════════════════════════════════════════════════════════════════

class TestCanonicalize:
    def test_valid_smiles(self):
        assert _canonicalize("CCO") == "CCO"

    def test_non_canonical_to_canonical(self):
        assert _canonicalize("OCC") == "CCO"

    def test_invalid_smiles_returns_none(self):
        assert _canonicalize("NOTASMILES!!!") is None

    def test_empty_string(self):
        result = _canonicalize("")
        assert result is None or result == ""

    def test_preserves_stereochemistry(self):
        smi = "C[C@@H](O)c1ccccc1"
        result = _canonicalize(smi)
        assert result is not None
        assert "@" in result

    def test_aspirin(self, aspirin_smiles):
        result = _canonicalize(aspirin_smiles)
        assert result is not None
        assert "C(=O)" in result


# ═════════════════════════════════════════════════════════════════════════════
# _resolve_name
# ═════════════════════════════════════════════════════════════════════════════

class TestResolveName:
    def test_returns_none_on_exception(self):
        with patch("mvp.tree_expansion.get_compound_properties",
                   side_effect=Exception("fail")):
            assert _resolve_name("CCO") is None

    def test_returns_iupac_name(self):
        with patch("mvp.tree_expansion.get_compound_properties",
                   return_value={"IUPACName": "ethanol", "Title": "Ethanol"}):
            assert _resolve_name("CCO") == "ethanol"

    def test_returns_title_if_no_iupac(self):
        with patch("mvp.tree_expansion.get_compound_properties",
                   return_value={"IUPACName": None, "Title": "Ethanol"}):
            assert _resolve_name("CCO") == "Ethanol"

    def test_returns_none_if_no_props(self):
        with patch("mvp.tree_expansion.get_compound_properties",
                   return_value={}):
            assert _resolve_name("CCO") is None


# ═════════════════════════════════════════════════════════════════════════════
# _find_top_routes
# ═════════════════════════════════════════════════════════════════════════════

class TestFindTopRoutes:
    def _ord_routes(self, n=2):
        return [
            {"reactants": f"R{i}.S{i}", "source": "ord", "final_score": 0.5 + i * 0.1}
            for i in range(n)
        ]

    def test_ord_hit_returns_routes(self):
        routes = self._ord_routes(2)
        with patch("mvp.tree_expansion.collect_candidate_routes", return_value=(routes, ["ord"])), \
             patch("mvp.tree_expansion.score_route", side_effect=lambda r: r):
            result = _find_top_routes("CCO")
            assert len(result) >= 1

    def test_ord_empty_falls_back_to_model(self):
        model_routes = [{"reactants": "X.Y", "source": "retro_model", "final_score": 0.7}]
        with patch("mvp.tree_expansion.collect_candidate_routes", return_value=(model_routes, ["retro_model"])), \
             patch("mvp.tree_expansion.score_route", side_effect=lambda r: r):
            result = _find_top_routes("CCO")
            assert len(result) >= 1

    def test_no_routes_returns_empty_list(self):
        with patch("mvp.tree_expansion.collect_candidate_routes", return_value=([], [])):
            result = _find_top_routes("CCO")
            assert result == []

    def test_model_exception_returns_empty(self):
        with patch("mvp.tree_expansion.collect_candidate_routes", side_effect=Exception("model error")):
            result = _find_top_routes("CCO")
            assert isinstance(result, list)

    def test_respects_top_n(self):
        routes = self._ord_routes(10)
        with patch("mvp.tree_expansion.collect_candidate_routes", return_value=(routes, ["ord"])), \
             patch("mvp.tree_expansion.score_route", side_effect=lambda r: r):
            result = _find_top_routes("CCO", top_n=3)
            assert len(result) <= 3


# ═════════════════════════════════════════════════════════════════════════════
# _build_node
# ═════════════════════════════════════════════════════════════════════════════

class TestBuildNode:
    def test_invalid_smiles(self):
        node = _build_node("INVALID!!!", 0, 6, set(), time.time(), 120)
        assert node["status"] == "invalid_smiles"
        assert node["children"] == []

    def test_cycle_detection(self):
        node = _build_node("CCO", 1, 6, {"CCO"}, time.time(), 120)
        assert node["status"] == "circular"

    def test_timeout(self):
        node = _build_node("CCO", 0, 6, set(), time.time() - 200, 120)
        assert node["status"] == "timeout"

    def test_banned_molecule(self):
        with patch("mvp.tree_expansion.banlist_check",
                   return_value={"status": "banned", "name": "Bad", "reason": "controlled"}), \
             patch("mvp.tree_expansion._resolve_name", return_value="Bad"):
            node = _build_node("CCO", 0, 6, set(), time.time(), 120)
        assert node["status"] == "banned"
        assert node["is_buyable"] is False

    def test_buyable_molecule(self):
        with patch("mvp.tree_expansion.banlist_check",
                   return_value={"status": "clear"}), \
             patch("mvp.tree_expansion._is_buyable", return_value=True), \
             patch("mvp.tree_expansion._resolve_name", return_value="ethanol"):
            node = _build_node("CCO", 0, 6, set(), time.time(), 120)
        assert node["status"] == "buyable"
        assert node["is_buyable"] is True

    def test_depth_limit(self):
        with patch("mvp.tree_expansion.banlist_check",
                   return_value={"status": "clear"}), \
             patch("mvp.tree_expansion._is_buyable", return_value=False), \
             patch("mvp.tree_expansion._resolve_name", return_value=None):
            node = _build_node("CCO", 6, 6, set(), time.time(), 120)
        assert node["status"] == "depth_limit"

    def test_unresolved_when_no_routes(self):
        with patch("mvp.tree_expansion.banlist_check",
                   return_value={"status": "clear"}), \
             patch("mvp.tree_expansion._is_buyable", return_value=False), \
             patch("mvp.tree_expansion._find_top_routes", return_value=[]), \
             patch("mvp.tree_expansion._resolve_name", return_value=None):
            node = _build_node("CCO", 0, 6, set(), time.time(), 120)
        assert node["status"] == "unresolved"

    def test_intermediate_has_children(self):
        route = {
            "reactants": "C.O",
            "reaction_smiles": "C.O>>CO",
            "source": "ord",
            "final_score": 0.8,
            "template": "should_be_stripped",
        }
        with patch("mvp.tree_expansion.banlist_check",
                   return_value={"status": "clear"}), \
             patch("mvp.tree_expansion._is_buyable",
                   side_effect=lambda s: s in ("C", "O")), \
             patch("mvp.tree_expansion._find_top_routes", return_value=[route]), \
             patch("mvp.tree_expansion._resolve_name", return_value="methanol"):
            node = _build_node("CO", 0, 6, set(), time.time(), 120)
        assert node["status"] == "intermediate"
        assert len(node["children"]) == 2
        assert "template" not in node["route"]

    def test_node_has_all_required_keys(self):
        with patch("mvp.tree_expansion.banlist_check",
                   return_value={"status": "clear"}), \
             patch("mvp.tree_expansion._is_buyable", return_value=True), \
             patch("mvp.tree_expansion._resolve_name", return_value="ethanol"):
            node = _build_node("CCO", 0, 6, set(), time.time(), 120)
        for key in ("smiles", "name", "status", "depth", "is_buyable", "guard", "route", "children"):
            assert key in node, f"Missing key: {key}"

    def test_banned_checked_before_buyable(self):
        call_order = []

        def mock_banlist(s):
            call_order.append("banlist")
            return {"status": "banned", "name": "Bad"}

        def mock_buyable(s):
            call_order.append("buyable")
            return True

        with patch("mvp.tree_expansion.banlist_check", side_effect=mock_banlist), \
             patch("mvp.tree_expansion._is_buyable", side_effect=mock_buyable), \
             patch("mvp.tree_expansion._resolve_name", return_value=None):
            node = _build_node("CCO", 0, 6, set(), time.time(), 120)
        assert node["status"] == "banned"
        assert "buyable" not in call_order

    def test_depth_stored_in_node(self):
        with patch("mvp.tree_expansion.banlist_check",
                   return_value={"status": "clear"}), \
             patch("mvp.tree_expansion._is_buyable", return_value=True), \
             patch("mvp.tree_expansion._resolve_name", return_value=None):
            node = _build_node("CCO", 3, 6, set(), time.time(), 120)
        assert node["depth"] == 3


# ═════════════════════════════════════════════════════════════════════════════
# expand_tree
# ═════════════════════════════════════════════════════════════════════════════

class TestExpandTree:
    def test_invalid_target_smiles(self):
        result = expand_tree("INVALID!!!", "A.B")
        assert result["tree"]["status"] == "invalid_smiles"
        assert result["stats"]["total_nodes"] == 1
        assert result["stats"]["unresolved_count"] == 1

    def test_basic_expansion_with_buyable_leaves(self):
        with patch("mvp.tree_expansion.banlist_check",
                   return_value={"status": "clear"}), \
             patch("mvp.tree_expansion._is_buyable", return_value=True), \
             patch("mvp.tree_expansion._resolve_name", return_value="test"):
            result = expand_tree("CC(=O)Oc1ccccc1C(=O)O", "CCO.CC(=O)O")
        tree = result["tree"]
        assert tree["status"] == "intermediate"
        assert tree["depth"] == 0
        assert len(tree["children"]) == 2
        for child in tree["children"]:
            assert child["status"] == "buyable"

    def test_stats_total_nodes(self):
        with patch("mvp.tree_expansion.banlist_check",
                   return_value={"status": "clear"}), \
             patch("mvp.tree_expansion._is_buyable", return_value=True), \
             patch("mvp.tree_expansion._resolve_name", return_value="test"):
            result = expand_tree("CC(=O)Oc1ccccc1C(=O)O", "CCO.CC(=O)O")
        stats = result["stats"]
        assert stats["total_nodes"] == 3
        assert stats["buyable_count"] == 2
        assert stats["banned_count"] == 0
        assert stats["elapsed_sec"] >= 0

    def test_root_has_selected_route(self):
        with patch("mvp.tree_expansion.banlist_check",
                   return_value={"status": "clear"}), \
             patch("mvp.tree_expansion._is_buyable", return_value=True), \
             patch("mvp.tree_expansion._resolve_name", return_value="test"):
            result = expand_tree("CC(=O)Oc1ccccc1C(=O)O", "CCO.CC(=O)O")
        root = result["tree"]
        assert root["route"]["source"] == "selected"
        assert "CCO" in root["route"]["reactants"]

    def test_max_depth_stops_recursion(self):
        with patch("mvp.tree_expansion.banlist_check",
                   return_value={"status": "clear"}), \
             patch("mvp.tree_expansion._is_buyable", return_value=False), \
             patch("mvp.tree_expansion._find_top_routes", return_value=[]), \
             patch("mvp.tree_expansion._resolve_name", return_value=None):
            result = expand_tree("CC(=O)Oc1ccccc1C(=O)O", "CCO", max_depth=1)
        child = result["tree"]["children"][0]
        assert child["status"] in ("depth_limit", "unresolved")

    def test_all_stats_keys_present(self):
        with patch("mvp.tree_expansion.banlist_check",
                   return_value={"status": "clear"}), \
             patch("mvp.tree_expansion._is_buyable", return_value=True), \
             patch("mvp.tree_expansion._resolve_name", return_value=None):
            result = expand_tree("CCO", "C.O")
        for key in ("total_nodes", "buyable_count", "banned_count", "unresolved_count",
                    "max_depth_reached", "elapsed_sec"):
            assert key in result["stats"], f"Missing stat key: {key}"


# ═════════════════════════════════════════════════════════════════════════════
# _collect_stats / _walk / _empty_stats
# ═════════════════════════════════════════════════════════════════════════════

class TestStats:
    def test_empty_stats(self):
        stats = _empty_stats(1.5)
        assert stats["total_nodes"] == 1
        assert stats["buyable_count"] == 0
        assert stats["banned_count"] == 0
        assert stats["unresolved_count"] == 1
        assert stats["elapsed_sec"] == 1.5

    def test_walk_counts_all_nodes(self):
        tree = {
            "status": "intermediate", "depth": 0,
            "children": [
                {"status": "buyable", "depth": 1, "children": []},
                {"status": "banned", "depth": 1, "children": []},
                {"status": "intermediate", "depth": 1, "children": [
                    {"status": "unresolved", "depth": 2, "children": []},
                    {"status": "buyable", "depth": 2, "children": []},
                ]},
            ],
        }
        counts = {"total": 0, "buyable": 0, "banned": 0, "unresolved": 0, "max_depth": 0}
        _walk(tree, counts)
        assert counts["total"] == 6
        assert counts["buyable"] == 2
        assert counts["banned"] == 1
        assert counts["unresolved"] == 1
        assert counts["max_depth"] == 2

    def test_collect_stats_full(self):
        tree = {
            "status": "intermediate", "depth": 0,
            "children": [
                {"status": "buyable", "depth": 1, "children": []},
                {"status": "timeout", "depth": 1, "children": []},
            ],
        }
        stats = _collect_stats(tree, 2.5)
        assert stats["total_nodes"] == 3
        assert stats["buyable_count"] == 1
        assert stats["unresolved_count"] == 1
        assert stats["max_depth_reached"] == 1
        assert stats["elapsed_sec"] == 2.5

    def test_circular_counts_as_unresolved(self):
        tree = {"status": "circular", "depth": 2, "children": []}
        counts = {"total": 0, "buyable": 0, "banned": 0, "unresolved": 0, "max_depth": 0}
        _walk(tree, counts)
        assert counts["unresolved"] == 1

    def test_depth_limit_counts_as_unresolved(self):
        tree = {"status": "depth_limit", "depth": 6, "children": []}
        counts = {"total": 0, "buyable": 0, "banned": 0, "unresolved": 0, "max_depth": 0}
        _walk(tree, counts)
        assert counts["unresolved"] == 1

