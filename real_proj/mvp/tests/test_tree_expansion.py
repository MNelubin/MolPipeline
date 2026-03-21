"""Tests for tree_expansion: recursive retrosynthesis tree building."""

from __future__ import annotations

import time
from unittest.mock import patch, MagicMock

import pytest

from ..tree_expansion import (
    _canonicalize,
    _resolve_name,
    _find_best_route,
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
        # RDKit may return empty or None for empty string
        result = _canonicalize("")
        assert result is None or result == ""

    def test_preserves_stereochemistry(self):
        smi = "C[C@@H](O)c1ccccc1"
        result = _canonicalize(smi)
        assert result is not None
        assert "@" in result  # stereochemistry preserved

    def test_aspirin(self, aspirin_smiles):
        result = _canonicalize(aspirin_smiles)
        assert result is not None
        assert "C(=O)" in result


# ═════════════════════════════════════════════════════════════════════════════
# _resolve_name
# ═════════════════════════════════════════════════════════════════════════════

class TestResolveName:
    def test_returns_none_on_exception(self):
        with patch("real_proj.mvp.tree_expansion.get_compound_properties", side_effect=Exception("fail")):
            assert _resolve_name("CCO") is None

    def test_returns_iupac_name(self):
        with patch("real_proj.mvp.tree_expansion.get_compound_properties",
                    return_value={"IUPACName": "ethanol", "Title": "Ethanol"}):
            assert _resolve_name("CCO") == "ethanol"

    def test_returns_title_if_no_iupac(self):
        with patch("real_proj.mvp.tree_expansion.get_compound_properties",
                    return_value={"IUPACName": None, "Title": "Ethanol"}):
            assert _resolve_name("CCO") == "Ethanol"

    def test_returns_none_if_no_props(self):
        with patch("real_proj.mvp.tree_expansion.get_compound_properties",
                    return_value={}):
            assert _resolve_name("CCO") is None


# ═════════════════════════════════════════════════════════════════════════════
# _find_best_route
# ═════════════════════════════════════════════════════════════════════════════

class TestFindBestRoute:
    def test_ord_hit_returns_best_scored(self):
        routes = [
            {"reactants": "A.B", "final_score": 0.5},
            {"reactants": "C.D", "final_score": 0.9},
        ]
        with patch("real_proj.mvp.tree_expansion.ord_search_by_product", return_value=routes), \
             patch("real_proj.mvp.tree_expansion.score_route"):
            result = _find_best_route("CCO")
            assert result is not None
            assert result["final_score"] == 0.9

    def test_no_ord_falls_back_to_model(self):
        model_routes = [{"reactants": "X.Y", "final_score": 0.7}]
        with patch("real_proj.mvp.tree_expansion.ord_search_by_product", return_value=[]), \
             patch("real_proj.mvp.tree_expansion._get_predict_retro",
                   return_value=lambda s, top_n: model_routes), \
             patch("real_proj.mvp.tree_expansion.score_route"):
            result = _find_best_route("CCO")
            assert result is not None
            assert result["reactants"] == "X.Y"

    def test_no_routes_returns_none(self):
        with patch("real_proj.mvp.tree_expansion.ord_search_by_product", return_value=[]), \
             patch("real_proj.mvp.tree_expansion._get_predict_retro",
                   return_value=lambda s, top_n: []):
            result = _find_best_route("CCO")
            assert result is None

    def test_model_exception_returns_none(self):
        with patch("real_proj.mvp.tree_expansion.ord_search_by_product", return_value=[]), \
             patch("real_proj.mvp.tree_expansion._get_predict_retro",
                   side_effect=Exception("model error")):
            result = _find_best_route("CCO")
            assert result is None


# ═════════════════════════════════════════════════════════════════════════════
# _build_node
# ═════════════════════════════════════════════════════════════════════════════

class TestBuildNode:
    def _make_args(self, smiles="CCO", depth=0, max_depth=6, visited=None, timeout=120):
        return {
            "smiles": smiles,
            "depth": depth,
            "max_depth": max_depth,
            "visited": visited or set(),
            "start_time": time.time(),
            "timeout_sec": timeout,
        }

    def test_invalid_smiles(self):
        node = _build_node("INVALID!!!", 0, 6, set(), time.time(), 120)
        assert node["status"] == "invalid_smiles"
        assert node["children"] == []

    def test_cycle_detection(self):
        node = _build_node("CCO", 1, 6, {"CCO"}, time.time(), 120)
        assert node["status"] == "circular"

    def test_timeout(self):
        # start_time far in the past → already timed out
        node = _build_node("CCO", 0, 6, set(), time.time() - 200, 120)
        assert node["status"] == "timeout"

    def test_banned_molecule(self):
        with patch("real_proj.mvp.tree_expansion.banlist_check",
                    return_value={"status": "banned", "name": "Bad stuff", "reason": "controlled"}), \
             patch("real_proj.mvp.tree_expansion._resolve_name", return_value="Bad stuff"):
            node = _build_node("CCO", 0, 6, set(), time.time(), 120)
            assert node["status"] == "banned"
            assert node["is_buyable"] is False

    def test_buyable_molecule(self):
        with patch("real_proj.mvp.tree_expansion.banlist_check",
                    return_value={"status": "clear"}), \
             patch("real_proj.mvp.tree_expansion._is_buyable", return_value=True), \
             patch("real_proj.mvp.tree_expansion._resolve_name", return_value="ethanol"):
            node = _build_node("CCO", 0, 6, set(), time.time(), 120)
            assert node["status"] == "buyable"
            assert node["is_buyable"] is True
            assert node["name"] == "ethanol"

    def test_depth_limit(self):
        with patch("real_proj.mvp.tree_expansion.banlist_check",
                    return_value={"status": "clear"}), \
             patch("real_proj.mvp.tree_expansion._is_buyable", return_value=False), \
             patch("real_proj.mvp.tree_expansion._resolve_name", return_value=None):
            node = _build_node("CCO", 6, 6, set(), time.time(), 120)
            assert node["status"] == "depth_limit"

    def test_unresolved(self):
        with patch("real_proj.mvp.tree_expansion.banlist_check",
                    return_value={"status": "clear"}), \
             patch("real_proj.mvp.tree_expansion._is_buyable", return_value=False), \
             patch("real_proj.mvp.tree_expansion._find_best_route", return_value=None), \
             patch("real_proj.mvp.tree_expansion._resolve_name", return_value=None):
            node = _build_node("CCO", 0, 6, set(), time.time(), 120)
            assert node["status"] == "unresolved"

    def test_intermediate_with_children(self):
        route = {"reactants": "C.O", "source": "ord", "final_score": 0.8}
        with patch("real_proj.mvp.tree_expansion.banlist_check",
                    return_value={"status": "clear"}), \
             patch("real_proj.mvp.tree_expansion._is_buyable", return_value=False), \
             patch("real_proj.mvp.tree_expansion._find_best_route", return_value=route), \
             patch("real_proj.mvp.tree_expansion._resolve_name", return_value="ethanol"):
            # C and O are simple enough to be buyable by default
            with patch("real_proj.mvp.tree_expansion._build_node",
                       wraps=_build_node) as mock_build:
                node = _build_node("CCO", 0, 6, set(), time.time(), 120)
                assert node["status"] == "intermediate"
                assert len(node["children"]) == 2
                assert node["route"] is not None
                assert "template" not in node["route"]  # template stripped

    def test_node_has_all_required_keys(self):
        with patch("real_proj.mvp.tree_expansion.banlist_check",
                    return_value={"status": "clear"}), \
             patch("real_proj.mvp.tree_expansion._is_buyable", return_value=True), \
             patch("real_proj.mvp.tree_expansion._resolve_name", return_value="ethanol"):
            node = _build_node("CCO", 0, 6, set(), time.time(), 120)
            for key in ("smiles", "name", "status", "depth", "is_buyable", "guard", "route", "children"):
                assert key in node, f"Missing key: {key}"

    def test_banned_before_buyable(self):
        """Banned check should happen before buyability check."""
        call_order = []
        def mock_banlist(s):
            call_order.append("banlist")
            return {"status": "banned", "name": "Bad"}
        def mock_buyable(s):
            call_order.append("buyable")
            return True

        with patch("real_proj.mvp.tree_expansion.banlist_check", side_effect=mock_banlist), \
             patch("real_proj.mvp.tree_expansion._is_buyable", side_effect=mock_buyable), \
             patch("real_proj.mvp.tree_expansion._resolve_name", return_value=None):
            node = _build_node("CCO", 0, 6, set(), time.time(), 120)
            assert node["status"] == "banned"
            assert "buyable" not in call_order  # buyable should not have been called


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
        with patch("real_proj.mvp.tree_expansion.banlist_check",
                    return_value={"status": "clear"}), \
             patch("real_proj.mvp.tree_expansion._is_buyable", return_value=True), \
             patch("real_proj.mvp.tree_expansion._resolve_name", return_value="test"):
            result = expand_tree("CC(=O)Oc1ccccc1C(=O)O", "CCO.CC(=O)O")
            tree = result["tree"]
            assert tree["status"] == "intermediate"
            assert tree["depth"] == 0
            assert len(tree["children"]) == 2
            for child in tree["children"]:
                assert child["status"] == "buyable"
                assert child["is_buyable"] is True

    def test_stats_counts(self):
        with patch("real_proj.mvp.tree_expansion.banlist_check",
                    return_value={"status": "clear"}), \
             patch("real_proj.mvp.tree_expansion._is_buyable", return_value=True), \
             patch("real_proj.mvp.tree_expansion._resolve_name", return_value="test"):
            result = expand_tree("CC(=O)Oc1ccccc1C(=O)O", "CCO.CC(=O)O")
            stats = result["stats"]
            assert stats["total_nodes"] == 3  # root + 2 children
            assert stats["buyable_count"] == 2
            assert stats["banned_count"] == 0
            assert stats["elapsed_sec"] >= 0

    def test_root_has_selected_route(self):
        with patch("real_proj.mvp.tree_expansion.banlist_check",
                    return_value={"status": "clear"}), \
             patch("real_proj.mvp.tree_expansion._is_buyable", return_value=True), \
             patch("real_proj.mvp.tree_expansion._resolve_name", return_value="test"):
            result = expand_tree("CC(=O)Oc1ccccc1C(=O)O", "CCO.CC(=O)O")
            root = result["tree"]
            assert root["route"]["source"] == "selected"
            assert root["route"]["reactants"] == "CCO.CC(=O)O"

    def test_max_depth_respected(self):
        """With max_depth=1, children should not recurse further."""
        route = {"reactants": "C.O", "source": "ord", "final_score": 0.8}
        with patch("real_proj.mvp.tree_expansion.banlist_check",
                    return_value={"status": "clear"}), \
             patch("real_proj.mvp.tree_expansion._is_buyable", return_value=False), \
             patch("real_proj.mvp.tree_expansion._find_best_route", return_value=None), \
             patch("real_proj.mvp.tree_expansion._resolve_name", return_value=None):
            result = expand_tree("CC(=O)Oc1ccccc1C(=O)O", "CCO", max_depth=1)
            # Child at depth=1 should be depth_limit or unresolved
            child = result["tree"]["children"][0]
            assert child["status"] in ("depth_limit", "unresolved")


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

    def test_walk_counts_correctly(self):
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
        assert counts["total"] == 5
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
        assert stats["unresolved_count"] == 1  # timeout counts as unresolved
        assert stats["max_depth_reached"] == 1
        assert stats["elapsed_sec"] == 2.5

    def test_circular_counts_as_unresolved(self):
        tree = {"status": "circular", "depth": 2, "children": []}
        counts = {"total": 0, "buyable": 0, "banned": 0, "unresolved": 0, "max_depth": 0}
        _walk(tree, counts)
        assert counts["unresolved"] == 1
