"""Tests for reagent_node, _collect_leaves, _check_via_tree, _check_via_immediate.

_is_buyable is mocked to avoid ASKCOS/network calls.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ..nodes.reagent_node import (
    reagent_node,
    _collect_leaves,
    _check_via_tree,
    _check_via_immediate,
)


def _patch_buyable(buyable_smiles: set):
    return patch(
        "mvp.nodes.reagent_node._is_buyable",
        side_effect=lambda smi: smi in buyable_smiles,
    )


# ═════════════════════════════════════════════════════════════════════════════
# _collect_leaves
# ═════════════════════════════════════════════════════════════════════════════

class TestCollectLeaves:
    def test_single_leaf(self):
        node = {"smiles": "CCO", "children": []}
        leaves = _collect_leaves(node)
        assert len(leaves) == 1
        assert leaves[0]["smiles"] == "CCO"

    def test_root_with_two_children(self):
        node = {
            "smiles": "ROOT",
            "children": [
                {"smiles": "A", "children": []},
                {"smiles": "B", "children": []},
            ],
        }
        leaves = _collect_leaves(node)
        smiles = [l["smiles"] for l in leaves]
        assert sorted(smiles) == ["A", "B"]

    def test_nested_tree_returns_only_leaves(self):
        node = {
            "smiles": "ROOT",
            "children": [
                {
                    "smiles": "INT",
                    "children": [
                        {"smiles": "LEAF1", "children": []},
                        {"smiles": "LEAF2", "children": []},
                    ],
                },
                {"smiles": "LEAF3", "children": []},
            ],
        }
        leaves = _collect_leaves(node)
        smiles = {l["smiles"] for l in leaves}
        assert smiles == {"LEAF1", "LEAF2", "LEAF3"}
        assert "ROOT" not in smiles
        assert "INT" not in smiles

    def test_empty_children_list_is_leaf(self):
        node = {"smiles": "X", "children": []}
        leaves = _collect_leaves(node)
        assert len(leaves) == 1


# ═════════════════════════════════════════════════════════════════════════════
# _check_via_tree
# ═════════════════════════════════════════════════════════════════════════════

class TestCheckViaTree:
    def _leaf(self, smiles, status="buyable"):
        return {"smiles": smiles, "status": status, "children": []}

    def _tree(self, *leaves):
        return {"smiles": "ROOT", "children": list(leaves)}

    def test_all_buyable(self):
        tree = self._tree(self._leaf("CCO", "buyable"), self._leaf("CC(=O)O", "buyable"))
        report = _check_via_tree(0, tree)
        assert report["all_available"] is True
        assert len(report["unavailable"]) == 0

    def test_one_banned_unavailable(self):
        tree = self._tree(self._leaf("CCO", "buyable"), self._leaf("EVIL", "banned"))
        report = _check_via_tree(0, tree)
        assert report["all_available"] is False
        assert "EVIL" in report["unavailable"]

    def test_unresolved_is_unavailable(self):
        tree = self._tree(self._leaf("CCO", "buyable"), self._leaf("UNKNOWN", "unresolved"))
        report = _check_via_tree(0, tree)
        assert report["all_available"] is False
        assert "UNKNOWN" in report["unavailable"]

    def test_pathway_index_stored(self):
        tree = self._tree(self._leaf("CCO"))
        report = _check_via_tree(3, tree)
        assert report["pathway_index"] == 3

    def test_source_is_tree(self):
        tree = self._tree(self._leaf("CCO"))
        report = _check_via_tree(0, tree)
        assert report["source"] == "tree"

    def test_total_reagents_count(self):
        tree = self._tree(self._leaf("A"), self._leaf("B"), self._leaf("C"))
        report = _check_via_tree(0, tree)
        assert report["total_reagents"] == 3


# ═════════════════════════════════════════════════════════════════════════════
# _check_via_immediate
# ═════════════════════════════════════════════════════════════════════════════

class TestCheckViaImmediate:
    def test_all_buyable(self):
        route = {"reactants": "CCO.CC(=O)O"}
        with _patch_buyable({"CCO", "CC(=O)O"}):
            report = _check_via_immediate(0, route)
        assert report["all_available"] is True

    def test_one_not_buyable(self):
        route = {"reactants": "CCO.OBSCURE"}
        with _patch_buyable({"CCO"}):
            report = _check_via_immediate(0, route)
        assert report["all_available"] is False
        assert "OBSCURE" in report["unavailable"]

    def test_source_is_immediate(self):
        route = {"reactants": "CCO"}
        with _patch_buyable({"CCO"}):
            report = _check_via_immediate(0, route)
        assert report["source"] == "immediate"

    def test_empty_reactants(self):
        route = {"reactants": ""}
        with _patch_buyable(set()):
            report = _check_via_immediate(0, route)
        assert report["total_reagents"] == 0
        assert report["all_available"] is True

    def test_pathway_index_stored(self):
        route = {"reactants": "CCO"}
        with _patch_buyable({"CCO"}):
            report = _check_via_immediate(5, route)
        assert report["pathway_index"] == 5


# ═════════════════════════════════════════════════════════════════════════════
# reagent_node — state wiring
# ═════════════════════════════════════════════════════════════════════════════

class TestReagentNode:
    def test_empty_state(self):
        result = reagent_node({})
        r = result["reagent_report"]
        assert r["pathway_reports"] == []
        assert r["all_available"] is True
        assert r["unavailable_reagents"] == []

    def test_empty_routes(self):
        result = reagent_node({"retro_result": {"routes": []}})
        r = result["reagent_report"]
        assert r["all_available"] is True

    def test_tree_route_used_when_tree_present(self):
        leaf = {"smiles": "CCO", "status": "buyable", "children": []}
        route = {"tree": {"smiles": "ROOT", "children": [leaf]}}
        result = reagent_node({"retro_result": {"routes": [route]}})
        r = result["reagent_report"]
        assert r["all_available"] is True

    def test_immediate_fallback_without_tree(self):
        route = {"reactants": "CCO.CC(=O)O"}
        with _patch_buyable({"CCO", "CC(=O)O"}):
            result = reagent_node({"retro_result": {"routes": [route]}})
        r = result["reagent_report"]
        assert r["all_available"] is True

    def test_multiple_pathways_aggregated(self):
        leaf_ok = {"smiles": "CCO", "status": "buyable", "children": []}
        leaf_bad = {"smiles": "EVIL", "status": "banned", "children": []}
        routes = [
            {"tree": {"smiles": "ROOT1", "children": [leaf_ok]}},
            {"tree": {"smiles": "ROOT2", "children": [leaf_bad]}},
        ]
        result = reagent_node({"retro_result": {"routes": routes}})
        r = result["reagent_report"]
        assert r["all_available"] is False
        assert "EVIL" in r["unavailable_reagents"]

    def test_unavailable_not_duplicated(self):
        leaf = {"smiles": "EVIL", "status": "banned", "children": []}
        routes = [
            {"tree": {"smiles": "R1", "children": [leaf]}},
            {"tree": {"smiles": "R2", "children": [leaf]}},
        ]
        result = reagent_node({"retro_result": {"routes": routes}})
        unavail = result["reagent_report"]["unavailable_reagents"]
        assert unavail.count("EVIL") == 1

