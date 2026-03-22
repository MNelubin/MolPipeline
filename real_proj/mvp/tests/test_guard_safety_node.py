"""Tests for guard_safety_node and _collect_all_smiles.

banlist_check is mocked to avoid filesystem/network access.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from ..nodes.guard_safety_node import guard_safety_node, _collect_all_smiles


def _clear(smiles=""):
    return {"status": "clear", "smiles": smiles}

def _banned(smiles="", reason="DEA Schedule I"):
    return {"status": "banned", "smiles": smiles, "reason": reason}

def _restricted(smiles="", reason="Class 3"):
    return {"status": "restricted", "smiles": smiles, "reason": reason}


def _patch_banlist(side_effect):
    return patch("real_proj.mvp.nodes.guard_safety_node.banlist_check", side_effect=side_effect)


# ═════════════════════════════════════════════════════════════════════════════
# _collect_all_smiles
# ═════════════════════════════════════════════════════════════════════════════

class TestCollectAllSmiles:
    def test_leaf_node(self):
        node = {"smiles": "CCO", "children": []}
        assert _collect_all_smiles(node) == ["CCO"]

    def test_root_with_two_children(self):
        node = {
            "smiles": "CC(=O)Oc1ccccc1C(=O)O",
            "children": [
                {"smiles": "OC(=O)c1ccccc1O", "children": []},
                {"smiles": "CC(=O)OC(C)=O", "children": []},
            ],
        }
        result = _collect_all_smiles(node)
        assert len(result) == 3
        assert "CC(=O)Oc1ccccc1C(=O)O" in result
        assert "OC(=O)c1ccccc1O" in result

    def test_nested_tree(self):
        node = {
            "smiles": "A",
            "children": [
                {"smiles": "B", "children": [
                    {"smiles": "C", "children": []},
                    {"smiles": "D", "children": []},
                ]},
                {"smiles": "E", "children": []},
            ],
        }
        result = _collect_all_smiles(node)
        assert set(result) == {"A", "B", "C", "D", "E"}

    def test_node_without_smiles_skipped(self):
        node = {"children": [{"smiles": "CCO", "children": []}]}
        result = _collect_all_smiles(node)
        assert result == ["CCO"]

    def test_empty_tree(self):
        result = _collect_all_smiles({"smiles": "", "children": []})
        assert result == []


# ═════════════════════════════════════════════════════════════════════════════
# guard_safety_node — empty state
# ═════════════════════════════════════════════════════════════════════════════

class TestGuardSafetyEmpty:
    def test_empty_state(self):
        result = guard_safety_node({})
        r = result["safety_report"]
        assert r["pathway_reports"] == []
        assert r["has_critical"] is False
        assert r["warnings"] == []

    def test_empty_routes(self):
        result = guard_safety_node({"retro_result": {"routes": []}})
        r = result["safety_report"]
        assert r["has_critical"] is False


# ═════════════════════════════════════════════════════════════════════════════
# guard_safety_node — via tree
# ═════════════════════════════════════════════════════════════════════════════

class TestGuardSafetyViaTree:
    def _make_tree_route(self, smiles_list):
        """Build a flat tree: root → children."""
        children = [{"smiles": s, "children": []} for s in smiles_list[1:]]
        return {
            "tree": {"smiles": smiles_list[0], "children": children},
            "reactants": ".".join(smiles_list[1:]),
        }

    def test_all_clear_no_critical(self):
        route = self._make_tree_route(["CC(=O)Oc1ccccc1C(=O)O", "OC(=O)c1ccccc1O", "CC(=O)OC(C)=O"])
        with _patch_banlist(lambda smi: _clear(smi)):
            result = guard_safety_node({"retro_result": {"routes": [route]}})
        r = result["safety_report"]
        assert r["has_critical"] is False
        assert r["warnings"] == []

    def test_banned_reactant_sets_critical(self):
        route = self._make_tree_route(["CC(=O)Oc1ccccc1C(=O)O", "CCO", "BAD_SMILES"])

        def banlist(smi):
            if smi == "BAD_SMILES":
                return _banned(smi)
            return _clear(smi)

        with _patch_banlist(banlist):
            result = guard_safety_node({"retro_result": {"routes": [route]}})
        r = result["safety_report"]
        assert r["has_critical"] is True
        assert len(r["warnings"]) >= 1

    def test_restricted_no_critical_but_warning(self):
        route = self._make_tree_route(["root", "restricted_smi"])

        def banlist(smi):
            if smi == "restricted_smi":
                return _restricted(smi)
            return _clear(smi)

        with _patch_banlist(banlist):
            result = guard_safety_node({"retro_result": {"routes": [route]}})
        r = result["safety_report"]
        assert r["has_critical"] is False
        assert len(r["warnings"]) >= 1

    def test_pathway_report_has_correct_index(self):
        routes = [
            self._make_tree_route(["A", "B"]),
            self._make_tree_route(["C", "D"]),
        ]
        with _patch_banlist(lambda smi: _clear(smi)):
            result = guard_safety_node({"retro_result": {"routes": routes}})
        reports = result["safety_report"]["pathway_reports"]
        assert reports[0]["pathway_index"] == 0
        assert reports[1]["pathway_index"] == 1


# ═════════════════════════════════════════════════════════════════════════════
# guard_safety_node — fallback (no tree)
# ═════════════════════════════════════════════════════════════════════════════

class TestGuardSafetyFallback:
    def _make_immediate_route(self, reactants):
        return {"reactants": reactants}

    def test_immediate_reactants_checked(self):
        route = self._make_immediate_route("CCO.CC(=O)O")
        checked = []

        def banlist(smi):
            checked.append(smi)
            return _clear(smi)

        with _patch_banlist(banlist):
            guard_safety_node({"retro_result": {"routes": [route]}})

        assert "CCO" in checked
        assert "CC(=O)O" in checked

    def test_banned_immediate_reactant_sets_critical(self):
        route = self._make_immediate_route("CCO.BANNED")

        def banlist(smi):
            return _banned(smi) if smi == "BANNED" else _clear(smi)

        with _patch_banlist(banlist):
            result = guard_safety_node({"retro_result": {"routes": [route]}})
        assert result["safety_report"]["has_critical"] is True
