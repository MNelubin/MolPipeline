"""Tests for aggregate_node: fan-in merge of safety + reagent reports.

Pure Python data manipulation — no external calls, no mocking required.
"""

from __future__ import annotations

import pytest

from ..nodes.aggregate_node import aggregate_node


def _make_route(score=0.8, unresolved=0, buyable=2, tree=True):
    node = {
        "reactants": "CCO.CC(=O)O",
        "reaction_smiles": "CCO.CC(=O)O>>CC(=O)OCC",
        "source": "ord",
        "final_score": score,
    }
    if tree:
        node["tree_stats"] = {
            "unresolved_count": unresolved,
            "buyable_count": buyable,
        }
    return node


def _make_reagent_report(available=True):
    return {"all_available": available, "unavailable": []}


def _make_safety_report(critical=False):
    return {"has_critical": critical}


# ═════════════════════════════════════════════════════════════════════════════
# Empty / missing state
# ═════════════════════════════════════════════════════════════════════════════

class TestAggregateEmpty:
    def test_empty_state_returns_empty_pathways(self):
        result = aggregate_node({})
        assert result["synthesis_pathways"] == []

    def test_no_routes_no_reports(self):
        result = aggregate_node({"retro_result": {"routes": []}})
        assert result["synthesis_pathways"] == []

    def test_missing_retro_result(self):
        result = aggregate_node({"reagent_report": {}, "safety_report": {}})
        assert result["synthesis_pathways"] == []


# ═════════════════════════════════════════════════════════════════════════════
# Single pathway
# ═════════════════════════════════════════════════════════════════════════════

class TestAggregateSingle:
    def _state(self, reagents_ok=True, safety_ok=True, score=0.8):
        return {
            "retro_result": {"routes": [_make_route(score=score)]},
            "reagent_report": {"pathway_reports": [_make_reagent_report(reagents_ok)]},
            "safety_report": {"pathway_reports": [_make_safety_report(not safety_ok)]},
        }

    def test_viable_when_both_ok(self):
        result = aggregate_node(self._state(True, True))
        assert result["synthesis_pathways"][0]["viable"] is True

    def test_not_viable_when_reagents_unavailable(self):
        result = aggregate_node(self._state(reagents_ok=False))
        assert result["synthesis_pathways"][0]["viable"] is False

    def test_not_viable_when_safety_critical(self):
        result = aggregate_node(self._state(safety_ok=False))
        assert result["synthesis_pathways"][0]["viable"] is False

    def test_pathway_has_reagents_available_flag(self):
        result = aggregate_node(self._state(True, True))
        p = result["synthesis_pathways"][0]
        assert "reagents_available" in p

    def test_pathway_has_safety_ok_flag(self):
        result = aggregate_node(self._state(True, True))
        p = result["synthesis_pathways"][0]
        assert "safety_ok" in p

    def test_pathway_has_buyable_leaves(self):
        result = aggregate_node(self._state())
        p = result["synthesis_pathways"][0]
        assert "buyable_leaves" in p

    def test_pathway_has_unresolved_leaves(self):
        result = aggregate_node(self._state())
        p = result["synthesis_pathways"][0]
        assert "unresolved_leaves" in p


# ═════════════════════════════════════════════════════════════════════════════
# Sorting logic
# ═════════════════════════════════════════════════════════════════════════════

class TestAggregateSorting:
    def _two_pathway_state(self, r1_ok, s1_ok, score1, r2_ok, s2_ok, score2):
        return {
            "retro_result": {"routes": [_make_route(score1, 0), _make_route(score2, 0)]},
            "reagent_report": {
                "pathway_reports": [_make_reagent_report(r1_ok), _make_reagent_report(r2_ok)]
            },
            "safety_report": {
                "pathway_reports": [_make_safety_report(not s1_ok), _make_safety_report(not s2_ok)]
            },
        }

    def test_viable_before_non_viable(self):
        # Route 1: not viable (low score), Route 2: viable (low score)
        state = self._two_pathway_state(
            r1_ok=False, s1_ok=True, score1=0.9,
            r2_ok=True,  s2_ok=True, score2=0.5,
        )
        result = aggregate_node(state)
        pathways = result["synthesis_pathways"]
        assert pathways[0]["viable"] is True
        assert pathways[1]["viable"] is False

    def test_higher_score_wins_among_viable(self):
        state = self._two_pathway_state(
            r1_ok=True, s1_ok=True, score1=0.6,
            r2_ok=True, s2_ok=True, score2=0.9,
        )
        result = aggregate_node(state)
        pathways = result["synthesis_pathways"]
        assert pathways[0]["final_score"] == pytest.approx(0.9)
        assert pathways[1]["final_score"] == pytest.approx(0.6)

    def test_fewer_unresolved_wins_among_viable(self):
        routes = [_make_route(0.8, unresolved=3), _make_route(0.8, unresolved=0)]
        state = {
            "retro_result": {"routes": routes},
            "reagent_report": {"pathway_reports": [
                _make_reagent_report(True), _make_reagent_report(True)
            ]},
            "safety_report": {"pathway_reports": [
                _make_safety_report(False), _make_safety_report(False)
            ]},
        }
        result = aggregate_node(state)
        pathways = result["synthesis_pathways"]
        assert pathways[0]["unresolved_leaves"] == 0

    def test_returns_all_pathways(self):
        routes = [_make_route(0.8), _make_route(0.7), _make_route(0.6)]
        state = {
            "retro_result": {"routes": routes},
            "reagent_report": {"pathway_reports": [_make_reagent_report()] * 3},
            "safety_report": {"pathway_reports": [_make_safety_report()] * 3},
        }
        result = aggregate_node(state)
        assert len(result["synthesis_pathways"]) == 3


# ═════════════════════════════════════════════════════════════════════════════
# Missing / mismatched reports (edge cases)
# ═════════════════════════════════════════════════════════════════════════════

class TestAggregateMissingReports:
    def test_no_reagent_report_defaults_to_available(self):
        state = {
            "retro_result": {"routes": [_make_route()]},
            "safety_report": {"pathway_reports": [_make_safety_report()]},
        }
        result = aggregate_node(state)
        assert result["synthesis_pathways"][0]["reagents_available"] is True

    def test_no_safety_report_defaults_to_ok(self):
        state = {
            "retro_result": {"routes": [_make_route()]},
            "reagent_report": {"pathway_reports": [_make_reagent_report()]},
        }
        result = aggregate_node(state)
        assert result["synthesis_pathways"][0]["safety_ok"] is True

    def test_fewer_reports_than_routes(self):
        state = {
            "retro_result": {"routes": [_make_route(), _make_route()]},
            "reagent_report": {"pathway_reports": [_make_reagent_report()]},
            "safety_report": {"pathway_reports": []},
        }
        result = aggregate_node(state)
        assert len(result["synthesis_pathways"]) == 2

    def test_route_without_tree_stats(self):
        route = _make_route(tree=False)
        state = {
            "retro_result": {"routes": [route]},
            "reagent_report": {"pathway_reports": [_make_reagent_report()]},
            "safety_report": {"pathway_reports": [_make_safety_report()]},
        }
        result = aggregate_node(state)
        p = result["synthesis_pathways"][0]
        assert p["unresolved_leaves"] == 0
        assert p["buyable_leaves"] == 0
