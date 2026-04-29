"""Tests for stoichiometry_node, _calc_from_tree, _calc_single_step, _calc_node.

stoichiometry_calc and AgentJournal are mocked to avoid external calls.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from ..nodes.stoichiometry_node import stoichiometry_node, _calc_single_step, _calc_from_tree


# ── Shared mocks ─────────────────────────────────────────────────────────────

def _mock_calc_result(reagents=None, target_moles=0.0055, warnings=None):
    """Build a mock StoichiometryCalcResult-like object."""
    m = MagicMock()
    m.model_dump.return_value = {
        "reagents": reagents or [
            {"smiles": "OC(=O)c1ccccc1O", "name": "salicylic acid",
             "mass_g": 0.72, "moles": 0.0055, "equivalents": 1.0, "volume_ml": None},
            {"smiles": "CC(=O)OC(C)=O", "name": "acetic anhydride",
             "mass_g": 0.56, "moles": 0.0055, "equivalents": 1.0, "volume_ml": None},
        ],
        "target_moles": target_moles,
        "target_mass_g": 1.0,
        "target_product_smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "warnings": warnings or [],
    }
    return m


@pytest.fixture(autouse=True)
def mock_journal():
    mock_j = MagicMock()
    mock_j.step.return_value.__enter__ = lambda s: None
    mock_j.step.return_value.__exit__ = MagicMock(return_value=False)
    with patch("mvp.journal.AgentJournal") as cls:
        cls.for_session.return_value = mock_j
        yield mock_j


@pytest.fixture()
def mock_stoichio():
    with patch("mvp.nodes.stoichiometry_node.stoichiometry_calc",
               return_value=_mock_calc_result()) as m:
        yield m


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def aspirin_pathway_single():
    """Single-step pathway, no tree."""
    return {
        "reactants": "OC(=O)c1ccccc1O.CC(=O)OC(C)=O",
        "reaction_smiles": "OC(=O)c1ccccc1O.CC(=O)OC(C)=O>>CC(=O)Oc1ccccc1C(=O)O",
        "source": "ord",
        "final_score": 0.85,
    }


@pytest.fixture()
def aspirin_pathway_with_tree():
    """Pathway with a two-level tree."""
    return {
        "reactants": "OC(=O)c1ccccc1O.CC(=O)OC(C)=O",
        "reaction_smiles": "OC(=O)c1ccccc1O.CC(=O)OC(C)=O>>CC(=O)Oc1ccccc1C(=O)O",
        "source": "ord",
        "final_score": 0.85,
        "tree": {
            "smiles": "CC(=O)Oc1ccccc1C(=O)O",
            "name": "aspirin",
            "status": "intermediate",
            "children": [
                {"smiles": "OC(=O)c1ccccc1O", "name": "salicylic acid", "status": "buyable", "children": [], "route": None},
                {"smiles": "CC(=O)OC(C)=O", "name": "acetic anhydride", "status": "buyable", "children": [], "route": None},
            ],
            "route": {
                "reactants": "OC(=O)c1ccccc1O.CC(=O)OC(C)=O",
                "reaction_smiles": "OC(=O)c1ccccc1O.CC(=O)OC(C)=O>>CC(=O)Oc1ccccc1C(=O)O",
            },
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# stoichiometry_node — error cases (no mocking needed)
# ═════════════════════════════════════════════════════════════════════════════

class TestStoichiometryNodeErrors:
    def test_no_selected_pathway_returns_error(self):
        result = stoichiometry_node({"synthesis_pathways": [{}]})
        assert "error" in result["calculations"]

    def test_empty_pathways_returns_error(self):
        result = stoichiometry_node({"synthesis_pathways": [], "selected_pathway": 0})
        assert "error" in result["calculations"]

    def test_out_of_range_index_returns_error(self):
        result = stoichiometry_node({
            "synthesis_pathways": [{}],
            "selected_pathway": 5,
        })
        assert "error" in result["calculations"]

    def test_negative_index_returns_error(self):
        result = stoichiometry_node({
            "synthesis_pathways": [{}],
            "selected_pathway": -1,
        })
        assert "error" in result["calculations"]


# ═════════════════════════════════════════════════════════════════════════════
# stoichiometry_node — single-step pathway
# ═════════════════════════════════════════════════════════════════════════════

class TestStoichiometryNodeSingleStep:
    def test_returns_calculations_key(self, aspirin_pathway_single, mock_stoichio):
        state = {
            "synthesis_pathways": [aspirin_pathway_single],
            "selected_pathway": 0,
            "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        }
        result = stoichiometry_node(state)
        assert "calculations" in result

    def test_target_mass_defaults_to_1g(self, aspirin_pathway_single, mock_stoichio):
        state = {
            "synthesis_pathways": [aspirin_pathway_single],
            "selected_pathway": 0,
        }
        stoichiometry_node(state)
        call_args = mock_stoichio.call_args[0][0]
        assert call_args.target_mass_g == pytest.approx(1.0)

    def test_custom_target_mass(self, aspirin_pathway_single, mock_stoichio):
        state = {
            "synthesis_pathways": [aspirin_pathway_single],
            "selected_pathway": 0,
            "target_amount": {"value": 5.0, "unit": "g"},
        }
        stoichiometry_node(state)
        call_args = mock_stoichio.call_args[0][0]
        assert call_args.target_mass_g == pytest.approx(5.0)

    def test_reagents_in_output(self, aspirin_pathway_single, mock_stoichio):
        state = {
            "synthesis_pathways": [aspirin_pathway_single],
            "selected_pathway": 0,
        }
        result = stoichiometry_node(state)
        calc = result["calculations"]
        assert "reagents" in calc
        assert len(calc["reagents"]) > 0


# ═════════════════════════════════════════════════════════════════════════════
# stoichiometry_node — tree pathway
# ═════════════════════════════════════════════════════════════════════════════

class TestStoichiometryNodeTree:
    def test_tree_pathway_returns_steps(self, aspirin_pathway_with_tree, mock_stoichio):
        state = {
            "synthesis_pathways": [aspirin_pathway_with_tree],
            "selected_pathway": 0,
            "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        }
        result = stoichiometry_node(state)
        calc = result["calculations"]
        assert "steps" in calc

    def test_tree_pathway_returns_buyable_reagents(self, aspirin_pathway_with_tree, mock_stoichio):
        state = {
            "synthesis_pathways": [aspirin_pathway_with_tree],
            "selected_pathway": 0,
        }
        result = stoichiometry_node(state)
        calc = result["calculations"]
        assert "all_buyable_reagents" in calc

    def test_tree_pathway_target_mass_stored(self, aspirin_pathway_with_tree, mock_stoichio):
        state = {
            "synthesis_pathways": [aspirin_pathway_with_tree],
            "selected_pathway": 0,
            "target_amount": {"value": 2.5, "unit": "g"},
        }
        result = stoichiometry_node(state)
        calc = result["calculations"]
        assert calc["target_mass_g"] == pytest.approx(2.5)


# ═════════════════════════════════════════════════════════════════════════════
# _calc_single_step — direct unit tests
# ═════════════════════════════════════════════════════════════════════════════

class TestCalcSingleStep:
    def test_valid_reaction_returns_calculations(self, mock_stoichio):
        pathway = {
            "reaction_smiles": "OC(=O)c1ccccc1O.CC(=O)OC(C)=O>>CC(=O)Oc1ccccc1C(=O)O",
        }
        result = _calc_single_step(pathway, 1.0, "CC(=O)Oc1ccccc1C(=O)O")
        assert "calculations" in result
        assert "error" not in result["calculations"]

    def test_no_reaction_smiles_returns_error(self):
        pathway = {}
        result = _calc_single_step(pathway, 1.0, "")
        assert "error" in result["calculations"]

    def test_reaction_smiles_built_from_reactants(self, mock_stoichio):
        pathway = {"reactants": "CCO.CC(=O)O"}
        result = _calc_single_step(pathway, 1.0, "CC(=O)OCC")
        # Should not error — builds reaction SMILES from reactants + target
        assert "calculations" in result

    def test_stoichio_exception_returns_error(self):
        pathway = {
            "reaction_smiles": "CCO.CC(=O)O>>CC(=O)OCC",
        }
        with patch("mvp.nodes.stoichiometry_node.stoichiometry_calc",
                   side_effect=Exception("calc failed")):
            result = _calc_single_step(pathway, 1.0, "CC(=O)OCC")
        assert "error" in result["calculations"]

