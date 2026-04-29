"""Tests for experiment_planner_node, helper builders, and _find_procedure_cascade.

Heavy external calls (ORD, RAG, LLM) are mocked.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from ..nodes.experiment_planner_node import (
    experiment_planner_node,
    _build_multistep_protocol,
    _build_single_step_protocol,
    _build_reagent_table_from_step,
    _build_reagent_table_from_list,
    _build_reagent_table_from_calc,
    _find_procedure_cascade,
    _format_protocol_text,
    _find_node_by_smiles,
)


# ── Journal patch ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_journal():
    mock_j = MagicMock()
    mock_j.step.return_value.__enter__ = lambda s: None
    mock_j.step.return_value.__exit__ = MagicMock(return_value=False)
    mock_j.export_markdown.return_value = None
    with patch("mvp.journal.AgentJournal") as cls:
        cls.for_session.return_value = mock_j
        yield mock_j


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def simple_pathway():
    return {
        "reactants": "OC(=O)c1ccccc1O.CC(=O)OC(C)=O",
        "reaction_smiles": "OC(=O)c1ccccc1O.CC(=O)OC(C)=O>>CC(=O)Oc1ccccc1C(=O)O",
        "procedure_details": "Mix salicylic acid with acetic anhydride and stir for 1 hour at room temperature.",
        "temperature": "25°C",
        "solvent": "CC(=O)O",
        "source": "ord",
    }


@pytest.fixture()
def simple_calculations():
    return {
        "reagents": [
            {"smiles": "OC(=O)c1ccccc1O", "name": "salicylic acid",
             "mass_g": 0.72, "moles": 0.0055, "equivalents": 1.0, "volume_ml": None},
            {"smiles": "CC(=O)OC(C)=O", "name": "acetic anhydride",
             "mass_g": 0.56, "moles": 0.0055, "equivalents": 1.0, "volume_ml": None},
        ],
        "target_mass_g": 1.0,
        "target_product_smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "target_moles": 0.0055,
        "warnings": [],
    }


@pytest.fixture()
def base_state(simple_pathway, simple_calculations):
    return {
        "synthesis_pathways": [simple_pathway],
        "selected_pathway": 0,
        "calculations": simple_calculations,
        "molecule_info": {"name": "Aspirin"},
    }


# ═════════════════════════════════════════════════════════════════════════════
# experiment_planner_node — error cases
# ═════════════════════════════════════════════════════════════════════════════

class TestExperimentPlannerErrors:
    def test_no_selected_pathway_returns_error(self):
        result = experiment_planner_node({"synthesis_pathways": [{}]})
        assert "error" in result["experiment_protocol"]

    def test_empty_pathways_returns_error(self):
        result = experiment_planner_node({
            "synthesis_pathways": [],
            "selected_pathway": 0,
        })
        assert "error" in result["experiment_protocol"]


# ═════════════════════════════════════════════════════════════════════════════
# experiment_planner_node — single-step protocol
# ═════════════════════════════════════════════════════════════════════════════

class TestExperimentPlannerSingleStep:
    def test_returns_experiment_protocol(self, base_state):
        with patch("mvp.nodes.experiment_planner_node.format_procedure_russian",
                   return_value=[{"step": "1", "description": "Mix reagents.", "reason": "ORD процедура"}]):
            result = experiment_planner_node(base_state)
        assert "experiment_protocol" in result
        assert "error" not in result["experiment_protocol"]

    def test_protocol_has_reaction_sections(self, base_state):
        with patch("mvp.nodes.experiment_planner_node.format_procedure_russian",
                   return_value=[{"step": "1", "description": "Mix.", "reason": "ORD процедура"}]):
            result = experiment_planner_node(base_state)
        sections = result["experiment_protocol"]["reaction_sections"]
        assert len(sections) == 1

    def test_protocol_title_contains_molecule(self, base_state):
        with patch("mvp.nodes.experiment_planner_node.format_procedure_russian",
                   return_value=[]):
            result = experiment_planner_node(base_state)
        assert "Aspirin" in result["experiment_protocol"]["title"]

    def test_sets_current_phase_experiment(self, base_state):
        with patch("mvp.nodes.experiment_planner_node.format_procedure_russian",
                   return_value=[]):
            result = experiment_planner_node(base_state)
        assert result["current_phase"] == "experiment"

    def test_final_answer_contains_protocol_marker(self, base_state):
        with patch("mvp.nodes.experiment_planner_node.format_procedure_russian",
                   return_value=[{"step": "1", "description": "Mix.", "reason": "ORD процедура"}]):
            result = experiment_planner_node(base_state)
        assert "ПРОТОКОЛ" in result["final_answer"]


# ═════════════════════════════════════════════════════════════════════════════
# _build_reagent_table_from_step
# ═════════════════════════════════════════════════════════════════════════════

class TestBuildReagentTableFromStep:
    def _step(self, reagents):
        return {"reagents": reagents}

    def test_empty_reagents(self):
        assert _build_reagent_table_from_step(self._step([])) == []

    def test_reagent_fields_present(self):
        step = self._step([{
            "smiles": "CCO", "name": "ethanol",
            "mass_g": 0.5, "moles": 0.011,
            "equivalents": 1.0, "volume_ml": 0.63,
        }])
        table = _build_reagent_table_from_step(step)
        assert len(table) == 1
        r = table[0]
        assert r["smiles"] == "CCO"
        assert r["mass_g"] == pytest.approx(0.5)
        assert r["volume_ml"] == pytest.approx(0.63)

    def test_name_truncated_to_50(self):
        long_name = "A" * 60
        step = self._step([{"smiles": "CCO", "name": long_name, "mass_g": 0, "moles": 0}])
        table = _build_reagent_table_from_step(step)
        assert len(table[0]["name"]) <= 50

    def test_missing_fields_use_defaults(self):
        step = self._step([{"smiles": "CCO"}])
        table = _build_reagent_table_from_step(step)
        assert table[0]["equivalents"] == pytest.approx(1.0)
        assert table[0]["moles"] == 0
        assert table[0]["mass_g"] == 0


# ═════════════════════════════════════════════════════════════════════════════
# _build_reagent_table_from_calc / _build_reagent_table_from_list
# ═════════════════════════════════════════════════════════════════════════════

class TestBuildReagentTableFromCalc:
    def test_flat_calc_reagents(self):
        calc = {
            "reagents": [
                {"smiles": "CCO", "name": "ethanol", "mass_g": 0.5, "moles": 0.011,
                 "equivalents": 1.0, "volume_ml": None},
            ]
        }
        table = _build_reagent_table_from_calc(calc)
        assert len(table) == 1
        assert table[0]["smiles"] == "CCO"

    def test_empty_calc(self):
        assert _build_reagent_table_from_calc({}) == []


class TestBuildReagentTableFromList:
    def test_buyable_list(self):
        reagents = [{"smiles": "CCO", "name": "ethanol", "mass_g": 0.5, "moles": 0.011}]
        table = _build_reagent_table_from_list(reagents)
        assert len(table) == 1

    def test_empty_list(self):
        assert _build_reagent_table_from_list([]) == []


# ═════════════════════════════════════════════════════════════════════════════
# _find_node_by_smiles
# ═════════════════════════════════════════════════════════════════════════════

class TestFindNodeBySmiles:
    def _tree(self):
        return {
            "smiles": "ROOT",
            "children": [
                {"smiles": "A", "children": [
                    {"smiles": "B", "children": []},
                ]},
                {"smiles": "C", "children": []},
            ],
        }

    def test_find_root(self):
        node = _find_node_by_smiles(self._tree(), "ROOT")
        assert node is not None
        assert node["smiles"] == "ROOT"

    def test_find_child(self):
        node = _find_node_by_smiles(self._tree(), "A")
        assert node is not None

    def test_find_grandchild(self):
        node = _find_node_by_smiles(self._tree(), "B")
        assert node is not None

    def test_not_found_returns_none(self):
        node = _find_node_by_smiles(self._tree(), "MISSING")
        assert node is None


# ═════════════════════════════════════════════════════════════════════════════
# _format_protocol_text
# ═════════════════════════════════════════════════════════════════════════════

class TestFormatProtocolText:
    def _minimal_protocol(self, is_multistep=False):
        return {
            "reaction_sections": [
                {
                    "step_number": 1,
                    "product_name": "aspirin",
                    "product_smiles": "CC(=O)Oc1ccccc1C(=O)O",
                    "product_mass_g": 1.0,
                    "reagent_table": [
                        {"name": "salicylic acid", "mass_g": 0.72,
                         "volume_ml": None, "equivalents": 1.0},
                    ],
                    "procedure_steps": [
                        {"step": "1", "description": "Mix reagents.", "reason": "ORD процедура"},
                    ],
                }
            ],
            "buyable_reagent_table": [],
            "calculations": {"target_mass_g": 1.0, "target_product_smiles": "CC(=O)O", "warnings": []},
            "is_multistep": is_multistep,
        }

    def test_contains_protocol_header(self):
        text = _format_protocol_text(self._minimal_protocol(), "aspirin")
        assert "ПРОТОКОЛ" in text

    def test_contains_molecule_name(self):
        text = _format_protocol_text(self._minimal_protocol(), "aspirin")
        assert "aspirin" in text

    def test_contains_reagent_name(self):
        text = _format_protocol_text(self._minimal_protocol(), "aspirin")
        assert "salicylic acid" in text

    def test_contains_procedure_step(self):
        text = _format_protocol_text(self._minimal_protocol(), "aspirin")
        assert "Mix reagents" in text

    def test_warnings_shown(self):
        protocol = self._minimal_protocol()
        protocol["calculations"]["warnings"] = ["Caution: exothermic reaction"]
        text = _format_protocol_text(protocol, "aspirin")
        assert "Caution" in text

    def test_multistep_shows_stage_header(self):
        protocol = self._minimal_protocol(is_multistep=True)
        # Add second section to trigger multi-step display
        protocol["reaction_sections"].append({
            "step_number": 2,
            "product_name": "intermediate",
            "product_smiles": "CCO",
            "product_mass_g": 0.5,
            "reagent_table": [],
            "procedure_steps": [],
        })
        text = _format_protocol_text(protocol, "aspirin")
        assert "СТАДИЯ" in text

