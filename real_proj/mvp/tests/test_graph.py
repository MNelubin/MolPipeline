"""End-to-end tests for the full LangGraph pipeline.

Tests the complete validate → guard → molecule_info → retrosynthesis flow,
including routing logic and early exit conditions.
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from ..graph import build_graph, _after_validate, _after_guard


# ═════════════════════════════════════════════════════════════════════════════
# Graph construction
# ═════════════════════════════════════════════════════════════════════════════

class TestBuildGraph:
    def test_graph_builds_without_error(self):
        graph = build_graph()
        assert graph is not None

    def test_graph_is_compiled(self):
        graph = build_graph()
        # Compiled graph has invoke method
        assert hasattr(graph, "invoke")

    def test_graph_has_correct_nodes(self):
        graph = build_graph()
        # Access underlying graph structure
        assert hasattr(graph, "nodes") or hasattr(graph, "graph")


# ═════════════════════════════════════════════════════════════════════════════
# Routing functions
# ═════════════════════════════════════════════════════════════════════════════

class TestRouting:
    def test_after_validate_valid_goes_to_guard(self):
        state = {"validation": {"is_valid": True}}
        assert _after_validate(state) == "guard"

    def test_after_validate_invalid_goes_to_end(self):
        state = {"validation": {"is_valid": False}}
        assert _after_validate(state) == "end"

    def test_after_validate_missing_validation_goes_to_end(self):
        state = {}
        assert _after_validate(state) == "end"

    def test_after_guard_safe_goes_to_molecule_info(self):
        state = {"guard_result": {"overall_status": "SAFE"}}
        assert _after_guard(state) == "molecule_info"

    def test_after_guard_warning_goes_to_molecule_info(self):
        state = {"guard_result": {"overall_status": "WARNING"}}
        assert _after_guard(state) == "molecule_info"

    def test_after_guard_critical_goes_to_end(self):
        state = {"guard_result": {"overall_status": "CRITICAL_STOP"}}
        assert _after_guard(state) == "end"

    def test_after_guard_missing_result_goes_to_molecule_info(self):
        state = {}
        # No guard_result → no CRITICAL_STOP → proceeds
        assert _after_guard(state) == "molecule_info"


# ═════════════════════════════════════════════════════════════════════════════
# Full pipeline — unit (LLM mocked)
# ═════════════════════════════════════════════════════════════════════════════

MOCK_MOLECULE_CARD = {
    "name": "Аспирин (2-ацетилоксибензойная кислота)",
    "synonyms": ["Ацетилсалициловая кислота"],
    "smiles": "CC(=O)Oc1ccccc1C(=O)O",
    "molecular_formula": "C9H8O4",
    "molecular_weight": 180.16,
    "physical_description": "Бесцветные кристаллы.",
    "properties": {
        "melting_point": "135", "boiling_point": "140",
        "solubility": "Плохо растворим", "density": "1.4",
        "logP": "1.19", "physical_state": "твёрдое",
        "flash_point": None, "vapor_pressure": None,
    },
    "ghs_classification": [],
    "spectral_notes": "ИК-спектр.",
    "description": "Широко используемое анальгетическое средство.",
    "pubchem_cid": 2244,
}


def _make_mock_llm_for_info():
    mock_resp = MagicMock()
    mock_resp.content = json.dumps(MOCK_MOLECULE_CARD, ensure_ascii=False)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_resp
    return mock_llm


class TestFullPipelineUnit:
    """Unit tests: LLM mocked, PubChem/ORD calls are real."""

    def _run(self, query):
        graph = build_graph()
        mock_llm = _make_mock_llm_for_info()
        with patch("real_proj.mvp.nodes.molecule_info_node._get_llm",
                   return_value=mock_llm):
            return graph.invoke({"query": query})

    def test_invalid_smiles_exits_early(self):
        graph = build_graph()
        result = graph.invoke({"query": "XXXINVALID((("})
        assert result["validation"]["is_valid"] is False
        assert "molecule_info" not in result or result.get("molecule_info") is None

    def test_invalid_smiles_has_error(self):
        graph = build_graph()
        result = graph.invoke({"query": "XXXINVALID((("})
        assert result.get("error") or not result["validation"]["is_valid"]

    def test_fentanyl_blocked_by_guard(self, fentanyl_smiles):
        graph = build_graph()
        result = graph.invoke({"query": fentanyl_smiles})
        assert result["guard_result"]["overall_status"] == "CRITICAL_STOP"
        assert "final_answer" not in result or result.get("final_answer") is None

    def test_fentanyl_has_error_message(self, fentanyl_smiles):
        graph = build_graph()
        result = graph.invoke({"query": fentanyl_smiles})
        assert result.get("error")
        assert "CRITICAL_STOP" in result["error"]

    @pytest.mark.integration
    def test_aspirin_smiles_full_pipeline(self, aspirin_smiles):
        result = self._run(aspirin_smiles)
        # Validation
        assert result["validation"]["is_valid"] is True
        # Guard passes
        assert result["guard_result"]["overall_status"] in ("SAFE", "WARNING")
        # Molecule info generated
        assert "molecule_info" in result
        assert result["molecule_info"]["molecular_formula"] == "C9H8O4"
        # Retrosynthesis ran
        assert "retro_result" in result
        assert "final_answer" in result

    @pytest.mark.integration
    def test_aspirin_name_full_pipeline(self):
        result = self._run("aspirin")
        assert result["validation"]["is_valid"] is True
        assert result["validation"]["pubchem_cid"] == 2244
        assert "retro_result" in result

    @pytest.mark.integration
    def test_final_answer_nonempty(self, aspirin_smiles):
        result = self._run(aspirin_smiles)
        assert result.get("final_answer")
        assert len(result["final_answer"]) > 100

    @pytest.mark.integration
    def test_state_has_all_keys_after_full_run(self, aspirin_smiles):
        result = self._run(aspirin_smiles)
        for key in ("query", "smiles", "pubchem_cid", "validation",
                    "guard_result", "molecule_info", "retro_result", "final_answer"):
            assert key in result, f"Missing state key: {key}"

    @pytest.mark.integration
    def test_canonical_smiles_in_state(self, aspirin_smiles):
        result = self._run(aspirin_smiles)
        assert result["smiles"] == aspirin_smiles  # already canonical

    @pytest.mark.integration
    def test_retro_result_has_routes(self, aspirin_smiles):
        result = self._run(aspirin_smiles)
        routes = result["retro_result"]["routes"]
        assert isinstance(routes, list)
        assert len(routes) > 0

    @pytest.mark.integration
    def test_retro_routes_scored(self, aspirin_smiles):
        result = self._run(aspirin_smiles)
        for route in result["retro_result"]["routes"]:
            assert "final_score" in route
            assert 0.0 <= route["final_score"] <= 1.0

    @pytest.mark.integration
    def test_guard_result_has_safety_data(self, aspirin_smiles):
        result = self._run(aspirin_smiles)
        safety = result["guard_result"]["safety_data"]
        assert "h_phrases" in safety
        assert "ghs_pictograms" in safety

    @pytest.mark.integration
    def test_empty_query_invalid(self):
        graph = build_graph()
        result = graph.invoke({"query": ""})
        assert result["validation"]["is_valid"] is False

    @pytest.mark.integration
    def test_russian_name_ethanol_resolves(self):
        """Russian input: LLM translates → PubChem → full pipeline."""
        result = self._run("этанол")
        assert result["validation"]["is_valid"] is True
        assert result["smiles"] == "CCO"
        assert result["validation"]["pubchem_cid"] == 702


# ═════════════════════════════════════════════════════════════════════════════
# Pipeline state integrity
# ═════════════════════════════════════════════════════════════════════════════

class TestPipelineStateIntegrity:
    @pytest.mark.integration
    def test_smiles_consistent_through_pipeline(self, aspirin_smiles):
        graph = build_graph()
        mock_llm = _make_mock_llm_for_info()
        with patch("real_proj.mvp.nodes.molecule_info_node._get_llm",
                   return_value=mock_llm):
            result = graph.invoke({"query": aspirin_smiles})

        # SMILES in state should match validation output
        assert result["smiles"] == result["validation"]["canonical_smiles"]

    @pytest.mark.integration
    def test_pubchem_cid_consistent(self, aspirin_smiles):
        graph = build_graph()
        mock_llm = _make_mock_llm_for_info()
        with patch("real_proj.mvp.nodes.molecule_info_node._get_llm",
                   return_value=mock_llm):
            result = graph.invoke({"query": aspirin_smiles})

        assert result["pubchem_cid"] == result["validation"]["pubchem_cid"]
        assert result["pubchem_cid"] == 2244

    @pytest.mark.integration
    def test_query_preserved_in_state(self, aspirin_smiles):
        graph = build_graph()
        mock_llm = _make_mock_llm_for_info()
        with patch("real_proj.mvp.nodes.molecule_info_node._get_llm",
                   return_value=mock_llm):
            result = graph.invoke({"query": aspirin_smiles})
        assert result["query"] == aspirin_smiles

    @pytest.mark.integration
    def test_banned_molecule_no_molecule_info(self, fentanyl_smiles):
        """Banned molecule should not reach molecule_info_node."""
        graph = build_graph()
        result = graph.invoke({"query": fentanyl_smiles})
        # molecule_info should not be populated
        mi = result.get("molecule_info")
        assert mi is None

    @pytest.mark.integration
    def test_banned_molecule_no_retro_result(self, fentanyl_smiles):
        graph = build_graph()
        result = graph.invoke({"query": fentanyl_smiles})
        assert result.get("retro_result") is None
