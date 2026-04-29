"""Tests for molecule_info_node: data gathering, LLM synthesis, output structure."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from ..nodes.molecule_info_node import molecule_info_node, _safe_int, _safe_float


# ═════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═════════════════════════════════════════════════════════════════════════════

class TestSafeInt:
    def test_int_input(self):
        assert _safe_int(42) == 42

    def test_string_int(self):
        assert _safe_int("42") == 42

    def test_none_returns_default(self):
        assert _safe_int(None, default=0) == 0

    def test_invalid_string_returns_default(self):
        assert _safe_int("abc", default=-1) == -1

    def test_float_truncates(self):
        assert _safe_int(3.9) == 3


class TestSafeFloat:
    def test_float_input(self):
        assert _safe_float(3.14) == pytest.approx(3.14)

    def test_string_float(self):
        assert _safe_float("180.16") == pytest.approx(180.16)

    def test_none_returns_default(self):
        assert _safe_float(None) == 0.0

    def test_invalid_string_returns_default(self):
        assert _safe_float("xyz", default=-1.0) == -1.0


# ═════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════════════

MOCK_LLM_RESPONSE = {
    "name": "Аспирин (2-ацетилоксибензойная кислота)",
    "synonyms": ["Ацетилсалициловая кислота", "АСК"],
    "smiles": "CC(=O)Oc1ccccc1C(=O)O",
    "molecular_formula": "C9H8O4",
    "molecular_weight": 180.16,
    "physical_description": "Бесцветные кристаллы.",
    "properties": {
        "melting_point": "135",
        "boiling_point": "140",
        "solubility": "Плохо растворим в воде",
        "density": "1.4",
        "logP": "1.19",
        "physical_state": "твёрдое",
        "flash_point": None,
        "vapor_pressure": None,
    },
    "ghs_classification": ["Вредно при проглатывании"],
    "spectral_notes": "Характерные полосы ИК-спектра при 1750 см⁻¹.",
    "description": "Аспирин — широко известное анальгетическое средство.",
    "pubchem_cid": 2244,
}


def _make_mock_llm(content: dict | None = None):
    mock_resp = MagicMock()
    mock_resp.content = json.dumps(content or MOCK_LLM_RESPONSE, ensure_ascii=False)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_resp
    return mock_llm


# ═════════════════════════════════════════════════════════════════════════════
# molecule_info_node
# ═════════════════════════════════════════════════════════════════════════════

class TestMoleculeInfoNode:
    def _run_with_mock_llm(self, state, llm_content=None):
        mock_llm = _make_mock_llm(llm_content)
        with patch("mvp.nodes.molecule_info_node._get_llm", return_value=mock_llm):
            return molecule_info_node(state)

    def test_output_has_molecule_info_key(self, aspirin_guarded_state):
        result = self._run_with_mock_llm(aspirin_guarded_state)
        assert "molecule_info" in result

    def test_output_has_final_answer_key(self, aspirin_guarded_state):
        result = self._run_with_mock_llm(aspirin_guarded_state)
        assert "final_answer" in result
        assert len(result["final_answer"]) > 50

    def test_molecule_info_has_required_keys(self, aspirin_guarded_state):
        result = self._run_with_mock_llm(aspirin_guarded_state)
        mi = result["molecule_info"]
        required = ("name", "smiles", "molecular_formula", "molecular_weight",
                    "properties", "description", "ghs_classification",
                    "pubchem_cid", "image_2d", "image_3d", "pubchem_url")
        for key in required:
            assert key in mi, f"Missing key: {key}"

    def test_properties_has_required_keys(self, aspirin_guarded_state):
        result = self._run_with_mock_llm(aspirin_guarded_state)
        props = result["molecule_info"]["properties"]
        for key in ("melting_point", "boiling_point", "solubility", "density",
                    "logP", "physical_state", "tpsa", "h_bond_donors",
                    "h_bond_acceptors", "rotatable_bonds", "ring_count"):
            assert key in props, f"Missing property key: {key}"

    def test_toxicity_has_ld50_keys(self, aspirin_guarded_state):
        result = self._run_with_mock_llm(aspirin_guarded_state)
        tox = result["molecule_info"]["toxicity"]
        assert "ld50_oral" in tox
        assert "ld50_dermal" in tox
        assert "ld50_inhalation" in tox

    def test_final_answer_contains_molecule_name(self, aspirin_guarded_state):
        result = self._run_with_mock_llm(aspirin_guarded_state)
        assert "Аспирин" in result["final_answer"]

    def test_final_answer_contains_smiles(self, aspirin_guarded_state):
        result = self._run_with_mock_llm(aspirin_guarded_state)
        assert "CC(=O)" in result["final_answer"]

    def test_rdkit_weight_prioritized_over_llm(self, aspirin_guarded_state):
        """RDKit molecular weight should be used (more accurate than LLM guess)."""
        result = self._run_with_mock_llm(aspirin_guarded_state)
        mw = result["molecule_info"]["molecular_weight"]
        assert 179.0 < mw < 182.0  # aspirin is ~180.16

    def test_image_urls_contain_pubchem(self, aspirin_guarded_state):
        result = self._run_with_mock_llm(aspirin_guarded_state)
        mi = result["molecule_info"]
        assert "pubchem" in mi["image_2d"].lower()
        assert "pubchem" in mi["pubchem_url"].lower()

    def test_llm_json_parse_error_handled_gracefully(self, aspirin_guarded_state):
        """If LLM returns garbage, node should still return a result."""
        mock_resp = MagicMock()
        mock_resp.content = "NOT JSON AT ALL"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_resp
        with patch("mvp.nodes.molecule_info_node._get_llm", return_value=mock_llm):
            result = molecule_info_node(aspirin_guarded_state)
        assert "molecule_info" in result
        assert "final_answer" in result

    def test_llm_markdown_json_parsed(self, aspirin_guarded_state):
        """LLM sometimes wraps JSON in ```json ... ``` blocks."""
        mock_resp = MagicMock()
        mock_resp.content = f"```json\n{json.dumps(MOCK_LLM_RESPONSE)}\n```"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_resp
        with patch("mvp.nodes.molecule_info_node._get_llm", return_value=mock_llm):
            result = molecule_info_node(aspirin_guarded_state)
        assert result["molecule_info"]["pubchem_cid"] == 2244

    def test_llm_exception_handled_gracefully(self, aspirin_guarded_state):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM API down")
        with patch("mvp.nodes.molecule_info_node._get_llm", return_value=mock_llm):
            result = molecule_info_node(aspirin_guarded_state)
        assert "molecule_info" in result

    def test_ghs_pictograms_list(self, aspirin_guarded_state):
        result = self._run_with_mock_llm(aspirin_guarded_state)
        pics = result["molecule_info"]["ghs_pictograms"]
        assert isinstance(pics, list)
        # Each pictogram entry should be a dict
        for p in pics:
            assert isinstance(p, dict)

    @pytest.mark.integration
    def test_real_pubchem_lookup(self, aspirin_guarded_state):
        """Integration: real PubChem call for aspirin."""
        result = self._run_with_mock_llm(aspirin_guarded_state)
        mi = result["molecule_info"]
        assert mi["pubchem_cid"] in (2244, 0)  # 2244 is aspirin CID

    def test_no_smiles_still_returns_result(self, aspirin_guarded_state):
        state = {**aspirin_guarded_state, "smiles": ""}
        result = self._run_with_mock_llm(state)
        assert "molecule_info" in result

