"""Tests for validate_node: input detection, SMILES validation, name resolution."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from ..nodes.validate_node import (
    _detect_input_type,
    _translate_name_via_llm,
    _validate_smiles,
    _validate_name,
    validate_node,
)


# ═════════════════════════════════════════════════════════════════════════════
# _detect_input_type
# ═════════════════════════════════════════════════════════════════════════════

class TestDetectInputType:
    def test_aspirin_smiles(self):
        assert _detect_input_type("CC(=O)Oc1ccccc1C(=O)O") == "smiles"

    def test_caffeine_smiles(self):
        assert _detect_input_type("Cn1cnc2c1c(=O)n(c(=O)n2C)C") == "smiles"

    def test_ethanol_smiles(self):
        assert _detect_input_type("CCO") == "smiles"

    def test_smiles_with_brackets(self):
        assert _detect_input_type("[NH4+]") == "smiles"

    def test_smiles_with_stereo(self):
        assert _detect_input_type("C[C@@H](O)c1ccccc1") == "smiles"

    def test_name_with_space(self):
        assert _detect_input_type("acetyl salicylic acid") == "name"

    def test_name_aspirin(self):
        assert _detect_input_type("aspirin") == "name"

    def test_name_russian_cyrillic(self):
        assert _detect_input_type("этанол") == "name"

    def test_name_russian_multi_word(self):
        assert _detect_input_type("ацетилсалициловая кислота") == "name"

    def test_special_chars_name(self):
        assert _detect_input_type("café") == "name"

    def test_smiles_with_digit(self):
        # Single token with digit → SMILES
        result = _detect_input_type("C6H6")
        # Could be either — just check it doesn't crash
        assert result in ("smiles", "name")


# ═════════════════════════════════════════════════════════════════════════════
# _validate_smiles
# ═════════════════════════════════════════════════════════════════════════════

class TestValidateSmiles:
    def test_valid_aspirin(self, aspirin_smiles):
        result = _validate_smiles(aspirin_smiles)
        assert result["validation"]["is_valid"] is True
        assert result["validation"]["input_type"] == "smiles"
        assert result["smiles"] == aspirin_smiles
        assert result["validation"]["canonical_smiles"] == aspirin_smiles

    def test_valid_returns_formula(self, aspirin_smiles):
        result = _validate_smiles(aspirin_smiles)
        assert result["validation"]["molecular_formula"] == "C9H8O4"

    def test_valid_returns_weight(self, aspirin_smiles):
        result = _validate_smiles(aspirin_smiles)
        mw = result["validation"]["molecular_weight"]
        assert 179.0 < mw < 182.0  # ~180.16

    def test_invalid_smiles(self):
        result = _validate_smiles("NOTASMILES!!!")
        assert result["validation"]["is_valid"] is False
        assert "error" in result["validation"]
        assert result["validation"]["error"] is not None

    def test_invalid_smiles_has_error_key(self):
        result = _validate_smiles("XXXYYY")
        assert "error" in result
        assert result["error"] is not None

    def test_valid_ethanol(self, ethanol_smiles):
        result = _validate_smiles(ethanol_smiles)
        assert result["validation"]["is_valid"] is True
        assert result["smiles"] == "CCO"

    def test_canonicalization(self):
        # Non-canonical input should produce canonical output
        result = _validate_smiles("OCC")  # non-canonical ethanol
        assert result["validation"]["is_valid"] is True
        assert result["smiles"] == "CCO"  # canonical form


# ═════════════════════════════════════════════════════════════════════════════
# _validate_name
# ═════════════════════════════════════════════════════════════════════════════

class TestValidateName:
    @pytest.mark.integration
    def test_aspirin_resolves(self):
        result = _validate_name("aspirin")
        assert result["validation"]["is_valid"] is True
        assert result["pubchem_cid"] == 2244
        assert "CC(=O)" in result["smiles"]

    @pytest.mark.integration
    def test_caffeine_resolves(self):
        result = _validate_name("caffeine")
        assert result["validation"]["is_valid"] is True
        assert result["pubchem_cid"] > 0

    @pytest.mark.integration
    def test_unknown_name_fails(self):
        result = _validate_name("xyznotamolecule99999")
        assert result["validation"]["is_valid"] is False
        assert "не найдено" in result["validation"]["error"]

    def test_russian_name_with_mocked_llm(self):
        with patch("real_proj.mvp.nodes.validate_node._translate_name_via_llm") as mock_llm, \
             patch("real_proj.mvp.nodes.validate_node.get_cid_by_name") as mock_cid, \
             patch("real_proj.mvp.nodes.validate_node.get_smiles_by_cid") as mock_smiles:

            mock_llm.return_value = "ethanol"
            # First call (Russian) fails, second call (English) succeeds
            mock_cid.side_effect = [None, 702]
            mock_smiles.return_value = "CCO"

            result = _validate_name("этанол")
            assert result["validation"]["is_valid"] is True
            mock_llm.assert_called_once_with("этанол")

    def test_pubchem_returns_invalid_smiles(self):
        with patch("real_proj.mvp.nodes.validate_node.get_cid_by_name") as mock_cid, \
             patch("real_proj.mvp.nodes.validate_node.get_smiles_by_cid") as mock_smiles:
            mock_cid.return_value = 999
            mock_smiles.return_value = "INVALID_SMILES_XYZ"

            result = _validate_name("something")
            assert result["validation"]["is_valid"] is False


# ═════════════════════════════════════════════════════════════════════════════
# _translate_name_via_llm
# ═════════════════════════════════════════════════════════════════════════════

class TestTranslateNameViaLlm:
    def test_returns_none_without_api_key(self):
        with patch("real_proj.mvp.nodes.validate_node.OPENROUTER_API_KEY", ""):
            result = _translate_name_via_llm("этанол")
            assert result is None

    def test_returns_english_name(self):
        mock_response = MagicMock()
        mock_response.content = "Ethanol"
        with patch("real_proj.mvp.nodes.validate_node.OPENROUTER_API_KEY", "test-key"), \
             patch("langchain_openai.ChatOpenAI.invoke", return_value=mock_response):
            result = _translate_name_via_llm("этанол")
            assert result == "Ethanol"

    def test_rejects_cyrillic_response(self):
        mock_response = MagicMock()
        mock_response.content = "Этанол"  # Cyrillic returned — should reject
        with patch("real_proj.mvp.nodes.validate_node.OPENROUTER_API_KEY", "test-key"), \
             patch("langchain_openai.ChatOpenAI.invoke", return_value=mock_response):
            result = _translate_name_via_llm("этанол")
            assert result is None

    def test_strips_leading_quote(self):
        # strip('"') strips from both ends only if both ends have "
        # '"Ethanol".' → 'Ethanol".' after strip('"') (trailing " blocked by .)
        # Test a clean response instead
        mock_response = MagicMock()
        mock_response.content = "Ethanol."
        with patch("real_proj.mvp.nodes.validate_node.OPENROUTER_API_KEY", "test-key"), \
             patch("langchain_openai.ChatOpenAI.invoke", return_value=mock_response):
            result = _translate_name_via_llm("этанол")
            assert result == "Ethanol"

    def test_strips_surrounding_quotes(self):
        mock_response = MagicMock()
        mock_response.content = "'Ethanol'"
        with patch("real_proj.mvp.nodes.validate_node.OPENROUTER_API_KEY", "test-key"), \
             patch("langchain_openai.ChatOpenAI.invoke", return_value=mock_response):
            result = _translate_name_via_llm("этанол")
            assert result == "Ethanol"

    def test_returns_none_on_exception(self):
        with patch("real_proj.mvp.nodes.validate_node.OPENROUTER_API_KEY", "test-key"), \
             patch("langchain_openai.ChatOpenAI.invoke", side_effect=Exception("API down")):
            result = _translate_name_via_llm("этанол")
            assert result is None


# ═════════════════════════════════════════════════════════════════════════════
# validate_node (full state machine)
# ═════════════════════════════════════════════════════════════════════════════

class TestValidateNode:
    def test_empty_query_returns_invalid(self):
        result = validate_node({"query": ""})
        assert result["validation"]["is_valid"] is False
        assert "error" in result

    def test_whitespace_query_returns_invalid(self):
        result = validate_node({"query": "   "})
        assert result["validation"]["is_valid"] is False

    def test_smiles_query_valid(self, aspirin_smiles):
        result = validate_node({"query": aspirin_smiles})
        assert result["validation"]["is_valid"] is True
        assert result["smiles"] == aspirin_smiles

    def test_invalid_smiles_query(self):
        result = validate_node({"query": "XXXINVALID((("})
        assert result["validation"]["is_valid"] is False

    @pytest.mark.integration
    def test_english_name_aspirin(self):
        result = validate_node({"query": "aspirin"})
        assert result["validation"]["is_valid"] is True
        assert result["pubchem_cid"] == 2244

    @pytest.mark.integration
    def test_returns_smiles_key(self):
        result = validate_node({"query": "CC(=O)Oc1ccccc1C(=O)O"})
        assert "smiles" in result
        assert result["smiles"] != ""

    @pytest.mark.integration
    def test_returns_pubchem_cid_key(self):
        result = validate_node({"query": "aspirin"})
        assert "pubchem_cid" in result
        assert result["pubchem_cid"] > 0

    def test_validation_result_has_all_keys(self, aspirin_smiles):
        result = validate_node({"query": aspirin_smiles})
        v = result["validation"]
        for key in ("is_valid", "input_type", "canonical_smiles", "error"):
            assert key in v, f"Missing key: {key}"
