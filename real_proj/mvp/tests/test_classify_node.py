"""Tests for classify_node: heuristic-based query classification.

No mocking needed — classify_node is pure heuristics (no LLM, no network).
Journal calls are patched to avoid filesystem writes.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from ..nodes.classify_node import _classify, classify_node


# ── Shared journal patch ────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_journal():
    """Prevent filesystem writes to logs/ during tests."""
    mock_j = MagicMock()
    mock_j.step.return_value.__enter__ = lambda s: None
    mock_j.step.return_value.__exit__ = MagicMock(return_value=False)
    with patch("real_proj.mvp.nodes.classify_node.AgentJournal") as cls:
        cls.for_session.return_value = mock_j
        yield mock_j


# ═════════════════════════════════════════════════════════════════════════════
# _classify — SMILES detection
# ═════════════════════════════════════════════════════════════════════════════

class TestClassifySmiles:
    def test_aspirin_smiles(self):
        assert _classify("CC(=O)Oc1ccccc1C(=O)O") == "molecule"

    def test_caffeine_smiles(self):
        assert _classify("Cn1cnc2c1c(=O)n(c(=O)n2C)C") == "molecule"

    def test_ethanol_smiles(self):
        assert _classify("CCO") == "molecule"

    def test_smiles_with_brackets(self):
        assert _classify("[NH4+]") == "molecule"

    def test_smiles_with_stereo(self):
        assert _classify("C[C@@H](O)c1ccccc1") == "molecule"

    def test_smiles_with_double_bond(self):
        assert _classify("C=C") == "molecule"

    def test_smiles_aromatic(self):
        assert _classify("c1ccccc1") == "molecule"


# ═════════════════════════════════════════════════════════════════════════════
# _classify — CAS / formula / name detection
# ═════════════════════════════════════════════════════════════════════════════

class TestClassifyName:
    def test_cas_number(self):
        assert _classify("50-78-2") == "molecule"       # aspirin CAS

    def test_cas_number_caffeine(self):
        assert _classify("58-08-2") == "molecule"

    def test_formula_water(self):
        assert _classify("H2O") == "molecule"

    def test_formula_co2(self):
        assert _classify("CO2") == "molecule"

    def test_name_aspirin_en(self):
        assert _classify("aspirin") == "molecule"

    def test_name_caffeine_en(self):
        assert _classify("caffeine") == "molecule"

    def test_name_ethanol_ru(self):
        assert _classify("этанол") == "molecule"

    def test_name_acetylsalicylic_ru(self):
        assert _classify("ацетилсалициловая кислота") == "molecule"

    def test_name_dopamine(self):
        assert _classify("dopamine") == "molecule"


# ═════════════════════════════════════════════════════════════════════════════
# _classify — research query detection
# ═════════════════════════════════════════════════════════════════════════════

class TestClassifyResearch:
    def test_ru_ищу(self):
        assert _classify("ищу антиоксидант для масла") == "research"

    def test_ru_нужен(self):
        assert _classify("нужен ингибитор коррозии") == "research"

    def test_ru_подбери(self):
        assert _classify("подбери аналог аспирина") == "research"

    def test_ru_похожее(self):
        assert _classify("похожее на морфин но не запрещённое") == "research"

    def test_en_looking_for(self):
        assert _classify("looking for an antioxidant") == "research"

    def test_en_suggest(self):
        assert _classify("suggest me an inhibitor for COX-2") == "research"

    def test_en_inhibitor_for(self):
        assert _classify("find an inhibitor for this enzyme") == "research"

    def test_ru_аналог(self):
        assert _classify("аналог парацетамола без побочек") == "research"


# ═════════════════════════════════════════════════════════════════════════════
# _classify — invalid
# ═════════════════════════════════════════════════════════════════════════════

class TestClassifyInvalid:
    def test_empty_string(self):
        assert _classify("") == "invalid"

    def test_single_digit(self):
        # Not a valid molecule identifier
        assert _classify("7") == "invalid"


# ═════════════════════════════════════════════════════════════════════════════
# classify_node — state wiring
# ═════════════════════════════════════════════════════════════════════════════

class TestClassifyNode:
    def test_molecule_query_sets_input_type(self):
        result = classify_node({"query": "aspirin"})
        assert result["input_type"] == "molecule"

    def test_research_query_sets_input_type(self):
        result = classify_node({"query": "ищу ингибитор"})
        assert result["input_type"] == "research"

    def test_invalid_sets_error(self):
        result = classify_node({"query": ""})
        assert result["input_type"] == "invalid"
        assert "error" in result

    def test_sets_current_phase(self):
        result = classify_node({"query": "aspirin"})
        assert result["current_phase"] == "identification"

    def test_sets_cycle_counts(self):
        result = classify_node({"query": "aspirin", "cycle_counts": {"x": 1}})
        assert "cycle_counts" in result

    def test_smiles_query(self):
        result = classify_node({"query": "CC(=O)Oc1ccccc1C(=O)O"})
        assert result["input_type"] == "molecule"

    def test_missing_query_key(self):
        result = classify_node({})
        assert result["input_type"] == "invalid"
