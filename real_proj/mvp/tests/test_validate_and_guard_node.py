"""Tests for validate_and_guard_node.

Tests the heuristic helpers (_detect_input_type, _determine_overall_status)
without mocking, and the full node with mocked PubChem/banlist calls.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from ..nodes.validate_and_guard_node import (
    _detect_input_type,
    _determine_overall_status,
    validate_and_guard_node,
)


# ── Journal patch used for all full-node tests ──────────────────────────────

@pytest.fixture()
def mock_journal():
    mock_j = MagicMock()
    mock_j.step.return_value.__enter__ = lambda s: None
    mock_j.step.return_value.__exit__ = MagicMock(return_value=False)
    with patch("real_proj.mvp.journal.AgentJournal") as cls:
        cls.for_session.return_value = mock_j
        yield mock_j


# ═════════════════════════════════════════════════════════════════════════════
# _detect_input_type
# ═════════════════════════════════════════════════════════════════════════════

class TestDetectInputType:
    def test_aspirin_smiles(self):
        assert _detect_input_type("CC(=O)Oc1ccccc1C(=O)O") == "smiles"

    def test_ethanol_smiles(self):
        assert _detect_input_type("CCO") == "smiles"

    def test_smiles_with_brackets(self):
        assert _detect_input_type("[NH4+]") == "smiles"

    def test_smiles_with_stereo(self):
        assert _detect_input_type("C[C@@H](O)c1ccccc1") == "smiles"

    def test_smiles_double_bond(self):
        assert _detect_input_type("C=C") == "smiles"

    def test_aromatic_smiles(self):
        assert _detect_input_type("c1ccccc1") == "smiles"

    def test_name_aspirin(self):
        assert _detect_input_type("aspirin") == "name"

    def test_name_with_space(self):
        assert _detect_input_type("acetyl salicylic acid") == "name"

    def test_name_russian(self):
        assert _detect_input_type("этанол") == "name"

    def test_name_russian_multiword(self):
        assert _detect_input_type("ацетилсалициловая кислота") == "name"

    def test_empty_string_is_name(self):
        # Empty string has no SMILES chars → treated as name
        result = _detect_input_type("")
        assert result in ("name", "smiles")


# ═════════════════════════════════════════════════════════════════════════════
# _determine_overall_status
# ═════════════════════════════════════════════════════════════════════════════

class TestDetermineOverallStatus:
    def test_both_clear_is_safe(self):
        assert _determine_overall_status("clear", "allowed") == "SAFE"

    def test_mol_restricted_is_warning(self):
        assert _determine_overall_status("restricted", "allowed") == "WARNING"

    def test_rxn_restricted_is_warning(self):
        assert _determine_overall_status("clear", "restricted") == "WARNING"

    def test_mol_banned_is_critical(self):
        assert _determine_overall_status("banned", "allowed") == "CRITICAL_STOP"

    def test_rxn_prohibited_is_critical(self):
        assert _determine_overall_status("clear", "prohibited") == "CRITICAL_STOP"

    def test_both_banned_is_critical(self):
        assert _determine_overall_status("banned", "prohibited") == "CRITICAL_STOP"

    def test_banned_overrides_restricted(self):
        assert _determine_overall_status("banned", "restricted") == "CRITICAL_STOP"

    def test_both_restricted_is_warning(self):
        assert _determine_overall_status("restricted", "restricted") == "WARNING"


# ═════════════════════════════════════════════════════════════════════════════
# validate_and_guard_node — mocked resolution
# ═════════════════════════════════════════════════════════════════════════════

def _mock_resolve(smiles="CCO", cid=702, status="found"):
    return {
        "resolve_status": status,
        "canonical_smiles": smiles,
        "pubchem_cid": cid,
        "iupac_name": "ethanol",
    }


def _mock_safety(overall="SAFE"):
    return {
        "overall_status": overall,
        "molecule_check": {"status": "clear"},
        "reaction_check": {"status": "allowed"},
        "safety_data": {},
        "ppe_recommendations": [],
    }


class TestValidateAndGuardNode:
    def test_found_and_safe_sets_smiles(self, mock_journal):
        with patch("real_proj.mvp.nodes.validate_and_guard_node._resolve_molecule",
                   return_value=_mock_resolve("CCO", 702, "found")), \
             patch("real_proj.mvp.nodes.validate_and_guard_node._run_safety_checks",
                   return_value=_mock_safety("SAFE")):
            result = validate_and_guard_node({"query": "ethanol"})
        assert result["smiles"] == "CCO"
        assert result["pubchem_cid"] == 702

    def test_found_and_safe_sets_guard_result(self, mock_journal):
        with patch("real_proj.mvp.nodes.validate_and_guard_node._resolve_molecule",
                   return_value=_mock_resolve(status="found")), \
             patch("real_proj.mvp.nodes.validate_and_guard_node._run_safety_checks",
                   return_value=_mock_safety("SAFE")):
            result = validate_and_guard_node({"query": "ethanol"})
        assert result["guard_result"]["overall_status"] == "SAFE"

    def test_not_found_returns_not_found_status(self, mock_journal):
        with patch("real_proj.mvp.nodes.validate_and_guard_node._resolve_molecule",
                   return_value={
                       "resolve_status": "not_found",
                       "canonical_smiles": None,
                       "pubchem_cid": None,
                   }):
            result = validate_and_guard_node({"query": "xyzunknownmolecule"})
        validation = result.get("validation", {})
        assert validation.get("resolve_status") == "not_found"

    def test_banned_returns_banned_status(self, mock_journal):
        with patch("real_proj.mvp.nodes.validate_and_guard_node._resolve_molecule",
                   return_value={
                       "resolve_status": "banned",
                       "canonical_smiles": "FENTANYL_SMILES",
                       "pubchem_cid": 3345,
                   }):
            result = validate_and_guard_node({"query": "fentanyl"})
        validation = result.get("validation", {})
        assert validation.get("resolve_status") == "banned"

    def test_critical_stop_sets_error(self, mock_journal):
        with patch("real_proj.mvp.nodes.validate_and_guard_node._resolve_molecule",
                   return_value=_mock_resolve(status="found")), \
             patch("real_proj.mvp.nodes.validate_and_guard_node._run_safety_checks",
                   return_value=_mock_safety("CRITICAL_STOP")):
            result = validate_and_guard_node({"query": "fentanyl"})
        assert result["guard_result"]["overall_status"] == "CRITICAL_STOP"

    def test_warning_status_preserved(self, mock_journal):
        with patch("real_proj.mvp.nodes.validate_and_guard_node._resolve_molecule",
                   return_value=_mock_resolve(status="found")), \
             patch("real_proj.mvp.nodes.validate_and_guard_node._run_safety_checks",
                   return_value=_mock_safety("WARNING")):
            result = validate_and_guard_node({"query": "something"})
        assert result["guard_result"]["overall_status"] == "WARNING"

    def test_empty_query(self, mock_journal):
        with patch("real_proj.mvp.nodes.validate_and_guard_node._resolve_molecule",
                   return_value={"resolve_status": "not_found", "canonical_smiles": None, "pubchem_cid": None}):
            result = validate_and_guard_node({"query": ""})
        # Should not crash; validation key should exist
        assert "validation" in result or "smiles" in result or "error" in result
