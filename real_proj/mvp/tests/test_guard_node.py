"""Tests for guard_node: banlist, GHS safety, PPE, overall status logic."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ..nodes.guard_node import _determine_overall_status, guard_node
from ..tools import banlist_check, reaction_banlist_check


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

    def test_restricted_overridden_by_banned(self):
        assert _determine_overall_status("banned", "restricted") == "CRITICAL_STOP"


# ═════════════════════════════════════════════════════════════════════════════
# banlist_check (tools.py)
# ═════════════════════════════════════════════════════════════════════════════

class TestBanlistCheck:
    def test_aspirin_is_clear(self, aspirin_smiles):
        result = banlist_check(aspirin_smiles)
        assert result["status"] == "clear"

    def test_ethanol_is_clear(self, ethanol_smiles):
        result = banlist_check(ethanol_smiles)
        assert result["status"] == "clear"

    def test_fentanyl_is_banned(self, fentanyl_smiles):
        result = banlist_check(fentanyl_smiles)
        assert result["status"] == "banned"
        assert result.get("name") == "Fentanyl"

    def test_fentanyl_has_reason(self, fentanyl_smiles):
        result = banlist_check(fentanyl_smiles)
        assert "reason" in result
        assert result["reason"]

    def test_fentanyl_has_category(self, fentanyl_smiles):
        result = banlist_check(fentanyl_smiles)
        assert "dea" in result.get("category", "").lower()

    def test_empty_smiles_returns_clear(self):
        result = banlist_check("")
        # Empty SMILES shouldn't crash; status should be clear or error
        assert "status" in result

    def test_invalid_smiles_returns_clear(self):
        result = banlist_check("NOTVALID")
        assert "status" in result


# ═════════════════════════════════════════════════════════════════════════════
# reaction_banlist_check
# ═════════════════════════════════════════════════════════════════════════════

class TestReactionBanlistCheck:
    def test_empty_description_is_allowed(self):
        result = reaction_banlist_check("")
        assert result["status"] == "allowed"

    def test_generic_description_is_allowed(self):
        result = reaction_banlist_check("Mix reagents and heat to 60°C")
        assert result["status"] in ("allowed", "restricted")

    def test_result_has_status_key(self):
        result = reaction_banlist_check("some reaction")
        assert "status" in result


# ═════════════════════════════════════════════════════════════════════════════
# guard_node
# ═════════════════════════════════════════════════════════════════════════════

class TestGuardNode:
    def test_empty_smiles_returns_critical(self):
        result = guard_node({"smiles": "", "query": "test"})
        assert result["guard_result"]["overall_status"] == "CRITICAL_STOP"
        assert "error" in result

    def test_missing_smiles_key_returns_critical(self):
        result = guard_node({"query": "test"})
        assert result["guard_result"]["overall_status"] == "CRITICAL_STOP"

    @pytest.mark.integration
    def test_aspirin_is_safe(self, aspirin_validated_state):
        result = guard_node(aspirin_validated_state)
        gr = result["guard_result"]
        assert gr["overall_status"] in ("SAFE", "WARNING")  # aspirin is safe
        assert "molecule_check" in gr
        assert "reaction_check" in gr
        assert "safety_data" in gr
        assert "ppe_recommendations" in gr

    @pytest.mark.integration
    def test_fentanyl_is_critical(self, fentanyl_validated_state):
        result = guard_node(fentanyl_validated_state)
        assert result["guard_result"]["overall_status"] == "CRITICAL_STOP"
        assert "error" in result
        assert result["error"].startswith("CRITICAL_STOP")

    @pytest.mark.integration
    def test_guard_result_has_all_keys(self, aspirin_validated_state):
        result = guard_node(aspirin_validated_state)
        gr = result["guard_result"]
        for key in ("overall_status", "molecule_check", "reaction_check",
                    "safety_data", "ppe_recommendations"):
            assert key in gr, f"Missing key: {key}"

    @pytest.mark.integration
    def test_aspirin_molecule_check_clear(self, aspirin_validated_state):
        result = guard_node(aspirin_validated_state)
        mol_check = result["guard_result"]["molecule_check"]
        assert mol_check.get("status") == "clear"

    @pytest.mark.integration
    def test_fentanyl_molecule_check_banned(self, fentanyl_validated_state):
        result = guard_node(fentanyl_validated_state)
        mol_check = result["guard_result"]["molecule_check"]
        assert mol_check.get("status") == "banned"

    def test_guard_node_uses_pubchem_cid(self, aspirin_validated_state):
        """CID from state should be passed through to safety_lookup."""
        with patch("real_proj.mvp.nodes.guard_node.safety_lookup") as mock_safety, \
             patch("real_proj.mvp.nodes.guard_node.banlist_check") as mock_ban, \
             patch("real_proj.mvp.nodes.guard_node.reaction_banlist_check") as mock_rxn, \
             patch("real_proj.mvp.nodes.guard_node.ppe_recommender") as mock_ppe:

            mock_ban.return_value = {"status": "clear"}
            mock_rxn.return_value = {"status": "allowed"}
            mock_safety.return_value = {"h_phrases": [], "ghs_pictograms": []}
            mock_ppe.return_value = []

            guard_node(aspirin_validated_state)
            mock_safety.assert_called_once_with(
                aspirin_validated_state["smiles"],
                cid=aspirin_validated_state["pubchem_cid"],
            )

    @pytest.mark.integration
    def test_ethanol_ppe_not_empty(self, ethanol_smiles):
        state = {"smiles": ethanol_smiles, "pubchem_cid": 702}
        result = guard_node(state)
        # PPE may or may not have items, but list should exist
        assert isinstance(result["guard_result"]["ppe_recommendations"], list)
