"""Tests for low-level safety tool heuristics."""

from __future__ import annotations

from ..tools.explosive import explosive_alias_check, explosive_hazard_check
from ..nodes.validate_and_guard_node import _run_safety_checks


def test_explosive_hazard_flags_tnt_like_trinitroaromatic_motif():
    result = explosive_hazard_check("Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]")

    assert result["hazard_type"] == "explosive"
    assert result["status"] == "blocked"
    assert result["hazard_family"] == "nitroaromatic_explosive"
    assert result["danger_level"] == "high"


def test_explosive_hazard_uses_ghs_h200_series():
    result = explosive_hazard_check("CCO", safety_data={"h_phrases": ["H201: Explosive; mass explosion hazard"]})

    assert result["status"] == "blocked"
    assert result["basis"] == "ghs"
    assert result["h_codes"] == ["H201"]


def test_explosive_alias_check_flags_common_names():
    result = explosive_alias_check("проверь тротил по ADMET")

    assert result["status"] == "blocked"
    assert result["name"] == "TNT"


def test_safety_gate_exposes_explosive_check_channel():
    result = _run_safety_checks(
        "Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]",
        cid=None,
        reaction_description="check TNT",
    )

    assert result["overall_status"] == "CRITICAL_STOP"
    assert result["explosive_check"]["hazard_type"] == "explosive"
    assert result["explosive_check"]["status"] == "blocked"
