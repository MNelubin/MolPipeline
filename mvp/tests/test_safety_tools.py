"""Tests for low-level safety tool heuristics."""

from __future__ import annotations

from ..tools.explosive import explosive_alias_check, explosive_hazard_check
from ..tools.safety_taxonomy import build_safety_taxonomy
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
    assert result["safety_taxonomy"]["status"] == "blocked"
    assert result["safety_taxonomy"]["blocked_categories"][0]["hazard_type"] == "explosive"


def test_safety_taxonomy_classifies_ghs_categories_without_banlist():
    taxonomy = build_safety_taxonomy(
        molecule_check={"status": "clear", "reason": "Not found in banlists."},
        reaction_check={"status": "allowed"},
        explosive_check={"status": "clear", "hazard_type": "explosive"},
        safety_data={"h_phrases": ["H301: Toxic if swallowed", "H225: Highly flammable liquid and vapor"]},
    )

    hazard_types = {item["hazard_type"] for item in taxonomy["categories"]}
    assert taxonomy["status"] == "warning"
    assert "acute_toxicity" in hazard_types
    assert "flammable" in hazard_types


def test_safety_taxonomy_blocks_controlled_substances():
    taxonomy = build_safety_taxonomy(
        molecule_check={"status": "banned", "danger_level": "high", "reason": "Exact match in banlist."},
        reaction_check={"status": "allowed"},
        explosive_check={"status": "clear", "hazard_type": "explosive"},
        safety_data={},
    )

    assert taxonomy["status"] == "blocked"
    assert taxonomy["blocked_categories"][0]["hazard_type"] == "controlled_substance"
