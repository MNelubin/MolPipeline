"""Tests for descriptor ADMET safety overlay."""

from __future__ import annotations

from ..admet import analyze_admet


def test_admet_safety_overlay_promotes_severe_ghs_toxicity_to_high_risk():
    safety_guard = {
        "overall_status": "SAFE",
        "molecule_check": {"status": "clear", "reason": "Not found in banlists."},
        "safety_data": {"h_phrases": ["H300: Fatal if swallowed", "H330: Fatal if inhaled"]},
        "ppe_recommendations": [],
    }

    result = analyze_admet("CN1CCCC1c1cccnc1", safety_guard=safety_guard)

    assert result["overall"]["risk_level"] == "high"
    assert result["sections"]["toxicity"]["score"] < 100
    assert any("GHS" in flag["message"] for flag in result["sections"]["toxicity"]["flags"])


def test_admet_safety_overlay_caps_banned_molecule_score():
    safety_guard = {
        "overall_status": "CRITICAL_STOP",
        "molecule_check": {"status": "banned", "reason": "Exact match in banlist: Cocaine."},
        "safety_data": {"h_phrases": ["H300: Fatal if swallowed"]},
        "ppe_recommendations": [],
    }

    result = analyze_admet(
        "COC(=O)C1C(OC(=O)c2ccccc2)CC2CCC1N2C",
        safety_guard=safety_guard,
    )

    assert result["overall"]["risk_level"] == "high"
    assert result["overall"]["score"] <= 40
    assert result["safety_overlay"]["molecule_status"] == "banned"
