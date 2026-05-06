"""Tests for low-level safety tool heuristics."""

from __future__ import annotations

from ..tools.safety import banlist_check


def test_banlist_flags_tnt_like_trinitroaromatic_motif():
    result = banlist_check("Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]")

    assert result["status"] == "banned"
    assert result["category"] == "explosive_synthesis"
    assert result["danger_level"] == "high"
    assert "Trinitroaromatic" in result["reason"]
