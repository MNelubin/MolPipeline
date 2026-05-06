"""Tests for the general chemistry chat orchestrator."""

from __future__ import annotations

from unittest.mock import patch

from ..chem_chat import classify_chem_intent, run_chem_chat


def test_general_question_uses_research_without_molecule_resolution():
    research_result = {
        "summary": "SN1 идет через карбкатион, SN2 через concerted backside attack.",
        "sources": [],
        "evidence": [],
    }

    with patch("mvp.chem_chat._resolve_molecule") as mock_resolve, \
         patch("mvp.chem_chat.run_research_workspace", return_value=research_result) as mock_research:
        result = run_chem_chat("Чем SN1 отличается от SN2?")

    assert result["intent"] == "general"
    assert result["tools_used"] == ["research_analyze"]
    assert "общий химический research-режим" in result["answer"]
    mock_resolve.assert_not_called()
    mock_research.assert_called_once()


def test_retrosynthesis_question_extracts_target_and_runs_safety_gate_first():
    resolved = {
        "validation": {
            "is_valid": True,
            "input_type": "name",
            "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
            "molecular_formula": "C9H8O4",
            "molecular_weight": 180.16,
        },
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "pubchem_cid": 2244,
    }
    safety = {
        "overall_status": "SAFE",
        "molecule_check": {"status": "clear", "reason": "Not found in banlists."},
        "reaction_check": {"status": "allowed"},
        "safety_data": {},
        "ppe_recommendations": [],
    }
    retro = {
        "routes": [{
            "source": "ord",
            "source_label": "ORD",
            "reactants": "CC(=O)O.Oc1ccccc1C(=O)O",
            "availability_summary": {"available_count": 2, "total": 2, "estimated_total_1g_usd": 1.2},
        }],
        "total_unique": 1,
        "source_errors": {},
    }

    with patch("mvp.chem_chat._resolve_molecule", return_value=resolved), \
         patch("mvp.chem_chat._run_safety_checks", return_value=safety), \
         patch("mvp.chem_chat.search_and_rank", return_value=retro), \
         patch("mvp.chem_chat._attach_procedure_steps"):
        result = run_chem_chat("Найди путь синтеза аспирина")

    assert result["intent"] == "retrosynthesis"
    assert result["tools_used"] == ["resolve_molecule", "safety_check", "retrosynthesis_search"]
    assert result["artifacts"]["molecule"]["query_used"] == "aspirin"
    assert "Найдено маршрутов: 1" in result["answer"]


def test_safety_stop_blocks_retrosynthesis_tool():
    resolved = {
        "validation": {"is_valid": True, "input_type": "name"},
        "smiles": "blocked",
        "pubchem_cid": 1,
    }
    safety = {
        "overall_status": "CRITICAL_STOP",
        "molecule_check": {"status": "banned", "reason": "Exact match in banlist."},
        "reaction_check": {},
        "safety_data": {},
        "ppe_recommendations": [],
    }

    with patch("mvp.chem_chat._resolve_molecule", return_value=resolved), \
         patch("mvp.chem_chat._run_safety_checks", return_value=safety), \
         patch("mvp.chem_chat.search_and_rank") as mock_retro:
        result = run_chem_chat("найди путь синтеза кокаина")

    assert "retrosynthesis_search" not in result["tools_used"]
    assert "заблокирован" in result["answer"]
    mock_retro.assert_not_called()


def test_availability_question_extracts_multiple_reagents_from_text():
    def fake_resolve(query: str):
        if query == "benzaldehyde":
            return {"validation": {"is_valid": True, "input_type": "name"}, "smiles": "O=Cc1ccccc1", "pubchem_cid": 240}
        if query == "ethanol":
            return {"validation": {"is_valid": True, "input_type": "name"}, "smiles": "CCO", "pubchem_cid": 702}
        return {"validation": {"is_valid": False, "error": "not found"}}

    def fake_availability(smiles: str, **kwargs):
        return {
            "input": kwargs.get("input_value"),
            "label": kwargs.get("label"),
            "smiles": smiles,
            "available": True,
            "availability_level": "catalog",
        }

    with patch("mvp.chem_chat._resolve_molecule", side_effect=fake_resolve), \
         patch("mvp.chem_chat.check_reagent_availability", side_effect=fake_availability):
        result = run_chem_chat("Можно ли купить бензальдегид и этанол?")

    items = result["artifacts"]["availability"]["items"]
    assert [item["smiles"] for item in items] == ["O=Cc1ccccc1", "CCO"]


def test_intent_classifier_routes_non_molecule_questions_to_general():
    assert classify_chem_intent("Почему реакция Гриньяра боится воды?") == "general"
    assert classify_chem_intent("safety aspirin") == "safety"
