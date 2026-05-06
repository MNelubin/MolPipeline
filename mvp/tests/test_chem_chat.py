"""Tests for the general chemistry chat orchestrator."""

from __future__ import annotations

from unittest.mock import patch

from ..chem_chat import CHEM_CHAT_MODEL, classify_chem_intent, run_chem_chat


def test_broad_general_question_answers_directly_without_research():
    plan = {
        "intent": "general",
        "target_molecules": [],
        "tools": ["research_analyze"],
        "source_mode": "auto",
        "research_mode": "literature",
        "reasoning": "broad educational question",
    }
    final = {"answer": "Химия изучает вещества и их превращения."}

    with patch("mvp.chem_chat._chat_llm_json", side_effect=[plan, final]), \
         patch("mvp.chem_chat._resolve_molecule") as mock_resolve, \
         patch("mvp.chem_chat.run_research_workspace") as mock_research:
        result = run_chem_chat("что такое химия и какие самые популярные элементы там")

    assert result["intent"] == "general"
    assert result["tools_used"] == []
    assert result["answer"] == "Химия изучает вещества и их превращения."
    mock_resolve.assert_not_called()
    mock_research.assert_not_called()


def test_real_world_material_composition_uses_research_even_without_explicit_sources():
    plan = {
        "intent": "general",
        "target_molecules": [],
        "tools": [],
        "source_mode": "auto",
        "research_mode": "literature",
        "reasoning": "general question",
    }
    research = {
        "summary": "Керамические кружки обычно состоят из глины, кварца, полевого шпата и глазури.",
        "sources": [{"title": "Ceramic materials overview", "url": "https://example.org/ceramics", "source_type": "web"}],
        "evidence": [{"id": "S1"}],
    }

    with patch("mvp.chem_chat._chat_llm_json", side_effect=[plan, {"answer": "research-backed answer"}]), \
         patch("mvp.chem_chat.run_research_workspace", return_value=research) as mock_research:
        result = run_chem_chat("из чего обычно состоят керамические кружки?")

    assert result["tools_used"] == ["research_analyze"]
    assert result["artifacts"]["research"]["summary"].startswith("Керамические кружки")
    assert "Перейти к вопросу про конкретную молекулу" not in result["suggested_next_actions"]
    mock_research.assert_called_once()


def test_broad_general_question_falls_back_when_model_refuses_russian():
    plan = {
        "intent": "general",
        "target_molecules": [],
        "tools": [],
        "source_mode": "auto",
        "research_mode": "literature",
        "reasoning": "broad educational question",
    }
    refusal = {"answer": "Извините, я не могу распознать ваш запрос. Пожалуйста, уточните."}

    with patch("mvp.chem_chat._chat_llm_json", side_effect=[plan, refusal]), \
         patch("mvp.chem_chat.run_research_workspace") as mock_research:
        result = run_chem_chat("что такое химия и какие самые популярные элементы там")

    assert "Химия — это наука о веществах" in result["answer"]
    assert "водород" in result["answer"]
    assert result["tools_used"] == []
    mock_research.assert_not_called()


def test_source_followup_research_uses_previous_topic():
    captured_queries = []
    plan = {
        "intent": "research",
        "target_molecules": [],
        "tools": ["research_analyze"],
        "source_mode": "auto",
        "research_mode": "literature",
        "reasoning": "source follow-up",
    }
    research = {
        "summary": "Found nitration sources.",
        "sources": [{"title": "Toluene nitration overview", "url": "https://example.org/toluene", "source_type": "web"}],
        "evidence": [{"id": "S1"}],
    }

    def fake_research(query: str, **kwargs):
        captured_queries.append(query)
        return research

    history = [{"role": "user", "content": "расскажи про нитрование толуола"}]
    with patch("mvp.chem_chat._chat_llm_json", side_effect=[plan, {"answer": "ok"}]), \
         patch("mvp.chem_chat.run_research_workspace", side_effect=fake_research):
        result = run_chem_chat("Попросить ответ со ссылками на источники", history=history)

    assert result["tools_used"] == ["research_analyze"]
    assert "нитрование толуола" in captured_queries[0]
    assert "citation formatting" in captured_queries[0]


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

    with patch("mvp.chem_chat._chat_llm_json", return_value=None), \
         patch("mvp.chem_chat._resolve_molecule", return_value=resolved), \
         patch("mvp.chem_chat._run_safety_checks", return_value=safety), \
         patch("mvp.chem_chat.search_and_rank", return_value=retro), \
         patch("mvp.chem_chat._attach_procedure_steps"):
        result = run_chem_chat("Найди путь синтеза аспирина")

    assert result["intent"] == "retrosynthesis"
    assert result["tools_used"] == ["resolve_molecule", "safety_check", "retrosynthesis_search"]
    assert result["artifacts"]["molecule"]["query_used"] == "aspirin"
    assert "Найдено маршрутов: 1" in result["answer"]


def test_availability_after_retrosynthesis_checks_route_reactants_not_target():
    resolved = {
        "validation": {"is_valid": True, "input_type": "name"},
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "pubchem_cid": 2244,
    }
    safety = {
        "overall_status": "SAFE",
        "molecule_check": {"status": "clear", "reason": "Not found in banlists."},
        "reaction_check": {"status": "allowed"},
        "safety_data": {},
    }
    retro = {
        "routes": [{"source": "ord", "reactants": "CC(=O)OC(C)=O.O=C(O)c1ccccc1O"}],
        "total_unique": 1,
        "source_errors": {},
    }
    availability = {"items": [], "summary": {"total": 2, "available_count": 2, "priced_count": 2}}

    with patch("mvp.chem_chat._chat_llm_json", return_value=None), \
         patch("mvp.chem_chat._resolve_molecule", return_value=resolved), \
         patch("mvp.chem_chat._run_safety_checks", return_value=safety), \
         patch("mvp.chem_chat.search_and_rank", return_value=retro), \
         patch("mvp.chem_chat._attach_procedure_steps"), \
         patch("mvp.chem_chat._availability_tool", return_value=availability) as mock_availability:
        run_chem_chat("Найди путь синтеза аспирина и оцени доступность реагентов")

    mock_availability.assert_called_once_with("CC(=O)OC(C)=O.O=C(O)c1ccccc1O")

def test_chat_uses_fixed_deepseek_model_for_planning_and_answering():
    plan = {
        "intent": "general",
        "target_molecules": [],
        "tools": [],
        "source_mode": "auto",
        "research_mode": "literature",
        "reasoning": "general chemistry question",
    }
    final = {"answer": "LLM final answer", "suggested_next_actions": []}

    with patch("mvp.chem_chat._chat_llm_json", side_effect=[plan, final]) as mock_llm, \
         patch("mvp.chem_chat.run_research_workspace") as mock_research:
        result = run_chem_chat("Explain SN1 vs SN2")

    assert result["model"] == CHEM_CHAT_MODEL == "deepseek/deepseek-v4-flash"
    assert result["plan"]["used_llm"] is True
    assert result["answer"] == "LLM final answer"
    assert mock_llm.call_count == 2
    mock_research.assert_not_called()


def test_chat_emits_progress_events():
    events = []
    plan = {
        "intent": "general",
        "target_molecules": [],
        "tools": [],
        "source_mode": "auto",
        "research_mode": "literature",
        "reasoning": "general chemistry question",
    }

    with patch("mvp.chem_chat._chat_llm_json", side_effect=[plan, {"answer": "ok"}]), \
         patch("mvp.chem_chat.run_research_workspace") as mock_research:
        run_chem_chat("Explain SN1 vs SN2", progress_callback=events.append)

    event_types = [event["type"] for event in events]
    assert "plan" in event_types
    assert not any(event["type"] == "tool_start" for event in events)
    assert any(event.get("stage") == "final_answer" for event in events)
    mock_research.assert_not_called()


def test_chat_passes_recent_history_to_planner_and_final_answer():
    captured_payloads = []
    plan = {
        "intent": "admet",
        "target_molecules": ["aspirin"],
        "tools": ["resolve_molecule", "research_analyze"],
        "source_mode": "auto",
        "research_mode": "literature",
        "reasoning": "follow-up asks for ADMET of previous aspirin target",
    }
    resolved = {
        "validation": {"is_valid": True, "input_type": "name"},
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "pubchem_cid": 2244,
    }
    safety = {
        "overall_status": "SAFE",
        "molecule_check": {"status": "clear", "reason": "Not found in banlists."},
        "reaction_check": {},
        "safety_data": {},
    }
    admet = {"overall": {"score": 91, "risk_level": "low"}, "safety_overlay": {}}

    def fake_llm(system, user, **kwargs):
        captured_payloads.append(user)
        return plan if len(captured_payloads) == 1 else {"answer": "ADMET по аспирину рассчитан."}

    history = [
        {"role": "user", "content": "Найди путь синтеза аспирина"},
        {"role": "assistant", "content": "Целевая молекула: aspirin."},
    ]

    with patch("mvp.chem_chat._chat_llm_json", side_effect=fake_llm), \
         patch("mvp.chem_chat._resolve_molecule", return_value=resolved), \
         patch("mvp.chem_chat._run_safety_checks", return_value=safety), \
         patch("mvp.chem_chat.run_research_workspace") as mock_research, \
         patch("mvp.chem_chat.analyze_admet", return_value=admet):
        result = run_chem_chat("а теперь проверь ADMET", history=history)

    assert result["tools_used"] == ["resolve_molecule", "safety_check", "admet_screen"]
    mock_research.assert_not_called()
    assert "conversation_history" in captured_payloads[0]
    assert "Найди путь синтеза аспирина" in captured_payloads[0]
    assert "conversation_history" in captured_payloads[1]


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

    with patch("mvp.chem_chat._chat_llm_json", return_value=None), \
         patch("mvp.chem_chat._resolve_molecule", return_value=resolved), \
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

    with patch("mvp.chem_chat._chat_llm_json", return_value=None), \
         patch("mvp.chem_chat._resolve_molecule", side_effect=fake_resolve), \
         patch("mvp.chem_chat.check_reagent_availability", side_effect=fake_availability):
        result = run_chem_chat("Можно ли купить бензальдегид и этанол?")

    items = result["artifacts"]["availability"]["items"]
    assert [item["smiles"] for item in items] == ["O=Cc1ccccc1", "CCO"]


def test_intent_classifier_routes_non_molecule_questions_to_general():
    assert classify_chem_intent("Почему реакция Гриньяра боится воды?") == "general"
    assert classify_chem_intent("safety aspirin") == "safety"
