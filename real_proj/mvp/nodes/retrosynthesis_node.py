"""Retrosynthesis node: find synthesis routes via ORD + retro model, score and rank.

Searches ORD SQLite for published reactions, uses standalone retro model
for prediction, deduplicates, scores, ranks, and formats procedure
details as step-by-step Russian instructions.
"""

from __future__ import annotations

import logging
from typing import Any

from ..retro_tools import search_and_rank
from ..procedure_inference import format_procedure_russian

logger = logging.getLogger(__name__)


def retrosynthesis_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: find and rank retrosynthesis routes.

    Reads: state["smiles"], state["molecule_info"]
    Writes: state["retro_result"], appends to state["final_answer"]
    """
    smiles = state.get("smiles", "")
    molecule_info = state.get("molecule_info", {})
    mol_name = molecule_info.get("name", "Неизвестно")

    if not smiles:
        return {
            "retro_result": {"routes": [], "error": "No SMILES available"},
        }

    logger.info("[retro] searching routes for %s (%s)", mol_name, smiles[:30])

    # Search and rank
    result = search_and_rank(smiles, top_n=5)
    routes = result.get("routes", [])
    sources = result.get("sources_used", [])
    total = result.get("total_found", 0)

    logger.info(
        "[retro] found %d total, showing top %d from %s",
        total, len(routes), ", ".join(sources) or "none",
    )

    # Generate procedure steps for each route
    for route in routes:
        procedure_steps = format_procedure_russian(route)
        route["procedure_steps_ru"] = procedure_steps

    # Build retro text (Russian)
    retro_text = _format_retro_text(mol_name, routes, sources, total)

    # Append to existing final_answer
    existing_answer = state.get("final_answer", "")

    return {
        "retro_result": result,
        "final_answer": existing_answer + "\n" + retro_text,
    }


def _format_retro_text(
    mol_name: str,
    routes: list[dict[str, Any]],
    sources: list[str],
    total: int,
) -> str:
    """Format retrosynthesis results as Russian text with step-by-step procedures."""
    source_labels = {
        "ord": "Open Reaction Database",
        "retro_model": "Ретросинтез-модель (template-relevance)",
    }
    source_str = ", ".join(source_labels.get(s, s) for s in sources)

    lines = [
        f"{'='*60}",
        f"  РЕТРОСИНТЕЗ: {mol_name}",
        f"{'='*60}",
        f"  Источники:      {source_str or 'нет данных'}",
        f"  Найдено путей:  {total}",
        f"  Показано лучших: {len(routes)}",
        "",
    ]

    if not routes:
        lines.append("  Пути синтеза не найдены.")
        lines.append(f"{'='*60}")
        return "\n".join(lines)

    for i, route in enumerate(routes, 1):
        scoring = route.get("scoring", {})
        source = route.get("source", "?")
        source_label = {
            "ord": "ORD",
            "retro_model": "МОДЕЛЬ",
        }.get(source, source.upper())

        lines.append(f"  ── Путь #{i} [{source_label}] " + "─" * 40)

        # Reactants (canonical)
        reactants = route.get("reactants", "")
        if len(reactants) > 80:
            reactants = reactants[:77] + "..."
        lines.append(f"  Реагенты:       {reactants}")

        # Reaction SMILES
        rxn_smi = route.get("reaction_smiles", "")
        if rxn_smi:
            if len(rxn_smi) > 80:
                rxn_smi = rxn_smi[:77] + "..."
            lines.append(f"  Реакция:        {rxn_smi}")

        # Conditions
        if route.get("temperature"):
            lines.append(f"  Температура:    {route['temperature']}")
        if route.get("solvent"):
            lines.append(f"  Растворитель:   {route['solvent']}")
        if route.get("catalyst"):
            lines.append(f"  Катализатор:    {route['catalyst']}")
        if route.get("expected_yield") is not None:
            lines.append(f"  Выход:          {route['expected_yield']:.0%}")

        # Score
        lines.append(f"  Оценка:         {route.get('final_score', 0):.3f}/1.00")
        lines.append(
            f"    Модель: {scoring.get('model_score', 0):.2f}  "
            f"Достоверность: {scoring.get('plausibility', 0):.2f}  "
            f"Доступность: {scoring.get('buyability', 0):.0%}  "
            f"Простота: {scoring.get('simplicity', 0):.2f}"
        )

        # Template info
        if route.get("num_examples"):
            lines.append(f"  Примеров в базе: {route['num_examples']}")

        if route.get("reaction_id"):
            lines.append(f"  ORD ID:         {route['reaction_id']}")

        # ── Procedure steps (structured Russian) ──
        procedure_steps = route.get("procedure_steps_ru", [])
        if procedure_steps:
            lines.append("")
            lines.append("  📋 ПРОЦЕДУРА СИНТЕЗА:")
            for step in procedure_steps:
                step_num = step.get("step", "?")
                desc = step.get("description", "")
                reason = step.get("reason", "")
                lines.append(f"    Шаг {step_num}. {desc}")
                if reason and reason != "ORD процедура":
                    lines.append(f"           ↳ {reason}")

        lines.append("")

    lines.append(f"{'='*60}")
    return "\n".join(lines)
