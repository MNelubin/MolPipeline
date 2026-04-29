"""Retrosynthesis node: find synthesis routes and expand into full trees.

1. Queries all enabled retrosynthesis sources via the shared runtime collector
2. For each top route, recursively expands non-buyable reactants via tree_expansion
3. Formats procedure details as step-by-step Russian instructions
"""

from __future__ import annotations

import logging
from typing import Any

from ..tools.retro_tools import search_and_rank
from ..tree_expansion import expand_tree
from ..procedure_inference import format_procedure_russian

logger = logging.getLogger(__name__)

_MAX_ROUTES_TO_EXPAND = 3
_TREE_MAX_DEPTH = 6
_TREE_TIMEOUT_SEC = 60.0


def retrosynthesis_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: find retrosynthesis routes and expand into trees.

    Reads:  state["smiles"], state["molecule_info"]
    Writes: state["retro_result"], state["final_answer"] (overwrites retro section)
    """
    import time as _time
    from ..journal import AgentJournal
    j = AgentJournal.for_session(state.get("session_id", "default"))

    smiles = state.get("smiles", "")
    molecule_info = state.get("molecule_info", {})
    mol_name = molecule_info.get("name", "Неизвестно")

    if not smiles:
        return {
            "retro_result": {"routes": [], "error": "No SMILES available"},
        }

    logger.info("[retro] searching routes for %s (%s)", mol_name, smiles[:30])

    with j.step("retrosynthesis"):
        t0 = _time.monotonic()
        result = search_and_rank(smiles, top_n=5)
        elapsed_search = int((_time.monotonic() - t0) * 1000)

        routes = result.get("routes", [])
        sources = result.get("sources_used", [])
        total = result.get("total_found", 0)

        j.tool_call(
            "retrosynthesis", "search_and_rank",
            args={"smiles": smiles[:40], "top_n": 5},
            result_summary=f"{len(routes)} маршрутов из {', '.join(sources) or 'нет'}",
            duration_ms=elapsed_search,
        )

        logger.info("[retro] found %d total, showing top %d from %s", total, len(routes), ", ".join(sources) or "none")

        for route in routes:
            procedure_steps = format_procedure_russian(route)
            route["procedure_steps_ru"] = procedure_steps

        # Expand top routes into full trees
        for i, route in enumerate(routes[:_MAX_ROUTES_TO_EXPAND]):
            reactants = route.get("reactants", "")
            if not reactants:
                continue
            try:
                t0 = _time.monotonic()
                tree_result = expand_tree(smiles, reactants, max_depth=_TREE_MAX_DEPTH, timeout_sec=_TREE_TIMEOUT_SEC)
                elapsed_tree = int((_time.monotonic() - t0) * 1000)
                route["tree"] = tree_result.get("tree", {})
                route["tree_stats"] = tree_result.get("stats", {})
                stats = route["tree_stats"]
                j.tool_call(
                    "retrosynthesis", "tree_expansion",
                    args={"route_idx": i},
                    result_summary=f"{stats.get('total_nodes', 0)} узлов, {stats.get('buyable_count', 0)} покупаемых",
                    duration_ms=elapsed_tree,
                )
                logger.info("[retro] tree #%d: %d nodes, %d buyable, %d unresolved, depth=%d",
                    i + 1, stats.get("total_nodes", 0), stats.get("buyable_count", 0),
                    stats.get("unresolved_count", 0), stats.get("max_depth_reached", 0))
            except Exception as e:
                logger.warning("[retro] tree expansion failed for route #%d: %s", i + 1, e)
                route["tree"] = None
                route["tree_stats"] = None

        j.decision(
            "retrosynthesis",
            f"Найдено {len(routes)} маршрутов синтеза для {mol_name}",
            {"routes_count": len(routes), "sources": sources, "total_found": total,
             "best_score": routes[0].get("final_score") if routes else None},
        )

    retro_text = _format_retro_text(mol_name, routes, sources, total)

    existing_answer = state.get("final_answer", "")
    retro_marker = "=" * 60 + "\n  РЕТРОСИНТЕЗ:"
    if retro_marker in existing_answer:
        existing_answer = existing_answer[:existing_answer.index(retro_marker)].rstrip()

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
    """Format retrosynthesis results as Russian text with tree details."""
    source_labels = {
        "ord": "Open Reaction Database",
        "web": "Веб-поиск синтеза",
        "retro_model": "Ретросинтез-модель (template-relevance)",
        "aizynthfinder": "AiZynthFinder",
        "retrocast": "RetroCast",
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
            "web": "WEB",
            "retro_model": "МОДЕЛЬ",
            "aizynthfinder": "AIZYNTH",
            "retrocast": "RETROCAST",
        }.get(source, source.upper())

        lines.append(f"  ── Путь #{i} [{source_label}] " + "─" * 40)

        reactants = route.get("reactants", "")
        if len(reactants) > 80:
            reactants = reactants[:77] + "..."
        lines.append(f"  Реагенты:       {reactants}")

        rxn_smi = route.get("reaction_smiles", "")
        if rxn_smi:
            if len(rxn_smi) > 80:
                rxn_smi = rxn_smi[:77] + "..."
            lines.append(f"  Реакция:        {rxn_smi}")

        if route.get("temperature"):
            lines.append(f"  Температура:    {route['temperature']}")
        if route.get("solvent"):
            lines.append(f"  Растворитель:   {route['solvent']}")
        if route.get("catalyst"):
            lines.append(f"  Катализатор:    {route['catalyst']}")
        if route.get("expected_yield") is not None:
            lines.append(f"  Выход:          {route['expected_yield']:.0%}")

        lines.append(f"  Оценка:         {route.get('final_score', 0):.3f}/1.00")
        lines.append(
            f"    Модель: {scoring.get('model_score', 0):.2f}  "
            f"Достоверность: {scoring.get('plausibility', 0):.2f}  "
            f"Доступность: {scoring.get('buyability', 0):.0%}  "
            f"Простота: {scoring.get('simplicity', 0):.2f}"
        )

        if route.get("num_examples"):
            lines.append(f"  Примеров в базе: {route['num_examples']}")
        if route.get("reaction_id"):
            lines.append(f"  ORD ID:         {route['reaction_id']}")

        procedure_steps = route.get("procedure_steps_ru", [])
        if procedure_steps:
            lines.append("")
            lines.append("  ПРОЦЕДУРА СИНТЕЗА:")
            for step in procedure_steps:
                step_num = step.get("step", "?")
                desc = step.get("description", "")
                reason = step.get("reason", "")
                lines.append(f"    Шаг {step_num}. {desc}")
                if reason and reason != "ORD процедура":
                    lines.append(f"           ↳ {reason}")

        tree_stats = route.get("tree_stats")
        if tree_stats:
            lines.append("")
            lines.append("  ДЕРЕВО РАЗЛОЖЕНИЯ:")
            lines.append(f"    Всего узлов:     {tree_stats.get('total_nodes', '?')}")
            lines.append(f"    Коммерческих:    {tree_stats.get('buyable_count', '?')}")
            lines.append(f"    Нерешённых:      {tree_stats.get('unresolved_count', '?')}")
            lines.append(f"    Запрещённых:     {tree_stats.get('banned_count', '?')}")
            lines.append(f"    Макс. глубина:   {tree_stats.get('max_depth_reached', '?')}")
            lines.append(f"    Время (сек):     {tree_stats.get('elapsed_sec', '?')}")

            tree = route.get("tree")
            if tree:
                lines.append("")
                lines.append("  ЛИСТЬЯ (конечные реагенты):")
                leaves = _collect_leaves(tree)
                for leaf in leaves:
                    status_ru = _STATUS_RU.get(leaf["status"], leaf["status"])
                    name = leaf.get("name") or leaf["smiles"][:40]
                    marker = "✓" if leaf["status"] == "buyable" else "✗"
                    guard = leaf.get("guard", {})
                    restricted_note = ""
                    if guard.get("status") == "restricted":
                        restricted_note = f" ⚠ ОГРАНИЧЕН: {guard.get('reason', '')}"
                    lines.append(f"    {marker} {name}  [{status_ru}]{restricted_note}")

        lines.append("")

    lines.append(f"{'='*60}")
    return "\n".join(lines)


_STATUS_RU = {
    "buyable": "коммерчески доступен",
    "banned": "ЗАПРЕЩЁН",
    "unresolved": "маршрут не найден",
    "depth_limit": "лимит глубины",
    "timeout": "таймаут",
    "circular": "цикл",
    "invalid_smiles": "невалидный SMILES",
}


def _collect_leaves(node: dict[str, Any]) -> list[dict[str, Any]]:
    """Walk the tree and return all leaf nodes (no children)."""
    children = node.get("children", [])
    if not children:
        return [node]
    leaves: list[dict[str, Any]] = []
    for child in children:
        leaves.extend(_collect_leaves(child))
    return leaves
