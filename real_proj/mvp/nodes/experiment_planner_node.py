"""Experiment planner node (Phase 3).

Builds a full experiment protocol for the selected synthesis pathway:
  1. If tree exists, generates procedures for each synthesis step (bottom-up)
  2. For each reaction, searches for procedures: ORD -> RAG -> inference
  3. Combines procedures with multi-step stoichiometry calculations
  4. Generates a structured protocol in Russian
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..procedure_inference import format_procedure_russian

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# LLM procedure enrichment
# ═════════════════════════════════════════════════════════════════════════════


def _build_section_prompt(sec: dict, mol_name: str) -> str:
    step_num = sec.get("step_number", 1)
    rxn = sec.get("reaction_smiles", "")
    product = sec.get("product_name") or sec.get("product_smiles", "")[:60]
    reagents = [r.get("name", r.get("smiles", "?")) for r in sec.get("reagent_table", [])]

    return "\n".join([
        f"Ты опытный химик-синтетик. Напиши подробный лабораторный протокол для стадии {step_num} синтеза {mol_name}.",
        "",
        f"Продукт: {product}",
        f"Реакция SMILES: {rxn}",
        f"Реагенты: {', '.join(reagents)}",
        "",
        "Напиши 6-8 конкретных шагов с учётом типа реакции. Включи:",
        "- подготовку реагентов и посуды, атмосферу (N₂/Ar или воздух)",
        "- порядок и скорость добавления реагентов, температуру и время",
        "- контроль реакции (ТСХ/GC), метод выделения продукта",
        "- очистку (перекристаллизация/хроматография/дистилляция), выход",
        "",
        "Верни ТОЛЬКО валидный JSON массив без пояснений:",
        '[{"step":"1","description":"...","reason":"..."},{"step":"2","description":"...","reason":"..."}]',
    ])


def _parse_llm_procedure(raw: str) -> list[dict]:
    """Strip markdown fences and parse JSON procedure list."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    # Find the JSON array
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1:
        raw = raw[start:end + 1]
    return json.loads(raw)


def _llm_enrich_procedures(sections: list[dict], mol_name: str, model: str | None = None) -> list[dict]:
    """Call LLM separately for each section so token limits never truncate results."""
    from ..config import make_llm

    llm = make_llm(model=model, temperature=0.2, max_tokens=2048)
    enriched = []

    for sec in sections:
        sn = sec.get("step_number", "?")
        try:
            prompt = _build_section_prompt(sec, mol_name)
            response = llm.invoke(prompt)
            steps = _parse_llm_procedure(response.content)
            if steps:
                sec = dict(sec)
                sec["procedure_steps"] = steps
                logger.info("[experiment_planner] section %s: LLM gave %d steps", sn, len(steps))
            else:
                logger.warning("[experiment_planner] section %s: LLM returned empty steps", sn)
        except Exception as e:
            logger.warning("[experiment_planner] section %s enrichment failed: %s", sn, e)
        enriched.append(sec)

    return enriched


def experiment_planner_node(state: dict[str, Any]) -> dict[str, Any]:
    """Build experiment protocol for the selected pathway.

    Reads:  state["synthesis_pathways"], state["selected_pathway"],
            state["calculations"], state["molecule_info"]
    Writes: state["experiment_protocol"], state["final_answer"]
    """
    from ..journal import AgentJournal
    j = AgentJournal.for_session(state.get("session_id", "default"))

    pathways = state.get("synthesis_pathways", [])
    selected_idx = state.get("selected_pathway")
    calculations = state.get("calculations", {})
    molecule_info = state.get("molecule_info", {})

    if selected_idx is None or not pathways:
        return {
            "experiment_protocol": {"error": "Путь синтеза не выбран."},
        }

    pathway = pathways[min(selected_idx, len(pathways) - 1)]
    mol_name = molecule_info.get("name", "целевая молекула")
    calc_steps = calculations.get("steps", [])

    with j.step("experiment_planner"):
        if calc_steps:
            protocol = _build_multistep_protocol(pathway, calc_steps, calculations, mol_name)
        else:
            protocol = _build_single_step_protocol(pathway, calculations, mol_name)

        sections = protocol.get("reaction_sections", [])

        # Always enrich procedures with LLM — heuristic inference is never good enough
        if sections:
            logger.info("[experiment_planner] enriching %d section(s) with LLM", len(sections))
            sections = _llm_enrich_procedures(sections, mol_name, model=state.get("llm_model"))
            protocol["reaction_sections"] = sections

        total_steps = sum(len(s.get("procedure_steps", [])) for s in sections)
        j.decision(
            "experiment_planner",
            f"Протокол сформирован: {len(sections)} стадий, ~{total_steps} шагов процедуры",
            {"sections_count": len(sections), "procedure_steps_total": total_steps,
             "is_multistep": protocol.get("is_multistep", False), "molecule": mol_name,
             "llm_enriched": True},
        )

    protocol_text = _format_protocol_text(protocol, mol_name)

    # Finalize journal with markdown export
    try:
        j.export_markdown()
    except Exception:
        pass

    existing_answer = state.get("final_answer", "")
    proto_marker = "=" * 60 + "\n  ПРОТОКОЛ ЭКСПЕРИМЕНТА:"
    if proto_marker in existing_answer:
        existing_answer = existing_answer[:existing_answer.index(proto_marker)].rstrip()

    return {
        "experiment_protocol": protocol,
        "final_answer": existing_answer + "\n" + protocol_text,
        "current_phase": "experiment",
    }


# ═════════════════════════════════════════════════════════════════════════════
# Multi-step protocol (from tree)
# ═════════════════════════════════════════════════════════════════════════════

def _build_multistep_protocol(
    pathway: dict[str, Any],
    calc_steps: list[dict[str, Any]],
    calculations: dict[str, Any],
    mol_name: str,
) -> dict[str, Any]:
    """Build protocol with one section per synthesis step."""
    reaction_sections: list[dict[str, Any]] = []

    for calc_step in calc_steps:
        route_info = {
            "reaction_smiles": calc_step.get("reaction_smiles", ""),
            "reactants": calc_step.get("reaction_smiles", "").split(">>")[0]
                if ">>" in calc_step.get("reaction_smiles", "") else "",
            "procedure_details": "",
            "temperature": None,
            "solvent": None,
            "catalyst": None,
        }

        _enrich_route_from_tree(route_info, pathway, calc_step.get("product_smiles", ""))

        # For root step, also check pathway-level procedure_details (from ORD)
        if not route_info.get("procedure_details") and calc_step.get("product_smiles") == pathway.get("tree", {}).get("smiles"):
            for key in ("procedure_details", "temperature", "solvent", "catalyst",
                        "expected_yield", "reaction_id"):
                if pathway.get(key):
                    route_info[key] = pathway[key]

        procedure_steps = _find_procedure_cascade(route_info)

        reagent_table = _build_reagent_table_from_step(calc_step)

        reaction_sections.append({
            "step_number": calc_step.get("step_number", 0),
            "product_smiles": calc_step.get("product_smiles", ""),
            "product_name": calc_step.get("product_name", ""),
            "product_mass_g": calc_step.get("product_mass_g", 0),
            "procedure_steps": procedure_steps,
            "reagent_table": reagent_table,
            "reaction_smiles": calc_step.get("reaction_smiles", ""),
        })

    all_buyable = calculations.get("all_buyable_reagents", [])
    buyable_table = _build_reagent_table_from_list(all_buyable)

    return {
        "title": f"Протокол синтеза: {mol_name}",
        "reaction_sections": reaction_sections,
        "buyable_reagent_table": buyable_table,
        "calculations": calculations,
        "is_multistep": True,
    }


def _enrich_route_from_tree(
    route_info: dict, pathway: dict, product_smiles: str,
) -> None:
    """Try to find the original route data in the tree for a given product."""
    tree = pathway.get("tree")
    if not tree:
        return
    node = _find_node_by_smiles(tree, product_smiles)
    if not node:
        return
    original_route = node.get("route", {})
    if not original_route:
        return
    for key in ("procedure_details", "temperature", "solvent", "catalyst",
                "expected_yield", "reaction_id"):
        if original_route.get(key):
            route_info[key] = original_route[key]


def _find_node_by_smiles(node: dict, smiles: str) -> dict | None:
    """DFS search for a node with matching SMILES."""
    if node.get("smiles") == smiles:
        return node
    for child in node.get("children", []):
        found = _find_node_by_smiles(child, smiles)
        if found:
            return found
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Single-step protocol (fallback, no tree)
# ═════════════════════════════════════════════════════════════════════════════

def _build_single_step_protocol(
    pathway: dict[str, Any],
    calculations: dict[str, Any],
    mol_name: str,
) -> dict[str, Any]:
    procedure_steps = _find_procedure_cascade(pathway)
    reagent_table = _build_reagent_table_from_calc(calculations)

    return {
        "title": f"Протокол синтеза: {mol_name}",
        "reaction_sections": [{
            "step_number": 1,
            "product_smiles": calculations.get("target_product_smiles", ""),
            "product_name": mol_name,
            "product_mass_g": calculations.get("target_mass_g", 0),
            "procedure_steps": procedure_steps,
            "reagent_table": reagent_table,
            "reaction_smiles": pathway.get("reaction_smiles", ""),
        }],
        "buyable_reagent_table": reagent_table,
        "calculations": calculations,
        "is_multistep": False,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Procedure cascade
# ═════════════════════════════════════════════════════════════════════════════

def _find_procedure_cascade(route: dict[str, Any]) -> list[dict[str, str]]:
    """Cascade: ORD procedure -> RAG -> inference."""
    procedure_details = route.get("procedure_details", "")
    if procedure_details and len(procedure_details) > 50:
        steps = format_procedure_russian(route)
        if steps and steps[0].get("description", "") != "Процедура синтеза не найдена.":
            logger.info("[experiment_planner] procedure found via ORD")
            return steps

    rag_steps = _try_rag_search(route)
    if rag_steps:
        logger.info("[experiment_planner] procedure found via RAG")
        return rag_steps

    logger.info("[experiment_planner] inferring procedure from conditions")
    return format_procedure_russian(route, use_inference=True)


def _try_rag_search(route: dict[str, Any]) -> list[dict[str, str]] | None:
    try:
        from ..tools.rag_search import search_synthesis_procedures
        reaction_smiles = route.get("reaction_smiles", "")
        if not reaction_smiles:
            return None
        results = search_synthesis_procedures(reaction_smiles)
        if results:
            return [{
                "step": str(i + 1),
                "description": r.get("text", r.get("answer", "")),
                "reason": f"Литература: {r.get('citation', r.get('title', 'RAG'))}",
            } for i, r in enumerate(results) if r.get("text") or r.get("answer")]
    except Exception as e:
        logger.debug("[experiment_planner] RAG search failed: %s", e)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Reagent tables
# ═════════════════════════════════════════════════════════════════════════════

def _build_reagent_table_from_step(calc_step: dict[str, Any]) -> list[dict[str, Any]]:
    """Build reagent table from a single stoichiometry step."""
    table = []
    for r in calc_step.get("reagents", []):
        table.append({
            "name": r.get("name", r.get("smiles", "?"))[:50],
            "smiles": r.get("smiles", ""),
            "equivalents": r.get("equivalents", 1.0),
            "moles": r.get("moles", 0),
            "mass_g": r.get("mass_g", 0),
            "volume_ml": r.get("volume_ml"),
            "notes": r.get("notes", ""),
            "is_leaf": r.get("is_leaf", True),
        })
    return table


def _build_reagent_table_from_list(reagents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build buyable reagent summary table."""
    table = []
    for r in reagents:
        table.append({
            "name": r.get("name", r.get("smiles", "?"))[:50],
            "smiles": r.get("smiles", ""),
            "equivalents": r.get("equivalents", 1.0),
            "moles": r.get("moles", 0),
            "mass_g": r.get("mass_g", 0),
            "volume_ml": r.get("volume_ml"),
            "notes": r.get("notes", ""),
        })
    return table


def _build_reagent_table_from_calc(calculations: dict[str, Any]) -> list[dict[str, Any]]:
    """Build reagent table from flat calculations (single-step fallback)."""
    table = []
    for r in calculations.get("reagents", []):
        table.append({
            "name": r.get("name", r.get("smiles", "?"))[:50],
            "smiles": r.get("smiles", ""),
            "equivalents": r.get("equivalents", 1.0),
            "moles": r.get("moles", 0),
            "mass_g": r.get("mass_g", 0),
            "volume_ml": r.get("volume_ml"),
            "notes": r.get("notes", ""),
        })
    return table


# ═════════════════════════════════════════════════════════════════════════════
# Text formatting
# ═════════════════════════════════════════════════════════════════════════════

def _format_protocol_text(protocol: dict[str, Any], mol_name: str) -> str:
    """Format protocol as human-readable Russian text."""
    lines = [
        f"\n{'='*60}",
        f"  ПРОТОКОЛ ЭКСПЕРИМЕНТА: {mol_name}",
        f"{'='*60}",
    ]

    sections = protocol.get("reaction_sections", [])
    is_multi = protocol.get("is_multistep", False)

    if is_multi and len(sections) > 1:
        buyable_table = protocol.get("buyable_reagent_table", [])
        if buyable_table:
            lines.append("\n  СВОДНАЯ ТАБЛИЦА ЗАКУПОК (коммерческие реагенты):")
            lines.append(f"  {'Реагент':<35} {'Масса, г':<12} {'Объём, мл':<12}")
            lines.append(f"  {'-'*59}")
            for r in buyable_table:
                name = r.get("name", "?")[:35]
                mass = f"{r.get('mass_g', 0):.3f}"
                vol = f"{r['volume_ml']:.2f}" if r.get("volume_ml") else "—"
                lines.append(f"  {name:<35} {mass:<12} {vol:<12}")

    for section in sections:
        step_num = section.get("step_number", "?")
        product_name = section.get("product_name", "")
        product_smi = section.get("product_smiles", "")[:40]
        product_mass = section.get("product_mass_g", 0)

        if is_multi and len(sections) > 1:
            lines.append(f"\n  {'─'*58}")
            lines.append(f"  СТАДИЯ {step_num}: {product_name or product_smi}")
            lines.append(f"  Продукт: {product_smi}  ({product_mass:.3f} г)")
            lines.append(f"  {'─'*58}")
        
        reagent_table = section.get("reagent_table", [])
        if reagent_table:
            lines.append(f"\n  Реагенты стадии {step_num}:")
            lines.append(f"  {'Реагент':<35} {'Масса, г':<12} {'Объём, мл':<12} {'Экв.':<8}")
            lines.append(f"  {'-'*67}")
            for r in reagent_table:
                name = r.get("name", "?")[:35]
                mass = f"{r.get('mass_g', 0):.3f}"
                vol = f"{r['volume_ml']:.2f}" if r.get("volume_ml") else "—"
                equiv = f"{r.get('equivalents', 1.0):.2f}"
                leaf = "" if r.get("is_leaf", True) else " [промежуточный]"
                lines.append(f"  {name:<35} {mass:<12} {vol:<12} {equiv:<8}{leaf}")

        proc_steps = section.get("procedure_steps", [])
        if proc_steps:
            lines.append(f"\n  Процедура стадии {step_num}:")
            for step in proc_steps:
                sn = step.get("step", "?")
                desc = step.get("description", "")
                reason = step.get("reason", "")
                lines.append(f"    Шаг {sn}. {desc}")
                if reason and reason not in ("ORD процедура", "inferred"):
                    lines.append(f"           -> Источник: {reason}")

    calc = protocol.get("calculations", {})
    target_mass = calc.get("target_mass_g")
    if target_mass:
        lines.append(f"\n  Целевой продукт: {calc.get('target_product_smiles', '?')}")
        lines.append(f"  Целевая масса:   {target_mass:.3f} г")
        target_moles = calc.get("target_moles", 0)
        if target_moles:
            lines.append(f"  Целевые моли:    {target_moles:.6f} моль")

    warnings = calc.get("warnings", [])
    if warnings:
        lines.append(f"\n  ПРЕДУПРЕЖДЕНИЯ:")
        for w in warnings:
            lines.append(f"    ! {w}")

    lines.append(f"\n{'='*60}")
    return "\n".join(lines)
