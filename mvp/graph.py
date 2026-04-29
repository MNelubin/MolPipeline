"""Multi-phase agent graph with interrupt points.

Phase 1 — Molecule identification:
  START -> classify -> validate_and_guard -> molecule_info -> INTERRUPT #1
  Fallback: validate(not_found) -> research -> validate_and_guard (retry)

Phase 2 — Retrosynthesis (user triggers "plan synthesis"):
  retrosynthesis (incl. tree expansion) -> [guard_safety, reagent_check] -> aggregate -> INTERRUPT #2
  No retry cycle: tree expansion already decomposes to buyable leaves.

Phase 3 — Experiment protocol (user selects pathway + target amount):
  stoichiometry -> experiment_planner -> END

Interrupt mechanism:
  Uses langgraph.types.interrupt() with MemorySaver checkpointer.
  Callers resume via Command(resume=<value>).
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from .state import MVPState
from .nodes.classify_node import classify_node
from .nodes.validate_and_guard_node import validate_and_guard_node
from .nodes.research_node import research_node
from .nodes.molecule_info_node import molecule_info_node
from .nodes.retrosynthesis_node import retrosynthesis_node
from .nodes.guard_safety_node import guard_safety_node
from .nodes.reagent_node import reagent_node
from .nodes.aggregate_node import aggregate_node
from .nodes.stoichiometry_node import stoichiometry_node
from .nodes.experiment_planner_node import experiment_planner_node


# ═════════════════════════════════════════════════════════════════════════════
# Routing functions
# ═════════════════════════════════════════════════════════════════════════════

def _after_classify(state: MVPState) -> str:
    input_type = state.get("input_type", "invalid")
    if input_type == "research":
        return "research"
    if input_type == "molecule":
        return "validate_and_guard"
    return "end"


def _after_validate(state: MVPState) -> str:
    validation = state.get("validation", {})
    resolve_status = validation.get("resolve_status", "not_found")

    if resolve_status == "found":
        return "molecule_info"
    if resolve_status == "banned":
        return "end"

    cycle_counts = state.get("cycle_counts", {})
    if cycle_counts.get("validate_research", 0) > 0:
        return "end"
    return "research_fallback"


def _after_research(state: MVPState) -> str:
    research_result = state.get("research_result", {})
    if research_result.get("is_successful"):
        return "validate_and_guard"
    return "end"


def _after_research_fallback(state: MVPState) -> str:
    research_result = state.get("research_result", {})
    if research_result.get("is_successful"):
        return "validate_and_guard"
    return "end"


# ═════════════════════════════════════════════════════════════════════════════
# Wrapper nodes for interrupt points and cycle bookkeeping
# ═════════════════════════════════════════════════════════════════════════════

def _research_fallback_node(state: dict) -> dict:
    """Wrap research_node: increment cycle counter, then delegate."""
    cycle_counts = dict(state.get("cycle_counts", {}))
    cycle_counts["validate_research"] = cycle_counts.get("validate_research", 0) + 1
    state_copy = dict(state)
    state_copy["cycle_counts"] = cycle_counts
    result = research_node(state_copy)
    result["cycle_counts"] = cycle_counts
    return result


def _interrupt_card_node(state: dict) -> dict:
    """Phase 1 complete: molecule card ready. Pauses graph for user decision.

    interrupt() suspends execution and returns the payload to the caller.
    The caller resumes with Command(resume=<anything truthy>) to proceed
    to retrosynthesis, or can simply stop.
    """
    mol_name = state.get("molecule_info", {}).get("name", "?")
    interrupt({
        "phase": "card_ready",
        "molecule": mol_name,
        "message": f"Молекула '{mol_name}' идентифицирована. Продолжить синтез?",
    })
    return {"current_phase": "identification"}


def _interrupt_select_pathway_node(state: dict) -> dict:
    """Phase 2 complete: synthesis pathways ready. Pauses for user selection.

    interrupt() returns pathway summary to the caller.
    The caller resumes with Command(resume={
        "selected_pathway": <int>,
        "target_amount": {"value": <float>, "unit": "g", ...}
    })
    """
    pathways = state.get("synthesis_pathways", [])
    pathway_summaries = []
    best_idx = 0
    best_score = -1.0
    for i, p in enumerate(pathways):
        viable = p.get("viable", False)
        score = p.get("final_score", 0)
        unresolved = p.get("unresolved_leaves", 999)
        rank = (1 if viable else 0, -unresolved, score)
        if rank > (1 if best_score >= 0 else 0, -999, best_score):
            best_idx = i
            best_score = score
        pathway_summaries.append({
            "index": i,
            "viable": viable,
            "score": score,
            "reactants": p.get("reactants", "")[:80],
            "buyable_leaves": p.get("buyable_leaves", 0),
            "unresolved_leaves": unresolved,
        })

    # Find true best: viable first, then fewest unresolved, then highest score
    ranked = sorted(
        range(len(pathways)),
        key=lambda i: (
            not pathways[i].get("viable", False),
            pathways[i].get("unresolved_leaves", 999),
            -pathways[i].get("final_score", 0),
        ),
    )
    best_idx = ranked[0] if ranked else 0

    user_choice = interrupt({
        "phase": "select_pathway",
        "pathways_count": len(pathways),
        "pathways": pathway_summaries,
        "best_index": best_idx,
        "message": f"Выберите путь синтеза (рекомендуется #{best_idx}). Укажите целевую массу продукта.",
    })

    if isinstance(user_choice, dict):
        selected = user_choice.get("selected_pathway", 0)
        target_raw = user_choice.get("target_amount", {})
    else:
        selected = 0
        target_raw = {}

    if isinstance(target_raw, dict) and target_raw.get("value"):
        target = target_raw
    else:
        target = {"value": 1.0, "unit": "g", "amount_type": "product_mass"}

    return {
        "current_phase": "experiment",
        "selected_pathway": selected,
        "target_amount": target,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Graph construction
# ═════════════════════════════════════════════════════════════════════════════

def build_graph(checkpointer=None):
    """Construct and compile the multi-phase graph.

    Args:
        checkpointer: LangGraph checkpointer instance. If None, creates
            a MemorySaver (in-memory, suitable for dev/CLI).
    """
    if not checkpointer or isinstance(checkpointer, dict):
        checkpointer = MemorySaver()

    graph = StateGraph(MVPState)

    # ── Phase 1 nodes ──
    graph.add_node("classify", classify_node)
    graph.add_node("validate_and_guard", validate_and_guard_node)
    graph.add_node("research", research_node)
    graph.add_node("research_fallback", _research_fallback_node)
    graph.add_node("molecule_info", molecule_info_node)
    graph.add_node("interrupt_card", _interrupt_card_node)

    # ── Phase 2 nodes ──
    graph.add_node("retrosynthesis", retrosynthesis_node)
    graph.add_node("guard_safety", guard_safety_node)
    graph.add_node("reagent_check", reagent_node)
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("interrupt_select_pathway", _interrupt_select_pathway_node)

    # ── Phase 3 nodes ──
    graph.add_node("stoichiometry", stoichiometry_node)
    graph.add_node("experiment_planner", experiment_planner_node)

    # ── Phase 1 edges ──
    graph.add_edge(START, "classify")

    graph.add_conditional_edges("classify", _after_classify, {
        "validate_and_guard": "validate_and_guard",
        "research": "research",
        "end": END,
    })

    graph.add_conditional_edges("research", _after_research, {
        "validate_and_guard": "validate_and_guard",
        "end": END,
    })

    graph.add_conditional_edges("validate_and_guard", _after_validate, {
        "molecule_info": "molecule_info",
        "research_fallback": "research_fallback",
        "end": END,
    })

    graph.add_conditional_edges("research_fallback", _after_research_fallback, {
        "validate_and_guard": "validate_and_guard",
        "end": END,
    })

    graph.add_edge("molecule_info", "interrupt_card")

    # Phase 1 -> Phase 2 (after user confirms)
    graph.add_edge("interrupt_card", "retrosynthesis")

    # ── Phase 2 edges ──
    graph.add_edge("retrosynthesis", "guard_safety")
    graph.add_edge("retrosynthesis", "reagent_check")

    graph.add_edge("guard_safety", "aggregate")
    graph.add_edge("reagent_check", "aggregate")

    graph.add_edge("aggregate", "interrupt_select_pathway")

    # Phase 2 -> Phase 3 (after user selects pathway + amount)
    graph.add_edge("interrupt_select_pathway", "stoichiometry")

    # ── Phase 3 edges ──
    graph.add_edge("stoichiometry", "experiment_planner")
    graph.add_edge("experiment_planner", END)

    return graph.compile(checkpointer=checkpointer)
