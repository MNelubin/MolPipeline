"""LangGraph StateGraph: validate → guard → (conditional) → molecule_info → END.

If guard returns CRITICAL_STOP, the graph skips molecule_info and goes to END.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from .state import MVPState
from .nodes.validate_node import validate_node
from .nodes.guard_node import guard_node
from .nodes.molecule_info_node import molecule_info_node


def _after_validate(state: MVPState) -> str:
    """Route after validation: if invalid → END, else → guard."""
    validation = state.get("validation", {})
    if not validation.get("is_valid", False):
        return "end"
    return "guard"


def _after_guard(state: MVPState) -> str:
    """Route after guard: CRITICAL_STOP → END, else → molecule_info."""
    guard_result = state.get("guard_result", {})
    if guard_result.get("overall_status") == "CRITICAL_STOP":
        return "end"
    return "molecule_info"


def build_graph() -> StateGraph:
    """Construct and compile the 3-node MVP graph."""
    graph = StateGraph(MVPState)

    # Nodes
    graph.add_node("validate", validate_node)
    graph.add_node("guard", guard_node)
    graph.add_node("molecule_info", molecule_info_node)

    # Edges
    graph.add_edge(START, "validate")
    graph.add_conditional_edges("validate", _after_validate, {
        "guard": "guard",
        "end": END,
    })
    graph.add_conditional_edges("guard", _after_guard, {
        "molecule_info": "molecule_info",
        "end": END,
    })
    graph.add_edge("molecule_info", END)

    return graph.compile()
