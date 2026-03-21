"""Reagent availability check node (Phase 2, runs in parallel with guard_safety).

If tree expansion was performed, uses tree leaf statuses directly (already
checked buyability during expansion). Otherwise falls back to _is_buyable
for immediate reactants.
"""

from __future__ import annotations

import logging
from typing import Any

from ..tools.retro_tools import _is_buyable

logger = logging.getLogger(__name__)


def reagent_node(state: dict[str, Any]) -> dict[str, Any]:
    """Check reagent availability for all synthesis pathways.

    Reads:  state["retro_result"]
    Writes: state["reagent_report"]
    """
    retro = state.get("retro_result", {})
    routes = retro.get("routes", [])

    if not routes:
        return {
            "reagent_report": {
                "pathway_reports": [],
                "all_available": True,
                "unavailable_reagents": [],
            }
        }

    pathway_reports: list[dict[str, Any]] = []
    all_unavailable: list[str] = []

    for i, route in enumerate(routes):
        tree = route.get("tree")
        if tree:
            report = _check_via_tree(i, tree)
        else:
            report = _check_via_immediate(i, route)

        pathway_reports.append(report)
        for smi in report["unavailable"]:
            if smi not in all_unavailable:
                all_unavailable.append(smi)

    overall_ok = all(p["all_available"] for p in pathway_reports)

    logger.info(
        "[reagent] checked %d pathways, all_available=%s, unavailable=%s",
        len(routes), overall_ok, all_unavailable[:5],
    )

    return {
        "reagent_report": {
            "pathway_reports": pathway_reports,
            "all_available": overall_ok,
            "unavailable_reagents": all_unavailable,
        }
    }


def _collect_leaves(node: dict[str, Any]) -> list[dict[str, Any]]:
    children = node.get("children", [])
    if not children:
        return [node]
    leaves: list[dict[str, Any]] = []
    for child in children:
        leaves.extend(_collect_leaves(child))
    return leaves


def _check_via_tree(idx: int, tree: dict[str, Any]) -> dict[str, Any]:
    """Use tree expansion results: leaves already have status from _build_node."""
    leaves = _collect_leaves(tree)
    available = []
    unavailable = []

    for leaf in leaves:
        smi = leaf.get("smiles", "")
        status = leaf.get("status", "")
        if status == "buyable":
            available.append(smi)
        elif status == "banned":
            unavailable.append(smi)
        else:
            unavailable.append(smi)

    return {
        "pathway_index": idx,
        "total_reagents": len(leaves),
        "available": available,
        "unavailable": unavailable,
        "all_available": len(unavailable) == 0,
        "source": "tree",
    }


def _check_via_immediate(idx: int, route: dict[str, Any]) -> dict[str, Any]:
    """Fallback: check immediate reactants via _is_buyable."""
    reactants_str = route.get("reactants", "")
    reactant_list = [r.strip() for r in reactants_str.split(".") if r.strip()]

    available = []
    unavailable = []

    for smi in reactant_list:
        if _is_buyable(smi):
            available.append(smi)
        else:
            unavailable.append(smi)

    return {
        "pathway_index": idx,
        "total_reagents": len(reactant_list),
        "available": available,
        "unavailable": unavailable,
        "all_available": len(unavailable) == 0,
        "source": "immediate",
    }
