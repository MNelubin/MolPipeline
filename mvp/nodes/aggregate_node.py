"""Aggregate node (Phase 2 fan-in): collect guard_safety + reagent_check results.

With tree expansion, retry cycles are no longer needed — the tree already
recursively decomposes intermediates. This node simply merges reports and
ranks pathways by viability.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def aggregate_node(state: dict[str, Any]) -> dict[str, Any]:
    """Fan-in: merge safety_report + reagent_report, rank pathways.

    Reads:  state["reagent_report"], state["safety_report"], state["retro_result"]
    Writes: state["synthesis_pathways"]
    """
    reagent_report = state.get("reagent_report", {})
    safety_report = state.get("safety_report", {})
    retro_result = state.get("retro_result", {})

    routes = retro_result.get("routes", [])

    reagent_pathway_reports = reagent_report.get("pathway_reports", [])
    safety_pathway_reports = safety_report.get("pathway_reports", [])

    pathways: list[dict[str, Any]] = []
    for i, route in enumerate(routes):
        r_report = reagent_pathway_reports[i] if i < len(reagent_pathway_reports) else {}
        s_report = safety_pathway_reports[i] if i < len(safety_pathway_reports) else {}

        reagents_ok = r_report.get("all_available", True)
        safety_ok = not s_report.get("has_critical", False)

        tree_stats = route.get("tree_stats") or {}
        unresolved = tree_stats.get("unresolved_count", 0)
        buyable = tree_stats.get("buyable_count", 0)

        pathway = dict(route)
        pathway["reagents_available"] = reagents_ok
        pathway["safety_ok"] = safety_ok
        pathway["viable"] = reagents_ok and safety_ok
        pathway["unresolved_leaves"] = unresolved
        pathway["buyable_leaves"] = buyable
        pathways.append(pathway)

    # Sort: viable first, then by fewer unresolved, then by score
    pathways.sort(
        key=lambda p: (
            not p["viable"],
            p["unresolved_leaves"],
            -(p.get("final_score", 0)),
        )
    )

    viable_count = sum(1 for p in pathways if p["viable"])
    logger.info(
        "[aggregate] %d/%d viable pathways (sorted by viability, unresolved, score)",
        viable_count, len(pathways),
    )

    return {
        "synthesis_pathways": pathways,
    }
