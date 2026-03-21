"""Guard safety node for synthesis pathways (Phase 2, parallel with reagent_node).

If tree expansion was performed, checks all tree nodes (intermediates + leaves)
against banlists. Otherwise falls back to checking immediate reactants.
"""

from __future__ import annotations

import logging
from typing import Any

from ..tools import banlist_check

logger = logging.getLogger(__name__)


def guard_safety_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run safety checks on all reagents across synthesis pathways.

    Reads:  state["retro_result"]
    Writes: state["safety_report"]
    """
    retro = state.get("retro_result", {})
    routes = retro.get("routes", [])

    if not routes:
        return {
            "safety_report": {
                "pathway_reports": [],
                "has_critical": False,
                "warnings": [],
            }
        }

    pathway_reports: list[dict[str, Any]] = []
    all_warnings: list[str] = []
    has_critical = False

    for i, route in enumerate(routes):
        tree = route.get("tree")
        if tree:
            smiles_to_check = _collect_all_smiles(tree)
        else:
            reactants_str = route.get("reactants", "")
            smiles_to_check = [r.strip() for r in reactants_str.split(".") if r.strip()]

        checks = []
        route_critical = False

        for smi in smiles_to_check:
            result = banlist_check(smi)
            status = result.get("status", "clear")
            checks.append(result)

            if status == "banned":
                route_critical = True
                has_critical = True
                all_warnings.append(
                    f"Путь #{i+1}: {smi} — {result.get('reason', 'в банлисте')}"
                )
            elif status == "restricted":
                all_warnings.append(
                    f"Путь #{i+1}: {smi} — ограничен ({result.get('reason', '')})"
                )

        pathway_reports.append({
            "pathway_index": i,
            "reagent_checks": checks,
            "has_critical": route_critical,
        })

    logger.info(
        "[guard_safety] checked %d pathways, critical=%s, warnings=%d",
        len(routes), has_critical, len(all_warnings),
    )

    return {
        "safety_report": {
            "pathway_reports": pathway_reports,
            "has_critical": has_critical,
            "warnings": all_warnings,
        }
    }


def _collect_all_smiles(node: dict[str, Any]) -> list[str]:
    """Collect SMILES of all nodes in the tree (intermediates + leaves)."""
    result = []
    smi = node.get("smiles")
    if smi:
        result.append(smi)
    for child in node.get("children", []):
        result.extend(_collect_all_smiles(child))
    return result
