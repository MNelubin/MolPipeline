"""Stoichiometry calculation node (Phase 3).

If the selected pathway has a tree (from expand_tree), walks it bottom-up
and calculates stoichiometry for every intermediate reaction step, chaining
product masses down the tree. Otherwise falls back to single-step calculation
for the root reaction only.
"""

from __future__ import annotations

import logging
from typing import Any

from ..tools.calculations import stoichiometry_calc
from ..models.calculations import StoichiometryRequest

logger = logging.getLogger(__name__)


def stoichiometry_node(state: dict[str, Any]) -> dict[str, Any]:
    """Calculate stoichiometry for the selected pathway (multi-step if tree exists).

    Reads:  state["synthesis_pathways"], state["selected_pathway"],
            state["target_amount"], state["smiles"]
    Writes: state["calculations"]
    """
    from ..journal import AgentJournal
    j = AgentJournal.for_session(state.get("session_id", "default"))

    pathways = state.get("synthesis_pathways", [])
    selected_idx = state.get("selected_pathway")
    target_amount = state.get("target_amount")
    target_smiles = state.get("smiles", "")

    if selected_idx is None or not pathways:
        return {"calculations": {"error": "Путь синтеза не выбран.", "steps": []}}
    if selected_idx < 0 or selected_idx >= len(pathways):
        return {"calculations": {"error": f"Неверный индекс пути: {selected_idx}", "steps": []}}

    pathway = pathways[selected_idx]
    target_mass_g = 1.0
    if target_amount:
        target_mass_g = target_amount.get("value", 1.0)

    with j.step("stoichiometry"):
        tree = pathway.get("tree")
        if tree and tree.get("children"):
            result = _calc_from_tree(tree, target_mass_g, target_smiles)
        else:
            result = _calc_single_step(pathway, target_mass_g, target_smiles)

        calc = result.get("calculations", {})
        steps_count = len(calc.get("steps", []))
        buyable_count = len(calc.get("all_buyable_reagents", []))
        j.decision(
            "stoichiometry",
            f"Стехиометрия рассчитана: {steps_count} стадий, цель {target_mass_g:.2f} г",
            {"steps_count": steps_count or 1, "target_mass_g": target_mass_g,
             "buyable_reagents": buyable_count, "pathway_idx": selected_idx},
        )

    return result


def _calc_from_tree(
    tree: dict[str, Any], target_mass_g: float, target_smiles: str,
) -> dict[str, Any]:
    """Walk tree bottom-up, calc stoichiometry for each intermediate step."""
    steps: list[dict[str, Any]] = []
    all_warnings: list[str] = []

    _calc_node(tree, target_mass_g, steps, all_warnings)

    all_buyable: list[dict[str, Any]] = []
    seen: set[str] = set()
    for step in steps:
        for r in step.get("reagents", []):
            smi = r.get("smiles", "")
            if r.get("is_leaf") and smi not in seen:
                all_buyable.append(r)
                seen.add(smi)

    logger.info(
        "[stoichiometry] tree: %d steps, %d unique buyable reagents, target=%.2fg",
        len(steps), len(all_buyable), target_mass_g,
    )

    return {
        "calculations": {
            "steps": steps,
            "all_buyable_reagents": all_buyable,
            "target_mass_g": target_mass_g,
            "target_product_smiles": target_smiles,
            "target_moles": steps[-1].get("target_moles", 0) if steps else 0,
            "warnings": all_warnings,
        }
    }


def _calc_node(
    node: dict[str, Any],
    needed_mass_g: float,
    steps: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    """Recursively calculate stoichiometry for one tree node."""
    route = node.get("route")
    children = node.get("children", [])

    if not route or not children:
        return

    reactants_str = route.get("reactants", "")
    product_smiles = node.get("smiles", "")
    if not reactants_str or not product_smiles:
        return

    reaction_smiles = route.get("reaction_smiles", "")
    if not reaction_smiles or ">" not in reaction_smiles:
        reaction_smiles = f"{reactants_str}>>{product_smiles}"

    try:
        request = StoichiometryRequest(
            reaction_smiles=reaction_smiles,
            target_mass_g=needed_mass_g,
            target_product_smiles=product_smiles,
        )
        result = stoichiometry_calc(request)
        calc = result.model_dump(mode="json")

        child_smiles_map = {c.get("smiles", ""): c for c in children}

        reagents_enriched = []
        for reagent in calc.get("reagents", []):
            r_smi = reagent.get("smiles", "")
            child = child_smiles_map.get(r_smi)
            is_intermediate = child is not None and child.get("status") == "intermediate"
            reagent["is_leaf"] = not is_intermediate

            if is_intermediate:
                _calc_node(child, reagent.get("mass_g", 0), steps, warnings)

            reagents_enriched.append(reagent)

        step = {
            "step_number": len(steps) + 1,
            "product_smiles": product_smiles,
            "product_name": node.get("name", ""),
            "product_mass_g": round(needed_mass_g, 4),
            "reaction_smiles": reaction_smiles,
            "reagents": reagents_enriched,
            "target_moles": calc.get("target_moles", 0),
            "warnings": calc.get("warnings", []),
        }
        steps.append(step)
        warnings.extend(calc.get("warnings", []))

    except Exception as e:
        logger.warning(
            "[stoichiometry] step calc failed for %s: %s",
            product_smiles[:30], e,
        )
        warnings.append(f"Расчёт для {product_smiles[:30]} не удался: {e}")


def _calc_single_step(
    pathway: dict[str, Any], target_mass_g: float, target_smiles: str,
) -> dict[str, Any]:
    """Fallback: single-step calculation when tree is not available."""
    reaction_smiles = pathway.get("reaction_smiles", "")
    if not reaction_smiles:
        reactants = pathway.get("reactants", "")
        if reactants and target_smiles:
            reaction_smiles = f"{reactants}>>{target_smiles}"

    if not reaction_smiles or ">" not in reaction_smiles:
        logger.warning("[stoichiometry] no valid reaction SMILES, skipping")
        return {
            "calculations": {
                "error": "Нет reaction SMILES для расчёта стехиометрии.",
                "steps": [],
            },
        }

    try:
        request = StoichiometryRequest(
            reaction_smiles=reaction_smiles,
            target_mass_g=target_mass_g,
            target_product_smiles=target_smiles or None,
        )
        result = stoichiometry_calc(request)
        calc_dict = result.model_dump(mode="json")
        logger.info(
            "[stoichiometry] single-step: %d reagents, target=%.2fg",
            len(calc_dict.get("reagents", [])), target_mass_g,
        )
        return {"calculations": calc_dict}

    except Exception as e:
        logger.warning("[stoichiometry] calculation failed: %s", e)
        return {
            "calculations": {"error": str(e), "steps": []},
        }
