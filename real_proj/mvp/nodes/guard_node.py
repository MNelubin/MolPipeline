"""Guard node: safety check via banlists + PubChem GHS + PPE."""

from __future__ import annotations

import logging
from typing import Any, Literal

from ..tools import banlist_check, reaction_banlist_check, safety_lookup, ppe_recommender

logger = logging.getLogger(__name__)


def _determine_overall_status(
    mol_status: str,
    rxn_status: str,
) -> Literal["SAFE", "WARNING", "CRITICAL_STOP"]:
    critical = {"banned", "prohibited"}
    warning = {"restricted"}
    if mol_status in critical or rxn_status in critical:
        return "CRITICAL_STOP"
    if mol_status in warning or rxn_status in warning:
        return "WARNING"
    return "SAFE"


def guard_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: run safety checks on the validated SMILES.

    Reads: state["smiles"], state.get("reaction_description", "")
    Writes: state["guard_result"], optionally state["error"]
    """
    smiles: str = state.get("smiles", "").strip()
    if not smiles:
        return {
            "guard_result": {
                "overall_status": "CRITICAL_STOP",
                "molecule_check": {},
                "reaction_check": {},
                "safety_data": {},
                "ppe_recommendations": [],
            },
            "error": "guard_node: no SMILES in state.",
        }

    reaction_description: str = state.get("reaction_description", "")
    cid: int | None = state.get("pubchem_cid") or None

    logger.info("[guard] checking smiles=%r cid=%s", smiles, cid)

    # 1. Molecule banlist
    mol_check = banlist_check(smiles)
    logger.info("[guard] banlist → %s", mol_check.get("status"))

    # 2. Reaction banlist
    rxn_check = reaction_banlist_check(reaction_description)
    logger.info("[guard] reaction_banlist → %s", rxn_check.get("status"))

    # 3. GHS / PubChem safety (pass CID to avoid re-resolving)
    safety = safety_lookup(smiles, cid=cid)
    logger.info(
        "[guard] safety → %d H-phrases, %d pictograms",
        len(safety.get("h_phrases", [])),
        len(safety.get("ghs_pictograms", [])),
    )

    # 4. PPE recommendations
    h_phrases_str = ",".join(safety.get("h_phrases", []))
    ppe = ppe_recommender(smiles, h_phrases_str)
    logger.info("[guard] ppe → %s", ppe)

    # 5. Aggregation
    overall = _determine_overall_status(
        mol_status=mol_check.get("status", "clear"),
        rxn_status=rxn_check.get("status", "allowed"),
    )
    logger.info("[guard] overall_status=%s", overall)

    guard_result = {
        "overall_status": overall,
        "molecule_check": mol_check,
        "reaction_check": rxn_check,
        "safety_data": safety,
        "ppe_recommendations": ppe,
    }

    result: dict[str, Any] = {"guard_result": guard_result}

    if overall == "CRITICAL_STOP":
        reason = mol_check.get("reason", "") or rxn_check.get("reason", "")
        result["error"] = f"CRITICAL_STOP: {reason}"

    return result
