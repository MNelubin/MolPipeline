"""Pathway scoring and reagent availability tools."""

import httpx
from langchain_core.tools import tool

from src.config import CHEAP_REAGENTS, EXPENSIVE_REAGENTS, PUBCHEM_BASE_URL


@tool
def reagent_availability(query: str) -> dict:
    """Check if a reagent is commercially available via PubChem vendors.

    Args:
        query: Reagent name or SMILES
    """
    try:
        # Try to find the compound in PubChem
        url = f"{PUBCHEM_BASE_URL}/compound/name/{query}/JSON"
        resp = httpx.get(url, timeout=15.0)

        if resp.status_code == 404:
            # Try SMILES
            url = f"{PUBCHEM_BASE_URL}/compound/smiles/{query}/JSON"
            resp = httpx.get(url, timeout=15.0)

        if resp.status_code != 200:
            return {
                "name": query,
                "available": None,
                "reason": "Could not find in PubChem",
            }

        data = resp.json()
        cid = data["PC_Compounds"][0]["id"]["id"]["cid"]

        # Check for vendors
        vendor_url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
            f"/data/compound/{cid}/JSON?heading=Chemical+Vendors"
        )
        vendor_resp = httpx.get(vendor_url, timeout=15.0)

        has_vendors = vendor_resp.status_code == 200
        name_lower = query.lower()

        # Price category heuristic
        price_category = "medium"
        if name_lower in CHEAP_REAGENTS or any(
            c in name_lower for c in CHEAP_REAGENTS
        ):
            price_category = "cheap"
        elif name_lower in EXPENSIVE_REAGENTS or any(
            c in name_lower for c in EXPENSIVE_REAGENTS
        ):
            price_category = "expensive"

        return {
            "name": query,
            "cid": cid,
            "available": has_vendors,
            "price_category": price_category,
        }

    except Exception as e:
        return {"name": query, "available": None, "error": str(e)}


@tool
def pathway_scorer(pathway_data: dict) -> dict:
    """Score a synthesis pathway on multiple metrics.

    Args:
        pathway_data: Dict with keys: total_steps, overall_yield, safety_score,
                      reagent_availability_ratio, rare_reagent_ratio
    """
    total_steps = pathway_data.get("total_steps", 1)
    overall_yield = pathway_data.get("overall_yield", 0.5)
    safety_score = pathway_data.get("safety_score", 0.5)
    availability_ratio = pathway_data.get("reagent_availability_ratio", 1.0)
    rare_ratio = pathway_data.get("rare_reagent_ratio", 0.0)
    confidence = pathway_data.get("confidence_score", 0.5)

    # Weights
    w_steps = 0.15
    w_yield = 0.25
    w_safety = 0.20
    w_cost = 0.20
    w_confidence = 0.20

    # Normalize step count (1 step = 1.0, 7 steps = ~0.14)
    step_score = 1.0 / total_steps

    # Cost score combines availability and rarity
    cost_score = 0.6 * availability_ratio + 0.4 * (1.0 - rare_ratio)

    # Weighted total
    total = (
        w_steps * step_score
        + w_yield * overall_yield
        + w_safety * safety_score
        + w_cost * cost_score
        + w_confidence * confidence
    )

    return {
        "total_score": round(total, 3),
        "breakdown": {
            "step_score": round(step_score, 3),
            "yield_score": round(overall_yield, 3),
            "safety_score": round(safety_score, 3),
            "cost_score": round(cost_score, 3),
            "confidence_score": round(confidence, 3),
        },
        "weights": {
            "steps": w_steps,
            "yield": w_yield,
            "safety": w_safety,
            "cost": w_cost,
            "confidence": w_confidence,
        },
    }
