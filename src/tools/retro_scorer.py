"""Autonomous retrosynthesis scoring system.

Two-level scoring inspired by ASKCOS:
  Level 1 — Precursor scoring: rank individual one-step disconnections
  Level 2 — Pathway scoring:   rank entire multi-step synthesis routes

Uses only RDKit (no ML models needed). Fully autonomous, no external API calls.
"""

import logging
import math
from functools import lru_cache

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, DataStructs

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    logger.warning("RDKit not available — scoring will use fallback heuristics")

# ---------------------------------------------------------------------------
# Buyability heuristic (no DB needed — uses structural complexity)
# ---------------------------------------------------------------------------

# Common cheap reagents: if we recognize the SMILES, it's "buyable"
_KNOWN_CHEAP: set[str] | None = None


@lru_cache(maxsize=1)
def _load_cheap_set() -> set[str]:
    """Simple set of common cheap reagent canonical SMILES."""
    reagents = [
        "O", "CO", "CCO", "CC(C)O", "CC(C)=O", "CC=O", "CC(O)=O",
        "CC(=O)OC(C)=O", "ClCCl", "ClC(Cl)Cl", "C(Cl)(Cl)(Cl)Cl",
        "C1CCOC1", "CCOCC", "CS(C)=O", "CN(C)C=O",
        "O=CO", "Cl", "O=C=O", "N", "[NH4+]", "O=S(=O)(O)O",
        "O=[N+]([O-])O", "[Na+].[OH-]", "[K+].[OH-]",
        "[Na+].[Cl-]", "[Na+].[O-]C(=O)C",
        "c1ccccc1", "Cc1ccccc1", "CCCCCC", "C1CCCCC1",
        "CC(=O)OCC", "COC(C)=O", "COC(=O)OC",
        "O=C(Cl)Cl",  # not banned in small-scale lab context
        "CCO", "CO", "O",
        "[Li]CCCC", "CC(C)(C)[Li]", "[Na]CC",
        "C([O-])(=O)[O-].[Na+].[Na+]",  # Na2CO3
        "[Mg+2].[Cl-].[Cl-]",
        "O=S(=O)([O-])[O-].[Mg+2]",  # MgSO4
        "O=P(O)(O)O", "O=P([O-])([O-])[O-]",
    ]
    canonical = set()
    if HAS_RDKIT:
        for smi in reagents:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                canonical.add(Chem.MolToSmiles(mol, isomericSmiles=True))
    else:
        canonical = set(reagents)
    return canonical


def is_likely_buyable(smiles: str) -> bool:
    """Fast heuristic: is this molecule likely commercially available?

    Uses structural complexity as proxy — simple small molecules are usually buyable.
    """
    cheap = _load_cheap_set()
    if smiles in cheap:
        return True

    if not HAS_RDKIT:
        return len(smiles) < 15

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    heavy = mol.GetNumHeavyAtoms()
    rings = Descriptors.RingCount(mol)
    chiral = len(Chem.FindMolChiralCenters(mol))

    # Simple molecules (< 10 heavy atoms, ≤ 1 ring, no chirality) are usually buyable
    if heavy <= 10 and rings <= 1 and chiral == 0:
        return True
    # Medium molecules with common patterns
    if heavy <= 6:
        return True

    return False


# ---------------------------------------------------------------------------
# Level 1: Precursor scoring (one-step retro results)
# ---------------------------------------------------------------------------

def score_precursor_set(
    precursor_smiles: str,
    model_score: float = 1.0,
    plausibility: float = 1.0,
) -> dict:
    """Score a set of precursors from a one-step retro prediction.

    Reimplements ASKCOS expand_one relevance heuristic:
        score = structural_penalty / model_score
        structural_penalty = sum over reactants of:
            -2.0 * atoms^1.5 - 1.0 * ring_bonds^1.5 - 2.0 * chiral^2.0
        buyable reactants get bonus: -ppg/1000 (we use -0.01 for cheap)

    Args:
        precursor_smiles: Dot-separated SMILES of all precursors.
        model_score: Score from retro model (higher = more confident).
        plausibility: Fast-filter plausibility score (0-1).

    Returns:
        Dict with score, breakdown, and buyability info.
    """
    if not HAS_RDKIT:
        return _score_precursor_fallback(precursor_smiles, model_score, plausibility)

    reactants = [s.strip() for s in precursor_smiles.split(".") if s.strip()]
    scores = []
    buyable_count = 0
    total_atoms = 0
    details = []

    for smi in reactants:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scores.append(-100.0)
            details.append({"smiles": smi, "error": "invalid SMILES"})
            continue

        heavy = mol.GetNumHeavyAtoms()
        total_atoms += heavy
        ring_bonds = sum(
            1 for b in mol.GetBonds() if b.IsInRing() and not b.GetIsAromatic()
        )
        chiral = len(Chem.FindMolChiralCenters(mol))
        rings = Descriptors.RingCount(mol)

        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        buyable = is_likely_buyable(canonical)
        if buyable:
            buyable_count += 1

        if buyable:
            component_score = -0.01  # cheap buyable bonus
        else:
            component_score = (
                -2.0 * math.pow(heavy, 1.5)
                - 1.0 * math.pow(ring_bonds, 1.5)
                - 2.0 * math.pow(chiral, 2.0)
            )

        scores.append(component_score)
        details.append({
            "smiles": canonical,
            "heavy_atoms": heavy,
            "rings": rings,
            "ring_bonds": ring_bonds,
            "chiral_centers": chiral,
            "buyable": buyable,
            "component_score": round(component_score, 3),
        })

    raw_score = sum(scores)
    # Normalize by model score (higher model score → less penalty)
    normalized = raw_score / max(model_score, 0.001)

    # Final composite: combine with plausibility
    # ASKCOS uses plausibility as a filter (threshold), we blend it in
    composite = normalized * plausibility

    n_reactants = len(reactants)
    buyability_ratio = buyable_count / max(n_reactants, 1)

    # Complexity: RMS molecular weight of precursors (ASKCOS uses this)
    rms_mw = 0.0
    if HAS_RDKIT:
        mws = []
        for smi in reactants:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                mws.append(Descriptors.ExactMolWt(mol))
        if mws:
            rms_mw = math.sqrt(sum(w * w for w in mws) / len(mws))

    return {
        "precursor_score": round(composite, 4),
        "raw_score": round(raw_score, 4),
        "normalized_score": round(normalized, 4),
        "model_score": round(model_score, 4),
        "plausibility": round(plausibility, 4),
        "num_reactants": n_reactants,
        "buyability_ratio": round(buyability_ratio, 3),
        "total_heavy_atoms": total_atoms,
        "rms_molecular_weight": round(rms_mw, 2),
        "reactants": details,
    }


def _score_precursor_fallback(
    precursor_smiles: str, model_score: float, plausibility: float
) -> dict:
    """Fallback scoring without RDKit — uses string length as complexity proxy."""
    reactants = [s.strip() for s in precursor_smiles.split(".") if s.strip()]
    total_len = sum(len(r) for r in reactants)
    # Longer SMILES = more complex = worse score
    raw = -total_len * 0.5
    normalized = raw / max(model_score, 0.001)
    composite = normalized * plausibility

    return {
        "precursor_score": round(composite, 4),
        "raw_score": round(raw, 4),
        "normalized_score": round(normalized, 4),
        "model_score": round(model_score, 4),
        "plausibility": round(plausibility, 4),
        "num_reactants": len(reactants),
        "buyability_ratio": 0.5,
        "total_heavy_atoms": total_len,
        "rms_molecular_weight": 0.0,
        "reactants": [{"smiles": r, "fallback": True} for r in reactants],
    }


# ---------------------------------------------------------------------------
# Level 2: Pathway scoring (multi-step routes)
# ---------------------------------------------------------------------------

# Weights for pathway composite score
PATHWAY_WEIGHTS = {
    "step_efficiency": 0.10,   # fewer steps = better
    "overall_yield": 0.20,     # higher cumulative yield = better
    "buyability": 0.20,        # more buyable starting materials = better
    "complexity_drop": 0.15,   # bigger complexity reduction per step = better
    "confidence": 0.15,        # DB-confirmed steps vs predicted
    "plausibility": 0.10,      # average plausibility across steps
    "safety": 0.10,            # safety score from guard agent
}


def score_pathway(
    steps: list[dict],
    target_smiles: str,
    safety_score: float | None = None,
) -> dict:
    """Score an entire synthesis pathway.

    Combines multiple metrics into a single composite score.
    Each step dict should have:
        reaction_smiles, score (model), source, expected_yield, plausibility

    Args:
        steps: List of step dicts from retrosynthesis.
        target_smiles: SMILES of the target molecule.
        safety_score: Optional safety score from guard agent (0-1, 1=safe).

    Returns:
        Comprehensive scoring breakdown.
    """
    n_steps = len(steps)
    if n_steps == 0:
        return {"total_score": 0.0, "error": "no steps"}

    # --- Step efficiency ---
    # 1 step = 1.0, 3 steps = 0.5, 6 steps = 0.25, 10 steps = 0.15
    step_efficiency = 1.0 / (1.0 + 0.3 * (n_steps - 1))

    # --- Overall yield ---
    yields = []
    for s in steps:
        y = s.get("expected_yield")
        if y is not None and 0 < y <= 1:
            yields.append(y)
    if yields:
        overall_yield = 1.0
        for y in yields:
            overall_yield *= y
    else:
        overall_yield = 0.5 ** n_steps  # assume 50% per step if unknown

    # --- Buyability of starting materials ---
    starting_materials = _extract_starting_materials(steps)
    if starting_materials:
        buyable = sum(1 for sm in starting_materials if is_likely_buyable(sm))
        buyability = buyable / len(starting_materials)
    else:
        buyability = 0.5

    # --- Complexity drop ---
    # How much simpler are starting materials vs target?
    complexity_drop = _compute_complexity_drop(target_smiles, starting_materials)

    # --- Confidence ---
    db_sources = {"askcos", "ibm_rxn", "ord", "rxn", "reaxys", "pistachio"}
    confirmed = sum(
        1 for s in steps if s.get("source", "").lower() in db_sources
    )
    confidence = confirmed / n_steps

    # --- Average plausibility ---
    plausibilities = [
        s.get("plausibility", s.get("score", 0.5)) for s in steps
    ]
    avg_plausibility = sum(plausibilities) / len(plausibilities)

    # --- Safety ---
    safety = safety_score if safety_score is not None else 0.7  # assume decent

    # --- Composite ---
    w = PATHWAY_WEIGHTS
    total = (
        w["step_efficiency"] * step_efficiency
        + w["overall_yield"] * overall_yield
        + w["buyability"] * buyability
        + w["complexity_drop"] * complexity_drop
        + w["confidence"] * confidence
        + w["plausibility"] * avg_plausibility
        + w["safety"] * safety
    )

    return {
        "total_score": round(total, 4),
        "breakdown": {
            "step_efficiency": round(step_efficiency, 4),
            "overall_yield": round(overall_yield, 4),
            "buyability": round(buyability, 4),
            "complexity_drop": round(complexity_drop, 4),
            "confidence": round(confidence, 4),
            "avg_plausibility": round(avg_plausibility, 4),
            "safety": round(safety, 4),
        },
        "weights": w,
        "num_steps": n_steps,
        "num_starting_materials": len(starting_materials),
        "starting_materials_buyable": sum(
            1 for sm in starting_materials if is_likely_buyable(sm)
        ),
        "estimated_overall_yield": round(overall_yield, 4),
    }


def _extract_starting_materials(steps: list[dict]) -> list[str]:
    """Extract leaf-node starting materials from a linear pathway.

    Starting materials = reactants in step 1 (deepest retro step).
    For tree-shaped pathways, collects all reactants that aren't products
    of any other step.
    """
    all_products = set()
    all_reactants = set()

    for s in steps:
        rxn = s.get("reaction_smiles", "")
        if ">>" not in rxn:
            continue
        reactant_str, product_str = rxn.split(">>", 1)
        for smi in reactant_str.split("."):
            smi = smi.strip()
            if smi:
                all_reactants.add(smi)
        for smi in product_str.split("."):
            smi = smi.strip()
            if smi:
                all_products.add(smi)

    # Starting materials = reactants that are not produced by any step
    starting = all_reactants - all_products
    return list(starting) if starting else list(all_reactants)


def _compute_complexity_drop(target_smiles: str, starting_materials: list[str]) -> float:
    """Compute normalized complexity reduction from target to starting materials.

    Uses heavy atom count as complexity proxy.
    Returns 0-1 where 1 = large reduction (good).
    """
    if not HAS_RDKIT or not starting_materials:
        return 0.5

    target_mol = Chem.MolFromSmiles(target_smiles)
    if target_mol is None:
        return 0.5

    target_complexity = target_mol.GetNumHeavyAtoms()
    if target_complexity == 0:
        return 0.5

    max_sm_complexity = 0
    for smi in starting_materials:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            max_sm_complexity = max(max_sm_complexity, mol.GetNumHeavyAtoms())

    # Ratio: how much simpler is the most complex SM vs target
    # If target has 30 atoms and biggest SM has 10, ratio = 1 - 10/30 = 0.67
    ratio = 1.0 - (max_sm_complexity / target_complexity)
    return max(0.0, min(1.0, ratio))


# ---------------------------------------------------------------------------
# Rank a batch of results
# ---------------------------------------------------------------------------

def rank_precursors(results: list[dict]) -> list[dict]:
    """Score and rank a list of one-step retro results.

    Each result should have at minimum: 'reactants' (SMILES string).
    Optional: 'score' (model score), 'plausibility'.

    Returns the same list sorted by precursor_score (best first),
    with scoring details added to each result.
    """
    scored = []
    for r in results:
        reactants = r.get("reactants", "")
        if not reactants:
            continue

        model_score = r.get("score", r.get("model_score", 0.5))
        plausibility = r.get("plausibility", 1.0)

        scoring = score_precursor_set(reactants, model_score, plausibility)
        r["precursor_scoring"] = scoring
        r["precursor_score"] = scoring["precursor_score"]
        scored.append(r)

    scored.sort(key=lambda x: x["precursor_score"], reverse=True)

    for i, r in enumerate(scored):
        r["rank"] = i + 1

    return scored


def rank_pathways(
    pathways: list[dict],
    target_smiles: str,
    safety_scores: dict[str, float] | None = None,
) -> list[dict]:
    """Score and rank a list of synthesis pathways.

    Each pathway dict should have: 'pathway_id', 'steps' (list of step dicts).

    Args:
        pathways: List of pathway dicts.
        target_smiles: Target molecule SMILES.
        safety_scores: Optional dict mapping pathway_id to safety score.

    Returns:
        Sorted list with scoring details.
    """
    safety_scores = safety_scores or {}

    scored = []
    for p in pathways:
        pid = p.get("pathway_id", "unknown")
        steps = p.get("steps", [])
        safety = safety_scores.get(pid)

        scoring = score_pathway(steps, target_smiles, safety)
        p["pathway_scoring"] = scoring
        p["total_score"] = scoring["total_score"]
        scored.append(p)

    scored.sort(key=lambda x: x["total_score"], reverse=True)

    for i, p in enumerate(scored):
        p["rank"] = i + 1

    return scored


# ---------------------------------------------------------------------------
# LangChain tools for agent use
# ---------------------------------------------------------------------------

@tool
def score_retro_precursors(precursor_smiles: str, model_score: float = 1.0) -> dict:
    """Score a set of retrosynthesis precursors for quality and feasibility.

    Evaluates structural complexity, buyability, and overall desirability
    of proposed precursors. Higher score = better disconnection.

    Based on ASKCOS relevance heuristic: penalizes complex, non-buyable
    precursors with many atoms, rings, and chiral centers.

    Args:
        precursor_smiles: Dot-separated SMILES of precursors (e.g. "CCO.CC(=O)Cl")
        model_score: Confidence score from the retro model (0-1, default 1.0)
    """
    return score_precursor_set(precursor_smiles, model_score)


@tool
def score_synthesis_pathway(
    steps: list[dict],
    target_smiles: str,
) -> dict:
    """Score a complete multi-step synthesis pathway.

    Evaluates step efficiency, overall yield, starting material buyability,
    complexity reduction, data source confidence, and plausibility.

    Each step should be a dict with keys:
        reaction_smiles, score, source, expected_yield

    Args:
        steps: List of reaction step dicts
        target_smiles: SMILES of the target molecule
    """
    return score_pathway(steps, target_smiles)
