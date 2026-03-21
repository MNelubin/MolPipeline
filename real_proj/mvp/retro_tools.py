"""Retrosynthesis tools for MVP: ORD SQLite search, local model, scoring.

Searches local ORD index for known synthesis routes (with procedure_details),
uses standalone retro model (extracted from ASKCOS) for prediction,
deduplicates, scores and ranks all candidates.
"""

from __future__ import annotations

import logging
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).parent.parent.parent
ORD_DB_PATH = _PROJECT_ROOT / "data" / "ord_reactions.db"

# RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# ═════════════════════════════════════════════════════════════════════════════
# Canonical deduplication
# ═════════════════════════════════════════════════════════════════════════════


def _canonical_reactant_key(reactants_str: str) -> str | None:
    """Create a canonical dedup key from dot-separated reactant SMILES."""
    if not HAS_RDKIT or not reactants_str:
        return reactants_str
    parts = []
    for smi in reactants_str.split("."):
        smi = smi.strip()
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        parts.append(Chem.MolToSmiles(mol, isomericSmiles=True))
    parts.sort()
    return ".".join(parts)


def _deduplicate_routes(routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate routes by canonical reactant set.

    Keeps the route with the higher score when duplicates are found.
    """
    seen: dict[str, dict[str, Any]] = {}
    for route in routes:
        key = _canonical_reactant_key(route.get("reactants", ""))
        if key is None:
            continue
        existing = seen.get(key)
        if existing is None or route.get("final_score", 0) > existing.get("final_score", 0):
            seen[key] = route
    return list(seen.values())


# ═════════════════════════════════════════════════════════════════════════════
# ORD SQLite search
# ═════════════════════════════════════════════════════════════════════════════


def _get_ord_db() -> sqlite3.Connection | None:
    if not ORD_DB_PATH.exists():
        logger.warning("ORD database not found at %s", ORD_DB_PATH)
        return None
    return sqlite3.connect(str(ORD_DB_PATH))


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor)


def ord_search_by_product(smiles: str, limit: int = 15) -> list[dict[str, Any]]:
    """Search ORD for reactions producing the target molecule.

    Returns list of dicts with: reaction_id, reaction_smiles, reactants,
    yield_pct, temperature, solvent, catalyst, procedure_details, source.
    """
    conn = _get_ord_db()
    if conn is None:
        return []

    has_procedure = _has_column(conn, "reactions", "procedure_details")

    cols = "r.id, r.reaction_smiles, r.yield_pct, r.temperature, r.solvent, r.catalyst"
    if has_procedure:
        cols += ", r.procedure_details"

    results: list[dict[str, Any]] = []

    # 1. Exact canonical match via product_index
    if HAS_RDKIT:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
            cursor = conn.execute(
                f"""
                SELECT {cols}
                FROM product_index pi
                JOIN reactions r ON r.id = pi.reaction_id
                WHERE pi.canonical_smiles = ?
                LIMIT ?
                """,
                (canonical, limit),
            )
            results = _rows_to_dicts(cursor, has_procedure)

    # 2. Fallback: direct component match
    if not results:
        cursor = conn.execute(
            f"""
            SELECT {cols}
            FROM components c
            JOIN reactions r ON r.id = c.reaction_id
            WHERE c.role = 'product' AND c.smiles = ?
            LIMIT ?
            """,
            (smiles, limit),
        )
        results = _rows_to_dicts(cursor, has_procedure)

    conn.close()
    logger.info("ORD search: %d results for %s", len(results), smiles[:30])
    return results[:limit]


def _rows_to_dicts(cursor, has_procedure: bool) -> list[dict[str, Any]]:
    results = []
    for row in cursor:
        if has_procedure:
            rxn_id, rxn_smi, yield_pct, temp, solvent, catalyst, procedure = row
        else:
            rxn_id, rxn_smi, yield_pct, temp, solvent, catalyst = row
            procedure = None

        if not rxn_smi:
            continue

        reactant_str = rxn_smi.split(">>")[0] if ">>" in rxn_smi else ""

        result: dict[str, Any] = {
            "reaction_id": rxn_id,
            "reaction_smiles": rxn_smi,
            "reactants": reactant_str,
            "source": "ord",
            "score": 0.85,
            "plausibility": 0.90,
        }
        if yield_pct is not None:
            result["expected_yield"] = yield_pct / 100.0
        if temp:
            result["temperature"] = temp
        if solvent:
            result["solvent"] = solvent
        if catalyst:
            result["catalyst"] = catalyst
        if procedure:
            result["procedure_details"] = procedure

        results.append(result)
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Scoring (adapted from src/tools/retro_scorer.py)
# ═════════════════════════════════════════════════════════════════════════════

_CHEAP_REAGENTS = {
    "O", "CO", "CCO", "CC(C)O", "CC(C)=O", "CC=O", "CC(O)=O",
    "CC(=O)OC(C)=O", "ClCCl", "ClC(Cl)Cl", "C(Cl)(Cl)(Cl)Cl",
    "C1CCOC1", "CCOCC", "CS(C)=O", "CN(C)C=O",
    "O=CO", "Cl", "O=C=O", "N", "[NH4+]", "O=S(=O)(O)O",
    "O=[N+]([O-])O", "[Na+].[OH-]", "[K+].[OH-]",
    "[Na+].[Cl-]", "c1ccccc1", "Cc1ccccc1", "CCCCCC", "C1CCCCC1",
    "CC(=O)OCC", "COC(C)=O", "COC(=O)OC",
    "O=P(O)(O)O",
}


@lru_cache(maxsize=1)
def _get_cheap_canonical() -> set[str]:
    if not HAS_RDKIT:
        return _CHEAP_REAGENTS
    canonical = set()
    for smi in _CHEAP_REAGENTS:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            canonical.add(Chem.MolToSmiles(mol, isomericSmiles=True))
    return canonical


def _is_buyable(smiles: str) -> bool:
    cheap = _get_cheap_canonical()
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
    return (heavy <= 10 and rings <= 1 and chiral == 0) or heavy <= 6


def score_route(route: dict[str, Any]) -> dict[str, Any]:
    """Score a single retrosynthesis route."""
    reactants_str = route.get("reactants", "")
    model_score = route.get("score", 0.5)
    plausibility = route.get("plausibility", 0.8)

    reactants = [s.strip() for s in reactants_str.split(".") if s.strip()]
    n_reactants = len(reactants) if reactants else 1

    buyable_count = 0
    total_atoms = 0
    max_atoms = 0
    total_chiral = 0

    for smi in reactants:
        if not HAS_RDKIT:
            total_atoms += len(smi)
            max_atoms = max(max_atoms, len(smi))
            if len(smi) < 15:
                buyable_count += 1
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        heavy = mol.GetNumHeavyAtoms()
        total_atoms += heavy
        max_atoms = max(max_atoms, heavy)
        total_chiral += len(Chem.FindMolChiralCenters(mol))
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        if _is_buyable(canonical):
            buyable_count += 1

    buyability_ratio = buyable_count / max(n_reactants, 1)
    simplicity = 1.0 / (1.0 + 0.08 * max_atoms)
    simplicity *= 1.0 / (1.0 + 0.3 * total_chiral)
    efficiency = 1.0 / (1.0 + 0.25 * (n_reactants - 1))

    yield_bonus = 0.0
    if route.get("expected_yield") is not None:
        yield_bonus = min(route["expected_yield"], 1.0) * 0.1

    procedure_bonus = 0.05 if route.get("procedure_details") else 0.0

    composite = (
        0.25 * min(model_score, 1.0)
        + 0.20 * min(plausibility, 1.0)
        + 0.20 * buyability_ratio
        + 0.15 * simplicity
        + 0.10 * efficiency
        + yield_bonus
        + procedure_bonus
    )

    route["final_score"] = round(composite, 4)
    route["scoring"] = {
        "model_score": round(min(model_score, 1.0), 4),
        "plausibility": round(min(plausibility, 1.0), 4),
        "buyability": round(buyability_ratio, 4),
        "simplicity": round(simplicity, 4),
        "efficiency": round(efficiency, 4),
        "yield_bonus": round(yield_bonus, 4),
        "procedure_bonus": round(procedure_bonus, 4),
        "num_reactants": n_reactants,
        "total_atoms": total_atoms,
        "buyable_count": buyable_count,
    }
    return route


# ═════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═════════════════════════════════════════════════════════════════════════════


def search_and_rank(smiles: str, top_n: int = 5) -> dict[str, Any]:
    """Full retrosynthesis pipeline: ORD → local retro model → deduplicate → score → rank.

    Returns dict with:
        routes: list of scored routes (best first), deduplicated
        best_route: the top route (or None)
        sources_used: list of sources that returned data
        total_found: total candidates before dedup/ranking
    """
    all_routes: list[dict[str, Any]] = []
    sources_used: list[str] = []

    # 1. ORD search (primary)
    ord_results = ord_search_by_product(smiles, limit=15)
    if ord_results:
        all_routes.extend(ord_results)
        sources_used.append("ord")
        logger.info("ORD: %d routes found", len(ord_results))

    # 2. Local retro model (standalone, extracted from ASKCOS)
    try:
        from .retro_predictor import predict_retro
        model_results = predict_retro(smiles, top_n=10)
        if model_results:
            all_routes.extend(model_results)
            sources_used.append("retro_model")
            logger.info("Retro model: %d routes found", len(model_results))
    except Exception as e:
        logger.warning("Retro model failed: %s", e)

    total_found = len(all_routes)

    if not all_routes:
        return {
            "routes": [],
            "best_route": None,
            "sources_used": [],
            "total_found": 0,
        }

    # 3. Score all routes
    for route in all_routes:
        score_route(route)

    # 4. Deduplicate by canonical reactant set
    all_routes = _deduplicate_routes(all_routes)
    logger.info("After dedup: %d routes (from %d)", len(all_routes), total_found)

    # 5. Sort by score (best first) and take top N
    all_routes.sort(key=lambda r: r.get("final_score", 0), reverse=True)
    top_routes = all_routes[:top_n]

    return {
        "routes": top_routes,
        "best_route": top_routes[0] if top_routes else None,
        "sources_used": sources_used,
        "total_found": total_found,
    }
