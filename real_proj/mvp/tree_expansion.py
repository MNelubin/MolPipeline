"""Recursive retrosynthesis tree expansion.

Expands a selected synthesis route into a full tree by recursively
breaking down non-buyable reactants until all leaves are either:
  - buyable (commercially available primitives)
  - banned (controlled substances)
  - unresolved (no route found)
  - depth/timeout/cycle limit reached
"""

from __future__ import annotations

import logging
import time
from typing import Any

from rdkit import Chem

from .retro_tools import (
    _is_buyable,
    ord_search_by_product,
    score_route,
)
from .tools import banlist_check, get_compound_properties

logger = logging.getLogger(__name__)

# Lazy import to avoid loading model weights at import time
_predict_retro = None


def _get_predict_retro():
    global _predict_retro
    if _predict_retro is None:
        from .retro_predictor import predict_retro
        _predict_retro = predict_retro
    return _predict_retro


def _canonicalize(smiles: str) -> str | None:
    """Canonicalize SMILES via RDKit. Returns None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _resolve_name(smiles: str) -> str | None:
    """Try to get a human-readable name for a molecule."""
    try:
        props = get_compound_properties(smiles)
        return props.get("IUPACName") or props.get("Title")
    except Exception:
        return None


def _find_best_route(smiles: str) -> dict[str, Any] | None:
    """Find the single best retrosynthesis route: ORD first, then model."""
    # 1. ORD search (authoritative)
    ord_results = ord_search_by_product(smiles, limit=5)
    if ord_results:
        for r in ord_results:
            score_route(r)
        ord_results.sort(key=lambda r: r.get("final_score", 0), reverse=True)
        logger.debug("[tree] ORD hit for %s: score=%.3f", smiles[:30], ord_results[0]["final_score"])
        return ord_results[0]

    # 2. ASKCOS model fallback
    try:
        predict = _get_predict_retro()
        model_results = predict(smiles, top_n=3)
        if model_results:
            for r in model_results:
                score_route(r)
            model_results.sort(key=lambda r: r.get("final_score", 0), reverse=True)
            logger.debug("[tree] Model hit for %s: score=%.3f", smiles[:30], model_results[0]["final_score"])
            return model_results[0]
    except Exception as e:
        logger.warning("[tree] Model failed for %s: %s", smiles[:30], e)

    return None


def _build_node(
    smiles: str,
    depth: int,
    max_depth: int,
    visited: set[str],
    start_time: float,
    timeout_sec: float,
) -> dict[str, Any]:
    """Recursively build a tree node for a single molecule."""

    canonical = _canonicalize(smiles)

    # Invalid SMILES
    if canonical is None:
        return {
            "smiles": smiles,
            "name": None,
            "status": "invalid_smiles",
            "depth": depth,
            "is_buyable": False,
            "guard": {"status": "error", "reason": "Invalid SMILES"},
            "route": None,
            "children": [],
        }

    # Cycle detection
    if canonical in visited:
        return {
            "smiles": canonical,
            "name": None,
            "status": "circular",
            "depth": depth,
            "is_buyable": False,
            "guard": {"status": "clear"},
            "route": None,
            "children": [],
        }

    # Timeout check
    elapsed = time.time() - start_time
    if elapsed > timeout_sec:
        return {
            "smiles": canonical,
            "name": None,
            "status": "timeout",
            "depth": depth,
            "is_buyable": False,
            "guard": {"status": "clear"},
            "route": None,
            "children": [],
        }

    # Guard check (banlist)
    guard = banlist_check(canonical)
    if guard.get("status") in ("banned", "restricted"):
        name = guard.get("name") or _resolve_name(canonical)
        return {
            "smiles": canonical,
            "name": name,
            "status": "banned",
            "depth": depth,
            "is_buyable": False,
            "guard": guard,
            "route": None,
            "children": [],
        }

    # Buyability check
    if _is_buyable(canonical):
        name = _resolve_name(canonical)
        return {
            "smiles": canonical,
            "name": name,
            "status": "buyable",
            "depth": depth,
            "is_buyable": True,
            "guard": guard,
            "route": None,
            "children": [],
        }

    # Depth limit
    if depth >= max_depth:
        return {
            "smiles": canonical,
            "name": _resolve_name(canonical),
            "status": "depth_limit",
            "depth": depth,
            "is_buyable": False,
            "guard": guard,
            "route": None,
            "children": [],
        }

    # Find best route for this intermediate
    route = _find_best_route(canonical)
    if route is None:
        return {
            "smiles": canonical,
            "name": _resolve_name(canonical),
            "status": "unresolved",
            "depth": depth,
            "is_buyable": False,
            "guard": guard,
            "route": None,
            "children": [],
        }

    # Parse reactants and recurse
    reactants_str = route.get("reactants", "")
    reactant_parts = [s.strip() for s in reactants_str.split(".") if s.strip()]

    visited_branch = visited | {canonical}
    children = []
    for reactant_smi in reactant_parts:
        child = _build_node(
            reactant_smi,
            depth=depth + 1,
            max_depth=max_depth,
            visited=visited_branch,
            start_time=start_time,
            timeout_sec=timeout_sec,
        )
        children.append(child)

    # Clean route for serialization (remove heavy template field)
    clean_route = {k: v for k, v in route.items() if k != "template"}

    return {
        "smiles": canonical,
        "name": _resolve_name(canonical),
        "status": "intermediate",
        "depth": depth,
        "is_buyable": False,
        "guard": guard,
        "route": clean_route,
        "children": children,
    }


def expand_tree(
    target_smiles: str,
    reactants: str,
    max_depth: int = 6,
    timeout_sec: float = 120.0,
) -> dict[str, Any]:
    """Expand a selected synthesis route into a full tree.

    Args:
        target_smiles: SMILES of the target molecule (root node).
        reactants: Dot-separated SMILES of reactants from the selected route.
        max_depth: Maximum recursion depth (default 6).
        timeout_sec: Maximum elapsed time in seconds (default 120).

    Returns:
        Dict with 'tree' (root TreeNode) and 'stats' (summary counts).
    """
    start_time = time.time()

    canonical_target = _canonicalize(target_smiles)
    if canonical_target is None:
        return {
            "tree": {
                "smiles": target_smiles,
                "name": None,
                "status": "invalid_smiles",
                "depth": 0,
                "is_buyable": False,
                "guard": {"status": "error"},
                "route": None,
                "children": [],
            },
            "stats": _empty_stats(time.time() - start_time),
        }

    # Build children from the selected route's reactants
    reactant_parts = [s.strip() for s in reactants.split(".") if s.strip()]
    visited = {canonical_target}

    children = []
    for reactant_smi in reactant_parts:
        child = _build_node(
            reactant_smi,
            depth=1,
            max_depth=max_depth,
            visited=visited,
            start_time=start_time,
            timeout_sec=timeout_sec,
        )
        children.append(child)

    # Root node
    guard = banlist_check(canonical_target)
    root = {
        "smiles": canonical_target,
        "name": _resolve_name(canonical_target),
        "status": "intermediate",
        "depth": 0,
        "is_buyable": False,
        "guard": guard,
        "route": {"reactants": reactants, "source": "selected"},
        "children": children,
    }

    elapsed = time.time() - start_time
    stats = _collect_stats(root, elapsed)

    logger.info(
        "[tree] expanded %s: %d nodes, %d buyable, %d banned, depth=%d, %.1fs",
        canonical_target[:30],
        stats["total_nodes"],
        stats["buyable_count"],
        stats["banned_count"],
        stats["max_depth_reached"],
        elapsed,
    )

    return {"tree": root, "stats": stats}


def _collect_stats(node: dict, elapsed: float) -> dict[str, Any]:
    """Walk the tree and collect summary statistics."""
    counts = {"total": 0, "buyable": 0, "banned": 0, "unresolved": 0, "max_depth": 0}
    _walk(node, counts)
    return {
        "total_nodes": counts["total"],
        "buyable_count": counts["buyable"],
        "banned_count": counts["banned"],
        "unresolved_count": counts["unresolved"],
        "max_depth_reached": counts["max_depth"],
        "elapsed_sec": round(elapsed, 2),
    }


def _walk(node: dict, counts: dict):
    counts["total"] += 1
    counts["max_depth"] = max(counts["max_depth"], node.get("depth", 0))
    status = node.get("status", "")
    if status == "buyable":
        counts["buyable"] += 1
    elif status == "banned":
        counts["banned"] += 1
    elif status in ("unresolved", "depth_limit", "timeout", "circular", "invalid_smiles"):
        counts["unresolved"] += 1
    for child in node.get("children", []):
        _walk(child, counts)


def _empty_stats(elapsed: float) -> dict[str, Any]:
    return {
        "total_nodes": 1,
        "buyable_count": 0,
        "banned_count": 0,
        "unresolved_count": 1,
        "max_depth_reached": 0,
        "elapsed_sec": round(elapsed, 2),
    }
