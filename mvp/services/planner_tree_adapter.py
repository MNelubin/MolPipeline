"""Planner route-tree adapters for the chat/runtime UI.

This is the integration seam for planner-specific tree formats. AiZynthFinder
uses atom-mapped retrosynthetic trees, while the frontend expects the project
TreeNode schema used by the molecule analysis graph.
"""

from __future__ import annotations

import time
from typing import Any

from ..smiles_normalization import canonicalize_smiles, canonicalize_smiles_list
from ..tools import banlist_check
from ..tools.retro_tools import _is_buyable


def _reaction_parts(metadata: dict[str, Any] | None) -> tuple[str, str] | None:
    if not isinstance(metadata, dict):
        return None
    mapped = metadata.get("mapped_reaction_smiles")
    if not isinstance(mapped, str) or ">>" not in mapped:
        return None
    product, reactants = mapped.split(">>", 1)
    product = canonicalize_smiles_list(product.strip()) or product.strip()
    reactants = canonicalize_smiles_list(reactants.strip()) or reactants.strip()
    if not product or not reactants:
        return None
    return reactants, product


def _is_molecule_node(node: dict[str, Any]) -> bool:
    node_type = str(node.get("type") or "").lower()
    return node_type in ("", "mol", "molecule") or bool(node.get("smiles"))


def _first_reaction_child(node: dict[str, Any]) -> dict[str, Any] | None:
    for child in node.get("children") or []:
        if not isinstance(child, dict):
            continue
        child_type = str(child.get("type") or "").lower()
        if child_type in ("reaction", "rxn") or _reaction_parts(child.get("metadata")):
            return child
    return None


def _name_for(node: dict[str, Any]) -> str | None:
    metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    value = node.get("name") or metadata.get("name") or metadata.get("label")
    return str(value) if value else None


def _status_for(smiles: str | None, *, has_children: bool, in_stock: bool) -> tuple[str, dict[str, Any], bool]:
    if not smiles:
        return "invalid_smiles", {"status": "error", "reason": "Invalid SMILES"}, False
    guard = banlist_check(smiles)
    if guard.get("status") == "banned":
        return "banned", guard, False
    if guard.get("status") == "restricted":
        return "restricted", guard, True
    if in_stock or _is_buyable(smiles):
        return "buyable", guard, True
    if has_children:
        return "intermediate", guard, False
    return "unresolved", guard, False


def _make_leaf(smiles: str, depth: int, visited: set[str]) -> dict[str, Any]:
    canonical = canonicalize_smiles(smiles)
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
    status, guard, buyable = _status_for(canonical, has_children=False, in_stock=False)
    return {
        "smiles": canonical or smiles,
        "name": None,
        "status": status,
        "depth": depth,
        "is_buyable": buyable,
        "guard": guard,
        "route": None,
        "children": [],
    }


def _adapt_aizynth_mol_node(
    node: dict[str, Any],
    *,
    depth: int,
    visited: set[str],
    selected_score: float | None = None,
) -> dict[str, Any]:
    raw_smiles = str(node.get("smiles") or "").strip()
    canonical = canonicalize_smiles(raw_smiles) if raw_smiles else None
    if canonical in visited:
        return {
            "smiles": canonical,
            "name": _name_for(node),
            "status": "circular",
            "depth": depth,
            "is_buyable": False,
            "guard": {"status": "clear"},
            "route": None,
            "children": [],
        }

    branch_visited = visited | ({canonical} if canonical else set())
    reaction = _first_reaction_child(node)
    reaction_parts = _reaction_parts(node.get("metadata"))
    reaction_children: list[dict[str, Any]] = []
    if reaction is not None:
        reaction_parts = _reaction_parts(reaction.get("metadata")) or reaction_parts
        reaction_children = [
            child
            for child in (reaction.get("children") or [])
            if isinstance(child, dict) and _is_molecule_node(child)
        ]

    children: list[dict[str, Any]] = []
    if reaction_children:
        children = [
            _adapt_aizynth_mol_node(child, depth=depth + 1, visited=branch_visited)
            for child in reaction_children
        ]
    elif reaction_parts:
        reactants, _product = reaction_parts
        children = [
            _make_leaf(part, depth + 1, branch_visited)
            for part in reactants.split(".")
            if part.strip()
        ]
    else:
        direct_children = [
            child
            for child in (node.get("children") or [])
            if isinstance(child, dict) and _is_molecule_node(child)
        ]
        children = [
            _adapt_aizynth_mol_node(child, depth=depth + 1, visited=branch_visited)
            for child in direct_children
        ]

    in_stock = bool(node.get("in_stock") or node.get("is_buyable"))
    status, guard, buyable = _status_for(canonical, has_children=bool(children), in_stock=in_stock)

    route = None
    if children and reaction_parts:
        reactants, product = reaction_parts
        route = {
            "source": "aizynthfinder",
            "reactants": reactants,
            "reaction_smiles": f"{reactants}>>{product}",
        }
        if selected_score is not None and depth == 0:
            route["final_score"] = selected_score

    return {
        "smiles": canonical or raw_smiles,
        "name": _name_for(node),
        "status": status,
        "depth": depth,
        "is_buyable": buyable,
        "guard": guard,
        "route": route,
        "children": children,
    }


def _walk_stats(node: dict[str, Any], counts: dict[str, int]) -> None:
    counts["total_nodes"] += 1
    counts["max_depth_reached"] = max(counts["max_depth_reached"], int(node.get("depth") or 0))
    status = node.get("status")
    children = node.get("children") or []
    if not children:
        counts["leaf_count"] += 1
        if status in ("buyable", "restricted"):
            counts["buyable_leaf_count"] += 1
        elif status in ("unresolved", "depth_limit", "timeout", "circular", "invalid_smiles"):
            counts["unresolved_leaf_count"] += 1
    if status in ("buyable", "restricted"):
        counts["buyable_count"] += 1
    elif status == "banned":
        counts["banned_count"] += 1
    elif status in ("unresolved", "depth_limit", "timeout", "circular", "invalid_smiles"):
        counts["unresolved_count"] += 1
    for child in children:
        if isinstance(child, dict):
            _walk_stats(child, counts)


def _collect_stats(tree: dict[str, Any], elapsed: float) -> dict[str, Any]:
    counts = {
        "total_nodes": 0,
        "buyable_count": 0,
        "banned_count": 0,
        "unresolved_count": 0,
        "leaf_count": 0,
        "buyable_leaf_count": 0,
        "unresolved_leaf_count": 0,
        "max_depth_reached": 0,
    }
    _walk_stats(tree, counts)
    counts["elapsed_sec"] = round(elapsed, 2)
    return counts


def adapt_aizynth_tree_to_runtime(
    raw_tree: dict[str, Any],
    *,
    target_smiles: str,
    selected_route: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Convert a raw AiZynthFinder route tree into the runtime TreeNode schema."""
    if not isinstance(raw_tree, dict):
        return None

    start = time.time()
    selected_score = None
    if selected_route and selected_route.get("final_score") is not None:
        selected_score = float(selected_route["final_score"])

    root = raw_tree
    if not root.get("smiles") and target_smiles:
        root = dict(raw_tree)
        root["smiles"] = target_smiles

    tree = _adapt_aizynth_mol_node(root, depth=0, visited=set(), selected_score=selected_score)
    if not tree.get("smiles"):
        return None
    return {
        "tree": tree,
        "stats": _collect_stats(tree, time.time() - start),
        "adapter": "aizynthfinder_tree",
    }
