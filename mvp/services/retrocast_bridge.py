"""Optional bridge to the real RetroCast package.

RetroCast is not a standalone route generator. It adapts and scores raw planner
outputs against a canonical multistep route schema. We use it as an optional
runtime bridge to enrich planner outputs such as AiZynthFinder with canonical
route metadata, route hashes, and topology-aware summaries.
"""

from __future__ import annotations

from typing import Any


def get_retrocast_runtime_info() -> dict[str, Any]:
    """Return availability and adapter metadata for the installed RetroCast package."""
    try:
        from retrocast import ADAPTER_MAP, __version__
    except ImportError as exc:
        return {
            "available": False,
            "version": None,
            "adapters": [],
            "error": str(exc),
        }

    return {
        "available": True,
        "version": __version__,
        "adapters": sorted(ADAPTER_MAP.keys()),
        "error": None,
    }


def _route_to_summary(route: Any, *, route_index: int) -> dict[str, Any] | None:
    """Convert a RetroCast Route object into a compact runtime summary."""
    target = getattr(route, "target", None)
    if target is None:
        return None

    product = getattr(target, "smiles", "") or ""
    synthesis_step = getattr(target, "synthesis_step", None)
    reactant_nodes = getattr(synthesis_step, "reactants", []) if synthesis_step is not None else []
    reactants = ".".join(
        reactant.smiles
        for reactant in reactant_nodes
        if getattr(reactant, "smiles", None)
    )
    reaction_smiles = f"{reactants}>>{product}" if reactants and product else ""

    leaves = getattr(route, "leaves", set()) or set()
    leaf_smiles = sorted(
        leaf.smiles
        for leaf in leaves
        if getattr(leaf, "smiles", None)
    )

    return {
        "route_index": route_index,
        "target_smiles": product,
        "reactants": reactants,
        "reaction_smiles": reaction_smiles,
        "num_steps": getattr(route, "length", 0),
        "leaf_count": len(leaf_smiles),
        "leaf_smiles": leaf_smiles,
        "has_convergent_reaction": bool(getattr(route, "has_convergent_reaction", False)),
        "content_hash": getattr(route, "content_hash", None),
        "signature": getattr(route, "signature", None),
        "retrocast_version": getattr(route, "retrocast_version", None),
        "canonical_route": route.model_dump(mode="json") if hasattr(route, "model_dump") else None,
    }


def adapt_aizynth_payload_with_retrocast(
    payload: dict[str, Any],
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Adapt AiZynthFinder route trees through RetroCast's canonical schema."""
    target_smiles = str(payload.get("smiles", "")).strip()
    if not target_smiles:
        return []

    from .aizynth_client import extract_route_trees

    route_trees = extract_route_trees(payload)
    if not route_trees:
        return []

    from retrocast import TargetInput, adapt_routes

    target = TargetInput(id=target_smiles, smiles=target_smiles)
    routes = adapt_routes(route_trees, target, "aizynth", max_routes=limit)

    summaries: list[dict[str, Any]] = []
    for idx, route in enumerate(routes):
        summary = _route_to_summary(route, route_index=idx)
        if summary is not None:
            summaries.append(summary)
    return summaries
