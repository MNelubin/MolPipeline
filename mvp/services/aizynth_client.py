"""AiZynthFinder service client and route normalization helpers."""

from __future__ import annotations

import logging
from typing import Any

import requests

from ..config import RETRO_ENABLE_RETROCAST

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 120


def _service_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


def get_aizynth_resources(base_url: str, *, timeout: int = _DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Fetch available models and stocks from the AiZynthFinder service."""
    resp = requests.get(
        _service_url(base_url, "/api/v1/resources"),
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def run_aizynth_retrosynthesis(
    base_url: str,
    smiles: str,
    *,
    max_transforms: int = 12,
    time_limit: int = 10,
    iterations: int = 100,
    expansion_model: str = "uspto",
    stock: str = "zinc",
    timeout: int = _DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Run multi-step retrosynthesis via the AiZynthFinder microservice."""
    payload = {
        "smiles": smiles,
        "max_transforms": max_transforms,
        "time_limit": time_limit,
        "iterations": iterations,
        "expansion_model": expansion_model,
        "stock": stock,
    }
    resp = requests.post(
        _service_url(base_url, "/api/v1/run"),
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("AiZynthFinder service returned a non-object payload")
    return data


def _iter_route_nodes(route_tree: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten all nodes of a retrosynthesis route tree."""
    nodes = [route_tree]
    for child in route_tree.get("children", []) or []:
        if isinstance(child, dict):
            nodes.extend(_iter_route_nodes(child))
    return nodes


def _count_route_steps(route_tree: dict[str, Any]) -> int:
    """Count route nodes containing mapped reaction data."""
    count = 0
    for node in _iter_route_nodes(route_tree):
        metadata = node.get("metadata") or {}
        if isinstance(metadata, dict) and metadata.get("mapped_reaction_smiles"):
            count += 1
    return count


def _extract_first_disconnection(route_tree: dict[str, Any]) -> tuple[str, str] | None:
    """Extract the first retrosynthetic step as (reactants, product)."""
    metadata = route_tree.get("metadata") or {}
    if isinstance(metadata, dict):
        mapped = metadata.get("mapped_reaction_smiles")
        if isinstance(mapped, str) and ">>" in mapped:
            product, reactants = mapped.split(">>", 1)
            product = product.strip()
            reactants = reactants.strip()
            if product and reactants:
                return reactants, product

    for child in route_tree.get("children", []) or []:
        if isinstance(child, dict):
            extracted = _extract_first_disconnection(child)
            if extracted is not None:
                return extracted
    return None


def extract_route_trees(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract raw route trees from the AiZynthFinder payload."""
    routes = payload.get("routes", [])
    if isinstance(routes, list):
        return [route for route in routes if isinstance(route, dict)]
    if isinstance(routes, dict):
        return [routes]
    return []


def normalize_aizynth_routes(
    payload: dict[str, Any],
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Convert AiZynthFinder tree results into the runtime route schema.

    The main pipeline expects one-step routes with a top-level disconnection.
    For each solved multi-step route tree, we extract the first retrosynthetic
    action as a candidate route and keep the full tree in provenance.
    """
    routes: list[dict[str, Any]] = []
    trees = extract_route_trees(payload)
    stats = payload.get("statistics") if isinstance(payload.get("statistics"), dict) else {}
    stock_info = payload.get("stock_info") if isinstance(payload.get("stock_info"), dict) else {}
    parameters = payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {}
    target_smiles = str(payload.get("smiles", "")).strip()
    retrocast_by_index: dict[int, dict[str, Any]] = {}

    if RETRO_ENABLE_RETROCAST:
        try:
            from .retrocast_bridge import adapt_aizynth_payload_with_retrocast

            retrocast_summaries = adapt_aizynth_payload_with_retrocast(payload, limit=limit)
            retrocast_by_index = {
                int(summary["route_index"]): summary
                for summary in retrocast_summaries
                if isinstance(summary, dict) and "route_index" in summary
            }
        except Exception as exc:
            logger.warning("[aizynth] RetroCast bridge failed: %s", exc)

    solved_bonus = 0.1 if stats.get("is_solved") else 0.0

    for idx, tree in enumerate(trees[:limit]):
        retrocast_summary = retrocast_by_index.get(idx, {})
        extracted = _extract_first_disconnection(tree)
        if extracted is None and retrocast_summary.get("reactants") and retrocast_summary.get("target_smiles"):
            reactants = str(retrocast_summary["reactants"]).strip()
            product = str(retrocast_summary["target_smiles"]).strip()
        elif extracted is None:
            continue
        else:
            reactants, product = extracted
        reaction_smiles = f"{reactants}>>{product}"
        route = {
            "reactants": reactants,
            "reaction_smiles": reaction_smiles,
            "source": "aizynthfinder",
            "score": min(0.65 + solved_bonus, 0.95),
            "plausibility": 0.75,
            "num_steps": retrocast_summary.get("num_steps") or _count_route_steps(tree),
            "target_smiles": target_smiles,
            "provenance": {
                "provider": "aizynthfinder",
                "retrieval_mode": "service_tree_search",
                "route_index": idx,
                "statistics": stats,
                "stock_info": stock_info,
                "parameters": parameters,
                "raw_tree": tree,
            },
        }
        if retrocast_summary:
            route["provenance"]["retrocast"] = retrocast_summary
        routes.append(route)

    logger.info(
        "[aizynth] normalized %d routes from %d raw trees for %s",
        len(routes),
        len(trees),
        target_smiles[:40],
    )
    return routes
