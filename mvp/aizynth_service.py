"""Local AiZynthFinder microservice for MolPipeline.

This service exposes the minimal API expected by ``mvp.services.aizynth_client``:

* ``GET /api/v1/resources`` — available stocks and policy names from config
* ``POST /api/v1/run`` — run tree-search and return route trees + statistics

The heavy ``aizynthfinder`` dependency is imported lazily so the main project can
still run without the optional planner service installed.
"""

from __future__ import annotations

import copy
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(os.getenv("AIZYNTH_CONFIG_PATH", "data/aizynth/config.yml"))
DEFAULT_FILTER_MODEL = os.getenv("AIZYNTH_DEFAULT_FILTER_MODEL", "")

app = FastAPI(title="MolPipeline AiZynthFinder Service")


class AiZynthRunRequest(BaseModel):
    smiles: str = Field(..., min_length=1)
    max_transforms: int = Field(12, ge=1, le=50)
    time_limit: int = Field(10, ge=1, le=3600)
    iterations: int = Field(100, ge=1, le=20000)
    expansion_model: str = Field("uspto", min_length=1)
    stock: str = Field("zinc", min_length=1)
    filter_model: str | None = Field(None)
    max_routes: int = Field(10, ge=1, le=100)


def _ensure_runtime_available() -> None:
    try:
        import aizynthfinder  # noqa: F401
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="AiZynthFinder runtime is not installed in this service environment",
        ) from exc


@lru_cache(maxsize=1)
def _load_base_config() -> dict[str, Any]:
    if not DEFAULT_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"AiZynthFinder config not found: {DEFAULT_CONFIG_PATH}. "
            "Run scripts/install_aizynth_service.sh first."
        )
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid AiZynthFinder config at {DEFAULT_CONFIG_PATH}")
    return loaded


def _named_section(config: dict[str, Any], name: str) -> list[str]:
    section = config.get(name) or {}
    if not isinstance(section, dict):
        return []
    return sorted(str(key) for key in section.keys())


def _resources_payload(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_path": str(DEFAULT_CONFIG_PATH),
        "stocks": _named_section(config, "stock"),
        "expansion_models": _named_section(config, "expansion"),
        "filter_models": _named_section(config, "filter"),
    }


def _runtime_config(base_config: dict[str, Any], request: AiZynthRunRequest) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    search = config.setdefault("search", {})
    if not isinstance(search, dict):
        raise ValueError("AiZynthFinder config.search must be a mapping")
    search["max_transforms"] = int(request.max_transforms)
    search["iteration_limit"] = int(request.iterations)
    search["time_limit"] = int(request.time_limit)
    return config


def _select_optional_policy(policy_collection: Any, name: str | None) -> None:
    if not name or policy_collection is None:
        return
    available = getattr(policy_collection, "items", {})
    if isinstance(available, dict) and name not in available:
        raise HTTPException(status_code=400, detail=f"Unknown AiZynthFinder policy: {name}")
    policy_collection.select(name)


def _extract_route_dicts(stats: dict[str, Any], finder: Any, limit: int) -> list[dict[str, Any]]:
    trees = stats.get("trees")
    if isinstance(trees, list):
        return [tree for tree in trees[:limit] if isinstance(tree, dict)]

    routes = getattr(finder, "routes", None)
    dicts = getattr(routes, "dicts", None)
    if isinstance(dicts, list):
        return [tree for tree in dicts[:limit] if isinstance(tree, dict)]

    reaction_trees = getattr(routes, "reaction_trees", None)
    if isinstance(reaction_trees, list):
        extracted: list[dict[str, Any]] = []
        for route_tree in reaction_trees[:limit]:
            if hasattr(route_tree, "to_dict"):
                extracted.append(route_tree.to_dict(include_metadata=True))
        return extracted

    return []


def _run_search(request: AiZynthRunRequest) -> dict[str, Any]:
    _ensure_runtime_available()

    from aizynthfinder.aizynthfinder import AiZynthFinder

    base_config = _load_base_config()
    config = _runtime_config(base_config, request)
    finder = AiZynthFinder(configdict=config)

    try:
        finder.stock.select(request.stock)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unknown AiZynthFinder stock: {request.stock}") from exc

    try:
        finder.expansion_policy.select(request.expansion_model)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown AiZynthFinder expansion model: {request.expansion_model}",
        ) from exc

    filter_model = request.filter_model or DEFAULT_FILTER_MODEL or request.expansion_model
    if getattr(finder, "filter_policy", None) is not None:
        try:
            _select_optional_policy(finder.filter_policy, filter_model)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown AiZynthFinder filter model: {filter_model}",
            ) from exc

    finder.target_smiles = request.smiles.strip()
    finder.tree_search()
    finder.build_routes()
    stats = finder.extract_statistics()
    if not isinstance(stats, dict):
        stats = {}

    routes = _extract_route_dicts(stats, finder, request.max_routes)
    payload = {
        "smiles": request.smiles.strip(),
        "parameters": {
            "stock": request.stock,
            "expansion_model": request.expansion_model,
            "filter_model": filter_model,
            "max_transforms": request.max_transforms,
            "iterations": request.iterations,
            "time_limit": request.time_limit,
            "max_routes": request.max_routes,
        },
        "statistics": stats,
        "stock_info": stats.get("stock_info", {}) if isinstance(stats.get("stock_info"), dict) else {},
        "routes": routes,
    }
    return payload


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        config = _load_base_config()
        resources = _resources_payload(config)
        _ensure_runtime_available()
        return {"status": "ok", "resources": resources}
    except Exception as exc:
        logger.warning("[aizynth_service] health degraded: %s", exc)
        return {"status": "degraded", "error": str(exc), "config_path": str(DEFAULT_CONFIG_PATH)}


@app.get("/api/v1/resources")
def resources() -> dict[str, Any]:
    try:
        return _resources_payload(_load_base_config())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/run")
def run(request: AiZynthRunRequest) -> dict[str, Any]:
    try:
        return _run_search(request)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[aizynth_service] search failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
