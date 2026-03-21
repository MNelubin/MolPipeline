"""FastAPI gateway for the Chemist Agent pipeline.

POST /analyze  — run full pipeline, return complete state + formatted output
GET  /health   — liveness check

Run:
    uvicorn real_proj.mvp.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Config must be imported first (sets LangSmith env vars)
from . import config as _cfg  # noqa: F401
from .graph import build_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("mvp.api")

app = FastAPI(
    title="Chemist Agent API",
    description="Molecule analysis: validation → safety → info → retrosynthesis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build graph once at startup (loads model weights, ~192 MB)
_graph = None


@app.on_event("startup")
async def _startup():
    global _graph
    logger.info("Building graph (loading model weights)…")
    _graph = build_graph()
    logger.info("Graph ready.")


# Thread pool for synchronous graph.invoke()
_executor = ThreadPoolExecutor(max_workers=4)


# ── Request / Response models ─────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    query: str  # SMILES or molecule name (any language)


class AnalyzeResponse(BaseModel):
    # Pipeline status
    status: str                     # "ok" | "invalid" | "banned" | "error"
    query: str

    # Formatted text output (same as CLI prints)
    output: str

    # Full pipeline state (all nodes)
    state: dict[str, Any]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_output(state: dict[str, Any]) -> str:
    """Reproduce the text output from run.py for the given final state."""
    lines: list[str] = []

    if state.get("error"):
        lines.append("!" * 60)
        lines.append(f"  ОШИБКА: {state['error']}")
        lines.append("!" * 60)

        guard = state.get("guard_result", {})
        if guard:
            mol_check = guard.get("molecule_check", {})
            rxn_check = guard.get("reaction_check", {})
            if mol_check.get("status") in ("banned", "restricted"):
                lines.append(f"\n  Вещество:   {mol_check.get('name', 'Неизвестно')}")
                lines.append(f"  Статус:     {mol_check.get('status')}")
                lines.append(f"  Категория:  {mol_check.get('category')}")
                lines.append(f"  Причина:    {mol_check.get('reason')}")
            if rxn_check.get("status") in ("prohibited", "restricted"):
                lines.append(f"\n  Реакция:    {rxn_check.get('reason')}")

        validation = state.get("validation", {})
        if validation and not validation.get("is_valid"):
            lines.append(f"\n  Ошибка валидации: {validation.get('error')}")

    elif state.get("final_answer"):
        lines.append(state["final_answer"])
    else:
        lines.append("  Результат не получен.")

    return "\n".join(lines)


def _derive_status(state: dict[str, Any]) -> str:
    if state.get("error"):
        guard = state.get("guard_result", {})
        if guard.get("overall_status") == "CRITICAL_STOP":
            return "banned"
        validation = state.get("validation", {})
        if validation and not validation.get("is_valid"):
            return "invalid"
        return "error"
    if state.get("final_answer"):
        return "ok"
    return "error"


def _sanitize(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable values to strings."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _run_pipeline(query: str) -> dict[str, Any]:
    """Run graph synchronously (called in thread pool)."""
    return _graph.invoke({"query": query})


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "graph_ready": _graph is not None}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be empty")

    logger.info("Received query: %r", query)

    import asyncio
    loop = asyncio.get_event_loop()
    try:
        state = await loop.run_in_executor(_executor, _run_pipeline, query)
    except Exception as exc:
        logger.exception("Pipeline crashed for query %r", query)
        raise HTTPException(status_code=500, detail=str(exc))

    output = _make_output(state)
    status = _derive_status(state)
    clean_state = _sanitize(state)

    logger.info("Query %r → status=%s", query, status)

    return AnalyzeResponse(
        status=status,
        query=query,
        output=output,
        state=clean_state,
    )
