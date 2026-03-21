"""FastAPI gateway for the Chemist Agent pipeline.

POST /analyze      — run full pipeline, return complete state + formatted output
POST /ord/search   — search ORD by molecule name or SMILES, return ranked reactions
GET  /health       — liveness check

Run:
    uvicorn real_proj.mvp.api:app --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Config must be imported first (sets LangSmith env vars)
from . import config as _cfg  # noqa: F401
from .graph import build_graph
from .retro_tools import ord_search_by_product, score_route, _deduplicate_routes

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


class TreeExpandRequest(BaseModel):
    smiles: str = Field(..., description="Target molecule SMILES")
    reactants: str = Field(..., description="Dot-separated reactant SMILES from the selected route")
    max_depth: int = Field(default=20, ge=1, le=25, description="Maximum recursion depth")
    timeout_sec: float = Field(default=120.0, ge=5, le=600, description="Maximum elapsed time in seconds")


class TreeExpandResponse(BaseModel):
    tree: dict[str, Any]
    stats: dict[str, Any]


class OrdSearchRequest(BaseModel):
    query: str = Field(..., description="Molecule name (any language) or SMILES string")
    limit: int = Field(default=15, ge=1, le=100, description="Max reactions to fetch from ORD before dedup/ranking")
    top_n: int = Field(default=15, ge=1, le=100, description="Max results to return after ranking")
    scored: bool = Field(default=True, description="Apply multi-factor scoring and ranking")


class OrdReaction(BaseModel):
    reaction_id: str | None = None
    reaction_smiles: str | None = None
    reactants: str | None = None
    expected_yield: float | None = None
    temperature: str | None = None
    solvent: str | None = None
    catalyst: str | None = None
    procedure_details: str | None = None
    source: str = "ord"
    final_score: float | None = None
    scoring: dict[str, Any] | None = None


class OrdSearchResponse(BaseModel):
    query: str                          # original input
    smiles: str                         # resolved canonical SMILES
    resolution: str                     # "smiles_direct" | "pubchem_name" | "pubchem_llm"
    total_found: int                    # reactions in ORD before dedup
    returned: int                       # reactions in this response
    reactions: list[dict[str, Any]]     # full reaction dicts


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


def _resolve_to_smiles(query: str) -> tuple[str, str]:
    """Resolve a molecule name or SMILES to canonical SMILES.

    Returns (canonical_smiles, resolution_method).
    Raises ValueError if resolution fails.

    Resolution methods:
      "smiles_direct"  — input was already valid SMILES
      "pubchem_name"   — resolved via PubChem name lookup
      "pubchem_llm"    — Cyrillic name translated by LLM, then PubChem
    """
    from rdkit import Chem
    from .nodes.validate_node import _detect_input_type, _translate_name_via_llm
    from .tools import get_cid_by_name, get_smiles_by_cid
    import re

    _CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁ]")
    query = query.strip()

    input_type = _detect_input_type(query)

    if input_type == "smiles":
        mol = Chem.MolFromSmiles(query)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {query}")
        return Chem.MolToSmiles(mol, isomericSmiles=True), "smiles_direct"

    # Name → CID → SMILES
    cid = get_cid_by_name(query)
    resolution = "pubchem_name"

    if cid is None and _CYRILLIC_RE.search(query):
        en_name = _translate_name_via_llm(query)
        if en_name:
            cid = get_cid_by_name(en_name)
            resolution = "pubchem_llm"

    if cid is None:
        raise ValueError(f"Molecule '{query}' not found in PubChem")

    smiles = get_smiles_by_cid(cid)
    if not smiles:
        raise ValueError(f"PubChem CID {cid} has no SMILES")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"PubChem returned invalid SMILES for CID {cid}")

    return Chem.MolToSmiles(mol, isomericSmiles=True), resolution


def _run_ord_search(query: str, limit: int, top_n: int, scored: bool) -> dict[str, Any]:
    """Resolve query → SMILES, search ORD, optionally score. Called in thread pool."""
    smiles, resolution = _resolve_to_smiles(query)

    reactions = ord_search_by_product(smiles, limit=limit)
    total_found = len(reactions)

    if scored and reactions:
        for r in reactions:
            score_route(r)
        reactions = _deduplicate_routes(reactions)
        reactions.sort(key=lambda r: r.get("final_score", 0), reverse=True)

    reactions = reactions[:top_n]

    return {
        "smiles": smiles,
        "resolution": resolution,
        "total_found": total_found,
        "reactions": reactions,
    }


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


@app.post("/ord/search", response_model=OrdSearchResponse)
async def ord_search(req: OrdSearchRequest):
    """Search Open Reaction Database for synthesis routes to the target molecule.

    Accepts a molecule name (English or Russian) or SMILES string.
    Returns all matching reactions from the local ORD SQLite index (2.38M reactions),
    optionally scored and ranked by multi-factor scoring.
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be empty")

    logger.info("[ord/search] query=%r limit=%d top_n=%d scored=%s",
                query, req.limit, req.top_n, req.scored)

    import asyncio
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _executor,
            lambda: _run_ord_search(query, req.limit, req.top_n, req.scored),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("[ord/search] crashed for query %r", query)
        raise HTTPException(status_code=500, detail=str(exc))

    clean_reactions = _sanitize(result["reactions"])
    logger.info("[ord/search] %r → smiles=%s total=%d returning=%d",
                query, result["smiles"][:30], result["total_found"], len(clean_reactions))

    return OrdSearchResponse(
        query=query,
        smiles=result["smiles"],
        resolution=result["resolution"],
        total_found=result["total_found"],
        returned=len(clean_reactions),
        reactions=clean_reactions,
    )


@app.post("/tree/expand", response_model=TreeExpandResponse)
async def tree_expand(req: TreeExpandRequest):
    """Recursively expand a selected synthesis route into a full tree.

    Takes the target molecule SMILES and the reactants string from a selected
    route, then recursively decomposes non-buyable reactants until all leaves
    are buyable, banned, or unresolvable.
    """
    smiles = req.smiles.strip()
    reactants = req.reactants.strip()
    if not smiles or not reactants:
        raise HTTPException(status_code=422, detail="smiles and reactants must not be empty")

    logger.info("[tree/expand] smiles=%s reactants=%s max_depth=%d timeout=%.0fs",
                smiles[:30], reactants[:40], req.max_depth, req.timeout_sec)

    import asyncio
    from .tree_expansion import expand_tree

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _executor,
            lambda: expand_tree(smiles, reactants, req.max_depth, req.timeout_sec),
        )
    except Exception as exc:
        logger.exception("[tree/expand] crashed for smiles %r", smiles[:30])
        raise HTTPException(status_code=500, detail=str(exc))

    clean_result = _sanitize(result)
    stats = clean_result.get("stats", {})
    logger.info("[tree/expand] %s → %d nodes, %d buyable, %d banned, %.1fs",
                smiles[:30], stats.get("total_nodes", 0),
                stats.get("buyable_count", 0), stats.get("banned_count", 0),
                stats.get("elapsed_sec", 0))

    return TreeExpandResponse(tree=clean_result["tree"], stats=clean_result["stats"])
