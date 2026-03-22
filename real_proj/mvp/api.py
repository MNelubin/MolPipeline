"""FastAPI gateway for the Chemist Agent pipeline.

POST /analyze      — run pipeline (mode=auto end-to-end, or mode=interactive with pauses)
POST /resume       — resume an interactive session at the next interrupt point
POST /ord/search   — search ORD by molecule name or SMILES, return ranked reactions
POST /tree/expand  — recursively expand a synthesis route into a full tree
GET  /health       — liveness check

Run:
    uvicorn mvp.api:app --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from . import config as _cfg  # noqa: F401
from .config import DATA_DIR
from .graph import build_graph
from .tools.retro_tools import _ord_search_via_api, score_route, _deduplicate_routes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("mvp.api")

app = FastAPI(
    title="Chemist Agent API",
    description=(
        "Multi-phase chemist agent with human-in-the-loop:\n"
        "  Phase 1: classify -> validate -> molecule_info -> INTERRUPT\n"
        "  Phase 2: retrosynthesis -> safety+reagent -> INTERRUPT\n"
        "  Phase 3: stoichiometry -> experiment_planner -> END\n\n"
        "Supports mode='auto' (end-to-end) and mode='interactive' (pause at interrupts)."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_graph = None
_executor = ThreadPoolExecutor(max_workers=4)


@app.on_event("startup")
async def _startup():
    global _graph
    logger.info("Building graph (loading model weights)...")
    try:
        import sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver
        db_path = DATA_DIR / "checkpoints.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(db_path), check_same_thread=False)
        checkpointer = SqliteSaver(_conn)
        logger.info("Using SqliteSaver checkpointer at %s", db_path)
    except Exception as e:
        logger.warning("SqliteSaver unavailable (%s), falling back to MemorySaver", e)
        checkpointer = None
    _graph = build_graph(checkpointer=checkpointer)
    logger.info("Graph ready.")


# ── Request / Response models ─────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    query: str = Field(..., description="SMILES or molecule name (any language)")
    mode: Literal["auto", "interactive"] = Field(
        default="auto",
        description="'auto' runs end-to-end with defaults; 'interactive' pauses at interrupts",
    )
    model: str | None = Field(
        default=None,
        description="LLM model override (e.g. 'openai/gpt-4o'). If omitted, uses server default.",
    )


class AnalyzeResponse(BaseModel):
    status: str
    query: str
    output: str
    state: dict[str, Any]
    thread_id: str | None = Field(
        default=None,
        description="Session ID for resuming interactive sessions (only in interactive mode)",
    )
    phase: str | None = Field(
        default=None,
        description="Current interrupt phase: 'card_ready', 'select_pathway', or 'completed'",
    )


class ResumeRequest(BaseModel):
    thread_id: str = Field(..., description="Session ID from a previous /analyze or /resume call")
    resume_data: Any = Field(
        default=True,
        description=(
            "Data to resume with. For card_ready: True (or any truthy value). "
            "For select_pathway: {selected_pathway: int, target_amount: {value: float, unit: str}}"
        ),
    )


class ResumeResponse(BaseModel):
    status: str
    output: str
    state: dict[str, Any]
    thread_id: str
    phase: str | None = None


class TreeExpandRequest(BaseModel):
    smiles: str = Field(..., description="Target molecule SMILES")
    reactants: str = Field(..., description="Dot-separated reactant SMILES from the selected route")
    max_depth: int = Field(default=6, ge=1, le=12, description="Maximum recursion depth")
    timeout_sec: float = Field(default=120.0, ge=5, le=600, description="Maximum elapsed time in seconds")


class TreeExpandResponse(BaseModel):
    tree: dict[str, Any]
    stats: dict[str, Any]


class OrdSearchRequest(BaseModel):
    query: str = Field(..., description="Molecule name (any language) or SMILES string")
    limit: int = Field(default=15, ge=1, le=100)
    top_n: int = Field(default=15, ge=1, le=100)
    scored: bool = Field(default=True)


class OrdSearchResponse(BaseModel):
    query: str
    smiles: str
    resolution: str
    total_found: int
    returned: int
    reactions: list[dict[str, Any]]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_state(config: dict) -> dict[str, Any]:
    """Get the current graph state for a given thread."""
    snapshot = _graph.get_state(config)
    return dict(snapshot.values) if snapshot and snapshot.values else {}


def _detect_phase(state: dict[str, Any]) -> str:
    """Detect which interrupt phase the graph is paused at, or 'completed'."""
    phase = state.get("current_phase", "")
    if state.get("experiment_protocol"):
        return "completed"
    if state.get("synthesis_pathways") and not state.get("selected_pathway") and state.get("selected_pathway") != 0:
        return "select_pathway"
    if state.get("molecule_info") and not state.get("retro_result"):
        return "card_ready"
    if phase == "experiment":
        return "completed"
    return phase or "unknown"


def _make_output(state: dict[str, Any]) -> str:
    """Build text output from state."""
    if state.get("error"):
        lines = ["!" * 60, f"  ОШИБКА: {state['error']}", "!" * 60]
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
        return "\n".join(lines)

    return state.get("final_answer", "  Результат не получен.")


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
    return "pending"


def _sanitize(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable values to strings."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _run_auto(query: str, model: str | None = None) -> dict[str, Any]:
    """Run graph end-to-end in auto mode: auto-resume all interrupts."""
    thread_id = f"auto-{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: dict[str, Any] = {"query": query}
    if model:
        initial_state["llm_model"] = model
    _graph.invoke(initial_state, config=config)
    state = _get_state(config)

    if state.get("error") or not state.get("molecule_info"):
        return state

    # Resume past card interrupt
    _graph.invoke(Command(resume=True), config=config)
    state = _get_state(config)

    pathways = state.get("synthesis_pathways", [])
    if not pathways:
        return state

    # Auto-pick best viable path: viable first, fewest unresolved, highest score
    best_idx = 0
    if len(pathways) > 1:
        ranked = sorted(
            range(len(pathways)),
            key=lambda i: (
                not pathways[i].get("viable", False),
                pathways[i].get("unresolved_leaves", 999),
                -pathways[i].get("final_score", 0),
            ),
        )
        best_idx = ranked[0]

    _graph.invoke(
        Command(resume={
            "selected_pathway": best_idx,
            "target_amount": {"value": 1.0, "unit": "g", "amount_type": "product_mass"},
        }),
        config=config,
    )
    return _get_state(config)


def _run_interactive_start(query: str, thread_id: str, model: str | None = None) -> dict[str, Any]:
    """Start an interactive session — runs until first interrupt."""
    from .journal import AgentJournal
    # Wipe any old journal for this thread so it starts fresh
    j = AgentJournal.for_session(thread_id)
    if j.path.exists():
        j.path.unlink()
    AgentJournal.close_session(thread_id)

    config = {"configurable": {"thread_id": thread_id}}
    initial_state: dict[str, Any] = {"query": query, "session_id": thread_id}
    if model:
        initial_state["llm_model"] = model
    _graph.invoke(initial_state, config=config)
    return _get_state(config)


def _run_resume(thread_id: str, resume_data: Any) -> dict[str, Any]:
    """Resume an interactive session with user-provided data."""
    config = {"configurable": {"thread_id": thread_id}}
    # Check if thread exists before resuming
    snapshot = _graph.get_state(config)
    if not snapshot or not snapshot.values:
        return {"error": "Сессия не найдена или истекла. Начните новый запрос."}
    _graph.invoke(Command(resume=resume_data), config=config)
    return _get_state(config)


def _resolve_to_smiles(query: str) -> tuple[str, str]:
    """Resolve a molecule name or SMILES to canonical SMILES."""
    from rdkit import Chem
    from .nodes.validate_and_guard_node import _detect_input_type, _translate_name_via_llm
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
    """Resolve query -> SMILES, search ORD, optionally score."""
    smiles, resolution = _resolve_to_smiles(query)

    reactions = _ord_search_via_api(smiles, limit=limit)
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
    """Start pipeline analysis.

    mode='auto': runs end-to-end, auto-selects best pathway, returns full protocol.
    mode='interactive': runs Phase 1 only, pauses at first interrupt, returns thread_id.
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be empty")

    logger.info("[analyze] query=%r mode=%s", query, req.mode)

    import asyncio
    loop = asyncio.get_event_loop()

    try:
        if req.mode == "auto":
            state = await loop.run_in_executor(
                _executor, _run_auto, query, req.model,
            )
            output = _make_output(state)
            status = _derive_status(state)
            return AnalyzeResponse(
                status=status,
                query=query,
                output=output,
                state=_sanitize(state),
                thread_id=None,
                phase="completed" if status == "ok" else None,
            )
        else:
            thread_id = f"session-{uuid.uuid4().hex[:12]}"
            state = await loop.run_in_executor(
                _executor, _run_interactive_start, query, thread_id, req.model,
            )
            output = _make_output(state)
            status = _derive_status(state)
            phase = _detect_phase(state)

            return AnalyzeResponse(
                status=status,
                query=query,
                output=output,
                state=_sanitize(state),
                thread_id=thread_id,
                phase=phase,
            )
    except Exception as exc:
        logger.exception("[analyze] crashed for query %r", query)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/resume", response_model=ResumeResponse)
async def resume(req: ResumeRequest):
    """Resume an interactive session at the next interrupt point.

    For 'card_ready' interrupt: send resume_data=true to continue to retrosynthesis.
    For 'select_pathway' interrupt: send resume_data={selected_pathway: 0, target_amount: {value: 1.0, unit: "g"}}.
    """
    logger.info("[resume] thread_id=%s resume_data=%s", req.thread_id, req.resume_data)

    import asyncio
    loop = asyncio.get_event_loop()

    try:
        state = await loop.run_in_executor(
            _executor, _run_resume, req.thread_id, req.resume_data,
        )
    except Exception as exc:
        logger.exception("[resume] crashed for thread_id=%s", req.thread_id)
        raise HTTPException(status_code=500, detail=str(exc))

    output = _make_output(state)
    status = _derive_status(state)
    phase = _detect_phase(state)

    return ResumeResponse(
        status=status,
        output=output,
        state=_sanitize(state),
        thread_id=req.thread_id,
        phase=phase,
    )


@app.post("/ord/search", response_model=OrdSearchResponse)
async def ord_search(req: OrdSearchRequest):
    """Search Open Reaction Database for synthesis routes."""
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
    logger.info("[ord/search] %r -> smiles=%s total=%d returning=%d",
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
    """Recursively expand a selected synthesis route into a full tree."""
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
    logger.info("[tree/expand] %s -> %d nodes, %d buyable, %d banned, %.1fs",
                smiles[:30], stats.get("total_nodes", 0),
                stats.get("buyable_count", 0), stats.get("banned_count", 0),
                stats.get("elapsed_sec", 0))

    return TreeExpandResponse(tree=clean_result["tree"], stats=clean_result["stats"])


# ── Journal endpoints ─────────────────────────────────────────────────────────

@app.get("/journal/{session_id}/md")
async def journal_markdown(session_id: str):
    """Return the agent journal for a session as a Markdown file download."""
    from .journal import AgentJournal, LOGS_DIR
    safe_id = session_id.replace("..", "").replace("/", "").replace("\\", "")
    j = AgentJournal.for_session(safe_id)
    if not j.path.exists():
        raise HTTPException(status_code=404, detail=f"No journal for session '{safe_id}'")
    try:
        md_path = j.export_markdown()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return FileResponse(
        path=str(md_path),
        media_type="text/markdown",
        filename=f"journal_{safe_id}.md",
    )


@app.get("/journal/{session_id}/jsonl")
async def journal_jsonl(session_id: str):
    """Return the raw JSONL journal file."""
    from .journal import AgentJournal
    safe_id = session_id.replace("..", "").replace("/", "").replace("\\", "")
    j = AgentJournal.for_session(safe_id)
    if not j.path.exists():
        raise HTTPException(status_code=404, detail=f"No journal for session '{safe_id}'")
    return FileResponse(
        path=str(j.path),
        media_type="application/x-ndjson",
        filename=f"journal_{safe_id}.jsonl",
    )
