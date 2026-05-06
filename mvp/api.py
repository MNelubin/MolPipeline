"""FastAPI gateway for the Chemist Agent pipeline.

POST /analyze      — run pipeline (mode=auto end-to-end, or mode=interactive with pauses)
POST /resume       — resume an interactive session at the next interrupt point
POST /ord/search   — search ORD by molecule name or SMILES, return ranked reactions
POST /retro/search — run additive retrosynthesis search across enabled sources
POST /research/analyze — run standalone molecule/literature/patent research
POST /admet/analyze    — run descriptor-based ADMET screening
POST /tree/expand  — recursively expand a synthesis route into a full tree
GET  /health       — liveness check
GET  /retro/sources — inspect enabled retrosynthesis sources and AiZynth service reachability

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
from .admet import analyze_admet
from .availability import check_reagent_availability, summarize_availability
from .chem_chat import TOOL_REGISTRY, run_chem_chat
from .config import DATA_DIR
from .graph import build_graph
from .research_workspace import run_research_workspace
from .services.aizynth_client import get_aizynth_resources
from .services.retrocast_bridge import get_retrocast_runtime_info
from .tools.retro_tools import (
    _ord_search_via_api,
    _deduplicate_routes,
    score_route,
    search_and_rank,
)

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


class RetroSearchRequest(BaseModel):
    query: str = Field(..., description="Molecule name (any language) or SMILES string")
    top_n: int = Field(default=5, ge=1, le=25)
    source_mode: Literal["auto", "ord", "web", "retro_model", "aizynthfinder", "all"] = Field(
        default="auto",
        description="Retrosynthesis source selection mode for ranking routes.",
    )


class RetroSearchResponse(BaseModel):
    query: str
    smiles: str
    resolution: str
    source_mode: str
    total_found: int
    total_unique: int
    returned: int
    sources_used: list[str]
    source_counts: dict[str, int]
    source_counts_deduped: dict[str, int]
    source_errors: dict[str, str] = Field(default_factory=dict)
    routes: list[dict[str, Any]]


class RetroAnalyzeRequest(BaseModel):
    query: str = Field(..., description="Molecule name (any language) or SMILES string")
    top_n: int = Field(default=5, ge=1, le=25)
    source_mode: Literal["auto", "ord", "web", "retro_model", "aizynthfinder", "all"] = Field(
        default="auto",
        description="Retrosynthesis source selection mode for the dedicated UI tab.",
    )
    model: str | None = Field(
        default=None,
        description="Optional LLM model override for molecule card generation.",
    )


class RetroAnalyzeResponse(BaseModel):
    status: Literal["ok", "blocked"]
    query: str
    smiles: str
    resolution: str
    source_mode: str
    molecule_info: dict[str, Any]
    guard_result: dict[str, Any]
    retro_result: dict[str, Any]
    source_errors: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


class RetroSourcesResponse(BaseModel):
    ord_authoritative: bool
    tree_include_experimental: bool
    source_modes: list[dict[str, Any]]
    sources: dict[str, dict[str, Any]]


class ResearchAnalyzeRequest(BaseModel):
    query: str = Field(..., description="Research question, molecule class, literature topic, or patent topic")
    mode: Literal["molecule", "literature", "patent"] = Field(
        default="literature",
        description="Standalone research mode for the dedicated UI workspace.",
    )
    max_sources: int = Field(default=8, ge=1, le=15)


class ResearchAnalyzeResponse(BaseModel):
    status: Literal["ok", "empty"]
    query: str
    mode: str
    interpreted_intent: str
    search_queries: list[str]
    summary: str
    analysis: dict[str, Any] = Field(default_factory=dict)
    candidates: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    rag_results: list[dict[str, Any]]
    source_errors: dict[str, str] = Field(default_factory=dict)


class AdmetAnalyzeRequest(BaseModel):
    query: str = Field(..., description="Molecule name or SMILES")


class AdmetAnalyzeResponse(BaseModel):
    query: str
    smiles: str
    resolution: str
    admet: dict[str, Any]


class AvailabilityCheckRequest(BaseModel):
    query: str | None = Field(
        default=None,
        description="Single molecule, comma/newline-separated reagents, or dot-separated reactant SMILES",
    )
    items: list[str] = Field(
        default_factory=list,
        description="Explicit reagent list. Used together with query when provided.",
    )


class AvailabilityCheckResponse(BaseModel):
    query: str | None
    items: list[dict[str, Any]]
    summary: dict[str, Any]


class ChemChatRequest(BaseModel):
    message: str = Field(..., description="Free-form chemistry-specific user task.")
    source_mode: Literal["auto", "ord", "web", "retro_model", "aizynthfinder", "all"] = Field(
        default="auto",
        description="Retrosynthesis source mode when the chat decides to run retrosynthesis.",
    )
    top_n: int = Field(default=5, ge=1, le=10)
    research_mode: Literal["molecule", "literature", "patent"] = Field(default="literature")
    max_sources: int = Field(default=6, ge=1, le=10)


class ChemChatResponse(BaseModel):
    status: str
    intent: str
    answer: str
    tools_used: list[str]
    artifacts: dict[str, Any]
    suggested_next_actions: list[str] = Field(default_factory=list)


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


def _parse_availability_items(query: str | None, items: list[str]) -> list[str]:
    """Parse free-form reagent input into individual molecules."""
    import re

    parsed: list[str] = []
    raw_chunks: list[str] = []
    if query:
        raw_chunks.append(query)
    raw_chunks.extend(items or [])

    for chunk in raw_chunks:
        for token in re.split(r"[\n;,]+", chunk):
            token = token.strip()
            if not token:
                continue
            if "." in token and not any(ch.isspace() for ch in token):
                parsed.extend(part.strip() for part in token.split(".") if part.strip())
            else:
                parsed.append(token)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in parsed:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _unresolved_availability_item(item: str, error: Exception) -> dict[str, Any]:
    return {
        "input": item,
        "label": item,
        "smiles": None,
        "canonical_smiles": None,
        "resolution": "unresolved",
        "available": False,
        "availability_level": "invalid",
        "basis": "resolution_failed",
        "confidence": "high",
        "ppg": None,
        "source": None,
        "source_label": None,
        "estimated_pack_prices": [],
        "supplier_search_links": [],
        "descriptors": {},
        "warnings": [str(error)],
    }


def _run_availability_check(query: str | None, items: list[str]) -> dict[str, Any]:
    parsed_items = _parse_availability_items(query, items)
    if not parsed_items:
        raise ValueError("query or items must contain at least one molecule")

    results: list[dict[str, Any]] = []
    for item in parsed_items:
        try:
            smiles, resolution = _resolve_to_smiles(item)
            results.append(
                check_reagent_availability(
                    smiles,
                    label=item,
                    input_value=item,
                    resolution=resolution,
                )
            )
        except Exception as exc:
            results.append(_unresolved_availability_item(item, exc))

    return {
        "query": query,
        "items": results,
        "summary": summarize_availability(results),
    }


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


def _attach_procedure_steps(routes: list[dict[str, Any]]) -> None:
    """Decorate retrosynthesis routes with structured Russian procedure steps."""
    from .procedure_inference import format_procedure_russian

    for route in routes:
        route["procedure_steps_ru"] = format_procedure_russian(route)


def _run_retro_search(query: str, top_n: int, source_mode: str = "auto") -> dict[str, Any]:
    """Resolve query -> SMILES, then run additive retrosynthesis search."""
    smiles, resolution = _resolve_to_smiles(query)
    result = search_and_rank(smiles, top_n=top_n, source_mode=source_mode)
    _attach_procedure_steps(result.get("routes", []))
    return {
        "smiles": smiles,
        "resolution": resolution,
        "source_mode": result.get("source_mode", source_mode),
        "total_found": result.get("total_found", 0),
        "total_unique": result.get("total_unique", 0),
        "sources_used": result.get("sources_used", []),
        "source_counts": result.get("source_counts", {}),
        "source_counts_deduped": result.get("source_counts_deduped", {}),
        "source_errors": result.get("source_errors", {}),
        "routes": result.get("routes", []),
    }


def _run_retro_analyze(
    query: str,
    top_n: int,
    source_mode: str = "auto",
    model: str | None = None,
) -> dict[str, Any]:
    """Resolve molecule, build card context, then run source-aware retrosynthesis."""
    from .nodes.molecule_info_node import molecule_info_node
    from .nodes.validate_and_guard_node import _resolve_molecule, _run_safety_checks

    resolved = _resolve_molecule(query)
    validation = resolved.get("validation", {})
    if not validation.get("is_valid"):
        raise ValueError(validation.get("error") or f"Could not resolve molecule: {query}")

    smiles = resolved.get("smiles", "")
    cid = resolved.get("pubchem_cid")
    guard_result = _run_safety_checks(smiles=smiles, cid=cid, reaction_description="")
    molecule_state = molecule_info_node({
        "query": query,
        "smiles": smiles,
        "pubchem_cid": cid,
        "guard_result": guard_result,
        "llm_model": model,
    })
    molecule_info = molecule_state.get("molecule_info", {})

    error = None
    if guard_result.get("overall_status") == "CRITICAL_STOP":
        error = (
            guard_result.get("molecule_check", {}).get("reason")
            or guard_result.get("reaction_check", {}).get("reason")
            or "Retrosynthesis blocked by safety policy."
        )
        retro_result = {
            "routes": [],
            "best_route": None,
            "sources_used": [],
            "total_found": 0,
            "total_unique": 0,
            "source_counts": {},
            "source_counts_deduped": {},
            "source_mode": source_mode,
            "source_errors": {},
            "error": error,
        }
        status = "blocked"
    else:
        retro_result = search_and_rank(smiles, top_n=top_n, source_mode=source_mode)
        _attach_procedure_steps(retro_result.get("routes", []))
        source_errors = retro_result.get("source_errors", {})
        if not retro_result.get("routes") and source_mode != "auto" and source_errors.get(source_mode):
            error = f"{source_mode} failed: {source_errors[source_mode]}"
        status = "ok"

    return {
        "status": status,
        "query": query,
        "smiles": smiles,
        "resolution": validation.get("input_type") or "resolved",
        "source_mode": retro_result.get("source_mode", source_mode),
        "molecule_info": molecule_info,
        "guard_result": guard_result,
        "retro_result": retro_result,
        "source_errors": retro_result.get("source_errors", {}),
        "error": error,
    }


def _build_source_modes(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Build frontend-facing source mode metadata from the runtime snapshot."""
    sources = snapshot["sources"]
    source_modes: list[dict[str, Any]] = [
        {
            "id": "auto",
            "label": "Авто",
            "description": "Стандартный режим как в основном UI.",
            "enabled": True,
        },
        {
            "id": "ord",
            "label": "ORD",
            "description": "Только Open Reaction Database.",
            "enabled": bool(sources["ord"]["enabled"]),
        },
        {
            "id": "retro_model",
            "label": "ASKCOS-derived model",
            "description": "Только локальная template-relevance модель.",
            "enabled": bool(sources["retro_model"]["enabled"]),
        },
        {
            "id": "web",
            "label": "Web Search",
            "description": "Только web-поиск синтетических маршрутов.",
            "enabled": bool(sources["web"]["enabled"]),
        },
        {
            "id": "aizynthfinder",
            "label": "AiZynthFinder",
            "description": "Только внешний multi-step planner AiZynthFinder.",
            "enabled": bool(
                sources["aizynthfinder"]["enabled"]
                and sources["aizynthfinder"]["configured"]
                and sources["aizynthfinder"].get("reachable") is not False
            ),
        },
    ]
    source_modes.append({
        "id": "all",
        "label": "All Sources",
        "description": "Явно собрать additive-пул из всех включённых источников.",
        "enabled": any(mode["enabled"] for mode in source_modes[1:]),
    })
    return source_modes


def _retro_sources_snapshot() -> dict[str, Any]:
    """Build a diagnostic snapshot of enabled retrosynthesis sources."""
    retrocast_info = get_retrocast_runtime_info()
    sources: dict[str, dict[str, Any]] = {
        "ord": {
            "enabled": _cfg.RETRO_ENABLE_ORD,
            "configured": True,
            "mode": "sqlite",
        },
        "web": {
            "enabled": _cfg.RETRO_ENABLE_WEB,
            "configured": True,
            "mode": "web_search",
        },
        "retro_model": {
            "enabled": _cfg.RETRO_ENABLE_RETRO_MODEL,
            "configured": True,
            "mode": "local_model",
        },
        "aizynthfinder": {
            "enabled": _cfg.RETRO_ENABLE_AIZYNTH,
            "configured": bool(_cfg.AIZYNTH_BASE_URL),
            "base_url": _cfg.AIZYNTH_BASE_URL or None,
            "reachable": None,
            "mode": "service_tree_search",
        },
        "retrocast": {
            "enabled": _cfg.RETRO_ENABLE_RETROCAST,
            "configured": retrocast_info["available"],
            "reachable": retrocast_info["available"],
            "mode": "canonicalization_bridge",
            "standalone_source": False,
            "version": retrocast_info["version"],
            "adapters": retrocast_info["adapters"],
        },
    }

    aizynth = sources["aizynthfinder"]
    if aizynth["enabled"] and aizynth["configured"]:
        try:
            resources = get_aizynth_resources(
                _cfg.AIZYNTH_BASE_URL,
                timeout=max(5, int(_cfg.AIZYNTH_TIMEOUT_SEC)),
            )
            aizynth["reachable"] = True
            aizynth["details"] = {
                "stocks": resources.get("stocks", []),
                "expansion_models": resources.get("expansion_models", []),
                "filter_models": resources.get("filter_models", []),
            }
        except Exception as exc:
            aizynth["reachable"] = False
            aizynth["error"] = str(exc)

    if retrocast_info["error"]:
        sources["retrocast"]["error"] = retrocast_info["error"]

    snapshot = {
        "ord_authoritative": _cfg.RETRO_ORD_AUTHORITATIVE,
        "tree_include_experimental": _cfg.RETRO_TREE_INCLUDE_EXPERIMENTAL,
        "source_modes": [],
        "sources": sources,
    }
    snapshot["source_modes"] = _build_source_modes(snapshot)
    return snapshot


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "graph_ready": _graph is not None}


@app.get("/chat/tools")
async def chat_tools():
    """Expose chemistry tools available to the general chat orchestrator."""
    return {
        "tools": [
            {
                "name": spec.name,
                "description": spec.description,
                "requires_safety_gate": spec.requires_safety_gate,
            }
            for spec in TOOL_REGISTRY.values()
        ]
    }


async def _run_chem_chat_request(req: ChemChatRequest) -> ChemChatResponse:
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="message must not be empty")

    logger.info("[chat/message] message=%r source_mode=%s", message[:120], req.source_mode)

    import asyncio
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _executor,
            lambda: run_chem_chat(
                message,
                source_mode=req.source_mode,
                top_n=req.top_n,
                research_mode=req.research_mode,
                max_sources=req.max_sources,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("[chat/message] crashed for message %r", message[:120])
        raise HTTPException(status_code=500, detail=str(exc))

    return ChemChatResponse(**_sanitize(result))


@app.post("/chat/message", response_model=ChemChatResponse)
async def chem_chat_message(req: ChemChatRequest):
    """Run the general chemistry chat orchestrator over MolPipeline tools."""
    return await _run_chem_chat_request(req)


@app.post("/research/chat", response_model=ChemChatResponse)
async def research_chat_alias(req: ChemChatRequest):
    """Deploy-compatible alias for ChemChat under the already proxied research prefix."""
    return await _run_chem_chat_request(req)


@app.get("/retro/sources", response_model=RetroSourcesResponse)
async def retro_sources():
    """Inspect currently enabled retrosynthesis sources and service availability."""
    return RetroSourcesResponse(**_sanitize(_retro_sources_snapshot()))


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


@app.post("/retro/search", response_model=RetroSearchResponse)
async def retro_search(req: RetroSearchRequest):
    """Run source-aware retrosynthesis search across enabled sources."""
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be empty")

    logger.info("[retro/search] query=%r top_n=%d source_mode=%s", query, req.top_n, req.source_mode)

    import asyncio
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _executor,
            lambda: _run_retro_search(query, req.top_n, req.source_mode),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("[retro/search] crashed for query %r", query)
        raise HTTPException(status_code=500, detail=str(exc))

    clean_routes = _sanitize(result["routes"])
    logger.info(
        "[retro/search] %r -> smiles=%s total=%d unique=%d returning=%d from %s",
        query,
        result["smiles"][:30],
        result["total_found"],
        result["total_unique"],
        len(clean_routes),
        ", ".join(result["sources_used"]) or "none",
    )

    return RetroSearchResponse(
        query=query,
        smiles=result["smiles"],
        resolution=result["resolution"],
        source_mode=result["source_mode"],
        total_found=result["total_found"],
        total_unique=result["total_unique"],
        returned=len(clean_routes),
        sources_used=result["sources_used"],
        source_counts=result["source_counts"],
        source_counts_deduped=result["source_counts_deduped"],
        routes=clean_routes,
    )


@app.post("/retro/analyze", response_model=RetroAnalyzeResponse)
async def retro_analyze(req: RetroAnalyzeRequest):
    """Build molecule card context and run source-aware retrosynthesis for the dedicated UI tab."""
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be empty")

    logger.info(
        "[retro/analyze] query=%r top_n=%d source_mode=%s model=%s",
        query,
        req.top_n,
        req.source_mode,
        req.model or "default",
    )

    import asyncio
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _executor,
            lambda: _run_retro_analyze(query, req.top_n, req.source_mode, req.model),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("[retro/analyze] crashed for query %r", query)
        raise HTTPException(status_code=500, detail=str(exc))

    return RetroAnalyzeResponse(**_sanitize(result))


@app.post("/research/analyze", response_model=ResearchAnalyzeResponse)
async def research_analyze(req: ResearchAnalyzeRequest):
    """Run standalone molecule/literature/patent research without mutating the graph."""
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be empty")

    logger.info(
        "[research/analyze] query=%r mode=%s max_sources=%d",
        query,
        req.mode,
        req.max_sources,
    )

    import asyncio
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _executor,
            lambda: run_research_workspace(query, mode=req.mode, max_sources=req.max_sources),
        )
    except Exception as exc:
        logger.exception("[research/analyze] crashed for query %r", query)
        raise HTTPException(status_code=500, detail=str(exc))

    return ResearchAnalyzeResponse(**_sanitize(result))


@app.post("/admet/analyze", response_model=AdmetAnalyzeResponse)
async def admet_analyze(req: AdmetAnalyzeRequest):
    """Run descriptor-based ADMET screening for a molecule."""
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be empty")

    logger.info("[admet/analyze] query=%r", query)

    import asyncio
    loop = asyncio.get_event_loop()
    try:
        def _run_admet_with_safety() -> tuple[str, str, dict[str, Any]]:
            from .nodes.validate_and_guard_node import _resolve_molecule, _run_safety_checks

            resolved = _resolve_molecule(query)
            validation = resolved.get("validation", {})
            if not validation.get("is_valid"):
                raise ValueError(validation.get("error") or f"Could not resolve molecule: {query}")

            smiles = resolved.get("smiles", "")
            cid = resolved.get("pubchem_cid")
            safety_guard = _run_safety_checks(smiles=smiles, cid=cid, reaction_description="")
            resolution = validation.get("input_type") or "resolved"
            return smiles, resolution, analyze_admet(smiles, safety_guard=safety_guard)

        smiles, resolution, admet = await loop.run_in_executor(_executor, _run_admet_with_safety)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("[admet/analyze] crashed for query %r", query)
        raise HTTPException(status_code=500, detail=str(exc))

    return AdmetAnalyzeResponse(
        query=query,
        smiles=smiles,
        resolution=resolution,
        admet=_sanitize(admet),
    )


@app.post("/availability/check", response_model=AvailabilityCheckResponse)
async def availability_check(req: AvailabilityCheckRequest):
    """Check local catalog availability and supplier search hints for reagents."""
    logger.info(
        "[availability/check] query=%r items=%d",
        (req.query or "")[:80],
        len(req.items),
    )

    import asyncio
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _executor,
            lambda: _run_availability_check(req.query, req.items),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("[availability/check] crashed for query %r", req.query)
        raise HTTPException(status_code=500, detail=str(exc))

    return AvailabilityCheckResponse(**_sanitize(result))


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


# ── Test runner ───────────────────────────────────────────────────────────────

@app.post("/tests/run")
async def run_tests():
    """Run the full pytest suite and return structured results."""
    import re
    import subprocess
    import time
    import asyncio

    loop = asyncio.get_event_loop()

    def _run():
        start = time.time()
        import sys
        python = sys.executable  # use same interpreter that runs the server
        result = subprocess.run(
            [python, "-m", "pytest", "mvp/tests/", "-v", "--tb=short", "--no-header", "-q"],
            capture_output=True,
            text=True,
            cwd="/opt/projects/chemist-agent",
        )
        elapsed = round(time.time() - start, 2)
        stdout = result.stdout + result.stderr

        # Parse individual test lines: "tests/foo.py::TestClass::test_name PASSED"
        tests = []
        for line in stdout.splitlines():
            m = re.match(r"^(.*::\S+)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)", line)
            if m:
                raw_name = m.group(1).strip()
                status = m.group(2)
                # Shorten path: keep only after last "tests/"
                short = re.sub(r"^.*tests/", "", raw_name)
                tests.append({"name": short, "status": status})

        # Summary line: "5 failed, 184 passed in 5.07s"
        counts = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
        m_sum = re.search(r"(\d+) passed", stdout)
        if m_sum:
            counts["passed"] = int(m_sum.group(1))
        m_sum = re.search(r"(\d+) failed", stdout)
        if m_sum:
            counts["failed"] = int(m_sum.group(1))
        m_sum = re.search(r"(\d+) error", stdout)
        if m_sum:
            counts["error"] = int(m_sum.group(1))
        m_sum = re.search(r"(\d+) skipped", stdout)
        if m_sum:
            counts["skipped"] = int(m_sum.group(1))

        # If parsing found no tests, derive from counts
        if not tests and counts["passed"] + counts["failed"] > 0:
            counts_total = counts["passed"] + counts["failed"] + counts["error"] + counts["skipped"]
        else:
            counts_total = len(tests)

        return {
            "passed": counts["passed"],
            "failed": counts["failed"],
            "error": counts["error"],
            "skipped": counts["skipped"],
            "total": counts_total or len(tests),
            "duration_sec": elapsed,
            "returncode": result.returncode,
            "tests": tests,
            "output": stdout[-8000:],  # last 8k chars
        }

    result = await loop.run_in_executor(_executor, _run)
    return result
