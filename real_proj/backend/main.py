"""FastAPI backend for the molecule pipeline web interface.

Endpoints:
  POST /api/run          — SSE stream of pipeline events
  GET  /api/molecule/2d  — RDKit SVG from SMILES
  GET  /api/molecule/3d  — PubChem / RDKit SDF for 3Dmol.js
  GET  /api/models       — Available OpenRouter models
  GET  /api/health       — Health check
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Add project root to path so we can import mvp ───────────────────────────
# Expected layout:
#   practice/
#     mvp/          ← your pipeline package
#     backend/      ← this file
#     frontend/
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("api")

app = FastAPI(title="Molecule Pipeline API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ────────────────────────────────────────────────

class RunRequest(BaseModel):
    query: str
    model: str = "openai/gpt-4o"


# ── Available models (OpenRouter) ────────────────────────────────────────────

AVAILABLE_MODELS = [
    {"id": "openai/gpt-4o",               "name": "GPT-4o",               "provider": "OpenAI"},
    {"id": "openai/gpt-4o-mini",          "name": "GPT-4o Mini",          "provider": "OpenAI"},
    {"id": "anthropic/claude-3.5-sonnet", "name": "Claude 3.5 Sonnet",    "provider": "Anthropic"},
    {"id": "anthropic/claude-3-haiku",    "name": "Claude 3 Haiku",       "provider": "Anthropic"},
    {"id": "google/gemini-2.0-flash-001", "name": "Gemini 2.0 Flash",     "provider": "Google"},
    {"id": "google/gemini-pro-1.5",       "name": "Gemini 1.5 Pro",       "provider": "Google"},
    {"id": "meta-llama/llama-3.1-70b-instruct", "name": "Llama 3.1 70B", "provider": "Meta"},
    {"id": "mistralai/mistral-large",     "name": "Mistral Large",        "provider": "Mistral"},
    {"id": "deepseek/deepseek-r1",        "name": "DeepSeek R1",          "provider": "DeepSeek"},
]


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ── Models ───────────────────────────────────────────────────────────────────

@app.get("/api/models")
async def get_models():
    return AVAILABLE_MODELS


# ── SSE Pipeline Stream ──────────────────────────────────────────────────────

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_pipeline(query: str, model: str) -> AsyncGenerator[str, None]:
    """Run the LangGraph pipeline and yield SSE events."""

    # Dynamically set model before building graph
    try:
        import mvp.config as cfg
        cfg.LLM_MODEL = model
    except ImportError:
        pass

    try:
        from mvp.graph import build_graph
    except ImportError as e:
        yield _sse("error", {"message": f"Import error: {e}"})
        return

    yield _sse("pipeline_start", {"query": query, "model": model})

    NODE_LABELS = {
        "validate":      "Валидация",
        "guard":         "Проверка безопасности",
        "molecule_info": "Сбор данных о молекуле",
    }

    try:
        graph = build_graph()

        # astream_events gives us fine-grained events including LLM token stream
        async for event in graph.astream_events({"query": query}, version="v2"):
            ev_type = event.get("event", "")
            ev_name = event.get("name", "")
            ev_data = event.get("data", {})
            tags    = event.get("tags", [])

            # ── Node lifecycle ────────────────────────────────────────────
            if ev_type == "on_chain_start" and ev_name in NODE_LABELS:
                yield _sse("node_start", {
                    "node":  ev_name,
                    "label": NODE_LABELS[ev_name],
                })

            elif ev_type == "on_chain_end" and ev_name in NODE_LABELS:
                output = ev_data.get("output", {})
                # Sanitise: remove large image payloads from stream (sent separately)
                safe_output = {k: v for k, v in output.items()
                               if k not in ("image_2d", "image_3d")}
                yield _sse("node_complete", {
                    "node":   ev_name,
                    "label":  NODE_LABELS[ev_name],
                    "output": safe_output,
                })

            # ── LLM token streaming (inside molecule_info node) ───────────
            elif ev_type == "on_chat_model_stream":
                chunk = ev_data.get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield _sse("token", {"text": chunk.content})

        yield _sse("pipeline_done", {})

    except Exception as exc:
        logger.exception("Pipeline error")
        yield _sse("error", {"message": str(exc)})


@app.post("/api/run")
async def run_pipeline(req: RunRequest):
    if not req.query.strip():
        raise HTTPException(400, "Query must not be empty")

    return StreamingResponse(
        _stream_pipeline(req.query.strip(), req.model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── 2D Molecule image (RDKit SVG) ────────────────────────────────────────────

@app.get("/api/molecule/2d")
async def molecule_2d(smiles: str = Query(..., description="SMILES string")):
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise HTTPException(400, "Invalid SMILES")

        drawer = rdMolDraw2D.MolDraw2DSVG(480, 320)
        drawer.drawOptions().addStereoAnnotation = True
        drawer.drawOptions().addAtomIndices = False

        # Dark theme colours
        drawer.drawOptions().backgroundColour = (0.05, 0.07, 0.10, 1.0)
        drawer.drawOptions().padding = 0.12

        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()

        return Response(content=svg, media_type="image/svg+xml")

    except ImportError:
        raise HTTPException(500, "RDKit not available")


# ── 3D Molecule SDF ──────────────────────────────────────────────────────────

@app.get("/api/molecule/3d")
async def molecule_3d(
    smiles: str | None = Query(None),
    cid: int | None = Query(None),
):
    # Try PubChem first (best quality 3D)
    if cid:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF?record_type=3d"
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                r = await client.get(url)
                if r.status_code == 200:
                    return Response(content=r.text, media_type="chemical/x-mdl-sdfile")
            except Exception:
                pass

    # Fallback: generate 3D conformer with RDKit
    if smiles:
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise HTTPException(400, "Invalid SMILES")

            mol = Chem.AddHs(mol)
            params = AllChem.EmbedParameters()
            params.randomSeed = 42
            AllChem.EmbedMolecule(mol, params)
            AllChem.MMFFOptimizeMolecule(mol)
            sdf = Chem.MolToMolBlock(mol)
            return Response(content=sdf, media_type="chemical/x-mdl-sdfile")

        except ImportError:
            raise HTTPException(500, "RDKit not available")

    raise HTTPException(400, "Provide smiles or cid")


# ── Calculator endpoint ───────────────────────────────────────────────────────

@app.post("/api/calculate")
async def calculate(request: dict):
    """Run stoichiometry_calc or equivalents_calc.

    Auto-detects mode from input fields:
      - reaction_smiles + target_mass_g  → stoichiometry_calc
      - reference_smiles + reagents      → equivalents_calc
    """
    import asyncio

    try:
        from mvp.calculator_combined import calculator_agent
    except ImportError as e:
        raise HTTPException(500, f"Calculator not available: {e}")

    try:
        # Run in thread pool (RDKit + PubChem calls are blocking)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, calculator_agent, dict(request))
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("Calculator error")
        raise HTTPException(500, str(e))


# ── Frontend SPA static files ───────────────────────────────────────────────

FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"

if FRONTEND_DIST.is_dir():
    from fastapi.responses import FileResponse

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve frontend SPA — try file first, fallback to index.html."""
        file_path = FRONTEND_DIST / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        index = FRONTEND_DIST / "index.html"
        if index.is_file():
            return FileResponse(index)
        raise HTTPException(404, "Not found")
