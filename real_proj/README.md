# MolPipeline — AI-Powered Retrosynthesis & Experiment Planning

> **Hackathon project** — Multi-agent LangGraph pipeline that takes a target molecule and produces a complete, step-by-step synthesis protocol backed by real chemical databases.

**Live demo:** `https://hack.humaneconomy.ru`
**Test runner:** `https://hack.humaneconomy.ru/test`

---

## What it does

A chemist types a molecule (name, SMILES, CAS, or Russian text). The system:

1. **Identifies** the molecule via PubChem — canonical SMILES, formula, properties, safety
2. **Checks** it against a controlled-substance banlist — stops immediately if banned
3. **Runs retrosynthesis** — searches Open Reaction Database (ORD) first, then ASKCOS neural model as fallback
4. **Expands a synthesis tree** recursively until every leaf is either commercially buyable or blocked
5. **Scores and ranks** up to 5 pathways by feasibility, yield, and buyability
6. **Calculates stoichiometry** — exact masses, volumes, equivalents for user-specified target mass
7. **Generates a full experimental protocol** — per-stage procedure steps enriched by LLM, in Russian

---

## Why it beats a plain LLM

| Criterion | Baseline LLM prompt | MolPipeline |
|---|---|---|
| Reaction source | Hallucinated / generic | ORD (real reactions) + ASKCOS model |
| Reagent quantities | Absent or approximate | Exact g, mL, mol, equiv for target mass |
| Procedure depth | 2–3 vague sentences | 8 detailed steps per stage, with rationale |
| Synthesis tree | Single step | Recursive tree to buyable leaves |
| Safety check | None | Banlist + GHS lookup on every node |
| Pathway ranking | None | Score = yield × buyability × procedure quality |
| Multi-stage | Not connected | Stages linked — product of stage N is reagent of stage N+1 |

---

## Architecture

```
User input
    │
    ▼
classify_node          ← heuristic: SMILES / name / research query / invalid
    │
    ▼
validate_and_guard_node   ← PubChem resolve + banlist check
    │ (not_found)
    ▼
research_node          ← web search + LLM synthesis of molecule identity
    │ (found)
    ▼
molecule_info_node     ← LLM enrichment of PubChem data → molecule card
    │
  [INTERRUPT #1 — user confirms to proceed]
    │
    ▼
retrosynthesis_node    ← ORD search → ASKCOS → score_route → tree expansion
    │
  ┌─┴─────────────────┐
  ▼                   ▼
guard_safety_node   reagent_node    ← parallel fan-out
  └─────────┬─────────┘
            ▼
      aggregate_node  ← merge, rank, select best pathway
            │
  [INTERRUPT #2 — user selects pathway + target mass]
            │
            ▼
    stoichiometry_node   ← exact mass/volume calculations
            │
            ▼
  experiment_planner_node  ← per-stage LLM procedure generation
            │
           END → PDF-ready protocol
```

### Routing logic

| Condition | Next node |
|---|---|
| SMILES/name resolves in PubChem | `molecule_info` |
| Not found, no prior research | `research_node` (web + LLM) |
| Not found after research | `END` with error |
| Banlist hit | `END` with warning |
| ORD has reaction for product | Use ORD route |
| ORD empty | ASKCOS neural retro model |
| Leaf is buyable | Mark `status=buyable`, stop branch |
| Depth ≥ 6 or timeout | Mark `status=depth_limit/timeout` |

---

## Tech stack

| Layer | Technology |
|---|---|
| Agent graph | LangGraph `StateGraph` with `MemorySaver` checkpointer |
| LLM | OpenRouter API (GPT-4o, GPT-4.5-nano, Claude, Gemini…) |
| Retrosynthesis (DB) | Open Reaction Database (ORD) — local SQLite, 1M+ reactions |
| Retrosynthesis (ML) | ASKCOS Molecular Transformer retro model |
| Molecule validation | RDKit + PubChem REST API |
| Safety | Custom banlist (FSKN, CWC, dual-use) + PubChem GHS |
| Backend API | FastAPI + SSE streaming, port 8765 |
| Frontend | React 18 + Vite, ReactFlow graph visualization |
| Proxy | SOCKS5 via httpx for geo-blocked OpenAI routes |
| Deployment | Systemd + Caddy reverse proxy |

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/analyze` | Start pipeline (SSE stream) |
| `GET` | `/stream/{thread_id}` | Resume interactive session |
| `POST` | `/resume/{thread_id}` | Resume after interrupt (pathway selection) |
| `POST` | `/tree/expand` | Recursive synthesis tree for a single route |
| `POST` | `/tests/run` | Run full pytest suite, returns JSON results |
| `GET` | `/health` | Health check |

### SSE events

```
pipeline_start  →  { query, model }
node_start      →  { node, label }
node_complete   →  { node, label, output }
interrupt       →  { phase: "card_ready" | "select_pathway", payload }
pipeline_done   →  {}
error           →  { message }
```

---

## Project structure

```
real_proj/
├── mvp/
│   ├── api.py                    ← FastAPI app (port 8765)
│   ├── graph.py                  ← LangGraph StateGraph definition
│   ├── state.py                  ← MVPState TypedDict
│   ├── config.py                 ← LLM factory, proxy, env vars
│   ├── nodes/
│   │   ├── classify_node.py      ← heuristic input classifier
│   │   ├── validate_and_guard_node.py  ← PubChem + banlist
│   │   ├── research_node.py      ← web search fallback
│   │   ├── molecule_info_node.py ← LLM molecule card enrichment
│   │   ├── retrosynthesis_node.py ← ORD + ASKCOS + tree expansion
│   │   ├── guard_safety_node.py  ← GHS safety per pathway
│   │   ├── reagent_node.py       ← buyability check
│   │   ├── aggregate_node.py     ← fan-in, rank pathways
│   │   ├── stoichiometry_node.py ← mass/volume calculations
│   │   └── experiment_planner_node.py  ← protocol generation
│   ├── tools/
│   │   ├── retro_tools.py        ← ORD search, scoring, buyability
│   │   ├── tree_expansion.py     ← recursive synthesis tree
│   │   └── askcos_api.py         ← ASKCOS model client
│   ├── services/
│   │   └── research_llm.py       ← web + LLM research service
│   └── tests/                    ← 284 unit tests (see TEST_REPORT.md)
├── backend/
│   └── main.py                   ← molecule viewer API (port 8002)
└── frontend/
    └── src/
        ├── App.jsx
        ├── components/
        │   ├── MoleculeCard.jsx   ← molecule info tabs
        │   ├── RetroCard.jsx      ← synthesis pathways
        │   ├── SynthesisGraph.jsx ← ReactFlow interactive tree
        │   ├── SynthesisTree.jsx  ← collapsible tree view
        │   ├── ProtocolGraph.jsx  ← experiment protocol
        │   └── ModelSelector.jsx  ← LLM model picker
        └── hooks/
            ├── useSSEPipeline.js
            └── useInteractivePipeline.js
```

---

## Environment variables

```env
# Required
OPENROUTER_API_KEY=sk-or-...
LLM_MODEL=openai/gpt-4.5-nano

# Optional
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LLM_TEMPERATURE=0.1
SOCKS_PROXY=socks5://user:pass@host:port   # for geo-blocked regions
LANGSMITH_API_KEY=ls__...                  # LangSmith tracing
```

---

## Running locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env && nano .env

# 3. Start the agent API
uvicorn real_proj.mvp.api:app --port 8765 --reload

# 4. Start the molecule viewer API
uvicorn real_proj.backend.main:app --port 8002 --reload

# 5. Start the frontend
cd real_proj/frontend
npm install && npm run dev
# → http://localhost:5173
```

---

## Running tests

```bash
# Fast unit tests only (no network, no LLM) — ~2.3 min, 284 tests
pytest real_proj/mvp/tests/ -m "not integration and not slow and not llm"

# All tests including integration (PubChem + ORD) — ~5 min, 312 tests
pytest real_proj/mvp/tests/

# Via browser (live server)
# → https://hack.humaneconomy.ru/test
```

See [TEST_REPORT.md](TEST_REPORT.md) for full breakdown.
