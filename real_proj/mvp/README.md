# MVP: Chemist Agent Pipeline

4-node LangGraph pipeline for chemical compound analysis, safety assessment, and retrosynthesis.

## Architecture

```
                 +-----------+
                 |   START   |
                 +-----+-----+
                       |
                       v
              +--------+--------+
              |   validate_node |  SMILES/name → canonical SMILES
              |                 |  (LLM translates Russian names)
              +--------+--------+
                       |
              is_valid? |
              +---------+---------+
              | Yes               | No → END (error)
              v                   |
        +-----+-------+          |
        |  guard_node  |         |
        |              |         |
        +-----+-------+         |
              |                  |
     status?  |                  |
     +--------+--------+        |
     | SAFE/WARNING     | CRITICAL_STOP → END (banned)
     v                  |
+----+---------------+  |
| molecule_info_node  |  |
| (LLM + PubChem)    |  |
+----+---------------+  |
     |                   |
     v                   |
+----+------------------+|
| retrosynthesis_node   ||
| (ORD + model fallback)||
+----+------------------+|
     |                   |
     v                   |
  +--+---+               |
  | END  | <-------------+
  +------+
```

## Nodes

### 1. validate_node
- **Input**: `query` (SMILES string or molecule name)
- **Output**: `smiles`, `pubchem_cid`, `validation`
- Auto-detects input type (SMILES vs name)
- Russian names automatically translated to English via LLM before PubChem lookup
- Validates SMILES via RDKit, resolves names via PubChem API

### 2. guard_node
- **Input**: `smiles`, `pubchem_cid`
- **Output**: `guard_result`, optionally `error`
- Checks against banlist of 150 controlled substances (CWC, DEA, EU)
- GHS classification, H/P-phrases from PubChem
- PPE recommendations based on hazard profile
- Three outcomes: `SAFE`, `WARNING`, `CRITICAL_STOP`

### 3. molecule_info_node
- **Input**: `smiles`, `pubchem_cid`, `query`
- **Output**: `molecule_info`, `final_answer`
- PubChem data: physical description, synonyms, images
- RDKit: TPSA, LogP, H-bond donors/acceptors, ring count
- Experimental properties: melting/boiling point, density, flash point
- LD50 toxicity data, CAS number, GHS pictograms
- LLM generates Russian description and name

### 4. retrosynthesis_node
- **Input**: `smiles`, `molecule_info`
- **Output**: `retro_result`, appends to `final_answer`
- **ORD priority**: if Open Reaction Database has results, model is NOT used
- **Model fallback**: standalone template-relevance model (163K templates, extracted from ASKCOS v2)
- Multi-factor scoring: model confidence (25%), plausibility (20%), buyability (20%), simplicity (15%), efficiency (10%), yield bonus, procedure bonus
- Canonical deduplication by reactant SMILES
- **Procedure translation**: ORD English procedures translated to Russian via LLM; model routes use rule-based inference

## Data Sources

| Source | Description | Size |
|--------|-------------|------|
| ORD SQLite | Open Reaction Database (local index) | 2.38M reactions |
| Retro model | Template-relevance NN (Morgan FP → 163K templates) | 192MB weights |
| PubChem API | Compound properties, safety, images | Online |
| Banlist | CWC/DEA/EU controlled substances | 150 entries |

## Scoring System

Each retrosynthesis route is scored on a 0–1 scale:

```
final_score = 0.25 * model_confidence
            + 0.20 * plausibility
            + 0.20 * buyability_ratio
            + 0.15 * simplicity
            + 0.10 * efficiency
            + yield_bonus (up to 0.10)
            + procedure_bonus (0.05)
```

- **Buyability**: fraction of reactants that are commercially available (cheap reagents list + RDKit heuristic)
- **Simplicity**: inversely proportional to heaviest reactant atom count and chiral centers
- **Efficiency**: penalty for many reactants

## Running

```bash
# With argument
python -m real_proj.mvp "aspirin"
python -m real_proj.mvp "CC(=O)Oc1ccccc1C(O)=O"
python -m real_proj.mvp "этанол"

# Interactive
python -m real_proj.mvp
```

## Configuration

Environment variables (`.env`):

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | API key for LLM (OpenRouter) |
| `LLM_MODEL` | Model name (default: `openai/gpt-4o`) |
| `LANGSMITH_API_KEY` | LangSmith tracing (optional) |

## File Structure

```
mvp/
├── config.py               # Environment config
├── state.py                # TypedDict state definitions
├── graph.py                # LangGraph 4-node pipeline
├── run.py                  # CLI entry point
├── tools.py                # PubChem, RDKit, safety utilities
├── retro_tools.py          # ORD search, scoring, dedup, pipeline
├── retro_predictor.py      # Standalone ASKCOS template-relevance model
├── procedure_inference.py  # LLM procedure translation + rule-based inference
├── nodes/
│   ├── validate_node.py    # Input validation + LLM Russian name resolver
│   ├── guard_node.py       # Safety guard (banlist, GHS, PPE)
│   ├── molecule_info_node.py  # Molecule card (PubChem + LLM)
│   └── retrosynthesis_node.py # Retrosynthesis search + formatting
└── data/
    ├── banned_chemicals.json  # 150 controlled substances
    └── banned_reactions.json  # Prohibited reaction patterns
```
