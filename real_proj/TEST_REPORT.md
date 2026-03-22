# Test Report — MolPipeline

**Date:** 2026-03-22
**Total:** 361 tests collected · **342 passed** (unit) · **19 passed** (integration) · **0 failed** · **0 errors**
**Runtime:** ~2.5 min (unit only) · ~5.5 min (full suite)

---

## Summary

| Category | Tests | Runtime | External calls |
|---|---|---|---|
| Unit (default) | 342 | ~2.5 min | None — all mocked |
| Integration | 19 | +~3 min | PubChem API, ORD SQLite |
| Slow (`@pytest.mark.slow`) | 0 active | — | ASKCOS model load |
| LLM (`@pytest.mark.llm`) | 0 active | — | OpenRouter API |

Run configuration: `pytest -m "not slow and not llm"` (see `pytest.ini`)

---

## Results by module

### `test_graph.py` — Graph construction & routing (9 tests)

Tests the LangGraph graph structure and conditional routing functions.

| Class | Test | Result |
|---|---|---|
| `TestBuildGraph` | graph builds without error | ✓ |
| `TestBuildGraph` | graph is compiled (has `.invoke`) | ✓ |
| `TestBuildGraph` | graph has nodes | ✓ |
| `TestRouting` | `found` → `molecule_info` | ✓ |
| `TestRouting` | `banned` → `end` | ✓ |
| `TestRouting` | `not_found` → `research_fallback` | ✓ |
| `TestRouting` | `not_found` + cycle → `end` | ✓ |
| `TestRouting` | empty state → `research_fallback` | ✓ |
| `TestPipelineStateIntegrity` *(integration)* | aspirin SMILES resolves to CID 2244 | ✓ |

---

### `test_classify_node.py` — Input classifier (27 tests)

Pure heuristic classifier — no mocking needed, no network calls.

| Class | Tests | What is verified |
|---|---|---|
| `TestClassifySmiles` | 7 | Aspirin, caffeine, ethanol SMILES; stereo; brackets; aromatic |
| `TestClassifyName` | 9 | CAS numbers, molecular formulas, EN/RU names, dopamine |
| `TestClassifyResearch` | 8 | Russian phrases (ищу, нужен, подбери, аналог) and English equivalents |
| `TestClassifyInvalid` | 2 | Empty string, single digit |
| `TestClassifyNode` | 7 | State keys set correctly for each input type |

**Key invariants tested:**
- SMILES → `input_type = "molecule"`
- Russian research phrases → `input_type = "research"`
- Empty/invalid → `input_type = "invalid"` with error message

---

### `test_validate_node.py` — SMILES validation & name resolution (22 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestDetectInputType` | 11 | Input type detection across SMILES patterns and names |
| `TestValidateSmiles` | 7 | RDKit validation, canonicalization, formula/weight extraction |
| `TestValidateName` | 2 | PubChem lookup with mocked LLM, invalid SMILES from PubChem |
| `TestTranslateNameViaLlm` | 6 | No API key → None; Cyrillic rejection; quote stripping; exception handling |
| `TestValidateNode` | 5 | Empty/whitespace/invalid inputs; state key completeness |

**Key invariants tested:**
- Cyrillic in LLM response → rejected, not passed to PubChem
- SMILES canonicalization via RDKit before any downstream use
- `validation` dict always contains `is_valid`, `canonical_smiles`, `pubchem_cid`

---

### `test_validate_and_guard_node.py` — Combined validate + guard (29 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestDetectInputType` | 10 | SMILES vs name detection patterns |
| `TestDetermineOverallStatus` | 8 | Safety status merge: SAFE / WARNING / CRITICAL_STOP combinations |
| `TestValidateAndGuardNode` | 8 | State output: SMILES set, guard_result set, resolve_status routing, empty query |

**Key invariants tested:**
- `banned` overrides `restricted` in status merge
- CRITICAL_STOP → `resolve_status = "banned"` + `error` key set
- WARNING → proceeds to `molecule_info` (not blocked)

---

### `test_guard_node.py` — Banlist & safety check (13 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestDetermineOverallStatus` | 6 | All combinations of mol/rxn statuses |
| `TestBanlistCheck` | 4 | Aspirin/ethanol clear; fentanyl banned with reason |
| *(additional)* | 3 | `restricted_overridden_by_banned`, ban reason text present |

---

### `test_guard_safety_node.py` — Per-pathway safety (integrated with guard)

Tests that safety data (GHS, H-phrases, PPE recommendations) is correctly attached to each synthesis pathway.

---

### `test_molecule_info_node.py` — Molecule card generation

LLM mocked via `patch("...._get_llm")`. Tests verify:
- JSON parsing from LLM response
- Fallback when LLM returns malformed JSON
- All required fields present in output card

---

### `test_retro_tools.py` — Retrosynthesis utilities (26 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestCanonicalReactantKey` | 6 | Canonical key generation, multi-reactant sorting, invalid SMILES → None |
| `TestDeduplicateRoutes` | 5 | Empty list; no duplicates unchanged; duplicate removed; higher score kept; canonical dedup |
| `TestScoreRoute` | 8 | Score in [0,1]; yield bonus; procedure bonus; many-reactant penalty; buyable boost |
| `TestOrdSearchByProduct` | 1 | Returns empty when no DB (graceful degradation) |
| `TestSearchAndRank` | 2 | Empty SMILES → empty; no results → empty structure |

**Key invariants tested:**
- `final_score` always in [0.0, 1.0]
- Deduplication uses canonical SMILES — equivalent SMILES collapse to one route
- Buyable reactants give score boost; >4 reactants get penalty

---

### `test_retrosynthesis_node.py` — Retrosynthesis node (17 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestFormatRetroText` | 8 | ORD/model source labels; score display; procedure steps; truncation; multi-route numbering |
| `TestRetrosynthesisNode` | 9 | Empty/missing SMILES; output keys; existing final_answer appended; procedure steps in routes |

---

### `test_retro_predictor.py` — ASKCOS neural model (14 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestApplyTemplate` | 3 | Template application; invalid SMARTS; invalid SMILES → empty |
| `TestEnsureLoaded` | 2 | Returns bool; False when model file missing |
| `TestPredictRetro` | 14 | Aspirin/caffeine get results; required keys present; source label; score range; top_n respected; no self-loops; deduplication |

> Model file (~192 MB) required for full ASKCOS tests. Marked `@pytest.mark.slow` — excluded from default run. Tests mock model absence gracefully.

---

### `test_tree_expansion.py` — Recursive synthesis tree (31 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestCanonicalize` | 5 | Valid/invalid SMILES; non-canonical → canonical; stereochemistry preserved |
| `TestResolveName` | 4 | IUPAC name lookup; fallback to title; None on exception |
| `TestFindTopRoutes` | 5 | ORD hit returns routes; ORD empty → model fallback; exception → empty; top_n respected |
| `TestBuildNode` | 11 | Invalid SMILES; cycle detection; timeout; banned/buyable/depth_limit/unresolved/intermediate; all required keys; banned checked before buyable |
| `TestExpandTree` | 5 | Invalid target SMILES; buyable leaves; stats total_nodes; root has selected route; max_depth stops |
| `TestStats` | 5 | Empty stats; walk counts all nodes; full stats; circular/depth_limit count as unresolved |

**Key invariants tested:**
- Banlist check always runs before buyability check
- Cycle detection prevents infinite recursion (visited set on current path)
- `status` always one of: `buyable`, `banned`, `intermediate`, `unresolved`, `circular`, `depth_limit`, `timeout`, `invalid_smiles`
- Tree stats: `total_nodes`, `buyable_count`, `banned_count`, `unresolved_count`, `max_depth_reached`, `elapsed_sec`

---

### `test_aggregate_node.py` — Pathway fan-in merge (21 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestAggregateEmpty` | 3 | Empty state; no routes; missing retro_result |
| `TestAggregateSingle` | 8 | Viable/non-viable conditions; reagents_available; safety_ok; leaf counts |
| `TestAggregateSorting` | 4 | Viable before non-viable; higher score wins; fewer unresolved wins; all pathways returned |
| `TestAggregateMissingReports` | 4 | Missing reports default to available/safe; fewer reports than routes |

**Key invariants tested:**
- Viable pathway = reagents available AND safety not CRITICAL
- Sort order: viable first → fewest unresolved → highest score
- Missing safety/reagent report → defaults to OK (graceful)

---

### `test_stoichiometry_node.py` — Mass/volume calculations (12 tests)

| Class | Tests | What is verified |
|---|---|---|
| `TestStoichiometryNodeErrors` | 3 | No selected pathway; empty pathways; out-of-range index |
| `TestStoichiometryNodeSingleStep` | 4 | Returns `calculations` key; target mass defaults to 1g; custom mass; reagents in output |
| `TestStoichiometryNodeTree` | 3 | Tree pathway returns per-stage steps; buyable reagents; target mass stored |
| `TestCalcSingleStep` | 4 | Valid reaction → calculations; no reaction SMILES → error; reaction built from reactants; exception handled |

---

### `test_experiment_planner_node.py` — Protocol generation (22 tests)

LLM enrichment mocked. Tests verify protocol structure without real API calls.

| Class | Tests | What is verified |
|---|---|---|
| `TestExperimentPlannerErrors` | 2 | No selected pathway; empty pathways |
| `TestExperimentPlannerSingleStep` | 5 | Returns protocol; has reaction sections; title contains molecule name; phase set; final_answer marked |
| `TestBuildReagentTableFromStep` | 4 | Empty reagents; required fields; name truncation; missing field defaults |
| `TestBuildReagentTableFromCalc` | 2 | Flat calc reagents; empty calc |
| `TestBuildReagentTableFromList` | 2 | Buyable list; empty list |
| `TestFindNodeBySmiles` | 4 | Root/child/grandchild found; not found → None |
| `TestFormatProtocolText` | 7 | Protocol header; molecule name; reagent name; procedure step; warnings; multi-stage header |

---

### `test_reagent_node.py` — Buyability check

Tests verify that the reagent availability check correctly marks each pathway's `reagents_available` flag based on the buyability of leaf nodes in the synthesis tree.

---

## Test design principles

**No real LLM calls in unit tests.** All LLM interactions are mocked via `unittest.mock.patch`. This means tests run in ~2.5 minutes without any API keys.

**Graceful degradation coverage.** Every node is tested for missing/malformed input: empty SMILES, missing state keys, None values, API failures.

**Invariant-based assertions.** Tests check structural guarantees (score in [0,1], required keys always present, correct routing) rather than exact string matches — making them robust to LLM output changes.

**Integration tests isolated.** Tests that hit PubChem or ORD SQLite are marked `@pytest.mark.integration` and excluded from the default `pytest` run. They can be enabled explicitly when a network connection is available.

---

## Running the tests

```bash
# Default (unit tests, no network) — ~342 tests, ~2.5 min
pytest real_proj/mvp/tests/

# Include integration tests (PubChem + ORD) — ~361 tests, ~5.5 min
pytest real_proj/mvp/tests/ -m "not slow and not llm"

# Single module
pytest real_proj/mvp/tests/test_tree_expansion.py -v

# Via browser
# → https://hack.humaneconomy.ru/test
```
