"""Retrosynthesis tools: multi-source route collection, scoring and ranking.

This module collects synthesis routes from all enabled sources, normalizes them
into a common internal schema, deduplicates by reactant set, scores, and ranks
the final route candidates consumed by the main pipeline.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from ..config import (
    AIZYNTH_BASE_URL,
    AIZYNTH_EXPANSION_MODEL,
    AIZYNTH_ITERATIONS,
    AIZYNTH_MAX_TRANSFORMS,
    AIZYNTH_STOCK,
    AIZYNTH_TIME_LIMIT,
    AIZYNTH_TIMEOUT_SEC,
    RETROCAST_BASE_URL,
    RETRO_ENABLE_AIZYNTH,
    RETRO_ENABLE_ORD,
    RETRO_ENABLE_RETROCAST,
    RETRO_ENABLE_RETRO_MODEL,
    RETRO_ENABLE_WEB,
    RETRO_ORD_AUTHORITATIVE,
)

logger = logging.getLogger(__name__)

RETRO_SOURCE_MODES = {
    "auto": None,
    "ord": {"ord"},
    "web": {"web"},
    "retro_model": {"retro_model"},
    "aizynthfinder": {"aizynthfinder"},
    "all": {"ord", "web", "retro_model", "aizynthfinder"},
}

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# ═════════════════════════════════════════════════════════════════════════════
# Canonical deduplication
# ═════════════════════════════════════════════════════════════════════════════


def _canonical_reactant_key(reactants_str: str) -> str | None:
    """Create a canonical dedup key from dot-separated reactant SMILES."""
    if not HAS_RDKIT or not reactants_str:
        return reactants_str
    parts = []
    for smi in reactants_str.split("."):
        smi = smi.strip()
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        parts.append(Chem.MolToSmiles(mol, isomericSmiles=True))
    parts.sort()
    return ".".join(parts)


def _deduplicate_routes(routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate routes by canonical reactant set."""
    seen: dict[str, dict[str, Any]] = {}
    for route in routes:
        key = _canonical_reactant_key(route.get("reactants", ""))
        if key is None:
            continue
        existing = seen.get(key)
        if existing is None or route.get("final_score", 0) > existing.get("final_score", 0):
            seen[key] = route
    return list(seen.values())


def _count_routes_by_source(routes: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize how many route candidates each source contributed."""
    counts: dict[str, int] = {}
    for route in routes:
        source = str(route.get("source") or "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


def get_enabled_sources_for_mode(source_mode: str | None) -> set[str] | None:
    """Return the source set implied by a requested UI/source mode."""
    if source_mode is None:
        return None
    normalized = source_mode.strip().lower()
    if normalized not in RETRO_SOURCE_MODES:
        raise ValueError(f"Unknown retrosynthesis source mode: {source_mode}")
    enabled = RETRO_SOURCE_MODES[normalized]
    return None if enabled is None else set(enabled)


def _merge_provenance(route: dict[str, Any], source: str, retrieval_mode: str) -> dict[str, Any]:
    """Attach a stable source/provenance block to a route dict."""
    result = dict(route)
    result["source"] = source
    provenance = dict(result.get("provenance") or {})
    provenance.setdefault("provider", source)
    provenance.setdefault("retrieval_mode", retrieval_mode)
    if "raw_score" not in provenance and result.get("score") is not None:
        provenance["raw_score"] = result.get("score")
    result["provenance"] = provenance
    return result


def _normalize_source_routes(
    routes: list[dict[str, Any]],
    source: str,
    retrieval_mode: str,
) -> list[dict[str, Any]]:
    """Normalize routes from a single source into a common internal schema."""
    normalized: list[dict[str, Any]] = []
    for route in routes:
        if not isinstance(route, dict):
            continue
        normalized.append(_merge_provenance(route, source, retrieval_mode))
    return normalized


# ═════════════════════════════════════════════════════════════════════════════
# ORD search via local SQLite database
# ═════════════════════════════════════════════════════════════════════════════

import sqlite3
from pathlib import Path

# DBs are at <project_root>/data/
# ord_reactions.db  — ORD reactions index
# buyables.db       — vendor catalogs (eMolecules, Mcule, ChemBridge, ChemSpace, SA)
_ORD_DB_PATH = Path(__file__).parent.parent.parent / "data" / "ord_reactions.db"


def _ord_search_via_api(smiles: str, limit: int = 15) -> list[dict]:
    """Search ORD local SQLite for reactions producing the target molecule."""
    if not _ORD_DB_PATH.exists():
        logger.warning("[ORD] SQLite DB not found at %s", _ORD_DB_PATH)
        return []

    try:
        conn = sqlite3.connect(str(_ORD_DB_PATH))
    except Exception as e:
        logger.warning("[ORD] Cannot open SQLite DB: %s", e)
        return []

    results: list[dict] = []
    try:
        canonical = smiles
        if HAS_RDKIT:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                canonical = Chem.MolToSmiles(mol, isomericSmiles=True)

        # 1. Exact canonical match via product_index
        cursor = conn.execute(
            """
            SELECT r.id, r.reaction_smiles, r.yield_pct,
                   r.temperature, r.solvent, r.catalyst
            FROM product_index pi
            JOIN reactions r ON r.id = pi.reaction_id
            WHERE pi.canonical_smiles = ?
            LIMIT ?
            """,
            (canonical, limit),
        )
        results = _rows_to_retro_dicts(cursor)

        # 2. Fallback: component role='product'
        if not results:
            cursor = conn.execute(
                """
                SELECT r.id, r.reaction_smiles, r.yield_pct,
                       r.temperature, r.solvent, r.catalyst
                FROM components c
                JOIN reactions r ON r.id = c.reaction_id
                WHERE c.role = 'product' AND c.smiles = ?
                LIMIT ?
                """,
                (canonical, limit),
            )
            results = _rows_to_retro_dicts(cursor)

    except Exception as e:
        logger.warning("[ORD] Query error: %s", e)
    finally:
        conn.close()

    logger.info("[ORD] SQLite: %d results for %s", len(results), smiles[:30])
    return results[:limit]


def _rows_to_retro_dicts(cursor) -> list[dict]:
    """Convert SQLite rows to retro_tools-format route dicts."""
    results = []
    for row in cursor:
        rxn_id, rxn_smi, yield_pct, temp, solvent, catalyst = row
        if not rxn_smi or (">" not in rxn_smi):
            continue
        # Handle both 'A.B>>C' and 'A.B>agent>C' formats
        if ">>" in rxn_smi:
            reactant_str = rxn_smi.split(">>")[0]
            agents_str = ""
        else:
            parts = rxn_smi.split(">")
            reactant_str = parts[0] if len(parts) >= 1 else ""
            agents_str = parts[1] if len(parts) >= 3 else ""
        # Join reactants + agents as dot-separated string
        all_parts = [s.strip() for s in reactant_str.split(".") if s.strip()]
        if agents_str:
            all_parts.extend(s.strip() for s in agents_str.split(".") if s.strip())
        reactants_clean = ".".join(all_parts)
        route: dict[str, Any] = {
            "reaction_id": rxn_id,
            "reaction_smiles": rxn_smi,
            "reactants": reactants_clean,
            "source": "ord",
            "num_examples": 1,
            "score": 0.95,
            "plausibility": 0.95,
        }
        if yield_pct is not None:
            route["expected_yield"] = float(yield_pct) / 100.0
        if temp:
            route["temperature"] = temp
        if solvent:
            route["solvent"] = solvent
        if catalyst:
            route["catalyst"] = catalyst
        results.append(_merge_provenance(route, "ord", "sqlite"))
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Scoring
# ═════════════════════════════════════════════════════════════════════════════

_CHEAP_REAGENTS = {
    "O", "CO", "CCO", "CC(C)O", "CC(C)=O", "CC=O", "CC(O)=O",
    "CC(=O)OC(C)=O", "ClCCl", "ClC(Cl)Cl", "C(Cl)(Cl)(Cl)Cl",
    "C1CCOC1", "CCOCC", "CS(C)=O", "CN(C)C=O",
    "O=CO", "Cl", "O=C=O", "N", "[NH4+]", "O=S(=O)(O)O",
    "O=[N+]([O-])O", "[Na+].[OH-]", "[K+].[OH-]",
    "[Na+].[Cl-]", "c1ccccc1", "Cc1ccccc1", "CCCCCC", "C1CCCCC1",
    "CC(=O)OCC", "COC(C)=O", "COC(=O)OC",
    "O=P(O)(O)O",
}


@lru_cache(maxsize=1)
def _get_cheap_canonical() -> set[str]:
    if not HAS_RDKIT:
        return _CHEAP_REAGENTS
    canonical = set()
    for smi in _CHEAP_REAGENTS:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            canonical.add(Chem.MolToSmiles(mol, isomericSmiles=True))
    return canonical


_BUYABLES_DB_PATH = Path(__file__).parent.parent.parent / "data" / "buyables.db"


def _buyables_lookup(smiles: str) -> dict | None:
    """Lookup SMILES in local buyables SQLite.

    DB built from 4 vendor catalogs (eMolecules, Mcule, ChemBridge, ChemSpace, SA)
    totaling ~390K commercially available molecules with price (ppg = $/g).
    Located at data/buyables.db alongside data/ord_reactions.db.

    Returns {"ppg": float, "source": str} or None if not in any catalog.
    """
    if not _BUYABLES_DB_PATH.exists():
        return None
    try:
        # canonicalize before lookup — DB stores RDKit canonical SMILES
        if HAS_RDKIT:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        conn = sqlite3.connect(str(_BUYABLES_DB_PATH), check_same_thread=False)
        row = conn.execute(
            "SELECT ppg, source FROM buyables WHERE smiles = ?", (smiles,)
        ).fetchone()
        conn.close()
        return {"ppg": row[0], "source": row[1]} if row else None
    except Exception:
        return None


def _is_buyable(smiles: str) -> bool:
    """Check commercial availability.

    Priority:
      1. Hardcoded common solvents/reagents (instant, no I/O)
      2. Local buyables SQLite — ~690K molecules from real vendor catalogs:
         eMolecules (EM), Mcule (MC), ChemBridge (CB), ChemSpace (CS),
         Sigma-Aldrich (SA). Same data/ directory as ORD index.
      3. Structural heuristic fallback (small/simple = likely available)
    """
    cheap = _get_cheap_canonical()
    if smiles in cheap:
        return True
    if _buyables_lookup(smiles) is not None:
        return True
    if not HAS_RDKIT:
        return len(smiles) < 15
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    heavy = mol.GetNumHeavyAtoms()
    rings = Descriptors.RingCount(mol)
    chiral = len(Chem.FindMolChiralCenters(mol))
    return (heavy <= 10 and rings <= 1 and chiral == 0) or heavy <= 6


def score_route(route: dict[str, Any]) -> dict[str, Any]:
    """Score a single retrosynthesis route."""
    reactants_str = route.get("reactants", "")
    model_score = route.get("score", 0.5)
    plausibility = route.get("plausibility", 0.8)

    reactants = [s.strip() for s in reactants_str.split(".") if s.strip()]
    n_reactants = len(reactants) if reactants else 1

    buyable_count = 0
    total_atoms = 0
    max_atoms = 0
    total_chiral = 0

    for smi in reactants:
        if not HAS_RDKIT:
            total_atoms += len(smi)
            max_atoms = max(max_atoms, len(smi))
            if len(smi) < 15:
                buyable_count += 1
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        heavy = mol.GetNumHeavyAtoms()
        total_atoms += heavy
        max_atoms = max(max_atoms, heavy)
        total_chiral += len(Chem.FindMolChiralCenters(mol))
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        if _is_buyable(canonical):
            buyable_count += 1

    buyability_ratio = buyable_count / max(n_reactants, 1)
    simplicity = 1.0 / (1.0 + 0.08 * max_atoms)
    simplicity *= 1.0 / (1.0 + 0.3 * total_chiral)
    efficiency = 1.0 / (1.0 + 0.25 * (n_reactants - 1))

    yield_bonus = 0.0
    if route.get("expected_yield") is not None:
        yield_bonus = min(route["expected_yield"], 1.0) * 0.1

    procedure_bonus = 0.05 if route.get("procedure_details") else 0.0

    composite = (
        0.25 * min(model_score, 1.0)
        + 0.20 * min(plausibility, 1.0)
        + 0.20 * buyability_ratio
        + 0.15 * simplicity
        + 0.10 * efficiency
        + yield_bonus
        + procedure_bonus
    )

    route["final_score"] = round(composite, 4)
    route["scoring"] = {
        "model_score": round(min(model_score, 1.0), 4),
        "plausibility": round(min(plausibility, 1.0), 4),
        "buyability": round(buyability_ratio, 4),
        "simplicity": round(simplicity, 4),
        "efficiency": round(efficiency, 4),
        "yield_bonus": round(yield_bonus, 4),
        "procedure_bonus": round(procedure_bonus, 4),
        "num_reactants": n_reactants,
        "total_atoms": total_atoms,
        "buyable_count": buyable_count,
    }
    return route


# ═════════════════════════════════════════════════════════════════════════════
# Web search for retrosynthesis
# ═════════════════════════════════════════════════════════════════════════════

_WEB_SEARCH_TIMEOUT = 15  # seconds for entire web search + LLM extraction


def _web_search_retro(smiles: str, target_name: str | None = None) -> list[dict[str, Any]]:
    """Search web for synthesis routes and extract structured reaction data.

    Returns routes in the same format as ORD/model results.
    All SMILES are validated via RDKit — invalid routes are discarded.
    """
    import time as _time

    t0 = _time.monotonic()

    # Resolve name for better search queries
    common_name = None
    iupac_name = None
    if not target_name:
        try:
            from ..tools import get_compound_properties, get_cid_by_smiles
            props = get_compound_properties(smiles)
            iupac_name = props.get("IUPACName")
            # Get common name via CID synonyms
            cid = get_cid_by_smiles(smiles)
            if cid:
                import requests
                try:
                    r = requests.get(
                        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON",
                        timeout=5,
                    )
                    if r.status_code == 200:
                        syns = r.json().get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])
                        if syns:
                            common_name = syns[0]
                except Exception:
                    pass
            target_name = common_name or iupac_name or smiles[:40]
        except Exception:
            target_name = smiles[:40]

    # Search — use common name (more search hits) + IUPAC as fallback
    try:
        from ..services.web_search import search_all
        search_name = common_name or target_name
        queries = [
            f"{search_name} synthesis procedure reagents step by step",
            f"{search_name} total synthesis starting materials SMILES",
        ]
        if iupac_name and iupac_name != search_name:
            queries.append(f"{iupac_name} synthesis")
        all_sources = []
        seen_urls: set[str] = set()
        for q in queries:
            if _time.monotonic() - t0 > _WEB_SEARCH_TIMEOUT * 0.5:
                break
            for s in search_all(q, max_results=4):
                if s.url not in seen_urls:
                    seen_urls.add(s.url)
                    all_sources.append(s)
    except Exception as e:
        logger.warning("[web_retro] search failed: %s", e)
        return []

    if not all_sources:
        return []

    # LLM extraction
    snippets = "\n---\n".join(
        f"Title: {s.title}\nSnippet: {s.snippet}" for s in all_sources[:6]
    )

    try:
        from ..services.research_llm import _chat_json
    except ImportError:
        logger.warning("[web_retro] research_llm not available")
        return []

    if _time.monotonic() - t0 > _WEB_SEARCH_TIMEOUT:
        logger.warning("[web_retro] timeout before LLM call")
        return []

    result = _chat_json(
        "You are a retrosynthesis expert. Given web search results about a molecule's synthesis, "
        "extract reaction routes. Return JSON: {\"routes\": [{\"reactants_smiles\": [\"SMI1\", \"SMI2\"], "
        "\"reaction_smiles\": \"SMI1.SMI2>>PRODUCT\", \"yield_pct\": number|null, "
        "\"procedure\": \"brief description\"}]}. "
        "CRITICAL: reactants_smiles MUST be valid SMILES strings, not names. "
        "If you cannot determine valid SMILES for reactants, omit that route. "
        "Return at most 3 routes. Product SMILES: " + smiles,
        f"Target molecule: {target_name}\nSMILES: {smiles}\n\nSearch results:\n{snippets}",
    )

    if not result or not isinstance(result.get("routes"), list):
        return []

    # Validate and convert to standard format
    routes: list[dict[str, Any]] = []
    for raw in result["routes"]:
        reactant_smiles_list = raw.get("reactants_smiles", [])
        if not isinstance(reactant_smiles_list, list) or not reactant_smiles_list:
            continue

        # Validate every SMILES with RDKit
        valid_parts: list[str] = []
        all_valid = True
        for smi in reactant_smiles_list:
            if not isinstance(smi, str) or not smi.strip():
                all_valid = False
                break
            mol = Chem.MolFromSmiles(smi.strip())
            if mol is None:
                all_valid = False
                break
            valid_parts.append(Chem.MolToSmiles(mol, isomericSmiles=True))

        if not all_valid or not valid_parts:
            continue

        # Check product SMILES is not among reactants (no self-loops)
        product_canon = smiles
        if HAS_RDKIT:
            pmol = Chem.MolFromSmiles(smiles)
            if pmol:
                product_canon = Chem.MolToSmiles(pmol, isomericSmiles=True)
        if product_canon in valid_parts:
            continue

        reactants_str = ".".join(valid_parts)
        rxn_smi = raw.get("reaction_smiles", f"{reactants_str}>>{product_canon}")

        # Validate reaction SMILES format
        if ">>" not in rxn_smi:
            rxn_smi = f"{reactants_str}>>{product_canon}"

        route: dict[str, Any] = {
            "reactants": reactants_str,
            "reaction_smiles": rxn_smi,
            "source": "web",
            "score": 0.5,
            "plausibility": 0.5,  # lower confidence than ORD/model
        }
        if raw.get("yield_pct") is not None:
            try:
                route["expected_yield"] = float(raw["yield_pct"]) / 100.0
            except (ValueError, TypeError):
                pass
        if raw.get("procedure"):
            route["procedure_details"] = str(raw["procedure"])[:500]

        routes.append(_merge_provenance(route, "web", "search_llm"))

    elapsed = _time.monotonic() - t0
    logger.info("[web_retro] %d valid routes extracted in %.1fs", len(routes), elapsed)
    return routes


def get_ord_routes(smiles: str, limit: int = 15) -> list[dict[str, Any]]:
    """Adapter for ORD-backed retrosynthesis routes."""
    if not RETRO_ENABLE_ORD:
        return []
    return _normalize_source_routes(_ord_search_via_api(smiles, limit=limit), "ord", "sqlite")


def get_web_routes(smiles: str, target_name: str | None = None) -> list[dict[str, Any]]:
    """Adapter for web-extracted retrosynthesis routes."""
    if not RETRO_ENABLE_WEB:
        return []
    return _normalize_source_routes(_web_search_retro(smiles, target_name=target_name), "web", "search_llm")


def get_retro_model_routes(smiles: str, top_n: int = 10) -> list[dict[str, Any]]:
    """Adapter for the current local template-relevance retrosynthesis model."""
    if not RETRO_ENABLE_RETRO_MODEL:
        return []
    from ..retro_predictor import predict_retro

    routes = predict_retro(smiles, top_n=top_n)
    return _normalize_source_routes(routes, "retro_model", "local_model")


def get_aizynthfinder_routes(smiles: str, top_n: int = 10) -> list[dict[str, Any]]:
    """Adapter for AiZynthFinder multi-step retrosynthesis routes."""
    if not RETRO_ENABLE_AIZYNTH:
        return []
    if not AIZYNTH_BASE_URL:
        logger.info("[retro] AiZynthFinder enabled but AIZYNTH_BASE_URL is not configured")
        return []
    from ..services.aizynth_client import normalize_aizynth_routes, run_aizynth_retrosynthesis

    payload = run_aizynth_retrosynthesis(
        AIZYNTH_BASE_URL,
        smiles,
        max_transforms=AIZYNTH_MAX_TRANSFORMS,
        time_limit=AIZYNTH_TIME_LIMIT,
        iterations=AIZYNTH_ITERATIONS,
        expansion_model=AIZYNTH_EXPANSION_MODEL,
        stock=AIZYNTH_STOCK,
        timeout=int(AIZYNTH_TIMEOUT_SEC),
    )
    routes = normalize_aizynth_routes(payload, limit=top_n)
    return _normalize_source_routes(routes, "aizynthfinder", "service_tree_search")


def get_retrocast_routes(smiles: str, top_n: int = 10) -> list[dict[str, Any]]:
    """RetroCast does not currently act as a standalone route generator here.

    The real RetroCast package is integrated as a canonicalization bridge for
    planner outputs such as AiZynthFinder. Standalone route collection remains a
    future seam because RetroCast itself is not a planner service.
    """
    if not RETRO_ENABLE_RETROCAST:
        return []
    logger.info("[retro] RetroCast is enabled as an adaptation bridge; no standalone route source is registered")
    return []


def collect_candidate_routes(
    smiles: str,
    *,
    ord_limit: int = 15,
    model_top_n: int = 10,
    use_web: bool = True,
    target_name: str | None = None,
    ord_authoritative: bool | None = None,
    include_experimental: bool = False,
    enabled_sources: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, str]]:
    """Collect raw route candidates from all enabled sources.

    The default policy preserves current behavior:
    if ORD returns routes and RETRO_ORD_AUTHORITATIVE is true, other sources are skipped.
    """
    if not smiles:
        return [], [], {}

    authoritative = RETRO_ORD_AUTHORITATIVE if ord_authoritative is None else ord_authoritative
    routes: list[dict[str, Any]] = []
    sources_used: list[str] = []
    source_errors: dict[str, str] = {}
    enabled = None if enabled_sources is None else set(enabled_sources)
    want_ord = enabled is None or "ord" in enabled
    want_web = enabled is None or "web" in enabled
    want_model = enabled is None or "retro_model" in enabled
    want_aizynth = enabled is None or "aizynthfinder" in enabled
    want_retrocast = enabled is None or "retrocast" in enabled

    try:
        ord_routes = get_ord_routes(smiles, limit=ord_limit) if want_ord else []
    except Exception as e:
        logger.warning("ORD route collection failed: %s", e)
        source_errors["ord"] = str(e)
        ord_routes = []

    if ord_routes:
        routes.extend(ord_routes)
        sources_used.append("ord")
        if authoritative and enabled is None:
            logger.info("ORD: %d routes found — authoritative mode, skipping fallback sources", len(ord_routes))
            return routes, sources_used, source_errors

    if use_web and want_web:
        try:
            web_routes = get_web_routes(smiles, target_name=target_name)
            if web_routes:
                routes.extend(web_routes)
                sources_used.append("web")
                logger.info("Web search: %d routes found", len(web_routes))
        except Exception as e:
            logger.warning("Web search retro failed: %s", e)
            source_errors["web"] = str(e)

    if want_model:
        try:
            model_routes = get_retro_model_routes(smiles, top_n=model_top_n)
            if model_routes:
                routes.extend(model_routes)
                sources_used.append("retro_model")
                logger.info("Retro model: %d routes found", len(model_routes))
        except Exception as e:
            logger.warning("Retro model failed: %s", e)
            source_errors["retro_model"] = str(e)

    if include_experimental and want_aizynth:
        try:
            aizynth_routes = get_aizynthfinder_routes(smiles, top_n=model_top_n)
            if aizynth_routes:
                routes.extend(aizynth_routes)
                sources_used.append("aizynthfinder")
                logger.info("AiZynthFinder: %d routes found", len(aizynth_routes))
        except Exception as e:
            logger.warning("AiZynthFinder route collection failed: %s", e)
            source_errors["aizynthfinder"] = str(e)

    if include_experimental and want_retrocast:
        try:
            retrocast_routes = get_retrocast_routes(smiles, top_n=model_top_n)
            if retrocast_routes:
                routes.extend(retrocast_routes)
                sources_used.append("retrocast")
                logger.info("RetroCast: %d routes found", len(retrocast_routes))
        except Exception as e:
            logger.warning("RetroCast route collection failed: %s", e)
            source_errors["retrocast"] = str(e)

    return routes, sources_used, source_errors


# ═════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═════════════════════════════════════════════════════════════════════════════


def search_and_rank(
    smiles: str,
    top_n: int = 5,
    *,
    source_mode: str = "auto",
    ord_authoritative: bool | None = None,
) -> dict[str, Any]:
    """Full retrosynthesis pipeline over enabled route sources."""
    enabled_sources = get_enabled_sources_for_mode(source_mode)
    all_routes, sources_used, source_errors = collect_candidate_routes(
        smiles,
        ord_limit=15,
        model_top_n=10,
        use_web=True,
        include_experimental=True,
        ord_authoritative=ord_authoritative,
        enabled_sources=enabled_sources,
    )
    total_found = len(all_routes)
    source_counts = _count_routes_by_source(all_routes)

    if not all_routes:
        return {
            "routes": [],
            "best_route": None,
            "sources_used": [],
            "total_found": 0,
            "total_unique": 0,
            "source_counts": {},
            "source_counts_deduped": {},
            "source_mode": source_mode,
            "source_errors": source_errors,
        }

    for route in all_routes:
        score_route(route)

    all_routes = _deduplicate_routes(all_routes)
    total_unique = len(all_routes)
    source_counts_deduped = _count_routes_by_source(all_routes)
    logger.info("After dedup: %d routes (from %d)", len(all_routes), total_found)

    all_routes.sort(key=lambda r: r.get("final_score", 0), reverse=True)
    top_routes = all_routes[:top_n]

    return {
        "routes": top_routes,
        "best_route": top_routes[0] if top_routes else None,
        "sources_used": sources_used,
        "total_found": total_found,
        "total_unique": total_unique,
        "source_counts": source_counts,
        "source_counts_deduped": source_counts_deduped,
        "source_mode": source_mode,
        "source_errors": source_errors,
    }
