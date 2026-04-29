"""Research node: resolve vague/unresolved queries into concrete molecule candidates.

Called in two scenarios:
  1. classify_node determined input_type == "research" (direct path)
  2. validate_and_guard could not find molecule in PubChem (fallback path)

Runs: search web -> scrape -> extract molecule names -> resolve via PubChem.
After finding candidates, sets state["query"] to the best candidate's SMILES
so that validate_and_guard can re-validate.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

from ..tools.research import (
    extract_molecules_from_text,
    formulate_search_queries,
    resolve_candidates,
)
from ..services.web_search import search_all
from ..services.web_scraper import extract_pubmed_abstract, fetch_and_extract

logger = logging.getLogger(__name__)

_MAX_SOURCES_TO_SCRAPE = 8
_MAX_CANDIDATES = 15
_PMID_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)")


def research_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: resolve vague queries to concrete molecules.

    Reads:  state["query"]
    Writes: state["research_result"], state["query"] (overwritten with best candidate SMILES)
    """
    query = state.get("query", "").strip()
    logger.info("[research] detected research query: %r", query)

    try:
        return _run_research(query)
    except Exception as e:
        logger.error("[research] pipeline failed: %s", e, exc_info=True)
        return {
            "research_result": {
                "is_successful": False,
                "error": f"Research pipeline error: {e}",
                "candidates": [],
            }
        }


def _run_research(query: str) -> dict[str, Any]:
    research_query = formulate_search_queries(query)

    all_sources = []
    seen_urls: set[str] = set()
    for q in research_query.search_queries:
        try:
            for src in search_all(q, max_results=5):
                if src.url not in seen_urls:
                    seen_urls.add(src.url)
                    all_sources.append(src)
        except Exception as e:
            logger.warning("[research] search_all failed for %r: %s", q, e)

    if not all_sources:
        return {
            "research_result": {
                "is_successful": False,
                "error": "No web sources found for the query.",
                "candidates": [],
            }
        }

    texts: list[str] = []
    scraped = 0
    for src in all_sources:
        if scraped >= _MAX_SOURCES_TO_SCRAPE:
            break
        try:
            text = None
            if src.source_type == "pubmed":
                m = _PMID_RE.search(src.url)
                if m:
                    text = extract_pubmed_abstract(m.group(1))
            if text is None:
                text = fetch_and_extract(src.url)
            if text:
                texts.append(text)
                scraped += 1
        except Exception as e:
            logger.warning("[research] scrape failed for %s: %s", src.url, e)

    counts: Counter[str] = Counter()
    for t in texts:
        names = extract_molecules_from_text(t)
        counts.update(names)
    ranked_names = [name for name, _ in counts.most_common()]

    if not ranked_names:
        return {
            "research_result": {
                "is_successful": False,
                "error": "Could not extract molecule names from the found sources.",
                "candidates": [],
            }
        }

    names_to_resolve = ranked_names[: _MAX_CANDIDATES * 2]
    try:
        candidates = resolve_candidates(names_to_resolve, query)
    except Exception as e:
        logger.warning("[research] resolve_candidates failed: %s", e)
        candidates = []

    source_urls = [s.url for s in all_sources[:5]]
    for c in candidates:
        c.source_urls = source_urls

    candidates.sort(key=lambda c: c.confidence, reverse=True)
    candidates = candidates[:_MAX_CANDIDATES]

    if not candidates:
        return {
            "research_result": {
                "is_successful": False,
                "error": "None of the extracted molecule names could be resolved via PubChem.",
                "candidates": [],
            }
        }

    try:
        from ..services.research_llm import llm_build_summary
        summary = llm_build_summary(
            research_query.original_input,
            research_query.interpreted_intent,
            candidates,
        )
    except Exception:
        names_str = ", ".join(c.name for c in candidates[:5])
        summary = (
            f'For the query "{query}", found {len(candidates)} candidate(s): '
            f"{names_str}{'...' if len(candidates) > 5 else ''}. "
            f"Each candidate was verified in PubChem."
        )

    logger.info("[research] found %d candidates", len(candidates))

    best = candidates[0]
    best_query = best.canonical_smiles or best.name

    return {
        "research_result": {
            "is_successful": True,
            "interpreted_intent": research_query.interpreted_intent,
            "candidates": [c.model_dump(mode="json") for c in candidates],
            "sources": [s.model_dump(mode="json") for s in all_sources[:10]],
            "summary": summary or "",
        },
        "query": best_query,
    }
