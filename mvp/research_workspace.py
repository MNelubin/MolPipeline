"""Dedicated literature/patent research workspace helpers.

This module reuses the existing research, web search, scraping and optional RAG
components, but does not mutate the main LangGraph molecule pipeline state.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Literal

from .models.research import WebSource
from .services.web_scraper import extract_pubmed_abstract, fetch_and_extract
from .services.web_search import search_all
from .tools.research import (
    extract_molecules_from_text,
    formulate_search_queries,
    resolve_candidates,
)

logger = logging.getLogger(__name__)

ResearchMode = Literal["molecule", "literature", "patent"]

_PMID_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)")
_MAX_QUERIES = 8
_MAX_CANDIDATES = 12


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        normalized = " ".join(item.split())
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            out.append(normalized)
    return out


def _mode_queries(base_queries: list[str], mode: ResearchMode) -> list[str]:
    expanded: list[str] = []
    if mode == "patent":
        for query in base_queries:
            expanded.extend([
                f"{query} patent synthesis preparation",
                f"{query} patents.google.com",
                f"{query} patent example reaction conditions",
            ])
    elif mode == "literature":
        for query in base_queries:
            expanded.extend([
                f"{query} synthesis literature PubMed",
                f"{query} reaction conditions experimental",
                f"{query} review chemistry",
            ])
    else:
        for query in base_queries:
            expanded.extend([
                f"{query} molecule PubChem pharmacology",
                f"{query} active compound SMILES",
            ])
    return _dedupe(expanded)[:_MAX_QUERIES]


def _make_excerpt(text: str, query: str, *, limit: int = 900) -> str:
    if not text:
        return ""
    lower = text.lower()
    keywords = [word for word in re.split(r"\W+", query.lower()) if len(word) > 3]
    start = 0
    for keyword in keywords:
        idx = lower.find(keyword)
        if idx >= 0:
            start = max(0, idx - 220)
            break
    excerpt = text[start:start + limit].strip()
    if start > 0:
        excerpt = "..." + excerpt
    if start + limit < len(text):
        excerpt += "..."
    return excerpt


def _fetch_source_text(source: WebSource) -> str:
    if source.source_type == "pubmed":
        match = _PMID_RE.search(source.url)
        if match:
            text = extract_pubmed_abstract(match.group(1))
            if text:
                return text
    return fetch_and_extract(source.url) or ""


def _optional_rag_results(query: str, mode: ResearchMode) -> list[dict[str, Any]]:
    if mode not in {"literature", "patent"}:
        return []
    try:
        from .tools.rag_search import KeywordSearchInput, literature_keyword_search

        return literature_keyword_search(KeywordSearchInput(query=query, top_k=5))
    except Exception as exc:
        logger.info("[research_workspace] optional RAG unavailable: %s", exc)
        return []


def _build_summary(
    query: str,
    interpreted_intent: str,
    candidates: list[Any],
    evidence: list[dict[str, Any]],
    mode: ResearchMode,
) -> str:
    try:
        from .services.research_llm import llm_build_summary

        summary = llm_build_summary(query, interpreted_intent, candidates)
        if summary:
            return summary
    except Exception:
        pass

    source_count = len(evidence)
    candidate_names = ", ".join(candidate.name for candidate in candidates[:5])
    mode_label = {
        "molecule": "поиска молекул",
        "literature": "литературного поиска",
        "patent": "патентного поиска",
    }[mode]
    if candidate_names:
        return (
            f"Режим {mode_label}: найдено {source_count} источников и "
            f"{len(candidates)} PubChem-кандидатов. Основные кандидаты: {candidate_names}."
        )
    return f"Режим {mode_label}: найдено {source_count} источников, но PubChem-кандидаты пока не извлечены."


def run_research_workspace(
    query: str,
    mode: ResearchMode = "literature",
    max_sources: int = 8,
) -> dict[str, Any]:
    """Run standalone research without changing the main synthesis flow."""
    research_query = formulate_search_queries(query)
    base_queries = research_query.search_queries or [query]
    search_queries = _mode_queries(base_queries, mode)

    all_sources: list[WebSource] = []
    seen_urls: set[str] = set()
    source_errors: dict[str, str] = {}
    for search_query in search_queries:
        try:
            for source in search_all(search_query, max_results=5):
                if source.url and source.url not in seen_urls:
                    seen_urls.add(source.url)
                    all_sources.append(source)
        except Exception as exc:
            source_errors[search_query] = str(exc)

    evidence: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for source in all_sources[:max_sources]:
        try:
            text = _fetch_source_text(source)
        except Exception as exc:
            logger.warning("[research_workspace] scrape failed for %s: %s", source.url, exc)
            text = ""
        if text:
            counts.update(extract_molecules_from_text(text))
        evidence.append({
            "url": source.url,
            "title": source.title,
            "snippet": source.snippet,
            "source_type": source.source_type,
            "excerpt": _make_excerpt(text or source.snippet, query),
        })

    ranked_names = [name for name, _ in counts.most_common()]
    candidates = resolve_candidates(ranked_names[: _MAX_CANDIDATES * 2], query) if ranked_names else []
    source_urls = [item["url"] for item in evidence[:5]]
    for candidate in candidates:
        candidate.source_urls = source_urls
    candidates.sort(key=lambda candidate: candidate.confidence, reverse=True)
    candidates = candidates[:_MAX_CANDIDATES]

    rag_results = _optional_rag_results(query, mode)
    summary = _build_summary(
        query,
        research_query.interpreted_intent,
        candidates,
        evidence,
        mode,
    )

    return {
        "status": "ok" if evidence or candidates or rag_results else "empty",
        "query": query,
        "mode": mode,
        "interpreted_intent": research_query.interpreted_intent,
        "search_queries": search_queries,
        "summary": summary,
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
        "sources": [source.model_dump(mode="json") for source in all_sources[:max_sources]],
        "evidence": evidence,
        "rag_results": rag_results,
        "source_errors": source_errors,
    }
