"""Dedicated literature/patent research workspace helpers.

This module reuses the existing research, web search, scraping and optional RAG
components, but does not mutate the main LangGraph molecule pipeline state.
"""

from __future__ import annotations

import logging
import re
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

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
_DATA_DIR = Path(__file__).resolve().parent / "data"
_CURATED_CORPUS_PATH = _DATA_DIR / "research_corpus.json"


def _source_domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def _safe_markdown_url(url: str) -> str:
    return str(url or "").replace("(", "%28").replace(")", "%29")


def _source_payload(source: WebSource, index: int) -> dict[str, Any]:
    citation_id = f"S{index}"
    title = source.title or source.url or citation_id
    url = source.url or ""
    return {
        **source.model_dump(mode="json"),
        "citation_id": citation_id,
        "domain": _source_domain(url),
        "citation_markdown": f"[{citation_id}]({_safe_markdown_url(url)})" if url else citation_id,
        "title_markdown": f"[{title}]({_safe_markdown_url(url)})" if url else title,
    }


@lru_cache(maxsize=1)
def _load_curated_corpus() -> list[dict[str, Any]]:
    if not _CURATED_CORPUS_PATH.exists():
        return []
    try:
        data = json.loads(_CURATED_CORPUS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("[research_workspace] curated corpus load failed: %s", exc)
        return []
    return data if isinstance(data, list) else []


def _query_tokens(query: str) -> set[str]:
    return {token for token in re.split(r"[^a-zA-Z0-9+-]+", query.lower()) if len(token) >= 3}


def _curated_score(query: str, doc: dict[str, Any]) -> int:
    normalized_query = " ".join(query.lower().split())
    tokens = _query_tokens(query)
    score = 0
    for keyword in doc.get("keywords") or []:
        key = str(keyword).lower().strip()
        if not key:
            continue
        if " " in key:
            if key in normalized_query:
                score += 3
        elif key in tokens:
            score += 2
    for author in doc.get("authors") or []:
        last_name = str(author).split()[-1].lower().strip(",.")
        if last_name and last_name in tokens:
            score += 3
    title = str(doc.get("title") or "").lower()
    score += len(tokens & _query_tokens(title))
    return score


def _curated_matches(query: str, *, limit: int = 3) -> list[dict[str, Any]]:
    scored = [
        (_curated_score(query, doc), doc)
        for doc in _load_curated_corpus()
    ]
    matches = [doc for score, doc in sorted(scored, key=lambda item: item[0], reverse=True) if score >= 4]
    return matches[:limit]


def _curated_text(doc: dict[str, Any]) -> str:
    lines: list[str] = []
    for excerpt in doc.get("excerpts") or []:
        page = excerpt.get("page")
        section = excerpt.get("section")
        prefix_parts = []
        if page:
            prefix_parts.append(f"page {page}")
        if section:
            prefix_parts.append(str(section))
        prefix = " - ".join(prefix_parts)
        text = str(excerpt.get("text") or "").strip()
        if text:
            lines.append(f"{prefix}: {text}" if prefix else text)
    return "\n\n".join(lines)


def _curated_source(doc: dict[str, Any]) -> WebSource:
    return WebSource(
        url=str(doc.get("url") or doc.get("related_url") or ""),
        title=str(doc.get("title") or doc.get("id") or "Curated research source"),
        snippet=_curated_text(doc)[:500],
        source_type="curated_pdf",
    )


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


def _fallback_analysis(
    query: str,
    mode: ResearchMode,
    candidates: list[Any],
    evidence: list[dict[str, Any]],
    rag_results: list[dict[str, Any]],
) -> dict[str, Any]:
    mode_label = {
        "molecule": "подбор молекул",
        "literature": "литературный обзор",
        "patent": "патентный поиск",
    }[mode]
    findings: list[dict[str, Any]] = []
    if evidence:
        findings.append({
            "claim": f"По запросу найдено {len(evidence)} релевантных web/PubMed источников для сценария: {mode_label}.",
            "evidence": [f"S{idx}" for idx in range(1, min(len(evidence), 3) + 1)],
            "confidence": "medium",
        })
    if candidates:
        names = ", ".join(candidate.name for candidate in candidates[:5])
        findings.append({
            "claim": f"Из найденных текстов извлечены и проверены через PubChem кандидаты: {names}.",
            "evidence": [f"S{idx}" for idx in range(1, min(len(evidence), 3) + 1)] or [],
            "confidence": "medium",
        })
    if rag_results:
        findings.append({
            "claim": f"Локальный RAG добавил {len(rag_results)} результатов для сопоставления с внешним поиском.",
            "evidence": [f"R{idx}" for idx in range(1, min(len(rag_results), 3) + 1)],
            "confidence": "medium",
        })

    return {
        "answer": (
            f"Агент собрал данные по запросу \"{query}\" и подготовил первичный анализ. "
            "Для окончательных химических решений нужно сверить первоисточники и экспериментальные условия."
        ),
        "key_findings": findings,
        "candidate_assessment": [
            {
                "name": candidate.name,
                "assessment": "Кандидат найден в источниках и разрешён через PubChem; требуется ручная проверка релевантности к запросу.",
                "confidence": "medium",
            }
            for candidate in candidates[:6]
        ],
        "limitations": [
            "Автоматический анализ использует извлечённые фрагменты, а не полный экспертный разбор статей.",
            "Внешний web/PubMed поиск может быть неполным или нестабильным.",
        ],
        "safety_notes": [
            "Не использовать вывод как готовый экспериментальный протокол без safety review.",
        ],
        "recommended_next_steps": [
            "Открыть ключевые источники и проверить экспериментальные разделы.",
            "Сопоставить найденные молекулы с safety guard и доступностью реагентов.",
            "Для перспективных кандидатов запустить ретросинтез в отдельной вкладке.",
        ],
        "source_quality": "heuristic_fallback",
        "analysis_engine": "heuristic",
    }


def _analyze_research_evidence(
    query: str,
    mode: ResearchMode,
    interpreted_intent: str,
    candidates: list[Any],
    evidence: list[dict[str, Any]],
    rag_results: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        from .services.research_llm import llm_analyze_research_evidence

        analysis = llm_analyze_research_evidence(
            query,
            mode,
            interpreted_intent,
            evidence,
            candidates,
            rag_results,
        )
        if analysis:
            return analysis
    except Exception as exc:
        logger.info("[research_workspace] LLM analysis unavailable: %s", exc)
    return _fallback_analysis(query, mode, candidates, evidence, rag_results)


def run_research_workspace(
    query: str,
    mode: ResearchMode = "literature",
    max_sources: int = 8,
) -> dict[str, Any]:
    """Run standalone research without changing the main synthesis flow."""
    research_query = formulate_search_queries(query)
    base_queries = research_query.search_queries or [query]
    search_queries = _mode_queries(base_queries, mode)
    curated_docs = _curated_matches(query)

    all_sources: list[WebSource] = []
    seen_urls: set[str] = set()
    source_errors: dict[str, str] = {}
    # Curated matches are primary-source evidence already available to the
    # workspace. Avoid slow/noisy web search for exact paper/supplement queries.
    web_search_queries = [] if curated_docs else search_queries
    for search_query in web_search_queries:
        try:
            for source in search_all(search_query, max_results=5):
                if source.url and source.url not in seen_urls:
                    seen_urls.add(source.url)
                    all_sources.append(source)
        except Exception as exc:
            source_errors[search_query] = str(exc)

    evidence: list[dict[str, Any]] = []
    source_payloads: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for doc in curated_docs[:max_sources]:
        source = _curated_source(doc)
        index = len(source_payloads) + 1
        source_info = _source_payload(source, index)
        source_payloads.append(source_info)
        text = _curated_text(doc)
        if text:
            counts.update(extract_molecules_from_text(text))
        evidence.append({
            "citation_id": source_info["citation_id"],
            "url": source.url,
            "title": source.title,
            "snippet": source.snippet,
            "source_type": source.source_type,
            "domain": source_info["domain"],
            "citation_markdown": source_info["citation_markdown"],
            "title_markdown": source_info["title_markdown"],
            "excerpt": _make_excerpt(text or source.snippet, query, limit=1600),
            "curated_source_id": doc.get("id"),
        })

    remaining_slots = max(max_sources - len(source_payloads), 0)
    for source in all_sources[:remaining_slots]:
        index = len(source_payloads) + 1
        try:
            text = _fetch_source_text(source)
        except Exception as exc:
            logger.warning("[research_workspace] scrape failed for %s: %s", source.url, exc)
            text = ""
        if text:
            counts.update(extract_molecules_from_text(text))
        source_info = _source_payload(source, index)
        source_payloads.append(source_info)
        evidence.append({
            "citation_id": source_info["citation_id"],
            "url": source.url,
            "title": source.title,
            "snippet": source.snippet,
            "source_type": source.source_type,
            "domain": source_info["domain"],
            "citation_markdown": source_info["citation_markdown"],
            "title_markdown": source_info["title_markdown"],
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
    analysis = _analyze_research_evidence(
        query,
        mode,
        research_query.interpreted_intent,
        candidates,
        evidence,
        rag_results,
    )

    return {
        "status": "ok" if evidence or candidates or rag_results else "empty",
        "query": query,
        "mode": mode,
        "interpreted_intent": research_query.interpreted_intent,
        "search_queries": search_queries,
        "summary": summary,
        "analysis": analysis,
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
        "sources": source_payloads,
        "citations": source_payloads,
        "evidence": evidence,
        "rag_results": rag_results,
        "source_errors": source_errors,
    }
