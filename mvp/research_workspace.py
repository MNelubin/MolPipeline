"""Dedicated literature/patent research workspace helpers.

This module reuses the existing research, web search, scraping and optional RAG
components, but does not mutate the main LangGraph molecule pipeline state.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Literal
from urllib.parse import urlparse

import requests

from .models.research import WebSource
from .services.web_scraper import discover_document_links, extract_pubmed_abstract, fetch_and_extract, fetch_page
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
_CROSSREF_API = "https://api.crossref.org/works"


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
                query,
                f"{query} patent synthesis preparation",
                f"{query} patents.google.com",
                f"{query} patent example reaction conditions",
            ])
    elif mode == "literature":
        for query in base_queries:
            expanded.extend([
                query,
                f"{query} PDF supplementary information",
                f"{query} synthesis literature PubMed",
                f"{query} review chemistry",
            ])
    else:
        for query in base_queries:
            expanded.extend([
                query,
                f"{query} molecule PubChem pharmacology",
                f"{query} active compound SMILES",
            ])
    return _dedupe(expanded)[:_MAX_QUERIES]


def _make_excerpt(text: str, query: str, *, limit: int = 900) -> str:
    if not text:
        return ""
    lower = text.lower()
    focus_query = query.rsplit(":", 1)[-1].strip()
    section_match = re.search(r"\bsection\s+([^:]{1,120}):", query, flags=re.IGNORECASE)
    if section_match:
        focus_query = f"{section_match.group(0)} {focus_query}"
    if len(focus_query) < 20:
        focus_query = query
    keywords = list(dict.fromkeys(word for word in re.split(r"\W+", focus_query.lower()) if len(word) > 2))
    start = 0
    best_score = -1
    for keyword in keywords:
        idx = lower.find(keyword)
        while idx >= 0:
            candidate_start = max(0, idx - 220)
            window = lower[candidate_start:candidate_start + limit]
            score = sum(1 for item in keywords if item in window)
            if score > best_score:
                best_score = score
                start = candidate_start
            idx = lower.find(keyword, idx + len(keyword))
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


def _crossref_sources(query: str, *, limit: int = 2) -> list[WebSource]:
    """Find likely primary article pages from a bibliographic query."""
    try:
        resp = requests.get(
            _CROSSREF_API,
            params={"query.bibliographic": query, "rows": limit},
            headers={"User-Agent": "MolPipeline/1.0 (mailto:molpipeline@example.org)"},
            timeout=12,
        )
        if resp.status_code != 200:
            return []
        items = (resp.json().get("message") or {}).get("items") or []
    except Exception as exc:
        logger.info("[research_workspace] Crossref lookup failed: %s", exc)
        return []

    sources: list[WebSource] = []
    for item in items[:limit]:
        doi = str(item.get("DOI") or "").strip()
        titles = item.get("title") or []
        title = str(titles[0] if titles else doi or "Crossref result").strip()
        url = str(item.get("URL") or (f"https://doi.org/{doi}" if doi else "")).strip()
        if not url:
            continue
        sources.append(WebSource(
            url=url,
            title=title,
            snippet=f"Crossref bibliographic match; DOI: {doi}" if doi else "Crossref bibliographic match",
            source_type="web",
        ))
    return sources


def _expand_linked_documents(sources: list[WebSource], *, limit: int) -> list[WebSource]:
    expanded: list[WebSource] = []
    seen: set[str] = set()
    for source in sources:
        if source.url and source.url not in seen:
            seen.add(source.url)
            expanded.append(source)
        if len(expanded) >= limit:
            break
        if source.source_type == "pubmed" or source.url.lower().split("?", 1)[0].endswith(".pdf"):
            continue
        html = fetch_page(source.url)
        if not html:
            continue
        linked_count = 0
        for link in discover_document_links(html, source.url, limit=4):
            url = link.get("url") or ""
            if not url or url in seen:
                continue
            seen.add(url)
            expanded.append(WebSource(
                url=url,
                title=link.get("title") or f"Linked document from {source.title}",
                snippet=f"Linked document discovered from {source.url}",
                source_type="pdf" if link.get("source_type") == "pdf" else "web",
            ))
            linked_count += 1
            if len(expanded) >= limit:
                return expanded
        if linked_count:
            return expanded
    return expanded


def _optional_rag_results(query: str, mode: ResearchMode) -> list[dict[str, Any]]:
    if mode not in {"literature", "patent"}:
        return []
    try:
        from .tools.rag_search import KeywordSearchInput, literature_keyword_search

        return literature_keyword_search(KeywordSearchInput(query=query, top_k=5))
    except Exception as exc:
        logger.info("[research_workspace] optional RAG unavailable: %s", exc)
        return []


def _has_strong_rag_hit(rag_results: list[dict[str, Any]]) -> bool:
    if not rag_results:
        return False
    try:
        return float(rag_results[0].get("score") or 0.0) >= 8.0
    except (TypeError, ValueError):
        return False


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
    base_queries = _dedupe([query] + (research_query.search_queries or []))
    search_queries = _mode_queries(base_queries, mode)
    rag_results = _optional_rag_results(query, mode)

    all_sources: list[WebSource] = []
    seen_urls: set[str] = set()
    source_errors: dict[str, str] = {}
    web_search_queries = [] if _has_strong_rag_hit(rag_results) else search_queries
    if mode == "literature" and web_search_queries:
        for source in _crossref_sources(query):
            if source.url and source.url not in seen_urls:
                seen_urls.add(source.url)
                all_sources.append(source)
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

    expanded_sources = _expand_linked_documents(all_sources, limit=max_sources)
    for source in expanded_sources[:max_sources]:
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
