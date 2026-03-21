"""Web search services: PubMed E-utilities and DuckDuckGo."""

from __future__ import annotations

import logging
import time
from typing import Any
from urllib.parse import quote_plus

import requests

from ..models.research import WebSource

logger = logging.getLogger(__name__)

_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_REQUEST_TIMEOUT = 15
_RETRY_DELAY = 0.4
_TOOL_NAME = "ChemSynthAssistant"
_TOOL_EMAIL = "chemsynthassistant@example.com"


def _get(url: str, *, retries: int = 2, timeout: int = _REQUEST_TIMEOUT) -> requests.Response | None:
    for attempt in range(1, retries + 2):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429 and attempt <= retries:
                time.sleep(_RETRY_DELAY * attempt * 2)
                continue
            if resp.status_code >= 500 and attempt <= retries:
                time.sleep(_RETRY_DELAY * attempt)
                continue
            logger.warning("GET %s returned %s", url, resp.status_code)
            return None
        except requests.RequestException as exc:
            logger.warning("Request failed (%s): %s", url, exc)
            if attempt <= retries:
                time.sleep(_RETRY_DELAY * attempt)
    return None


def _get_json(url: str, **kwargs: Any) -> dict[str, Any] | None:
    resp = _get(url, **kwargs)
    if resp is None:
        return None
    try:
        return resp.json()
    except ValueError:
        return None


def _esearch(query: str, max_results: int = 10) -> list[str]:
    url = (
        f"{_EUTILS_BASE}/esearch.fcgi?"
        f"db=pubmed&term={quote_plus(query)}&retmax={max_results}"
        f"&retmode=json&tool={_TOOL_NAME}&email={_TOOL_EMAIL}"
    )
    data = _get_json(url)
    if data is None:
        return []
    try:
        return data["esearchresult"]["idlist"]
    except (KeyError, TypeError):
        return []


def _efetch_summaries(pmids: list[str]) -> list[dict[str, Any]]:
    if not pmids:
        return []
    ids_str = ",".join(pmids)
    url = (
        f"{_EUTILS_BASE}/esummary.fcgi?"
        f"db=pubmed&id={ids_str}&retmode=json"
        f"&tool={_TOOL_NAME}&email={_TOOL_EMAIL}"
    )
    data = _get_json(url)
    if data is None:
        return []
    result_block = data.get("result", {})
    summaries: list[dict[str, Any]] = []
    for pmid in pmids:
        info = result_block.get(pmid)
        if info and isinstance(info, dict):
            summaries.append(info)
    return summaries


def search_pubmed(query: str, max_results: int = 10) -> list[WebSource]:
    """Search PubMed and return structured WebSource objects."""
    pmids = _esearch(query, max_results=max_results)
    if not pmids:
        return []

    summaries = _efetch_summaries(pmids)
    sources: list[WebSource] = []
    for s in summaries:
        uid = s.get("uid", "")
        title = s.get("title", "").strip()
        snippet_parts: list[str] = []
        authors = s.get("authors", [])
        if authors:
            first_author = authors[0].get("name", "") if isinstance(authors[0], dict) else str(authors[0])
            snippet_parts.append(first_author + " et al.")
        source_journal = s.get("source", "")
        pubdate = s.get("pubdate", "")
        if source_journal:
            snippet_parts.append(source_journal)
        if pubdate:
            snippet_parts.append(pubdate)

        sources.append(WebSource(
            url=f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
            title=title,
            snippet="; ".join(snippet_parts),
            source_type="pubmed",
        ))
    return sources


def search_web(query: str, max_results: int = 10) -> list[WebSource]:
    """General web search via duckduckgo-search library."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.warning("duckduckgo-search not installed; skipping web search")
        return []

    sources: list[WebSource] = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        for r in results:
            sources.append(WebSource(
                url=r.get("href", r.get("link", "")),
                title=r.get("title", ""),
                snippet=r.get("body", r.get("snippet", "")),
                source_type="web",
            ))
    except Exception as exc:
        logger.warning("DuckDuckGo search failed: %s", exc)
    return sources


def search_all(query: str, max_results: int = 10) -> list[WebSource]:
    """Run PubMed and general web search, merge results."""
    pubmed = search_pubmed(query, max_results=max_results)
    web = search_web(query, max_results=max_results)

    seen_urls: set[str] = set()
    merged: list[WebSource] = []
    for src in pubmed + web:
        if src.url not in seen_urls:
            seen_urls.add(src.url)
            merged.append(src)
    return merged
