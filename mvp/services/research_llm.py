"""OpenAI-compatible chat completions for research agent (VseGPT / OpenAI)."""

from __future__ import annotations

import json
import logging
from typing import Any

from ..config import get_openai_api_key, get_openai_base_url, get_openai_model, SOCKS_PROXY
from ..models.research import CandidateMolecule, ResearchQuery

logger = logging.getLogger(__name__)


def _client():
    """Return OpenAI client or None if not configured."""
    key = get_openai_api_key()
    if not key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed; LLM disabled")
        return None
    kwargs: dict = dict(api_key=key, base_url=get_openai_base_url())
    if SOCKS_PROXY:
        import httpx
        transport = httpx.HTTPTransport(proxy=SOCKS_PROXY)
        kwargs["http_client"] = httpx.Client(transport=transport, timeout=120.0)
    return OpenAI(**kwargs)


def _chat_json(system: str, user: str) -> dict[str, Any] | None:
    """Call chat completions and parse JSON from assistant message."""
    client = _client()
    if client is None:
        return None
    model = get_openai_model()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        text = (resp.choices[0].message.content or "").strip()
        return json.loads(text)
    except Exception as exc:
        logger.warning("LLM chat failed: %s", exc)
        return None


def llm_formulate_search_queries(user_input: str) -> ResearchQuery | None:
    """Produce interpreted intent + English search queries for PubMed/web."""
    system = (
        "You are a chemistry/pharmacology assistant. The user describes what kind of "
        "compound they want (not a specific molecule name). "
        "Reply with a JSON object only, keys: "
        '"interpreted_intent" (string, short, English), '
        '"search_queries" (array of 4-8 distinct English search strings for PubMed and web; '
        "include drug names, mechanism, protein targets where relevant), "
        '"language" ("ru" or "en" — user\'s primary language).'
    )
    data = _chat_json(system, f"User request:\n{user_input}")
    if not data:
        return None
    try:
        intent = str(data.get("interpreted_intent", "")).strip()
        queries = data.get("search_queries")
        if not isinstance(queries, list) or not queries:
            return None
        queries = [str(q).strip() for q in queries if str(q).strip()]
        if not queries:
            return None
        lang = data.get("language", "en")
        if lang not in ("ru", "en"):
            lang = "en"
        return ResearchQuery(
            original_input=user_input,
            interpreted_intent=intent or f"Search for compounds matching: {user_input}",
            search_queries=queries,
            language=lang,
        )
    except Exception as exc:
        logger.warning("llm_formulate_search_queries parse error: %s", exc)
        return None


def llm_extract_molecule_names(text: str) -> list[str] | None:
    """Extract chemical/drug names from text; returns None on LLM failure."""
    if not text.strip():
        return []
    snippet = text[:12000]
    system = (
        "You extract names of chemical compounds, drugs, and small-molecule inhibitors "
        "mentioned in scientific text. "
        "Return JSON only: {\"names\": [\"...\", ...]} — use common English or international "
        "nonproprietary names when possible. Exclude proteins/genes as standalone targets "
        "unless they are also drug names. No duplicates."
    )
    data = _chat_json(system, f"Text:\n{snippet}")
    if not data:
        return None
    names = data.get("names")
    if not isinstance(names, list):
        return None
    out: list[str] = []
    seen: set[str] = set()
    for n in names:
        s = str(n).strip()
        if not s or s.lower() in seen:
            continue
        seen.add(s.lower())
        out.append(s)
    return out


def llm_build_summary(
    original_query: str,
    interpreted_intent: str,
    candidates: list[CandidateMolecule],
) -> str | None:
    """Short Russian/English summary for the user."""
    if not candidates:
        return None
    lines = []
    for c in candidates[:12]:
        lines.append(f"- {c.name} (PubChem CID {c.pubchem_cid})")
    system = (
        "You write a concise 2-4 sentence summary for a chemist. "
        "Use the same language as the user's original query (Russian or English). "
        "Explain what was inferred and list example molecules as candidates — they are "
        "resolved in PubChem. Reply with JSON only: {\"summary\": \"...\"}."
    )
    user = (
        f"Original query: {original_query}\n"
        f"Interpreted intent: {interpreted_intent}\n"
        f"Candidates:\n" + "\n".join(lines)
    )
    data = _chat_json(system, user)
    if not data:
        return None
    s = data.get("summary")
    if isinstance(s, str) and s.strip():
        return s.strip()
    return None
