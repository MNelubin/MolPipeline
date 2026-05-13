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


def llm_analyze_research_evidence(
    original_query: str,
    mode: str,
    interpreted_intent: str,
    evidence: list[dict[str, Any]],
    candidates: list[CandidateMolecule],
    rag_results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Analyze collected evidence into a structured research-agent report."""
    if not evidence and not candidates and not rag_results:
        return None

    source_lines: list[str] = []
    for idx, item in enumerate(evidence[:8], start=1):
        citation_id = item.get("citation_id") or f"S{idx}"
        source_lines.append(
            "\n".join([
                f"[{citation_id}] {item.get('title') or item.get('url') or 'Untitled source'}",
                f"Type: {item.get('source_type', 'web')}",
                f"URL: {item.get('url', '')}",
                f"Markdown citation: {item.get('citation_markdown') or ''}",
                f"Excerpt: {(item.get('excerpt') or item.get('snippet') or '')[:1200]}",
            ])
        )

    candidate_lines = [
        f"- {candidate.name}; CID={candidate.pubchem_cid}; SMILES={candidate.canonical_smiles or 'N/A'}"
        for candidate in candidates[:10]
    ]
    rag_lines = [
        "\n".join([
            f"[R{idx}] {item.get('title') or 'Local RAG result'}",
            f"Score: {item.get('score', '')}",
            f"Text: {(item.get('child_text') or item.get('parent_text') or '')[:1000]}",
        ])
        for idx, item in enumerate(rag_results[:5], start=1)
    ]

    system = (
        "You are a chemistry research analyst embedded in a molecule synthesis assistant. "
        "Use only the provided evidence, candidates and local RAG snippets. "
        "Do not invent citations. If evidence is weak, say so. "
        "If the user asks about supplementary information, a specific section, appendix, table, figure, or PDF, "
        "prioritize matching PDF/RAG excerpts over main article abstracts or general web summaries. "
        "When a claim is based on a web source, include the matching source marker like [S1] in the claim text. "
        "Reply with JSON only using this schema: "
        "{"
        "\"answer\": string, "
        "\"key_findings\": [{\"claim\": string, \"evidence\": [\"S1\"], \"confidence\": \"high|medium|low\"}], "
        "\"candidate_assessment\": [{\"name\": string, \"assessment\": string, \"confidence\": \"high|medium|low\"}], "
        "\"limitations\": [string], "
        "\"safety_notes\": [string], "
        "\"recommended_next_steps\": [string], "
        "\"source_quality\": string"
        "}. "
        "Use the same language as the user's query where possible."
    )
    user = (
        f"Original query: {original_query}\n"
        f"Mode: {mode}\n"
        f"Interpreted intent: {interpreted_intent}\n\n"
        "Evidence sources:\n"
        + ("\n\n".join(source_lines) or "No web evidence.")
        + "\n\nPubChem-resolved candidates:\n"
        + ("\n".join(candidate_lines) or "No candidates.")
        + "\n\nLocal RAG snippets:\n"
        + ("\n\n".join(rag_lines) or "No local RAG snippets.")
    )
    data = _chat_json(system, user)
    if not data:
        return None

    def _list(value: Any) -> list[Any]:
        return value if isinstance(value, list) else []

    return {
        "answer": str(data.get("answer") or "").strip(),
        "key_findings": _list(data.get("key_findings")),
        "candidate_assessment": _list(data.get("candidate_assessment")),
        "limitations": [str(item) for item in _list(data.get("limitations"))],
        "safety_notes": [str(item) for item in _list(data.get("safety_notes"))],
        "recommended_next_steps": [str(item) for item in _list(data.get("recommended_next_steps"))],
        "source_quality": str(data.get("source_quality") or "").strip(),
        "analysis_engine": "llm",
    }
