"""General chemistry chat orchestrator backed by MolPipeline tools.

The chat layer is intentionally tool-first: the LLM can later improve intent
classification and wording, but chemistry facts, safety decisions, routes,
ADMET and availability are always produced by deterministic project modules.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Literal

from .admet import analyze_admet
from .availability import check_reagent_availability, summarize_availability
from .config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, SOCKS_PROXY
from .nodes.validate_and_guard_node import _resolve_molecule, _run_safety_checks
from .research_workspace import run_research_workspace
from .tools.retro_tools import search_and_rank

logger = logging.getLogger(__name__)

CHEM_CHAT_MODEL = "deepseek/deepseek-v4-flash"
VALID_INTENTS: set[str] = {"general", "retrosynthesis", "admet", "availability", "research", "safety", "molecule", "mixed"}
VALID_TOOLS: set[str] = {"resolve_molecule", "safety_check", "retrosynthesis_search", "admet_screen", "availability_check", "research_analyze"}

Intent = Literal[
    "general",
    "retrosynthesis",
    "admet",
    "availability",
    "research",
    "safety",
    "molecule",
    "mixed",
]

COMMON_RU_MOLECULE_ALIASES = {
    "аспирин": "aspirin",
    "аспирина": "aspirin",
    "кофеин": "caffeine",
    "кофеина": "caffeine",
    "этанол": "ethanol",
    "этанола": "ethanol",
    "дофамин": "dopamine",
    "дофамина": "dopamine",
    "кокаин": "cocaine",
    "кокаина": "cocaine",
    "никотин": "nicotine",
    "никотина": "nicotine",
    "бензальдегид": "benzaldehyde",
    "бензальдегида": "benzaldehyde",
}


@dataclass(frozen=True)
class ChemToolSpec:
    name: str
    description: str
    runner: Callable[..., dict[str, Any]]
    requires_safety_gate: bool = False


def _attach_procedure_steps(routes: list[dict[str, Any]]) -> None:
    from .procedure_inference import format_procedure_russian

    for route in routes:
        route["procedure_steps_ru"] = format_procedure_russian(route)


def _resolve_tool(query: str) -> dict[str, Any]:
    resolved = _resolve_molecule(query)
    validation = resolved.get("validation", {})
    if not validation.get("is_valid"):
        return {
            "status": "not_found",
            "query": query,
            "validation": validation,
            "error": validation.get("error") or "Molecule could not be resolved.",
        }
    return {
        "status": "ok",
        "query": query,
        "smiles": resolved.get("smiles", ""),
        "pubchem_cid": resolved.get("pubchem_cid"),
        "validation": validation,
    }


def _candidate_molecule_queries(message: str) -> list[str]:
    import re

    text = message.strip()
    candidates: list[str] = []

    lowered = text.casefold()
    for alias, canonical in COMMON_RU_MOLECULE_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            candidates.append(canonical)

    for match in re.findall(r"[`\"']([^`\"']{2,120})[`\"']", text):
        candidates.append(match.strip())

    marker_pattern = (
        r"(?:для|по|про|вещества|молекулы|соединения|синтез|ретросинтез|"
        r"получить|получения|маршрут|admet|доступность|поставщики)\s+"
        r"([A-Za-zА-Яа-яЁё0-9@\+\-\[\]\(\)\\/=#$%.:,\s]{2,120})"
    )
    for match in re.findall(marker_pattern, text, flags=re.IGNORECASE):
        candidates.append(match.strip())

    smiles_like = re.findall(r"[A-Za-z0-9@\+\-\[\]\(\)\\/=#$%.:]{3,120}", text)
    candidates.extend(smiles_like)

    words = re.findall(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9\-]{2,}", text)
    candidates.extend(words[-4:])
    candidates.append(text)

    normalized: list[str] = []
    seen: set[str] = set()
    stop = {
        "найди", "путь", "маршрут", "синтез", "ретросинтез", "проверь",
        "для", "мне", "сделай", "admet", "поставщики", "доступность",
    }
    for candidate in candidates:
        clean = re.sub(r"[?!.]+$", "", candidate).strip(" ,;:")
        if not clean:
            continue
        low = clean.casefold()
        if low in stop:
            continue
        clean = COMMON_RU_MOLECULE_ALIASES.get(low, clean)
        key = clean.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(clean)
    return normalized


def _resolve_from_message(message: str) -> dict[str, Any]:
    errors: list[str] = []
    for candidate in _candidate_molecule_queries(message):
        result = _resolve_tool(candidate)
        if result.get("status") == "ok":
            result["query_used"] = candidate
            return result
        if result.get("error"):
            errors.append(f"{candidate}: {result['error']}")
    return {
        "status": "not_found",
        "query": message,
        "validation": {"is_valid": False, "input_type": "freeform", "error": errors[:3]},
        "error": "Не удалось выделить и распознать целевую молекулу из сообщения.",
    }


def _safety_tool(smiles: str, cid: int | None = None, reaction_description: str = "") -> dict[str, Any]:
    return _run_safety_checks(smiles=smiles, cid=cid, reaction_description=reaction_description)


def _retro_tool(smiles: str, top_n: int = 5, source_mode: str = "auto") -> dict[str, Any]:
    result = search_and_rank(smiles, top_n=top_n, source_mode=source_mode)
    _attach_procedure_steps(result.get("routes", []))
    return result


def _admet_tool(smiles: str, safety_guard: dict[str, Any] | None = None) -> dict[str, Any]:
    return analyze_admet(smiles, safety_guard=safety_guard)


def _split_availability_query(query: str) -> list[str]:
    import re

    parts: list[str] = []
    for token in re.split(r"[\n;,]+", query):
        token = token.strip()
        if not token:
            continue
        if "." in token and not any(ch.isspace() for ch in token):
            parts.extend(part.strip() for part in token.split(".") if part.strip())
        else:
            lowered = token.casefold()
            alias_matches: list[tuple[int, str]] = []
            for alias, canonical in COMMON_RU_MOLECULE_ALIASES.items():
                match = re.search(rf"\b{re.escape(alias)}\b", lowered)
                if match:
                    alias_matches.append((match.start(), canonical))
            if alias_matches:
                parts.extend(canonical for _, canonical in sorted(alias_matches))
            else:
                parts.append(token)

    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        key = part.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(part)
    return deduped


def _availability_tool(query: str) -> dict[str, Any]:
    items = _split_availability_query(query)
    results: list[dict[str, Any]] = []
    for item in items:
        resolved = _resolve_tool(item)
        if resolved.get("status") != "ok":
            for candidate in _candidate_molecule_queries(item):
                if candidate.casefold() == item.casefold():
                    continue
                resolved = _resolve_tool(candidate)
                if resolved.get("status") == "ok":
                    item = candidate
                    break
        if resolved.get("status") != "ok":
            results.append({
                "input": item,
                "label": item,
                "available": False,
                "availability_level": "invalid",
                "basis": resolved.get("error") or "Не удалось распознать молекулу.",
            })
            continue
        results.append(
            check_reagent_availability(
                resolved["smiles"],
                label=item,
                input_value=item,
                resolution=resolved.get("validation", {}).get("input_type") or "resolved",
            )
        )
    return {"query": query, "items": results, "summary": summarize_availability(results)}


def _research_tool(query: str, mode: str = "literature", max_sources: int = 6) -> dict[str, Any]:
    return run_research_workspace(query, mode=mode, max_sources=max_sources)


def _openrouter_client():
    if not OPENROUTER_API_KEY:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package is not installed; ChemChat LLM disabled")
        return None

    kwargs: dict[str, Any] = {
        "api_key": OPENROUTER_API_KEY,
        "base_url": OPENROUTER_BASE_URL,
    }
    if SOCKS_PROXY:
        import httpx

        transport = httpx.HTTPTransport(proxy=SOCKS_PROXY)
        kwargs["http_client"] = httpx.Client(transport=transport, timeout=120.0)
    return OpenAI(**kwargs)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _chat_llm_json(system: str, user: str, *, max_tokens: int = 1200) -> dict[str, Any] | None:
    client = _openrouter_client()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=CHEM_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        text = (resp.choices[0].message.content or "").strip()
        return _extract_json_object(text)
    except Exception as exc:
        logger.warning("ChemChat LLM call failed: %s", exc)
        return None


def _normalize_llm_plan(plan: dict[str, Any] | None, message: str) -> dict[str, Any]:
    fallback_intent = classify_chem_intent(message)
    if not isinstance(plan, dict):
        plan = {}

    intent = str(plan.get("intent") or fallback_intent).strip()
    if intent not in VALID_INTENTS:
        intent = fallback_intent

    raw_tools = plan.get("tools")
    tools = [str(tool).strip() for tool in raw_tools] if isinstance(raw_tools, list) else []
    tools = [tool for tool in tools if tool in VALID_TOOLS]
    if not tools:
        if intent == "general":
            tools = ["research_analyze"]
        elif intent == "retrosynthesis":
            tools = ["resolve_molecule", "safety_check", "retrosynthesis_search"]
        elif intent == "admet":
            tools = ["resolve_molecule", "safety_check", "admet_screen"]
        elif intent == "availability":
            tools = ["availability_check"]
        elif intent == "research":
            tools = ["research_analyze"]
        elif intent == "safety":
            tools = ["resolve_molecule", "safety_check"]
        elif intent == "molecule":
            tools = ["resolve_molecule", "safety_check"]
        else:
            tools = ["resolve_molecule", "safety_check", "retrosynthesis_search", "availability_check", "admet_screen"]

    # Safety is a hard gate for molecule-specific tools.
    gated_tools = {"retrosynthesis_search", "admet_screen"}
    if any(tool in tools for tool in gated_tools):
        for required in ("resolve_molecule", "safety_check"):
            if required not in tools:
                tools.insert(0 if required == "resolve_molecule" else 1, required)

    targets = plan.get("target_molecules")
    if not isinstance(targets, list):
        targets = []
    targets = [str(target).strip() for target in targets if str(target).strip()]

    source_mode = str(plan.get("source_mode") or "auto").strip()
    if source_mode not in {"auto", "ord", "web", "retro_model", "aizynthfinder", "all"}:
        source_mode = "auto"

    research_mode = str(plan.get("research_mode") or "literature").strip()
    if research_mode not in {"molecule", "literature", "patent"}:
        research_mode = "literature"

    return {
        "intent": intent,
        "tools": tools,
        "target_molecules": targets,
        "source_mode": source_mode,
        "research_mode": research_mode,
        "reasoning": str(plan.get("reasoning") or "").strip(),
        "used_llm": isinstance(plan, dict) and bool(plan),
    }


def _plan_with_llm(message: str, source_mode: str, research_mode: str) -> dict[str, Any]:
    system = (
        "You are the planner for MolPipeline ChemChat, a chemistry-specific assistant. "
        "Choose which project tools must be called before answering. "
        "Do not answer chemistry facts directly in this planning step. "
        "Return JSON only with keys: intent, target_molecules, tools, source_mode, research_mode, reasoning. "
        "intent must be one of: general, molecule, safety, retrosynthesis, availability, admet, research, mixed. "
        "tools must be selected from: resolve_molecule, safety_check, retrosynthesis_search, "
        "availability_check, admet_screen, research_analyze. "
        "For synthesis/route/retrosynthesis requests include resolve_molecule, safety_check, retrosynthesis_search. "
        "For general chemistry questions that do not need a concrete molecule, use research_analyze only. "
        "For supplier/price/buyability requests use availability_check and extract every reagent if possible. "
        "For safety/ADMET requests include resolve_molecule and safety_check. "
        "target_molecules should contain English/common molecule names or SMILES extracted from the user text."
    )
    user = json.dumps(
        {
            "message": message,
            "default_source_mode": source_mode,
            "default_research_mode": research_mode,
        },
        ensure_ascii=False,
    )
    plan = _chat_llm_json(system, user, max_tokens=900)
    normalized = _normalize_llm_plan(plan, message)
    normalized["llm_raw_plan"] = plan or None
    return normalized


def _compact_artifacts_for_llm(artifacts: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    molecule = artifacts.get("molecule")
    if molecule:
        validation = molecule.get("validation") or {}
        compact["molecule"] = {
            "status": molecule.get("status"),
            "query_used": molecule.get("query_used") or molecule.get("query"),
            "smiles": molecule.get("smiles"),
            "formula": validation.get("molecular_formula"),
            "molecular_weight": validation.get("molecular_weight"),
            "pubchem_cid": molecule.get("pubchem_cid"),
            "error": molecule.get("error"),
        }
    safety = artifacts.get("safety")
    if safety:
        compact["safety"] = {
            "overall_status": safety.get("overall_status"),
            "molecule_status": (safety.get("molecule_check") or {}).get("status"),
            "reason": (safety.get("molecule_check") or {}).get("reason")
            or (safety.get("reaction_check") or {}).get("reason"),
            "h_phrases": (safety.get("safety_data") or {}).get("h_phrases", [])[:6],
        }
    retro = artifacts.get("retrosynthesis")
    if retro:
        routes = retro.get("routes") or []
        best = routes[0] if routes else {}
        compact["retrosynthesis"] = {
            "total_found": retro.get("total_found"),
            "total_unique": retro.get("total_unique"),
            "sources_used": retro.get("sources_used"),
            "source_errors": retro.get("source_errors"),
            "best_route": {
                "source": best.get("source_label") or best.get("source"),
                "reactants": best.get("reactants"),
                "score": best.get("final_score"),
                "availability_summary": best.get("availability_summary"),
            } if best else None,
        }
    admet = artifacts.get("admet")
    if admet:
        compact["admet"] = {
            "overall": admet.get("overall"),
            "safety_overlay": admet.get("safety_overlay"),
        }
    availability = artifacts.get("availability")
    if availability:
        compact["availability"] = {
            "summary": availability.get("summary"),
            "items": [
                {
                    "label": item.get("label") or item.get("input"),
                    "smiles": item.get("smiles"),
                    "available": item.get("available"),
                    "level": item.get("availability_level"),
                    "source": item.get("source_label") or item.get("source"),
                    "ppg": item.get("ppg"),
                }
                for item in (availability.get("items") or [])[:8]
            ],
        }
    research = artifacts.get("research")
    if research:
        compact["research"] = {
            "summary": research.get("summary"),
            "analysis": research.get("analysis"),
            "sources_count": len(research.get("sources") or []),
            "evidence_count": len(research.get("evidence") or []),
        }
    return compact


def _final_answer_with_llm(message: str, plan: dict[str, Any], artifacts: dict[str, Any], fallback: str) -> str:
    system = (
        "You are MolPipeline ChemChat running on deepseek/deepseek-v4-flash. "
        "Answer in the user's language, usually Russian. "
        "Use ONLY the provided tool outputs for molecule data, safety, retrosynthesis, ADMET, availability and sources. "
        "Do not invent synthesis routes, prices, safety classifications or citations. "
        "If tools did not find enough data, say exactly what is missing and what to try next. "
        "Keep the answer concise but useful for a chemist."
    )
    user = json.dumps(
        {
            "user_message": message,
            "plan": {k: v for k, v in plan.items() if k != "llm_raw_plan"},
            "tool_outputs": _compact_artifacts_for_llm(artifacts),
            "fallback_summary": fallback,
        },
        ensure_ascii=False,
    )
    data = _chat_llm_json(
        system,
        user + "\nReturn JSON only: {\"answer\": string, \"suggested_next_actions\": [string, ...]}.",
        max_tokens=1600,
    )
    if not data:
        return fallback
    answer = data.get("answer")
    return answer.strip() if isinstance(answer, str) and answer.strip() else fallback


TOOL_REGISTRY: dict[str, ChemToolSpec] = {
    "resolve_molecule": ChemToolSpec(
        name="resolve_molecule",
        description="Resolve name, SMILES or CAS-like query to canonical molecule metadata.",
        runner=_resolve_tool,
    ),
    "safety_check": ChemToolSpec(
        name="safety_check",
        description="Run banlist, GHS and PPE safety checks.",
        runner=_safety_tool,
    ),
    "retrosynthesis_search": ChemToolSpec(
        name="retrosynthesis_search",
        description="Run MolPipeline retrosynthesis across ORD, web, local model and external planners.",
        runner=_retro_tool,
        requires_safety_gate=True,
    ),
    "admet_screen": ChemToolSpec(
        name="admet_screen",
        description="Run descriptor ADMET screening with safety overlay.",
        runner=_admet_tool,
        requires_safety_gate=True,
    ),
    "availability_check": ChemToolSpec(
        name="availability_check",
        description="Check local buyables and supplier hints for reagents.",
        runner=_availability_tool,
    ),
    "research_analyze": ChemToolSpec(
        name="research_analyze",
        description="Collect literature, patent and open-source evidence.",
        runner=_research_tool,
    ),
}


def classify_chem_intent(message: str) -> Intent:
    text = message.casefold()
    retro_words = (
        "ретросинтез", "синтез", "маршрут", "путь", "получить", "получи",
        "synthesis", "retrosynthesis", "route", "pathway",
    )
    admet_words = ("admet", "фармако", "абсорб", "токсикокин", "drug", "bbb", "lipinski")
    availability_words = ("постав", "купить", "цена", "стоимость", "доступ", "supplier", "buy", "price")
    research_words = ("литератур", "патент", "стать", "исслед", "pubmed", "patent", "paper", "evidence")
    safety_words = (
        "безопас", "опас", "ghs", "сиз", "ppe", "ban", "запрещ",
        "safety", "hazard", "risk", "toxic", "toxicity",
    )
    molecule_words = ("молекул", "веществ", "соединен", "smiles", "cas", "формул", "масса")

    hits = {
        "retrosynthesis": any(word in text for word in retro_words),
        "admet": any(word in text for word in admet_words),
        "availability": any(word in text for word in availability_words),
        "research": any(word in text for word in research_words),
        "safety": any(word in text for word in safety_words),
    }
    active = [name for name, hit in hits.items() if hit]
    if len(active) > 1:
        return "mixed"
    if active:
        return active[0]  # type: ignore[return-value]
    if any(word in text for word in molecule_words):
        return "molecule"
    return "general"


def _route_summary(retro: dict[str, Any]) -> list[str]:
    routes = retro.get("routes") or []
    if not routes:
        errors = retro.get("source_errors") or {}
        if errors:
            return ["Маршруты не найдены; часть источников вернула ошибки."]
        return ["Маршруты ретросинтеза не найдены."]

    best = routes[0]
    summary = [
        f"Найдено маршрутов: {len(routes)}; уникальных: {retro.get('total_unique', len(routes))}.",
        f"Лучший источник: {best.get('source_label') or best.get('source') or 'не указан'}.",
    ]
    availability = best.get("availability_summary") or {}
    if availability:
        summary.append(
            "Доступность реагентов: "
            f"{availability.get('available_count', 0)}/{availability.get('total', 0)}, "
            f"ориентир на 1 г: {availability.get('estimated_total_1g_usd', 'n/a')} USD."
        )
    reactants = best.get("reactants")
    if reactants:
        summary.append(f"Первый вариант исходников: {reactants}.")
    return summary


def _admet_summary(admet: dict[str, Any]) -> list[str]:
    overall = admet.get("overall", {})
    return [
        f"ADMET score: {overall.get('score', 'n/a')}/100.",
        f"Уровень риска: {overall.get('risk_level', 'unknown')}.",
    ]


def _availability_summary(availability: dict[str, Any]) -> list[str]:
    summary = availability.get("summary") or {}
    return [
        f"Проверено реагентов: {summary.get('total', 0)}.",
        f"Доступно: {summary.get('available_count', 0)}.",
        f"С ценами: {summary.get('priced_count', 0)}.",
    ]


def _research_summary(research: dict[str, Any]) -> list[str]:
    summary = research.get("summary")
    if summary:
        return [summary]
    return [
        f"Найдено источников: {len(research.get('sources') or [])}.",
        f"Evidence-блоков: {len(research.get('evidence') or [])}.",
    ]


def run_chem_chat(
    message: str,
    *,
    source_mode: str = "auto",
    top_n: int = 5,
    research_mode: str = "literature",
    max_sources: int = 6,
) -> dict[str, Any]:
    query = message.strip()
    if not query:
        raise ValueError("message must not be empty")

    plan = _plan_with_llm(query, source_mode=source_mode, research_mode=research_mode)
    intent = plan["intent"]
    selected_tools = set(plan["tools"])
    selected_source_mode = plan.get("source_mode") or source_mode
    selected_research_mode = plan.get("research_mode") or research_mode
    tools_used: list[str] = []
    artifacts: dict[str, Any] = {}
    answer_lines: list[str] = []

    resolved: dict[str, Any] = {"status": "skipped", "query": query}
    resolved_ok = False
    should_resolve = "resolve_molecule" in selected_tools
    if should_resolve:
        targets = plan.get("target_molecules") or []
        resolved = _resolve_from_message(str(targets[0])) if targets else _resolve_from_message(query)
        tools_used.append("resolve_molecule")
        artifacts["molecule"] = resolved
        resolved_ok = resolved.get("status") == "ok"
    smiles = resolved.get("smiles") if resolved_ok else None
    cid = resolved.get("pubchem_cid") if resolved_ok else None

    safety_guard: dict[str, Any] | None = None
    if resolved_ok:
        safety_guard = _safety_tool(smiles, cid=cid, reaction_description=query)
        tools_used.append("safety_check")
        artifacts["safety"] = safety_guard

        status = safety_guard.get("overall_status", "SAFE")
        query_used = resolved.get("query_used") or resolved.get("query") or query
        answer_lines.append(f"Целевая молекула: {query_used}; SMILES `{smiles}`. Safety gate: {status}.")
        if status == "CRITICAL_STOP":
            reason = (
                safety_guard.get("molecule_check", {}).get("reason")
                or safety_guard.get("reaction_check", {}).get("reason")
                or "критический safety-stop"
            )
            answer_lines.append(f"Дальнейший синтетический сценарий заблокирован: {reason}")
    elif should_resolve and intent != "availability":
        answer_lines.append(resolved.get("error") or "Не удалось распознать молекулу через PubChem/RDKit.")

    wants_retro = "retrosynthesis_search" in selected_tools
    wants_admet = "admet_screen" in selected_tools
    wants_availability = "availability_check" in selected_tools
    wants_research = "research_analyze" in selected_tools or (should_resolve and not resolved_ok)
    wants_safety_only = intent == "safety" or selected_tools == {"resolve_molecule", "safety_check"}

    if resolved_ok and wants_retro and safety_guard and safety_guard.get("overall_status") != "CRITICAL_STOP":
        retro = _retro_tool(smiles, top_n=top_n, source_mode=selected_source_mode)
        tools_used.append("retrosynthesis_search")
        artifacts["retrosynthesis"] = retro
        answer_lines.extend(_route_summary(retro))

    if resolved_ok and wants_admet:
        admet = _admet_tool(smiles, safety_guard=safety_guard)
        tools_used.append("admet_screen")
        artifacts["admet"] = admet
        answer_lines.extend(_admet_summary(admet))

    if wants_availability:
        availability = _availability_tool(query)
        tools_used.append("availability_check")
        artifacts["availability"] = availability
        answer_lines.extend(_availability_summary(availability))

    if wants_research:
        research = _research_tool(query, mode=selected_research_mode, max_sources=max_sources)
        tools_used.append("research_analyze")
        artifacts["research"] = research
        if intent == "general":
            answer_lines.append("Вопрос не требует целевой молекулы, поэтому использую общий химический research-режим.")
        answer_lines.extend(_research_summary(research))

    if resolved_ok and wants_safety_only and safety_guard:
        mol_check = safety_guard.get("molecule_check", {})
        h_phrases = safety_guard.get("safety_data", {}).get("h_phrases") or []
        answer_lines.append(mol_check.get("reason") or "Критичных banlist-флагов не найдено.")
        if h_phrases:
            answer_lines.append(f"GHS-фразы: {', '.join(h_phrases[:5])}.")

    if intent == "molecule" and resolved_ok:
        validation = resolved.get("validation") or {}
        answer_lines.append(
            "Базовые данные: "
            f"формула {validation.get('molecular_formula', 'n/a')}, "
            f"молекулярная масса {validation.get('molecular_weight', 'n/a')}."
        )
        answer_lines.append(
            "Могу продолжить в режим ретросинтеза, ADMET, поставщиков или исследования по этой молекуле."
        )

    suggestions = [
        "Построить ретросинтез и сравнить маршруты",
        "Проверить доступность исходных реагентов",
        "Сделать ADMET и safety-разбор",
    ]
    if artifacts.get("retrosynthesis", {}).get("routes"):
        suggestions.insert(0, "Выбрать лучший маршрут и посчитать масштаб")

    fallback_answer = "\n".join(answer_lines)
    answer = _final_answer_with_llm(query, plan, artifacts, fallback_answer)

    return {
        "status": "ok",
        "intent": intent,
        "model": CHEM_CHAT_MODEL,
        "plan": plan,
        "answer": answer,
        "tools_used": tools_used,
        "artifacts": artifacts,
        "suggested_next_actions": suggestions[:4],
    }
