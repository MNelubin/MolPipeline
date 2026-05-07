"""General chemistry chat orchestrator backed by MolPipeline tools.

The chat layer is intentionally tool-first: the LLM can later improve intent
classification and wording, but chemistry facts, safety decisions, routes,
ADMET and availability are always produced by deterministic project modules.
"""

from __future__ import annotations

import json
import logging
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Literal

from .admet import analyze_admet
from .availability import check_reagent_availability, summarize_availability
from .config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, SOCKS_PROXY
from .nodes.validate_and_guard_node import _resolve_molecule, _run_safety_checks
from .research_workspace import run_research_workspace
from .tools.retro_tools import search_and_rank
from .tree_expansion import expand_tree

logger = logging.getLogger(__name__)

CHEM_CHAT_MODEL = "deepseek/deepseek-v4-flash"
VALID_INTENTS: set[str] = {"general", "retrosynthesis", "admet", "availability", "research", "safety", "molecule", "mixed"}
VALID_TOOLS: set[str] = {"resolve_molecule", "safety_check", "retrosynthesis_search", "admet_screen", "availability_check", "research_analyze"}
VALID_RETRO_DEPTH_MODES: set[str] = {"one_step", "multi_step"}

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


def _multi_step_tree_tool(
    target_smiles: str,
    route: dict[str, Any],
    *,
    max_depth: int = 6,
    timeout_sec: float = 45.0,
) -> dict[str, Any]:
    reactants = route.get("reactants") or ""
    if not reactants:
        return {
            "status": "skipped",
            "reason": "Selected route has no reactants to expand.",
            "tree": None,
            "stats": {},
        }
    result = expand_tree(target_smiles, reactants, max_depth=max_depth, timeout_sec=timeout_sec)
    result["status"] = "ok"
    result["selected_route"] = {
        "source": route.get("source_label") or route.get("source"),
        "reactants": reactants,
        "score": route.get("final_score"),
    }
    result["max_depth"] = max_depth
    result["timeout_sec"] = timeout_sec
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


def _last_substantive_user_message(history: list[dict[str, str]]) -> str:
    for item in reversed(history):
        if item.get("role") != "user":
            continue
        content = (item.get("content") or "").strip()
        if content and not _is_source_followup_request(content):
            return content
    return ""


def _is_source_followup_request(message: str) -> bool:
    text = message.casefold()
    return any(marker in text for marker in (
        "со ссылк", "ссылки", "источник", "источники", "sources", "citations", "references",
    )) and len(text) < 180


def _contextual_research_query(message: str, history: list[dict[str, str]]) -> str:
    previous = _last_substantive_user_message(history)
    if not previous:
        return message
    if _is_source_followup_request(message) or _looks_context_dependent(message):
        return (
            f"{previous}\n\n"
            f"Follow-up request: {message}\n"
            "Find sources for the previous chemistry topic, not for citation formatting itself."
        )
    return message


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


def _compact_chat_history(history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    compact: list[dict[str, str]] = []
    for item in history or []:
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        compact.append({"role": role, "content": content[:2000]})
    return compact[-8:]


def _normalize_llm_plan(
    plan: dict[str, Any] | None,
    message: str,
    default_source_mode: str = "auto",
    context: str | None = None,
) -> dict[str, Any]:
    planning_context = context or message
    fallback_intent = classify_chem_intent(message)
    if not isinstance(plan, dict):
        plan = {}

    intent = str(plan.get("intent") or fallback_intent).strip()
    if intent not in VALID_INTENTS:
        intent = fallback_intent

    source_mode = str(plan.get("source_mode") or default_source_mode or "auto").strip()
    if source_mode not in {"auto", "ord", "web", "retro_model", "aizynthfinder", "all"}:
        source_mode = "auto"

    raw_tools = plan.get("tools")
    tools = [str(tool).strip() for tool in raw_tools] if isinstance(raw_tools, list) else []
    tools = [tool for tool in tools if tool in VALID_TOOLS]

    # Keep the model as planner, but reconcile impossible plans where intent and
    # tools disagree. Otherwise a follow-up like "now check ADMET" can be routed
    # to research without ever calling the ADMET pipeline.
    if intent == "retrosynthesis" and "retrosynthesis_search" not in tools:
        tools.append("retrosynthesis_search")
    if intent == "admet" and "admet_screen" not in tools:
        tools.append("admet_screen")
    if intent in {"safety", "molecule"} and "safety_check" not in tools:
        tools.append("safety_check")
    if any(tool in tools for tool in {"retrosynthesis_search", "admet_screen", "safety_check"}) and "resolve_molecule" not in tools:
        tools.insert(0, "resolve_molecule")
    if intent in {"retrosynthesis", "admet", "safety", "molecule"} and not _mentions_external_evidence(message):
        tools = [tool for tool in tools if tool != "research_analyze"]

    if (
        intent == "general"
        and tools == ["research_analyze"]
        and source_mode == "auto"
        and _is_broad_educational_question(planning_context)
    ):
        tools = []
    if intent == "general" and not tools and source_mode in {"web", "all"}:
        tools = ["research_analyze"]
    if intent == "general" and not tools and _looks_like_real_world_composition_question(planning_context):
        tools = ["research_analyze"]
    if not tools:
        if intent == "general":
            tools = []
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

    research_mode = str(plan.get("research_mode") or "literature").strip()
    if research_mode not in {"molecule", "literature", "patent"}:
        research_mode = "literature"

    retro_depth_mode = _normalize_retrosynthesis_depth_mode(
        plan.get("retrosynthesis_depth_mode"),
        message=message,
        context=planning_context,
        source_mode=source_mode,
    )

    return {
        "intent": intent,
        "tools": tools,
        "target_molecules": targets,
        "source_mode": source_mode,
        "research_mode": research_mode,
        "retrosynthesis_depth_mode": retro_depth_mode,
        "reasoning": str(plan.get("reasoning") or "").strip(),
        "used_llm": isinstance(plan, dict) and bool(plan),
    }


def _normalize_retrosynthesis_depth_mode(
    raw_mode: Any,
    *,
    message: str,
    context: str,
    source_mode: str,
) -> str:
    value = str(raw_mode or "").strip().lower().replace("-", "_")
    aliases = {
        "one": "one_step",
        "single": "one_step",
        "single_step": "one_step",
        "onestep": "one_step",
        "one_step": "one_step",
        "multi": "multi_step",
        "multistep": "multi_step",
        "multi_step": "multi_step",
        "tree": "multi_step",
        "full": "multi_step",
    }
    if value in aliases:
        return aliases[value]

    text = f"{context}\n{message}".casefold()
    multi_markers = (
        "multi-step", "multistep", "full route", "route tree", "synthetic tree",
        "from buyable", "from purchasable", "starting materials", "complete route",
        "многостадий", "многошаг", "полный маршрут", "полную схему", "дерево",
        "от доступных", "из доступных", "до доступных", "исходных реагентов",
        "все стадии", "цепочку", "полный путь", "план синтеза",
    )
    one_step_markers = (
        "one-step", "onestep", "single-step", "first disconnection",
        "direct precursors", "precursors only", "одношаг", "одностадий",
        "первый шаг", "первую реакцию", "предшественники", "только реакцию",
        "одно разбиение", "disconnection",
    )
    if any(marker in text for marker in one_step_markers):
        return "one_step"
    if any(marker in text for marker in multi_markers):
        return "multi_step"
    if source_mode == "aizynthfinder":
        return "multi_step"
    return "one_step"


def _is_broad_educational_question(message: str) -> bool:
    text = message.casefold()
    broad_markers = (
        "что такое", "расскажи про", "объясни", "какие самые", "в чем разница",
        "what is", "explain", "overview", "basics",
    )
    return (
        any(marker in text for marker in broad_markers)
        and not _mentions_external_evidence(message)
        and not _looks_like_real_world_composition_question(message)
    )


def _looks_like_real_world_composition_question(message: str) -> bool:
    text = message.casefold()
    composition_markers = (
        "из чего", "состав", "состоят", "состоит", "что используют", "используется",
        "made of", "made from", "composition", "what are", "what is used",
    )
    material_markers = (
        "керами", "фарфор", "фаянс", "стекл", "глазур", "глина", "каолин", "кварц",
        "полевой шпат", "пластик", "полимер", "металл", "сплав", "бетон", "цемент",
        "ceramic", "porcelain", "glass", "glaze", "clay", "kaolin", "quartz",
        "feldspar", "plastic", "polymer", "metal", "alloy", "concrete", "cement",
    )
    product_markers = (
        "кружк", "чашк", "посуда", "плитк", "упаковк", "бутылк", "краск", "клей",
        "mug", "cup", "tableware", "tile", "packaging", "bottle", "paint", "adhesive",
    )
    has_composition = any(marker in text for marker in composition_markers)
    has_material_context = any(marker in text for marker in material_markers + product_markers)
    return has_composition and has_material_context


def _looks_context_dependent(message: str) -> bool:
    text = message.casefold().strip()
    markers = (
        "это", "этот", "эта", "эту", "его", "ее", "её", "она", "он", "они", "котор",
        "а теперь", "теперь", "дальше", "следующий", "там", "в ней", "в нем",
        "ссылк", "источник", "источники", "подробнее",
        "it", "that", "this", "they", "them", "now", "next", "sources", "citations",
    )
    return len(text) < 120 and any(marker in text for marker in markers)


def _mentions_external_evidence(message: str) -> bool:
    text = message.casefold()
    research_markers = (
        "источник", "источники", "ссылка", "ссылки", "литература", "статья", "статьи",
        "web", "pubmed", "patent", "paper", "evidence", "source", "sources", "citation",
    )
    return any(marker in text for marker in research_markers)


def _plan_with_llm(
    message: str,
    source_mode: str,
    research_mode: str,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    system = (
        "You are the planner for MolPipeline ChemChat, a chemistry-specific assistant. "
        "Choose which project tools must be called before answering. "
        "Do not answer chemistry facts directly in this planning step. "
        "Use conversation_history to resolve follow-up questions, but current user message has priority. "
        "Return JSON only with keys: intent, target_molecules, tools, source_mode, research_mode, retrosynthesis_depth_mode, reasoning. "
        "intent must be one of: general, molecule, safety, retrosynthesis, availability, admet, research, mixed. "
        "tools must be selected from: resolve_molecule, safety_check, retrosynthesis_search, "
        "availability_check, admet_screen, research_analyze. "
        "For broad educational questions such as definitions, basic concepts, or simple comparisons, usually use an empty tools list. "
        "Use research_analyze when the user asks for sources, literature, web evidence, PubMed, papers, patents, current/external data, or selects a web/all source mode. "
        "Also use research_analyze for real-world product/material composition questions, industrial formulations, consumer goods, and 'what is this usually made of/used in' questions, because those require external/web evidence rather than pure textbook recall. "
        "For synthesis/route/retrosynthesis requests include resolve_molecule, safety_check, retrosynthesis_search. "
        "For retrosynthesis_depth_mode use one_step for direct precursor/disconnection requests and multi_step for full route/tree/from-buyable-starting-materials requests. "
        "For supplier/price/buyability requests use availability_check and extract every reagent if possible. "
        "For safety/ADMET requests include resolve_molecule and safety_check. "
        "target_molecules should contain English/common molecule names or SMILES extracted from the user text."
    )
    user = json.dumps(
        {
            "message": message,
            "conversation_history": _compact_chat_history(history),
            "default_source_mode": source_mode,
            "default_research_mode": research_mode,
            "default_retrosynthesis_depth_mode": "one_step",
        },
        ensure_ascii=False,
    )
    plan = _chat_llm_json(system, user, max_tokens=900)
    history_context = "\n".join(item["content"] for item in _compact_chat_history(history))
    planning_context = f"{history_context}\n{message}" if history_context and _looks_context_dependent(message) else message
    normalized = _normalize_llm_plan(
        plan,
        message,
        default_source_mode=source_mode,
        context=planning_context,
    )
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
        taxonomy = safety.get("safety_taxonomy") or {}
        compact["safety"] = {
            "overall_status": safety.get("overall_status"),
            "taxonomy_status": taxonomy.get("status"),
            "taxonomy_categories": [
                {
                    "type": item.get("hazard_type"),
                    "status": item.get("status"),
                    "level": item.get("danger_level"),
                    "reason": item.get("reason"),
                    "h_codes": item.get("h_codes"),
                }
                for item in (taxonomy.get("categories") or [])[:8]
            ],
            "molecule_status": (safety.get("molecule_check") or {}).get("status"),
            "explosive_status": (safety.get("explosive_check") or {}).get("status"),
            "explosive_reason": (safety.get("explosive_check") or {}).get("reason"),
            "reason": (((taxonomy.get("blocked_categories") or taxonomy.get("warning_categories") or [{}])[0]).get("reason"))
            or (safety.get("explosive_check") or {}).get("reason")
            or (safety.get("molecule_check") or {}).get("reason")
            or (safety.get("reaction_check") or {}).get("reason"),
            "h_phrases": (safety.get("safety_data") or {}).get("h_phrases", [])[:6],
        }
    retro = artifacts.get("retrosynthesis")
    if retro:
        routes = retro.get("routes") or []
        best = routes[0] if routes else {}
        compact["retrosynthesis"] = {
            "depth_mode": retro.get("depth_mode"),
            "total_found": retro.get("total_found"),
            "total_unique": retro.get("total_unique"),
            "sources_used": retro.get("sources_used"),
            "source_errors": retro.get("source_errors"),
            "tree_stats": (retro.get("multi_step_tree") or {}).get("stats"),
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
            "sources": [
                {
                    "citation_id": source.get("citation_id"),
                    "title": source.get("title") or source.get("name") or source.get("url"),
                    "url": source.get("url"),
                    "type": source.get("source_type") or source.get("type"),
                    "domain": source.get("domain"),
                    "citation_markdown": source.get("citation_markdown"),
                }
                for source in (research.get("sources") or [])[:8]
            ],
            "evidence": [
                {
                    "citation_id": item.get("citation_id"),
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "excerpt": (item.get("excerpt") or item.get("snippet") or "")[:600],
                }
                for item in (research.get("evidence") or [])[:6]
            ],
            "sources_count": len(research.get("sources") or []),
            "evidence_count": len(research.get("evidence") or []),
        }
    return compact


def _final_answer_with_llm(
    message: str,
    plan: dict[str, Any],
    artifacts: dict[str, Any],
    fallback: str,
    history: list[dict[str, Any]] | None = None,
) -> str:
    system = (
        "You are MolPipeline ChemChat running on deepseek/deepseek-v4-flash. "
        "Answer in the user's language, usually Russian; Russian Cyrillic text is valid user input. "
        "Use conversation_history to keep continuity in follow-up questions, but do not override current tool outputs. "
        "Use ONLY the provided tool outputs for molecule data, safety, retrosynthesis, ADMET, availability and sources. "
        "For retrosynthesis, explicitly distinguish one-step disconnection results from multi-step route-tree results when a depth_mode or tree_stats field is present. "
        "If no tools were selected, answer as a normal chemistry tutor from general chemistry knowledge and use fallback_summary as guidance. "
        "For real-world materials and product composition, do not force a single-molecule framing: explain mixtures, mineral phases, additives, coatings and likely variability. "
        "For hazardous or dual-use synthesis topics, including nitration of toluene, explosives and narcotics, keep the answer high-level and non-operational: do not provide temperatures, reagent ratios, step-by-step procedures, purification instructions, yields, procurement advice or scale-up guidance. "
        "Do not invent synthesis routes, prices, safety classifications or citations. "
        "When research/web sources include citation_id fields, attach source markers like [S1] to source-backed claims. "
        "When research/web sources include URLs, cite them as Markdown links: [title](url) or [S1](url). "
        "If tools did not find enough data, say exactly what is missing and what to try next. "
        "Keep the answer concise but useful for a chemist."
    )
    user = json.dumps(
        {
            "user_message": message,
            "conversation_history": _compact_chat_history(history),
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
        return _append_research_source_links(fallback, artifacts)
    answer = data.get("answer")
    answer_text = answer.strip() if isinstance(answer, str) and answer.strip() else fallback
    if fallback and _looks_like_unhelpful_refusal(answer_text):
        answer_text = fallback
    return _append_research_source_links(answer_text, artifacts)


def _looks_like_unhelpful_refusal(answer: str) -> bool:
    text = answer.casefold()
    markers = (
        "не могу распознать",
        "не могу понять",
        "уточните, что именно",
        "please clarify",
        "cannot recognize",
        "can't recognize",
    )
    return any(marker in text for marker in markers)


def _direct_general_fallback(message: str) -> str:
    text = message.casefold()
    if "хим" in text and ("элемент" in text or "популяр" in text or "что такое" in text):
        return (
            "Химия — это наука о веществах: из чего они состоят, как устроены, "
            "какими свойствами обладают и как превращаются друг в друга в реакциях.\n\n"
            "Если под «популярными элементами» понимать самые часто встречающиеся и важные в учебной/практической химии, "
            "то обычно называют: **водород (H)**, **кислород (O)**, **углерод (C)**, **азот (N)**, "
            "**натрий (Na)**, **хлор (Cl)**, **железо (Fe)**, **алюминий (Al)**, **кремний (Si)**, "
            "**кальций (Ca)**, **сера (S)** и **фосфор (P)**.\n\n"
            "Важно: «популярность» можно понимать по-разному — распространенность в земной коре, роль в живых организмах, "
            "частота в промышленности или частота в школьной программе."
        )
    return (
        "Это общий химический вопрос, для него не нужен отдельный инструмент MolPipeline. "
        "Я могу объяснить базовую теорию, привести примеры или перейти к конкретной молекуле/реакции."
    )


def _link_citation_markers(answer: str, sources: list[dict[str, Any]]) -> str:
    linked = answer
    for index, source in enumerate(sources, start=1):
        citation_id = str(source.get("citation_id") or f"S{index}")
        url = source.get("url")
        if not url:
            continue
        safe_url = str(url).replace("(", "%28").replace(")", "%29")
        linked = linked.replace(f"[{citation_id}]", f"[{citation_id}]({safe_url})")
        linked = linked.replace(f"[{citation_id}]({safe_url})({safe_url})", f"[{citation_id}]({safe_url})")
    return linked


def _append_research_source_links(answer: str, artifacts: dict[str, Any]) -> str:
    research = artifacts.get("research") or {}
    sources = research.get("citations") or research.get("sources") or []
    answer = _link_citation_markers(answer, sources)
    links: list[str] = []
    seen: set[str] = set()
    for index, source in enumerate(sources, start=1):
        url = source.get("url")
        title = source.get("title") or source.get("name") or url
        if not url or not title or url in seen:
            continue
        seen.add(url)
        citation_id = source.get("citation_id") or f"S{index}"
        if url in answer or f"]({url})" in answer:
            continue
        safe_url = str(url).replace("(", "%28").replace(")", "%29")
        links.append(f"- [{citation_id}] [{title}]({safe_url})")
        if len(links) >= 6:
            break
    if not links:
        return answer
    return answer.rstrip() + "\n\n### Источники\n" + "\n".join(links)


def _emit_progress(callback: Callable[[dict[str, Any]], None] | None, event: dict[str, Any]) -> None:
    if callback is None:
        return
    try:
        callback(event)
    except Exception:
        logger.debug("ChemChat progress callback failed", exc_info=True)


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


def _multi_step_summary(tree_result: dict[str, Any]) -> list[str]:
    if not tree_result or tree_result.get("status") != "ok":
        reason = (tree_result or {}).get("reason") or "multi-step expansion did not run."
        return [f"Multi-step expansion: {reason}"]
    stats = tree_result.get("stats") or {}
    return [
        "Multi-step route tree built: "
        f"nodes={stats.get('total_nodes', 0)}, "
        f"buyable_leaves={stats.get('buyable_count', 0)}, "
        f"unresolved={stats.get('unresolved_count', 0)}, "
        f"max_depth={stats.get('max_depth_reached', 0)}."
    ]


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


def _stable_suggestion_sample(seed: str, suggestions: list[str], limit: int = 4) -> list[str]:
    scored = []
    seen: set[str] = set()
    for suggestion in suggestions:
        clean = " ".join(suggestion.split())
        if not clean or clean in seen:
            continue
        seen.add(clean)
        digest = hashlib.sha256(f"{seed}|{clean}".encode("utf-8")).hexdigest()
        scored.append((digest, clean))
    return [item for _, item in sorted(scored)[:limit]]


def _build_suggestions(
    query: str,
    intent: str,
    tools_used: list[str],
    artifacts: dict[str, Any],
    history: list[dict[str, str]],
) -> list[str]:
    context = query
    if _looks_context_dependent(query):
        context = "\n".join([query, *(item["content"] for item in history[-4:])])
    pool: list[str] = []
    safety = artifacts.get("safety") or {}
    blocked = safety.get("overall_status") == "CRITICAL_STOP"

    if blocked:
        return _stable_suggestion_sample(f"{query}|blocked", [
            "Объяснить, почему запрос заблокирован",
            "Предложить безопасную учебную альтернативу",
            "Показать только общую теорию без процедур",
            "Перейти к разрешенному веществу",
        ], limit=4)

    if artifacts.get("retrosynthesis", {}).get("routes"):
        pool.extend([
            "Выбрать лучший маршрут и посчитать масштаб",
            "Проверить доступность всех исходных реагентов",
            "Сравнить маршруты по безопасности и цене",
            "Показать подробности первого маршрута",
        ])

    if _looks_like_real_world_composition_question(context) or "research_analyze" in tools_used:
        pool.extend([
            "Показать источники и ссылки подробнее",
            "Разложить состав на вещества и минералы",
            "Объяснить, какая часть отвечает за прочность",
            "Найти типичный промышленный состав",
            "Сравнить варианты материалов между собой",
            "Проверить безопасность контакта с пищей",
        ])

    if intent in {"admet", "safety"} or any(tool in tools_used for tool in ("admet_screen", "safety_check")):
        pool.extend([
            "Пояснить риск простыми словами",
            "Сравнить с похожим веществом",
            "Показать, какие параметры сильнее всего влияют на оценку",
            "Проверить доступность вещества и ограничения",
        ])

    if intent in {"availability", "mixed"} or "availability_check" in tools_used:
        pool.extend([
            "Показать только доступные позиции",
            "Сравнить поставщиков по цене",
            "Проверить альтернативные реагенты",
            "Связать доступность с маршрутом синтеза",
        ])

    if not pool:
        pool.extend([
            "Попросить ответ со ссылками на источники",
            "Попросить короткое объяснение на примере",
            "Уточнить практическое применение",
            "Перейти к конкретному веществу или материалу",
            "Сравнить два похожих случая",
            "Попросить таблицу с ключевыми отличиями",
        ])

    return _stable_suggestion_sample(f"{query}|{intent}|{','.join(tools_used)}", pool, limit=4)


def run_chem_chat(
    message: str,
    *,
    history: list[dict[str, Any]] | None = None,
    source_mode: str = "auto",
    top_n: int = 5,
    research_mode: str = "literature",
    max_sources: int = 6,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    query = message.strip()
    if not query:
        raise ValueError("message must not be empty")

    _emit_progress(progress_callback, {
        "type": "status",
        "stage": "planning",
        "label": "Модель планирует инструменты",
        "model": CHEM_CHAT_MODEL,
    })
    compact_history = _compact_chat_history(history)
    plan = _plan_with_llm(query, source_mode=source_mode, research_mode=research_mode, history=compact_history)
    intent = plan["intent"]
    selected_tools = set(plan["tools"])
    selected_source_mode = plan.get("source_mode") or source_mode
    selected_research_mode = plan.get("research_mode") or research_mode
    selected_retro_depth_mode = plan.get("retrosynthesis_depth_mode") or "one_step"
    tools_used: list[str] = []
    artifacts: dict[str, Any] = {}
    answer_lines: list[str] = []
    _emit_progress(progress_callback, {
        "type": "plan",
        "stage": "planned",
        "label": "План готов",
        "intent": intent,
        "tools": plan["tools"],
        "target_molecules": plan.get("target_molecules", []),
        "retrosynthesis_depth_mode": selected_retro_depth_mode,
        "used_llm": plan.get("used_llm", False),
    })

    resolved: dict[str, Any] = {"status": "skipped", "query": query}
    resolved_ok = False
    should_resolve = "resolve_molecule" in selected_tools
    if should_resolve:
        _emit_progress(progress_callback, {
            "type": "tool_start",
            "tool": "resolve_molecule",
            "label": "Распознаю целевую молекулу",
        })
        targets = plan.get("target_molecules") or []
        resolved = _resolve_from_message(str(targets[0])) if targets else _resolve_from_message(query)
        tools_used.append("resolve_molecule")
        artifacts["molecule"] = resolved
        resolved_ok = resolved.get("status") == "ok"
        _emit_progress(progress_callback, {
            "type": "tool_done",
            "tool": "resolve_molecule",
            "label": "Молекула распознана" if resolved_ok else "Молекулу распознать не удалось",
            "status": resolved.get("status"),
            "smiles": resolved.get("smiles"),
            "query_used": resolved.get("query_used") or resolved.get("query"),
        })
    smiles = resolved.get("smiles") if resolved_ok else None
    cid = resolved.get("pubchem_cid") if resolved_ok else None

    safety_guard: dict[str, Any] | None = None
    if resolved_ok:
        _emit_progress(progress_callback, {
            "type": "tool_start",
            "tool": "safety_check",
            "label": "Проверяю banlist, GHS и PPE",
        })
        safety_guard = _safety_tool(smiles, cid=cid, reaction_description=query)
        tools_used.append("safety_check")
        artifacts["safety"] = safety_guard
        _emit_progress(progress_callback, {
            "type": "tool_done",
            "tool": "safety_check",
            "label": "Safety gate завершен",
            "status": safety_guard.get("overall_status", "UNKNOWN"),
        })

        status = safety_guard.get("overall_status", "SAFE")
        query_used = resolved.get("query_used") or resolved.get("query") or query
        answer_lines.append(f"Целевая молекула: {query_used}; SMILES `{smiles}`. Safety gate: {status}.")
        if status == "CRITICAL_STOP":
            reason = (
                (((safety_guard.get("safety_taxonomy") or {}).get("blocked_categories") or [{}])[0].get("reason"))
                or safety_guard.get("explosive_check", {}).get("reason")
                or safety_guard.get("molecule_check", {}).get("reason")
                or safety_guard.get("reaction_check", {}).get("reason")
                or "критический safety-stop"
            )
            answer_lines.append(f"Действия, связанные с синтезом, маршрутом или доступностью, заблокированы: {reason}")
    elif should_resolve and intent != "availability":
        answer_lines.append(resolved.get("error") or "Не удалось распознать молекулу через PubChem/RDKit.")

    wants_retro = "retrosynthesis_search" in selected_tools
    wants_admet = "admet_screen" in selected_tools
    wants_availability = "availability_check" in selected_tools
    wants_research = "research_analyze" in selected_tools or (should_resolve and not resolved_ok)
    wants_safety_only = intent == "safety" or selected_tools == {"resolve_molecule", "safety_check"}

    if resolved_ok and wants_retro and safety_guard and safety_guard.get("overall_status") != "CRITICAL_STOP":
        _emit_progress(progress_callback, {
            "type": "tool_start",
            "tool": "retrosynthesis_search",
            "label": "Ищу и ранжирую маршруты ретросинтеза",
            "source_mode": selected_source_mode,
        })
        retro = _retro_tool(smiles, top_n=top_n, source_mode=selected_source_mode)
        retro["depth_mode"] = selected_retro_depth_mode
        tools_used.append("retrosynthesis_search")
        artifacts["retrosynthesis"] = retro
        answer_lines.extend(_route_summary(retro))
        _emit_progress(progress_callback, {
            "type": "tool_done",
            "tool": "retrosynthesis_search",
            "label": "Ретросинтез завершен",
            "routes": len(retro.get("routes") or []),
            "sources_used": retro.get("sources_used", []),
        })

        best_route = retro.get("best_route") or ((retro.get("routes") or [None])[0])
        if selected_retro_depth_mode == "multi_step" and best_route:
            _emit_progress(progress_callback, {
                "type": "tool_start",
                "tool": "retrosynthesis_tree_expand",
                "label": "Расширяю лучший маршрут до multi-step дерева",
                "source_mode": selected_source_mode,
            })
            tree_result = _multi_step_tree_tool(smiles, best_route)
            tools_used.append("retrosynthesis_tree_expand")
            retro["multi_step_tree"] = tree_result
            answer_lines.extend(_multi_step_summary(tree_result))
            _emit_progress(progress_callback, {
                "type": "tool_done",
                "tool": "retrosynthesis_tree_expand",
                "label": "Multi-step дерево построено",
                "status": tree_result.get("status"),
                "stats": tree_result.get("stats", {}),
            })

    if resolved_ok and wants_admet:
        _emit_progress(progress_callback, {
            "type": "tool_start",
            "tool": "admet_screen",
            "label": "Считаю ADMET-дескрипторы",
        })
        admet = _admet_tool(smiles, safety_guard=safety_guard)
        tools_used.append("admet_screen")
        artifacts["admet"] = admet
        answer_lines.extend(_admet_summary(admet))
        _emit_progress(progress_callback, {
            "type": "tool_done",
            "tool": "admet_screen",
            "label": "ADMET завершен",
            "score": (admet.get("overall") or {}).get("score"),
            "risk_level": (admet.get("overall") or {}).get("risk_level"),
        })

    if wants_availability:
        if safety_guard and safety_guard.get("overall_status") == "CRITICAL_STOP":
            answer_lines.append("Проверку доступности и поставщиков для заблокированного опасного сценария не выполняю.")
            wants_availability = False

    if wants_availability:
        _emit_progress(progress_callback, {
            "type": "tool_start",
            "tool": "availability_check",
            "label": "Проверяю доступность реагентов",
        })
        availability_query = query
        retro_routes = (artifacts.get("retrosynthesis") or {}).get("routes") or []
        if retro_routes and retro_routes[0].get("reactants"):
            availability_query = retro_routes[0]["reactants"]
        availability = _availability_tool(availability_query)
        tools_used.append("availability_check")
        artifacts["availability"] = availability
        answer_lines.extend(_availability_summary(availability))
        _emit_progress(progress_callback, {
            "type": "tool_done",
            "tool": "availability_check",
            "label": "Проверка поставщиков завершена",
            "summary": availability.get("summary"),
        })

    if wants_research:
        research_query = _contextual_research_query(query, compact_history)
        _emit_progress(progress_callback, {
            "type": "tool_start",
            "tool": "research_analyze",
            "label": "Собираю web/PubMed/RAG evidence",
            "query": research_query,
        })
        research = _research_tool(research_query, mode=selected_research_mode, max_sources=max_sources)
        tools_used.append("research_analyze")
        artifacts["research"] = research
        if intent == "general":
            answer_lines.append("Вопрос не требует целевой молекулы, поэтому использую общий химический research-режим.")
        answer_lines.extend(_research_summary(research))
        _emit_progress(progress_callback, {
            "type": "tool_done",
            "tool": "research_analyze",
            "label": "Research завершен",
            "sources": len(research.get("sources") or []),
            "evidence": len(research.get("evidence") or []),
        })

    if resolved_ok and wants_safety_only and safety_guard:
        mol_check = safety_guard.get("molecule_check", {})
        taxonomy = safety_guard.get("safety_taxonomy") or {}
        h_phrases = safety_guard.get("safety_data", {}).get("h_phrases") or []
        safety_reason = (
            (((taxonomy.get("blocked_categories") or taxonomy.get("warning_categories") or [{}])[0]).get("reason"))
            or (safety_guard.get("explosive_check") or {}).get("reason")
            or mol_check.get("reason")
            or "Критичных safety-флагов не найдено."
        )
        answer_lines.append(safety_reason)
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

    if intent == "general" and not tools_used:
        answer_lines.append(_direct_general_fallback(query))

    suggestions = _build_suggestions(query, intent, tools_used, artifacts, compact_history)

    fallback_answer = "\n".join(answer_lines)
    _emit_progress(progress_callback, {
        "type": "status",
        "stage": "final_answer",
        "label": "Модель формирует финальный Markdown-ответ",
        "model": CHEM_CHAT_MODEL,
    })
    answer = _final_answer_with_llm(query, plan, artifacts, fallback_answer, history=compact_history)

    return {
        "status": "ok",
        "intent": intent,
        "model": CHEM_CHAT_MODEL,
        "plan": plan,
        "answer": answer,
        "tools_used": tools_used,
        "artifacts": artifacts,
        "suggested_next_actions": suggestions,
    }
