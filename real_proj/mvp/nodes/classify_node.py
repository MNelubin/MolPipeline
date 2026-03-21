"""Classify user input as molecule identifier, research query, or invalid.

Uses heuristics (no LLM). Does NOT need to be perfect — if it misclassifies
a misspelled name as "molecule", validate_and_guard will fail and fallback
to research_node automatically.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_SMILES_CHARS = set("=()[]@/\\#%+.")
_CAS_PATTERN = re.compile(r"^\d{2,7}-\d{2}-\d$")
_FORMULA_PATTERN = re.compile(r"^[A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?(?:\d+)?)*$")

_RESEARCH_MARKERS_RU = [
    "хочу", "хотел бы", "ищу", "нужен", "нужна", "нужно", "нужны",
    "подбери", "подобрать", "найди", "найти", "предложи", "предложить",
    "посоветуй", "порекомендуй", "какой", "какая", "какое", "какие",
    "чем заменить", "аналог", "альтернатив",
    "ингибитор", "активатор", "антагонист", "агонист",
    "антиоксидант", "катализатор для", "стабилизатор",
    "противовоспалительн", "антибактериальн", "противоопухолев",
    "анальгетик", "антибиотик", "антивирусн",
    "вещество для", "препарат для", "соединение для",
    "похожее на", "что-то вроде", "типа",
]

_RESEARCH_MARKERS_EN = [
    "i want", "i need", "looking for", "find me", "suggest",
    "recommend", "which", "what is a good",
    "inhibitor", "activator", "antagonist", "agonist",
    "drug for", "molecule for", "compound for",
    "alternative to", "similar to", "substitute for",
]


def classify_node(state: dict[str, Any]) -> dict[str, Any]:
    """Classify user query into molecule / research / invalid.

    Reads:  state["query"]
    Writes: state["input_type"], state["current_phase"], state["cycle_counts"]
    """
    query = state.get("query", "").strip()

    if not query:
        return {
            "input_type": "invalid",
            "current_phase": "identification",
            "cycle_counts": {},
            "error": "Пустой запрос.",
        }

    input_type = _classify(query)
    logger.info("[classify] query=%r -> input_type=%s", query[:80], input_type)

    result: dict[str, Any] = {
        "input_type": input_type,
        "current_phase": "identification",
        "cycle_counts": state.get("cycle_counts", {}),
    }

    if input_type == "invalid":
        result["error"] = "Не удалось распознать запрос."

    return result


def _classify(query: str) -> str:
    stripped = query.strip()

    if len(stripped) < 2:
        return "invalid"

    if _CAS_PATTERN.match(stripped):
        return "molecule"

    if " " not in stripped and _SMILES_CHARS & set(stripped):
        return "molecule"

    if _FORMULA_PATTERN.match(stripped) and len(stripped) <= 30:
        return "molecule"

    lower = stripped.lower()

    for marker in _RESEARCH_MARKERS_RU:
        if marker in lower:
            return "research"
    for marker in _RESEARCH_MARKERS_EN:
        if marker in lower:
            return "research"

    words = lower.split()
    if len(words) >= 4:
        return "research"

    return "molecule"
