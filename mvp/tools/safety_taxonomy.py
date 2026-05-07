"""Safety taxonomy assembled from banlists, GHS data and hazard channels."""

from __future__ import annotations

import re
from typing import Any

_H_CODE_RE = re.compile(r"H\d{3}[A-Za-z]?")

_GHS_RULES: tuple[dict[str, Any], ...] = (
    {
        "hazard_type": "explosive",
        "label_ru": "Взрывоопасность",
        "status": "blocked",
        "danger_level": "high",
        "h_codes": {"H200", "H201", "H202", "H203", "H204", "H205"},
    },
    {
        "hazard_type": "acute_toxicity",
        "label_ru": "Острая токсичность",
        "status": "warning",
        "danger_level": "high",
        "h_codes": {"H300", "H301", "H310", "H311", "H330", "H331"},
    },
    {
        "hazard_type": "chronic_health_hazard",
        "label_ru": "Хронический вред здоровью",
        "status": "warning",
        "danger_level": "high",
        "h_codes": {"H340", "H350", "H360", "H370", "H372"},
    },
    {
        "hazard_type": "chronic_health_hazard",
        "label_ru": "Подозрение на хронический вред здоровью",
        "status": "warning",
        "danger_level": "medium",
        "h_codes": {"H341", "H351", "H361", "H371", "H373"},
    },
    {
        "hazard_type": "corrosive_or_irritant",
        "label_ru": "Коррозионное/раздражающее действие",
        "status": "warning",
        "danger_level": "medium",
        "h_codes": {"H314", "H315", "H317", "H318", "H319", "H334", "H335"},
    },
    {
        "hazard_type": "flammable",
        "label_ru": "Пожароопасность",
        "status": "warning",
        "danger_level": "medium",
        "h_codes": {
            "H220", "H221", "H222", "H223", "H224", "H225", "H226", "H228",
            "H242", "H250", "H251", "H252", "H260", "H261",
        },
    },
    {
        "hazard_type": "oxidizer",
        "label_ru": "Окислитель",
        "status": "warning",
        "danger_level": "medium",
        "h_codes": {"H270", "H271", "H272"},
    },
    {
        "hazard_type": "environmental_hazard",
        "label_ru": "Опасность для окружающей среды",
        "status": "warning",
        "danger_level": "medium",
        "h_codes": {"H400", "H410", "H411", "H412", "H413"},
    },
)


def build_safety_taxonomy(
    *,
    molecule_check: dict[str, Any],
    reaction_check: dict[str, Any],
    explosive_check: dict[str, Any],
    safety_data: dict[str, Any],
) -> dict[str, Any]:
    """Build normalized safety categories for the UI and agent memory."""
    categories: list[dict[str, Any]] = []

    mol_status = molecule_check.get("status")
    if mol_status in {"banned", "prohibited", "restricted"}:
        categories.append({
            "hazard_type": "controlled_substance",
            "label_ru": "Регуляторное ограничение",
            "status": "blocked" if mol_status in {"banned", "prohibited"} else "warning",
            "danger_level": molecule_check.get("danger_level") or ("high" if mol_status != "restricted" else "medium"),
            "basis": "banlist",
            "reason": molecule_check.get("reason") or "Matched controlled/restricted substance policy.",
        })

    rxn_status = reaction_check.get("status")
    if rxn_status in {"banned", "prohibited", "blocked", "restricted", "warning"}:
        categories.append({
            "hazard_type": "reaction_policy",
            "label_ru": "Ограниченная реакция",
            "status": "blocked" if rxn_status in {"banned", "prohibited", "blocked"} else "warning",
            "danger_level": reaction_check.get("danger_level") or "high",
            "basis": "reaction_policy",
            "reason": reaction_check.get("reason") or "Matched restricted reaction policy.",
        })

    if explosive_check.get("status") != "clear":
        categories.append({
            "hazard_type": explosive_check.get("hazard_type") or "explosive",
            "label_ru": "Взрывоопасность",
            "status": "blocked" if explosive_check.get("status") == "blocked" else "warning",
            "danger_level": explosive_check.get("danger_level") or "high",
            "hazard_family": explosive_check.get("hazard_family"),
            "basis": explosive_check.get("basis") or "explosive_channel",
            "reason": explosive_check.get("reason") or "Explosive hazard detected.",
            "h_codes": explosive_check.get("h_codes") or [],
        })

    categories.extend(_ghs_categories(safety_data, has_explosive_channel=explosive_check.get("status") != "clear"))

    status = _rollup_status(categories)
    return {
        "status": status,
        "categories": categories,
        "blocked_categories": [item for item in categories if item.get("status") == "blocked"],
        "warning_categories": [item for item in categories if item.get("status") == "warning"],
        "h_codes": sorted(_extract_h_codes(safety_data)),
    }


def _ghs_categories(safety_data: dict[str, Any], *, has_explosive_channel: bool) -> list[dict[str, Any]]:
    h_codes = _extract_h_codes(safety_data)
    categories: list[dict[str, Any]] = []
    seen_types: set[tuple[str, str]] = set()
    for rule in _GHS_RULES:
        if rule["hazard_type"] == "explosive" and has_explosive_channel:
            continue
        matched = sorted(h_codes & rule["h_codes"])
        if not matched:
            continue
        key = (rule["hazard_type"], rule["danger_level"])
        if key in seen_types:
            continue
        seen_types.add(key)
        categories.append({
            "hazard_type": rule["hazard_type"],
            "label_ru": rule["label_ru"],
            "status": rule["status"],
            "danger_level": rule["danger_level"],
            "basis": "ghs",
            "h_codes": matched,
            "reason": f"GHS hazard statements: {', '.join(matched)}.",
        })
    return categories


def _extract_h_codes(safety_data: dict[str, Any]) -> set[str]:
    codes: set[str] = set()
    for phrase in safety_data.get("h_phrases", []) or []:
        codes.update(code[:4] for code in _H_CODE_RE.findall(str(phrase)))
    for code in safety_data.get("ghs_codes", []) or []:
        codes.update(match[:4] for match in _H_CODE_RE.findall(str(code)))
    return codes


def _rollup_status(categories: list[dict[str, Any]]) -> str:
    if any(item.get("status") == "blocked" for item in categories):
        return "blocked"
    if any(item.get("status") == "warning" for item in categories):
        return "warning"
    return "clear"
