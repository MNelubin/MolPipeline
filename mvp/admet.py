"""Heuristic ADMET analysis built on top of RDKit descriptors.

This is an interpretable screening layer, not a QSAR replacement. It gives
fast, source-transparent flags that can later be augmented with learned models.
"""

from __future__ import annotations

import re
from typing import Any

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


def _band(value: float, low: float, high: float) -> str:
    if value < low:
        return "низкая"
    if value > high:
        return "высокая"
    return "умеренная"


def _score_from_flags(flags: list[dict[str, Any]], base: int = 100) -> int:
    score = base
    for flag in flags:
        severity = flag.get("severity")
        if severity == "high":
            score -= 25
        elif severity == "medium":
            score -= 12
        elif severity == "low":
            score -= 5
    return max(0, min(100, score))


def _flag(condition: bool, severity: str, message: str, evidence: str) -> dict[str, Any] | None:
    if not condition:
        return None
    return {
        "severity": severity,
        "message": message,
        "evidence": evidence,
    }


def _filter_none(items: list[dict[str, Any] | None]) -> list[dict[str, Any]]:
    return [item for item in items if item is not None]


ACUTE_FATAL_H_CODES = {"H300", "H310", "H330"}
HIGH_TOX_H_CODES = {"H300", "H310", "H330", "H340", "H350"}
MEDIUM_TOX_H_CODES = {
    "H301", "H302", "H311", "H312", "H314", "H317", "H318", "H331", "H332",
    "H334", "H341", "H351", "H360", "H361", "H370", "H371", "H372", "H373",
}


def _extract_h_codes(safety_data: dict[str, Any] | None) -> set[str]:
    codes: set[str] = set()
    if not safety_data:
        return codes
    for phrase in safety_data.get("h_phrases", []) or []:
        codes.update(re.findall(r"H\d{3}", str(phrase)))
    for code in safety_data.get("ghs_codes", []) or []:
        if re.fullmatch(r"H\d{3}", str(code)):
            codes.add(str(code))
    return codes


def _safety_overlay_flags(safety_guard: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not safety_guard:
        return []

    molecule_check = safety_guard.get("molecule_check", {}) or {}
    safety_data = safety_guard.get("safety_data", {}) or {}
    h_codes = _extract_h_codes(safety_data)
    flags: list[dict[str, Any] | None] = []

    mol_status = molecule_check.get("status")
    if mol_status == "banned":
        flags.append(_flag(
            True,
            "high",
            "Проверка безопасности: вещество запрещено или контролируется.",
            molecule_check.get("reason") or "совпадение со списком ограничений",
        ))
    elif mol_status == "restricted":
        flags.append(_flag(
            True,
            "medium",
            "Проверка безопасности: вещество ограничено или имеет двойное назначение.",
            molecule_check.get("reason") or "совпадение со списком ограничений",
        ))

    high_codes = sorted(h_codes & HIGH_TOX_H_CODES)
    medium_codes = sorted((h_codes & MEDIUM_TOX_H_CODES) - set(high_codes))
    if high_codes:
        flags.append(_flag(
            True,
            "high",
            "Данные GHS содержат признаки выраженной острой или хронической токсичности.",
            ", ".join(high_codes),
        ))
    if medium_codes:
        flags.append(_flag(
            True,
            "medium",
            "Данные GHS содержат предупреждения о токсичности или серьезном вреде здоровью.",
            ", ".join(medium_codes[:8]),
        ))

    return _filter_none(flags)


def _build_safety_overlay(safety_guard: dict[str, Any] | None) -> dict[str, Any]:
    if not safety_guard:
        return {
            "available": False,
            "overall_status": "UNKNOWN",
            "molecule_status": "unknown",
            "h_codes": [],
        }
    molecule_check = safety_guard.get("molecule_check", {}) or {}
    safety_data = safety_guard.get("safety_data", {}) or {}
    taxonomy = safety_guard.get("safety_taxonomy", {}) or {}
    return {
        "available": True,
        "overall_status": safety_guard.get("overall_status", "SAFE"),
        "taxonomy_status": taxonomy.get("status", "clear"),
        "taxonomy_categories": (taxonomy.get("categories") or [])[:8],
        "molecule_status": molecule_check.get("status", "clear"),
        "molecule_reason": molecule_check.get("reason"),
        "molecule_category": molecule_check.get("category"),
        "danger_level": molecule_check.get("danger_level"),
        "h_codes": sorted(_extract_h_codes(safety_data)),
        "h_phrases": (safety_data.get("h_phrases", []) or [])[:8],
        "ppe_recommendations": safety_guard.get("ppe_recommendations", []),
    }


def analyze_admet(smiles: str, safety_guard: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return interpretable ADMET heuristics for a canonical SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    mw = round(Descriptors.MolWt(mol), 2)
    logp = round(Crippen.MolLogP(mol), 2)
    tpsa = round(rdMolDescriptors.CalcTPSA(mol), 2)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rotatable = Lipinski.NumRotatableBonds(mol)
    rings = Lipinski.RingCount(mol)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()
    formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())

    lipinski_violations = sum([
        mw > 500,
        logp > 5,
        hbd > 5,
        hba > 10,
    ])
    veber_violations = sum([
        rotatable > 10,
        tpsa > 140,
    ])

    absorption_flags = _filter_none([
        _flag(lipinski_violations >= 2, "high", "Вероятны слабые лекарственно-подобные свойства при пероральном применении по правилам Lipinski.", f"{lipinski_violations} Lipinski violations"),
        _flag(lipinski_violations == 1, "medium", "Есть одно нарушение правил Lipinski; нужна проверка проницаемости и растворимости.", f"{lipinski_violations} Lipinski violation"),
        _flag(veber_violations > 0, "medium", "Фильтр Veber указывает на возможное снижение всасывания при пероральном применении.", f"rotatable={rotatable}, TPSA={tpsa}"),
        _flag(logp < 0, "medium", "Низкий LogP может снижать пассивную мембранную проницаемость.", f"LogP={logp}"),
        _flag(logp > 4.5, "medium", "Высокий LogP может снижать растворимость и повышать nonspecific binding.", f"LogP={logp}"),
    ])

    distribution_flags = _filter_none([
        _flag(tpsa < 90 and 1 <= logp <= 4 and mw < 450, "low", "Параметры совместимы с возможным проникновением в ЦНС через гематоэнцефалический барьер.", f"TPSA={tpsa}, LogP={logp}, MW={mw}"),
        _flag(tpsa > 120, "medium", "Высокая TPSA обычно снижает проникновение через гематоэнцефалический барьер.", f"TPSA={tpsa}"),
        _flag(logp > 4, "medium", "Высокая липофильность может повышать связывание с белками плазмы.", f"LogP={logp}"),
        _flag(abs(formal_charge) > 0, "medium", "Формальный заряд может влиять на распределение и проницаемость.", f"formal_charge={formal_charge}"),
    ])

    metabolism_flags = _filter_none([
        _flag(logp > 3 and aromatic_rings >= 2, "medium", "Липофильная ароматическая структура может иметь повышенный риск метаболизма через CYP-ферменты.", f"LogP={logp}, aromatic_rings={aromatic_rings}"),
        _flag(rotatable > 8, "medium", "Высокая гибкость может повышать число метаболически доступных конформаций.", f"rotatable_bonds={rotatable}"),
        _flag(heavy_atoms > 45, "medium", "Большой размер молекулы повышает риск сложного профиля метаболизма и выведения.", f"heavy_atoms={heavy_atoms}"),
    ])

    excretion_flags = _filter_none([
        _flag(mw < 300 and tpsa > 75, "low", "Профиль совместим с частичной renal clearance вероятностью.", f"MW={mw}, TPSA={tpsa}"),
        _flag(mw > 500 or logp > 5, "medium", "Высокий MW/LogP может снижать renal clearance и повышать biliary clearance вероятность.", f"MW={mw}, LogP={logp}"),
    ])

    toxicity_flags = _filter_none([
        _flag(logp > 5, "medium", "Высокий LogP связан с риском фосфолипидоза и неспецифической токсичности.", f"LogP={logp}"),
        _flag(aromatic_rings >= 4, "medium", "Большое число ароматических колец может повышать риск hERG и внецелевых эффектов.", f"aromatic_rings={aromatic_rings}"),
        _flag(mw > 700, "high", "Очень высокая молекулярная масса выходит за пределы обычного пространства малых молекул.", f"MW={mw}"),
        _flag(hba + hbd > 14, "medium", "Высокая суммарная способность к водородным связям может ухудшать проницаемость и общий ADMET-баланс.", f"HBA+HBD={hba + hbd}"),
    ])
    toxicity_flags.extend(_safety_overlay_flags(safety_guard))

    sections = {
        "absorption": {
            "score": _score_from_flags(absorption_flags),
            "interpretation": "Оценка всасывания при пероральном применении и лекарственно-подобных свойств по эвристикам Lipinski/Veber.",
            "flags": absorption_flags,
        },
        "distribution": {
            "score": _score_from_flags(distribution_flags),
            "interpretation": "Оценка распределения, вероятности проникновения в ЦНС и риска неспецифического связывания.",
            "flags": distribution_flags,
        },
        "metabolism": {
            "score": _score_from_flags(metabolism_flags),
            "interpretation": "Оценка структурных факторов, связанных с метаболической уязвимостью.",
            "flags": metabolism_flags,
        },
        "excretion": {
            "score": _score_from_flags(excretion_flags),
            "interpretation": "Грубая оценка направления выведения по молекулярной массе, TPSA и LogP.",
            "flags": excretion_flags,
        },
        "toxicity": {
            "score": _score_from_flags(toxicity_flags),
            "interpretation": "Структурные предупреждения о возможной неспецифической токсичности.",
            "flags": toxicity_flags,
        },
    }

    overall = round(sum(section["score"] for section in sections.values()) / len(sections))
    risk_level = "low"
    if overall < 55:
        risk_level = "high"
    elif overall < 75:
        risk_level = "medium"

    safety_overlay = _build_safety_overlay(safety_guard)
    if safety_overlay["overall_status"] == "CRITICAL_STOP":
        risk_level = "high"
        overall = min(overall, 40)
    elif set(safety_overlay.get("h_codes", [])) & ACUTE_FATAL_H_CODES:
        risk_level = "high"
        overall = min(overall, 55)
    elif safety_overlay["overall_status"] == "WARNING":
        risk_level = "medium" if risk_level == "low" else risk_level
        overall = min(overall, 70)

    recommendations = [
        "Использовать ADMET как первичный фильтр, а не как финальное QSAR-предсказание.",
        "Для перспективных кандидатов сверить проверку безопасности, GHS/LD50 и доступность синтеза.",
    ]
    if lipinski_violations:
        recommendations.append("Проверить дизайн аналогов: снижение молекулярной массы, LogP и числа доноров/акцепторов водородных связей может улучшить пероральный профиль.")
    if tpsa > 120:
        recommendations.append("Если нужен профиль для ЦНС, рассмотреть снижение TPSA и числа доноров водородных связей.")
    if logp > 4.5:
        recommendations.append("Проверить растворимость и риск высокого связывания с белками из-за липофильности.")

    return {
        "smiles": canonical,
        "descriptors": {
            "molecular_weight": mw,
            "logp": logp,
            "tpsa": tpsa,
            "h_bond_donors": hbd,
            "h_bond_acceptors": hba,
            "rotatable_bonds": rotatable,
            "ring_count": rings,
            "aromatic_rings": aromatic_rings,
            "heavy_atoms": heavy_atoms,
            "formal_charge": formal_charge,
            "lipinski_violations": lipinski_violations,
            "veber_violations": veber_violations,
            "solubility_band": _band(logp, 1, 4),
            "permeability_band": "благоприятная" if tpsa <= 90 and 1 <= logp <= 4 else "неопределенная",
        },
        "sections": sections,
        "overall": {
            "score": overall,
            "risk_level": risk_level,
            "summary": f"ADMET-оценка с проверкой безопасности: {overall}/100; общий риск: {risk_level}.",
        },
        "safety_overlay": safety_overlay,
        "recommendations": recommendations,
        "method": "rdkit_descriptor_heuristics_v2_with_safety_overlay",
    }
