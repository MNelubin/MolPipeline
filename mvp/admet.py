"""Heuristic ADMET analysis built on top of RDKit descriptors.

This is an interpretable screening layer, not a QSAR replacement. It gives
fast, source-transparent flags that can later be augmented with learned models.
"""

from __future__ import annotations

from typing import Any

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


def _band(value: float, low: float, high: float) -> str:
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "moderate"


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


def analyze_admet(smiles: str) -> dict[str, Any]:
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
        _flag(lipinski_violations >= 2, "high", "Вероятно слабая oral drug-likeness по Lipinski.", f"{lipinski_violations} Lipinski violations"),
        _flag(lipinski_violations == 1, "medium", "Есть одно нарушение Lipinski; нужна проверка проницаемости/растворимости.", f"{lipinski_violations} Lipinski violation"),
        _flag(veber_violations > 0, "medium", "Veber-фильтр указывает на возможное снижение oral absorption.", f"rotatable={rotatable}, TPSA={tpsa}"),
        _flag(logp < 0, "medium", "Низкий LogP может снижать пассивную мембранную проницаемость.", f"LogP={logp}"),
        _flag(logp > 4.5, "medium", "Высокий LogP может снижать растворимость и повышать nonspecific binding.", f"LogP={logp}"),
    ])

    distribution_flags = _filter_none([
        _flag(tpsa < 90 and 1 <= logp <= 4 and mw < 450, "low", "Параметры совместимы с возможной CNS/BBB-проницаемостью.", f"TPSA={tpsa}, LogP={logp}, MW={mw}"),
        _flag(tpsa > 120, "medium", "Высокая TPSA обычно снижает BBB penetration.", f"TPSA={tpsa}"),
        _flag(logp > 4, "medium", "Высокая липофильность может повышать связывание с белками плазмы.", f"LogP={logp}"),
        _flag(abs(formal_charge) > 0, "medium", "Формальный заряд может влиять на распределение и проницаемость.", f"formal_charge={formal_charge}"),
    ])

    metabolism_flags = _filter_none([
        _flag(logp > 3 and aromatic_rings >= 2, "medium", "Липофильная ароматическая структура может иметь высокий CYP-mediated metabolism risk.", f"LogP={logp}, aromatic_rings={aromatic_rings}"),
        _flag(rotatable > 8, "medium", "Высокая гибкость может повышать число метаболически доступных конформаций.", f"rotatable_bonds={rotatable}"),
        _flag(heavy_atoms > 45, "medium", "Большой размер молекулы повышает риск сложного metabolism/clearance профиля.", f"heavy_atoms={heavy_atoms}"),
    ])

    excretion_flags = _filter_none([
        _flag(mw < 300 and tpsa > 75, "low", "Профиль совместим с частичной renal clearance вероятностью.", f"MW={mw}, TPSA={tpsa}"),
        _flag(mw > 500 or logp > 5, "medium", "Высокий MW/LogP может снижать renal clearance и повышать biliary clearance вероятность.", f"MW={mw}, LogP={logp}"),
    ])

    toxicity_flags = _filter_none([
        _flag(logp > 5, "medium", "Высокий LogP связан с риском phospholipidosis/nonspecific toxicity.", f"LogP={logp}"),
        _flag(aromatic_rings >= 4, "medium", "Много ароматических колец может повышать hERG/off-target risk.", f"aromatic_rings={aromatic_rings}"),
        _flag(mw > 700, "high", "Очень высокий MW выходит за пределы обычного small-molecule ADMET пространства.", f"MW={mw}"),
        _flag(hba + hbd > 14, "medium", "Высокая H-bond capacity может ухудшать permeability и ADMET balance.", f"HBA+HBD={hba + hbd}"),
    ])

    sections = {
        "absorption": {
            "score": _score_from_flags(absorption_flags),
            "interpretation": "Оценка oral absorption и drug-likeness по Lipinski/Veber-like эвристикам.",
            "flags": absorption_flags,
        },
        "distribution": {
            "score": _score_from_flags(distribution_flags),
            "interpretation": "Оценка распределения, BBB/CNS вероятности и nonspecific binding риска.",
            "flags": distribution_flags,
        },
        "metabolism": {
            "score": _score_from_flags(metabolism_flags),
            "interpretation": "Оценка структурных факторов, связанных с метаболической уязвимостью.",
            "flags": metabolism_flags,
        },
        "excretion": {
            "score": _score_from_flags(excretion_flags),
            "interpretation": "Грубая оценка clearance-направления по MW, TPSA и LogP.",
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

    recommendations = [
        "Использовать ADMET как screening layer, не как финальное QSAR-предсказание.",
        "Для promising candidates сверить safety guard, GHS/LD50 и доступность синтеза.",
    ]
    if lipinski_violations:
        recommendations.append("Проверить analog design: снижение MW/LogP/H-bond capacity может улучшить oral profile.")
    if tpsa > 120:
        recommendations.append("Если нужен CNS/BBB профиль, рассмотреть снижение TPSA и числа доноров H-связей.")
    if logp > 4.5:
        recommendations.append("Проверить растворимость и риск высокой protein binding из-за липофильности.")

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
            "permeability_band": "favorable" if tpsa <= 90 and 1 <= logp <= 4 else "uncertain",
        },
        "sections": sections,
        "overall": {
            "score": overall,
            "risk_level": risk_level,
            "summary": f"ADMET screening score {overall}/100; aggregate risk: {risk_level}.",
        },
        "recommendations": recommendations,
        "method": "rdkit_descriptor_heuristics_v1",
    }
