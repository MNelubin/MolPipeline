"""Safety tools: banlist checks, PPE recommendations."""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path

from rdkit import Chem

from ..config import DATA_DIR

logger = logging.getLogger(__name__)


def _load_json(filename: str) -> dict:
    path = DATA_DIR / filename
    if not path.exists():
        logger.warning("Data file not found: %s", path)
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _banned_chemicals() -> list[dict]:
    data = _load_json("banned_chemicals.json")
    return data.get("chemicals", [])


@lru_cache(maxsize=1)
def _banned_reactions() -> list[dict]:
    data = _load_json("banned_reactions.json")
    return data.get("reactions", [])


# ═══════════════════════════════════════════════════════════════════════════════
# banlist_check
# ═══════════════════════════════════════════════════════════════════════════════

def banlist_check(smiles: str) -> dict:
    """Check if a SMILES is in the banned chemicals list."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "smiles": smiles, "name": None,
            "status": "clear", "category": None,
            "danger_level": None, "reason": "Invalid SMILES — skipped.",
        }

    canon = Chem.MolToSmiles(mol, isomericSmiles=True)

    for entry in _banned_chemicals():
        entry_canon = entry.get("canonical_smiles") or entry.get("smiles", "")
        if entry_canon == canon:
            dl = entry.get("danger_level", "medium")
            status = "restricted" if dl == "medium" else "banned"
            return {
                "smiles": canon,
                "name": entry.get("name"),
                "status": status,
                "category": entry.get("category"),
                "danger_level": dl,
                "reason": f"Exact match in banlist: {entry.get('name')}.",
            }

    for entry in _banned_reactions():
        smarts = entry.get("smarts", "")
        if not smarts:
            continue
        pat = Chem.MolFromSmarts(smarts)
        if pat and mol.HasSubstructMatch(pat):
            dl = entry.get("danger_level", "medium")
            status = "restricted" if dl == "medium" else "banned"
            return {
                "smiles": canon,
                "name": entry.get("name"),
                "status": status,
                "category": entry.get("category"),
                "danger_level": dl,
                "reason": f"Substructure match: {entry.get('name')}.",
            }

    nitro = Chem.MolFromSmarts("[N+](=O)[O-]")
    aromatic_nitro = Chem.MolFromSmarts("[c:1][N+](=O)[O-]")
    if nitro and aromatic_nitro:
        nitro_count = len(mol.GetSubstructMatches(nitro))
        aromatic_nitro_count = len(mol.GetSubstructMatches(aromatic_nitro))
        if nitro_count >= 3 and aromatic_nitro_count >= 3:
            return {
                "smiles": canon,
                "name": "trinitroaromatic explosive motif",
                "status": "banned",
                "category": "explosive_synthesis",
                "danger_level": "high",
                "reason": "Trinitroaromatic explosive motif detected.",
            }

    return {
        "smiles": canon, "name": None,
        "status": "clear", "category": None,
        "danger_level": None, "reason": "Not found in banlists.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# reaction_banlist_check
# ═══════════════════════════════════════════════════════════════════════════════

def reaction_banlist_check(reaction_description: str) -> dict:
    """Check if a reaction description matches any banned pattern."""
    if not reaction_description or not reaction_description.strip():
        return {
            "status": "allowed",
            "reason": "No reaction description provided.",
            "matched_pattern": None,
        }

    desc_lower = reaction_description.lower()

    for entry in _banned_reactions():
        name_lower = entry.get("name", "").lower()
        desc_entry = entry.get("description", "").lower()
        keywords = set(name_lower.split()) | set(desc_entry.split())
        keywords -= {"the", "of", "a", "an", "in", "for", "and", "or", "with", "is", "to"}

        match_count = sum(1 for kw in keywords if len(kw) > 3 and kw in desc_lower)
        if match_count >= 3:
            dl = entry.get("danger_level", "medium")
            status = "prohibited" if dl in ("critical", "high") else "restricted"
            return {
                "status": status,
                "reason": f"Semantic match: {entry.get('name')}.",
                "matched_pattern": entry.get("smarts"),
            }

    return {
        "status": "allowed",
        "reason": "No prohibited patterns.",
        "matched_pattern": None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PPE recommender
# ═══════════════════════════════════════════════════════════════════════════════

_PPE_MAP: dict[str, list[str]] = {
    "H200": ["Взрывозащищённое оборудование"],
    "H201": ["Взрывозащищённое оборудование"],
    "H202": ["Взрывозащищённое оборудование"],
    "H220": ["Огнестойкий халат", "Взрывозащищённое оборудование"],
    "H221": ["Огнестойкий халат"],
    "H224": ["Огнестойкий халат", "Взрывозащищённое оборудование"],
    "H225": ["Огнестойкий халат"],
    "H226": ["Огнестойкий халат"],
    "H290": ["Химически стойкие перчатки"],
    "H300": ["Нитриловые перчатки", "Полный лицевой щиток", "Вытяжной шкаф"],
    "H301": ["Нитриловые перчатки", "Вытяжной шкаф"],
    "H302": ["Нитриловые перчатки"],
    "H310": ["Химзащитный костюм", "Нитриловые перчатки", "Полный лицевой щиток"],
    "H311": ["Нитриловые перчатки", "Химзащитный фартук"],
    "H312": ["Нитриловые перчатки"],
    "H314": ["Защитные очки от брызг", "Лицевой щиток", "Химически стойкие перчатки"],
    "H315": ["Нитриловые перчатки"],
    "H317": ["Нитриловые перчатки"],
    "H318": ["Защитные очки от брызг"],
    "H319": ["Защитные очки"],
    "H330": ["Автономный дыхательный аппарат (SCBA)", "Вытяжной шкаф"],
    "H331": ["Вытяжной шкаф", "Респиратор"],
    "H332": ["Вытяжной шкаф"],
    "H334": ["Респиратор", "Вытяжной шкаф"],
    "H335": ["Вытяжной шкаф"],
    "H336": ["Вытяжной шкаф"],
    "H340": ["Вытяжной шкаф", "Нитриловые перчатки", "Минимизировать воздействие"],
    "H341": ["Вытяжной шкаф", "Нитриловые перчатки"],
    "H350": ["Вытяжной шкаф", "Нитриловые перчатки", "Минимизировать воздействие"],
    "H351": ["Вытяжной шкаф", "Нитриловые перчатки"],
    "H360": ["Полный комплект СИЗ", "Минимизировать воздействие"],
    "H370": ["Полный комплект СИЗ", "Вытяжной шкаф"],
    "H372": ["Полный комплект СИЗ", "Вытяжной шкаф"],
    "H400": ["Средства локализации разливов", "Химически стойкие перчатки"],
    "H410": ["Средства локализации разливов", "Химически стойкие перчатки"],
}


def ppe_recommender(substances: str, h_phrases: str) -> list[str]:
    """Рекомендации СИЗ на основе H-фраз."""
    ppe_set: set[str] = {"Лабораторный халат", "Нитриловые перчатки", "Защитные очки"}

    codes = re.findall(r"H\d{3}[A-Za-z]?", h_phrases)
    for code in codes:
        base = code[:4]
        if base in _PPE_MAP:
            ppe_set.update(_PPE_MAP[base])

    return sorted(ppe_set)
