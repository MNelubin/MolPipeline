"""Self-contained tools for the MVP pipeline.

No imports from outside real_proj/mvp/.  All data files live in ./data/.
"""

from __future__ import annotations

import json
import logging
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from .config import PUBCHEM_BASE_URL, PUBCHEM_VIEW_URL, DATA_DIR

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 15
_RETRY_DELAY = 0.3


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP helper
# ═══════════════════════════════════════════════════════════════════════════════

def _get_json(url: str, *, retries: int = 2) -> dict[str, Any] | None:
    for attempt in range(1, retries + 2):
        try:
            resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                return None
            if resp.status_code == 503 and attempt <= retries:
                time.sleep(_RETRY_DELAY * attempt)
                continue
            return None
        except requests.RequestException:
            if attempt <= retries:
                time.sleep(_RETRY_DELAY * attempt)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Banned chemicals / reactions data (loaded once)
# ═══════════════════════════════════════════════════════════════════════════════

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
# banlist_check — exact + SMARTS substructure match
# ═══════════════════════════════════════════════════════════════════════════════

def banlist_check(smiles: str) -> dict:
    """Check if a SMILES is in the banned chemicals list.

    Returns dict with keys: smiles, name, status, category, danger_level, reason.
    """
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
            return {
                "smiles": canon,
                "name": entry.get("name"),
                "status": "banned",
                "category": entry.get("category"),
                "danger_level": entry.get("danger_level"),
                "reason": f"Exact match in banlist: {entry.get('name')}.",
            }

    # SMARTS substructure check from banned reactions patterns
    for entry in _banned_reactions():
        smarts = entry.get("smarts", "")
        if not smarts:
            continue
        pat = Chem.MolFromSmarts(smarts)
        if pat and mol.HasSubstructMatch(pat):
            dl = entry.get("danger_level", "medium")
            status = "banned" if dl == "critical" else "restricted"
            return {
                "smiles": canon,
                "name": entry.get("name"),
                "status": status,
                "category": entry.get("category"),
                "danger_level": dl,
                "reason": f"Substructure match: {entry.get('name')}.",
            }

    return {
        "smiles": canon, "name": None,
        "status": "clear", "category": None,
        "danger_level": None, "reason": "Not found in banlists.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# reaction_banlist_check — SMARTS pattern matching on reaction description
# ═══════════════════════════════════════════════════════════════════════════════

def reaction_banlist_check(reaction_description: str) -> dict:
    """Check if a reaction description matches any banned pattern.

    For MVP we do keyword matching against banned reaction names/descriptions.
    Returns dict with keys: status, reason, matched_pattern.
    """
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
        # Remove common words
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
# safety_lookup — GHS data from PubChem
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_h_p_phrases(text: str) -> tuple[list[str], list[str]]:
    """Extract H- and P- phrases from a PubChem text block."""
    h_phrases = re.findall(r"(H\d{3}[A-Za-z]?(?:\s*\+\s*H\d{3}[A-Za-z]?)*[^HP]*?)(?=(?:H\d|P\d|$))", text)
    p_phrases = re.findall(r"(P\d{3}[A-Za-z]?(?:\s*\+\s*P\d{3}[A-Za-z]?)*[^HP]*?)(?=(?:H\d|P\d|$))", text)
    return (
        [h.strip() for h in h_phrases if h.strip()],
        [p.strip() for p in p_phrases if p.strip()],
    )


def safety_lookup(smiles: str, cid: int | None = None) -> dict:
    """Fetch GHS safety data from PubChem.

    Args:
        smiles: SMILES string (used as fallback if cid is None).
        cid: PubChem CID (preferred, avoids re-resolving).

    Returns dict with: ghs_pictograms, h_phrases, p_phrases, ld50, flash_point.
    """
    result: dict[str, Any] = {
        "ghs_pictograms": [],
        "h_phrases": [],
        "p_phrases": [],
        "ld50": None,
        "flash_point": None,
    }

    if cid is None:
        encoded = quote(smiles, safe="")
        cid_url = f"{PUBCHEM_BASE_URL}/compound/smiles/{encoded}/cids/JSON"
        cid_data = _get_json(cid_url)
        if not cid_data:
            return result
        try:
            cid = cid_data["IdentifierList"]["CID"][0]
        except (KeyError, IndexError, TypeError):
            return result

    # GHS Classification
    ghs_url = f"{PUBCHEM_VIEW_URL}/data/compound/{cid}/JSON?heading=GHS+Classification"
    ghs_data = _get_json(ghs_url)
    if not ghs_data:
        return result

    try:
        sections = ghs_data["Record"]["Section"]
    except (KeyError, TypeError):
        return result

    pictograms: list[str] = []
    h_phrases: list[str] = []
    p_phrases: list[str] = []

    def _walk(sections_list: list[dict]) -> None:
        for sec in sections_list:
            for info in sec.get("Information", []):
                name = info.get("Name", "").lower()
                val = info.get("Value", {})

                for swm in val.get("StringWithMarkup", []):
                    text = swm.get("String", "")

                    # Pictograms — check both URL and Extra fields
                    if "pictogram" in name:
                        for markup in swm.get("Markup", []):
                            for field in ("URL", "Extra"):
                                val_str = markup.get(field, "")
                                m = re.search(r"GHS\d{2}", val_str)
                                if m and m.group() not in pictograms:
                                    pictograms.append(m.group())
                        if text:
                            m = re.search(r"GHS\d{2}", text)
                            if m and m.group() not in pictograms:
                                pictograms.append(m.group())

                    # H-phrases
                    if "hazard" in name and "statement" in name and text:
                        found_h = re.findall(r"H\d{3}[A-Za-z]?[^;,\[]*", text)
                        h_phrases.extend(s.strip() for s in found_h if s.strip())

                    # P-phrases
                    if "precautionary" in name and text:
                        found_p = re.findall(r"P\d{3}(?:\+P\d{3})*", text)
                        p_phrases.extend(s.strip() for s in found_p if s.strip())

            children = sec.get("Section", [])
            if children:
                _walk(children)

    _walk(sections)

    result["ghs_pictograms"] = sorted(set(pictograms))
    result["h_phrases"] = sorted(set(h_phrases))
    result["p_phrases"] = sorted(set(p_phrases))

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# GHS pictogram info (image URLs + descriptions)
# ═══════════════════════════════════════════════════════════════════════════════

GHS_PICTOGRAMS: dict[str, dict[str, str]] = {
    "GHS01": {
        "name_ru": "Взрывающаяся бомба",
        "name_en": "Exploding Bomb",
        "description": "Взрывчатые вещества, самореактивные вещества, органические пероксиды",
        "image_svg": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS01.svg",
        "image_gif": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS01.gif",
    },
    "GHS02": {
        "name_ru": "Пламя",
        "name_en": "Flame",
        "description": "Воспламеняющиеся газы, аэрозоли, жидкости, твёрдые вещества; пирофорные; самонагревающиеся",
        "image_svg": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS02.svg",
        "image_gif": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS02.gif",
    },
    "GHS03": {
        "name_ru": "Пламя над кругом",
        "name_en": "Flame Over Circle",
        "description": "Окисляющие газы, жидкости, твёрдые вещества",
        "image_svg": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS03.svg",
        "image_gif": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS03.gif",
    },
    "GHS04": {
        "name_ru": "Газовый баллон",
        "name_en": "Gas Cylinder",
        "description": "Сжатые, сжиженные, охлаждённые или растворённые газы под давлением",
        "image_svg": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS04.svg",
        "image_gif": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS04.gif",
    },
    "GHS05": {
        "name_ru": "Коррозия",
        "name_en": "Corrosion",
        "description": "Коррозийно для металлов; вызывает тяжёлые ожоги кожи и повреждение глаз",
        "image_svg": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS05.svg",
        "image_gif": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS05.gif",
    },
    "GHS06": {
        "name_ru": "Череп и кости",
        "name_en": "Skull and Crossbones",
        "description": "Острая токсичность (смертельно/токсично при проглатывании, контакте с кожей, вдыхании)",
        "image_svg": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS06.svg",
        "image_gif": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS06.gif",
    },
    "GHS07": {
        "name_ru": "Восклицательный знак",
        "name_en": "Exclamation Mark",
        "description": "Раздражение кожи/глаз; острая токсичность (вредно); наркотические эффекты",
        "image_svg": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS07.svg",
        "image_gif": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS07.gif",
    },
    "GHS08": {
        "name_ru": "Опасность для здоровья",
        "name_en": "Health Hazard",
        "description": "Канцерогенность, мутагенность, репродуктивная токсичность, поражение органов-мишеней",
        "image_svg": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS08.svg",
        "image_gif": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS08.gif",
    },
    "GHS09": {
        "name_ru": "Окружающая среда",
        "name_en": "Environment",
        "description": "Опасно для водной среды (острая и хроническая токсичность)",
        "image_svg": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS09.svg",
        "image_gif": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS09.gif",
    },
}


def get_ghs_pictogram_info(code: str) -> dict[str, str] | None:
    """Get pictogram info by GHS code (e.g. 'GHS02').

    Returns dict with: name_ru, name_en, description, image_svg, image_gif.
    """
    return GHS_PICTOGRAMS.get(code)


def enrich_ghs_pictograms(codes: list[str]) -> list[dict[str, str]]:
    """Convert list of GHS codes to enriched pictogram data for frontend.

    Returns list of dicts, each with: code, name_ru, description, image_svg, image_gif.
    """
    result = []
    for code in codes:
        info = GHS_PICTOGRAMS.get(code)
        if info:
            result.append({"code": code, **info})
        else:
            result.append({
                "code": code,
                "name_ru": code,
                "name_en": code,
                "description": "",
                "image_svg": f"https://pubchem.ncbi.nlm.nih.gov/images/ghs/{code}.svg",
                "image_gif": f"https://pubchem.ncbi.nlm.nih.gov/images/ghs/{code}.gif",
            })
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ppe_recommender — PPE based on H-phrases
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
    """Рекомендации СИЗ на основе H-фраз.

    Args:
        substances: SMILES (только для логирования).
        h_phrases: H-фразы через запятую (напр. "H225,H319").

    Returns:
        Отсортированный список рекомендаций СИЗ на русском.
    """
    ppe_set: set[str] = {"Лабораторный халат", "Нитриловые перчатки", "Защитные очки"}

    codes = re.findall(r"H\d{3}[A-Za-z]?", h_phrases)
    for code in codes:
        base = code[:4]
        if base in _PPE_MAP:
            ppe_set.update(_PPE_MAP[base])

    return sorted(ppe_set)


# ═══════════════════════════════════════════════════════════════════════════════
# PubChem lookup (compound info)
# ═══════════════════════════════════════════════════════════════════════════════

def pubchem_lookup(name_or_smiles: str) -> dict:
    """Get compound data from PubChem by name or SMILES.

    Returns dict with: cid, formula, weight, iupac, smiles, logp, tpsa, synonyms.
    """
    # Try by name first
    base_url = f"{PUBCHEM_BASE_URL}/compound/name"
    encoded = quote(name_or_smiles, safe="")
    props_url = (
        f"{base_url}/{encoded}/property/"
        "MolecularFormula,MolecularWeight,IUPACName,IsomericSMILES,CanonicalSMILES,XLogP,TPSA/JSON"
    )

    data = _get_json(props_url)

    # If name lookup fails, try SMILES
    if data is None:
        base_url = f"{PUBCHEM_BASE_URL}/compound/smiles"
        props_url = (
            f"{base_url}/{encoded}/property/"
            "MolecularFormula,MolecularWeight,IUPACName,IsomericSMILES,CanonicalSMILES,XLogP,TPSA/JSON"
        )
        data = _get_json(props_url)

    if data is None:
        return {"error": f"Not found in PubChem: {name_or_smiles}"}

    try:
        props = data["PropertyTable"]["Properties"][0]
    except (KeyError, IndexError, TypeError):
        return {"error": "Unexpected PubChem response format"}

    cid = props.get("CID")
    synonyms: list[str] = []
    if cid:
        syn_url = f"{PUBCHEM_BASE_URL}/compound/cid/{cid}/synonyms/JSON"
        syn_data = _get_json(syn_url)
        if syn_data:
            try:
                all_syns = (
                    syn_data.get("InformationList", {})
                    .get("Information", [{}])[0]
                    .get("Synonym", [])
                )
                synonyms = all_syns[:5]
            except (IndexError, TypeError):
                pass

    return {
        "cid": cid,
        "formula": props.get("MolecularFormula"),
        "weight": props.get("MolecularWeight"),
        "iupac": props.get("IUPACName"),
        "smiles": props.get("IsomericSMILES") or props.get("CanonicalSMILES"),
        "logp": props.get("XLogP"),
        "tpsa": props.get("TPSA"),
        "synonyms": synonyms,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RDKit properties
# ═══════════════════════════════════════════════════════════════════════════════

def rdkit_properties(smiles: str) -> dict:
    """Calculate molecular properties from SMILES using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: {smiles}"}

    return {
        "molecular_weight": round(Descriptors.MolWt(mol), 4),
        "logp": round(Descriptors.MolLogP(mol), 4),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "h_bond_acceptors": Lipinski.NumHAcceptors(mol),
        "h_bond_donors": Lipinski.NumHDonors(mol),
        "heavy_atoms": Descriptors.HeavyAtomCount(mol),
        "ring_count": Lipinski.RingCount(mol),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Physical description from PubChem PUG View
# ═══════════════════════════════════════════════════════════════════════════════

def get_physical_description(smiles: str, cid: int | None = None) -> list[str]:
    """Fetch physical description texts from PubChem (color, form, odor, etc.).

    Args:
        smiles: SMILES (fallback for CID resolution).
        cid: PubChem CID (preferred).

    Returns a list of unique, meaningful description strings.
    """
    if cid is None:
        encoded = quote(smiles, safe="")
        cid_url = f"{PUBCHEM_BASE_URL}/compound/smiles/{encoded}/cids/JSON"
        cid_data = _get_json(cid_url)
        if not cid_data:
            return []
        try:
            cid = cid_data["IdentifierList"]["CID"][0]
        except (KeyError, IndexError, TypeError):
            return []

    url = f"{PUBCHEM_VIEW_URL}/data/compound/{cid}/JSON?heading=Physical+Description"
    data = _get_json(url)
    if not data:
        return []

    raw_descriptions: list[str] = []
    try:
        sections = data["Record"]["Section"]
    except (KeyError, TypeError):
        return []

    def _walk_phys(secs: list[dict]) -> None:
        for sec in secs:
            for info in sec.get("Information", []):
                val = info.get("Value", {})
                for swm in val.get("StringWithMarkup", []):
                    text = swm.get("String", "").strip()
                    if text:
                        raw_descriptions.append(text)
            children = sec.get("Section", [])
            if children:
                _walk_phys(children)

    _walk_phys(sections)

    # Filter: skip too short (<10 chars like "Solid"), deduplicate by content
    seen_lower: set[str] = set()
    filtered: list[str] = []
    for desc in raw_descriptions:
        lower = desc.lower().strip(" .")
        if len(desc) < 10:
            continue
        if lower in seen_lower:
            continue
        # Skip if already covered by a longer description
        if any(lower in existing for existing in seen_lower):
            continue
        seen_lower.add(lower)
        filtered.append(desc)

    return filtered


# ═══════════════════════════════════════════════════════════════════════════════
# Molecule image URLs (2D + 3D)
# ═══════════════════════════════════════════════════════════════════════════════

def get_molecule_images(smiles: str, cid: int | None = None) -> dict[str, str]:
    """Get URLs for 2D structure image and 3D conformer from PubChem.

    Returns dict with keys: image_2d, image_3d, pubchem_url.
    """
    result: dict[str, str] = {
        "image_2d": "",
        "image_3d": "",
        "pubchem_url": "",
    }

    if cid is None:
        encoded = quote(smiles, safe="")
        cid_url = f"{PUBCHEM_BASE_URL}/compound/smiles/{encoded}/cids/JSON"
        cid_data = _get_json(cid_url)
        if cid_data:
            try:
                cid = cid_data["IdentifierList"]["CID"][0]
            except (KeyError, IndexError, TypeError):
                pass

    if cid:
        result["image_2d"] = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG?image_size=300x300"
        result["image_3d"] = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG?record_type=3d&image_size=300x300"
        result["pubchem_url"] = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"
    elif smiles:
        encoded = quote(smiles, safe="")
        result["image_2d"] = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded}/PNG?image_size=300x300"

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PubChem CID resolvers (for validation node)
# ═══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=512)
def get_cid_by_smiles(smiles: str) -> int | None:
    encoded = quote(smiles, safe="")
    url = f"{PUBCHEM_BASE_URL}/compound/smiles/{encoded}/cids/JSON"
    data = _get_json(url)
    if not data:
        return None
    try:
        return data["IdentifierList"]["CID"][0]
    except (KeyError, IndexError, TypeError):
        return None


@lru_cache(maxsize=512)
def get_cid_by_name(name: str) -> int | None:
    encoded = quote(name, safe="")
    url = f"{PUBCHEM_BASE_URL}/compound/name/{encoded}/cids/JSON"
    data = _get_json(url)
    if not data:
        return None
    try:
        return data["IdentifierList"]["CID"][0]
    except (KeyError, IndexError, TypeError):
        return None


@lru_cache(maxsize=512)
def get_smiles_by_cid(cid: int) -> str | None:
    url = f"{PUBCHEM_BASE_URL}/compound/cid/{cid}/property/CanonicalSMILES,IsomericSMILES/JSON"
    data = _get_json(url)
    if not data:
        return None
    try:
        props = data["PropertyTable"]["Properties"][0]
        return (
            props.get("CanonicalSMILES")
            or props.get("IsomericSMILES")
            or props.get("ConnectivitySMILES")
            or props.get("SMILES")
        )
    except (KeyError, IndexError, TypeError):
        return None


def get_compound_properties(smiles: str) -> dict[str, Any]:
    encoded = quote(smiles, safe="")
    url = (
        f"{PUBCHEM_BASE_URL}/compound/smiles/{encoded}/property/"
        "MolecularWeight,MolecularFormula,IUPACName,IsomericSMILES,InChI,InChIKey/JSON"
    )
    data = _get_json(url)
    if not data:
        return {}
    try:
        return data["PropertyTable"]["Properties"][0]
    except (KeyError, IndexError, TypeError):
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# Experimental properties from PubChem PUG View
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_numeric(text: str) -> float | None:
    """Pull the first number from text."""
    m = re.search(r"(-?\d+\.?\d*)", text)
    return float(m.group(1)) if m else None


def _walk_pug_view(sections: list[dict], target_heading: str) -> list[str]:
    """Recursively find a section by heading and extract all StringWithMarkup texts."""
    results: list[str] = []

    def _walk(secs: list[dict]) -> None:
        for sec in secs:
            heading = sec.get("TOCHeading", "")
            if heading.lower() == target_heading.lower():
                for info in sec.get("Information", []):
                    val = info.get("Value", {})
                    for swm in val.get("StringWithMarkup", []):
                        text = swm.get("String", "").strip()
                        if text:
                            results.append(text)
                    # Also check numeric values
                    nums = val.get("Number")
                    unit = val.get("Unit", "")
                    if nums is not None:
                        ns = nums if isinstance(nums, list) else [nums]
                        results.append(f"{ns[0]} {unit}".strip())
            children = sec.get("Section", [])
            if children:
                _walk(children)

    _walk(sections)
    return results


def get_experimental_properties(cid: int) -> dict[str, Any]:
    """Fetch experimental properties from PubChem PUG View.

    Returns dict with: melting_point, boiling_point, density, solubility,
    flash_point, vapor_pressure, logp, cas_number.
    """
    result: dict[str, Any] = {
        "melting_point": None,
        "boiling_point": None,
        "density": None,
        "solubility": None,
        "flash_point": None,
        "vapor_pressure": None,
        "logp": None,
    }

    url = f"{PUBCHEM_VIEW_URL}/data/compound/{cid}/JSON?heading=Experimental+Properties"
    data = _get_json(url)
    if not data:
        return result

    try:
        sections = data["Record"]["Section"]
    except (KeyError, TypeError):
        return result

    # Melting Point
    mp_texts = _walk_pug_view(sections, "Melting Point")
    if mp_texts:
        for t in mp_texts:
            if "°C" in t or "deg" in t.lower() or "°" in t:
                val = _extract_numeric(t)
                if val is not None and -300 < val < 5000:
                    result["melting_point"] = val
                    break
        if result["melting_point"] is None:
            val = _extract_numeric(mp_texts[0])
            if val is not None:
                result["melting_point"] = val

    # Boiling Point
    bp_texts = _walk_pug_view(sections, "Boiling Point")
    if bp_texts:
        for t in bp_texts:
            if "°C" in t or "deg" in t.lower() or "°" in t:
                val = _extract_numeric(t)
                if val is not None and -300 < val < 5000:
                    result["boiling_point"] = val
                    break
        if result["boiling_point"] is None:
            val = _extract_numeric(bp_texts[0])
            if val is not None:
                result["boiling_point"] = val

    # Density
    density_texts = _walk_pug_view(sections, "Density")
    if density_texts:
        val = _extract_numeric(density_texts[0])
        if val is not None and 0 < val < 25:
            result["density"] = val

    # Solubility
    sol_texts = _walk_pug_view(sections, "Solubility")
    if sol_texts:
        result["solubility"] = sol_texts[0]

    # Flash Point
    fp_texts = _walk_pug_view(sections, "Flash Point")
    if fp_texts:
        val = _extract_numeric(fp_texts[0])
        if val is not None:
            result["flash_point"] = val

    # Vapor Pressure
    vp_texts = _walk_pug_view(sections, "Vapor Pressure")
    if vp_texts:
        result["vapor_pressure"] = vp_texts[0]

    # LogP
    logp_texts = _walk_pug_view(sections, "LogP")
    if logp_texts:
        val = _extract_numeric(logp_texts[0])
        if val is not None:
            result["logp"] = val

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# LD50 / Toxicity from PubChem
# ═══════════════════════════════════════════════════════════════════════════════

def get_ld50(cid: int) -> dict[str, Any]:
    """Fetch LD50 / acute toxicity data from PubChem.

    Returns dict with: ld50_oral, ld50_dermal, ld50_inhalation (strings).
    """
    result: dict[str, Any] = {
        "ld50_oral": None,
        "ld50_dermal": None,
        "ld50_inhalation": None,
    }

    url = f"{PUBCHEM_VIEW_URL}/data/compound/{cid}/JSON?heading=Acute+Effects"
    data = _get_json(url)
    if not data:
        return result

    try:
        sections = data["Record"]["Section"]
    except (KeyError, TypeError):
        return result

    all_texts: list[str] = []

    def _walk_ld50(secs: list[dict]) -> None:
        for sec in secs:
            for info in sec.get("Information", []):
                val = info.get("Value", {})
                for swm in val.get("StringWithMarkup", []):
                    text = swm.get("String", "").strip()
                    if text and "LD50" in text.upper():
                        all_texts.append(text)
            children = sec.get("Section", [])
            if children:
                _walk_ld50(children)

    _walk_ld50(sections)

    for text in all_texts:
        text_lower = text.lower()
        if "oral" in text_lower and result["ld50_oral"] is None:
            result["ld50_oral"] = text
        elif "dermal" in text_lower and result["ld50_dermal"] is None:
            result["ld50_dermal"] = text
        elif "inhal" in text_lower and result["ld50_inhalation"] is None:
            result["ld50_inhalation"] = text

    # If no route-specific found, take first one
    if all(v is None for v in result.values()) and all_texts:
        result["ld50_oral"] = all_texts[0]

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CAS number from PubChem synonyms
# ═══════════════════════════════════════════════════════════════════════════════

_CAS_PATTERN = re.compile(r"^\d{2,7}-\d{2}-\d$")


def get_cas_number(cid: int) -> str | None:
    """Extract CAS Registry Number from PubChem synonyms.

    CAS numbers appear as synonyms in format: NNNNN-NN-N
    """
    url = f"{PUBCHEM_BASE_URL}/compound/cid/{cid}/synonyms/JSON"
    data = _get_json(url)
    if not data:
        return None

    try:
        synonyms = (
            data.get("InformationList", {})
            .get("Information", [{}])[0]
            .get("Synonym", [])
        )
    except (IndexError, TypeError):
        return None

    for syn in synonyms:
        if _CAS_PATTERN.match(syn.strip()):
            return syn.strip()

    return None
