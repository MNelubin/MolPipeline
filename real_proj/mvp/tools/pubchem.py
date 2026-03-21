"""PubChem tools: lookup, properties, safety, images, experimental data."""

from __future__ import annotations

import logging
import re
import time
from functools import lru_cache
from typing import Any
from urllib.parse import quote

import requests

from ..config import PUBCHEM_BASE_URL, PUBCHEM_VIEW_URL

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
# CID resolvers
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
# pubchem_lookup (compound info)
# ═══════════════════════════════════════════════════════════════════════════════

def pubchem_lookup(name_or_smiles: str) -> dict:
    """Get compound data from PubChem by name or SMILES."""
    base_url = f"{PUBCHEM_BASE_URL}/compound/name"
    encoded = quote(name_or_smiles, safe="")
    props_url = (
        f"{base_url}/{encoded}/property/"
        "MolecularFormula,MolecularWeight,IUPACName,IsomericSMILES,CanonicalSMILES,XLogP,TPSA/JSON"
    )

    data = _get_json(props_url)

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
# Physical description from PubChem PUG View
# ═══════════════════════════════════════════════════════════════════════════════

def get_physical_description(smiles: str, cid: int | None = None) -> list[str]:
    """Fetch physical description texts from PubChem (color, form, odor, etc.)."""
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

    seen_lower: set[str] = set()
    filtered: list[str] = []
    for desc in raw_descriptions:
        lower = desc.lower().strip(" .")
        if len(desc) < 10:
            continue
        if lower in seen_lower:
            continue
        if any(lower in existing for existing in seen_lower):
            continue
        seen_lower.add(lower)
        filtered.append(desc)

    return filtered


# ═══════════════════════════════════════════════════════════════════════════════
# Molecule image URLs (2D + 3D)
# ═══════════════════════════════════════════════════════════════════════════════

def get_molecule_images(smiles: str, cid: int | None = None) -> dict[str, str]:
    """Get URLs for 2D structure image and 3D conformer from PubChem."""
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
# Safety lookup — GHS data from PubChem
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_h_p_phrases(text: str) -> tuple[list[str], list[str]]:
    h_phrases = re.findall(r"(H\d{3}[A-Za-z]?(?:\s*\+\s*H\d{3}[A-Za-z]?)*[^HP]*?)(?=(?:H\d|P\d|$))", text)
    p_phrases = re.findall(r"(P\d{3}[A-Za-z]?(?:\s*\+\s*P\d{3}[A-Za-z]?)*[^HP]*?)(?=(?:H\d|P\d|$))", text)
    return (
        [h.strip() for h in h_phrases if h.strip()],
        [p.strip() for p in p_phrases if p.strip()],
    )


def safety_lookup(smiles: str, cid: int | None = None) -> dict:
    """Fetch GHS safety data from PubChem."""
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

                    if "hazard" in name and "statement" in name and text:
                        found_h = re.findall(r"H\d{3}[A-Za-z]?[^;,\[]*", text)
                        h_phrases.extend(s.strip() for s in found_h if s.strip())

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
# GHS pictogram info
# ═══════════════════════════════════════════════════════════════════════════════

GHS_PICTOGRAMS: dict[str, dict[str, str]] = {
    "GHS01": {"name_ru": "Взрывающаяся бомба", "name_en": "Exploding Bomb", "description": "Взрывчатые вещества, самореактивные вещества, органические пероксиды", "image_svg": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS01.svg", "image_gif": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS01.gif"},
    "GHS02": {"name_ru": "Пламя", "name_en": "Flame", "description": "Воспламеняющиеся газы, аэрозоли, жидкости, твёрдые вещества; пирофорные; самонагревающиеся", "image_svg": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS02.svg", "image_gif": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS02.gif"},
    "GHS03": {"name_ru": "Пламя над кругом", "name_en": "Flame Over Circle", "description": "Окисляющие газы, жидкости, твёрдые вещества", "image_svg": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS03.svg", "image_gif": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS03.gif"},
    "GHS04": {"name_ru": "Газовый баллон", "name_en": "Gas Cylinder", "description": "Сжатые, сжиженные, охлаждённые или растворённые газы под давлением", "image_svg": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS04.svg", "image_gif": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS04.gif"},
    "GHS05": {"name_ru": "Коррозия", "name_en": "Corrosion", "description": "Коррозийно для металлов; вызывает тяжёлые ожоги кожи и повреждение глаз", "image_svg": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS05.svg", "image_gif": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS05.gif"},
    "GHS06": {"name_ru": "Череп и кости", "name_en": "Skull and Crossbones", "description": "Острая токсичность (смертельно/токсично при проглатывании, контакте с кожей, вдыхании)", "image_svg": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS06.svg", "image_gif": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS06.gif"},
    "GHS07": {"name_ru": "Восклицательный знак", "name_en": "Exclamation Mark", "description": "Раздражение кожи/глаз; острая токсичность (вредно); наркотические эффекты", "image_svg": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS07.svg", "image_gif": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS07.gif"},
    "GHS08": {"name_ru": "Опасность для здоровья", "name_en": "Health Hazard", "description": "Канцерогенность, мутагенность, репродуктивная токсичность, поражение органов-мишеней", "image_svg": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS08.svg", "image_gif": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS08.gif"},
    "GHS09": {"name_ru": "Окружающая среда", "name_en": "Environment", "description": "Опасно для водной среды (острая и хроническая токсичность)", "image_svg": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS09.svg", "image_gif": "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS09.gif"},
}


def get_ghs_pictogram_info(code: str) -> dict[str, str] | None:
    return GHS_PICTOGRAMS.get(code)


def enrich_ghs_pictograms(codes: list[str]) -> list[dict[str, str]]:
    """Convert list of GHS codes to enriched pictogram data for frontend."""
    result = []
    for code in codes:
        info = GHS_PICTOGRAMS.get(code)
        if info:
            result.append({"code": code, **info})
        else:
            result.append({
                "code": code, "name_ru": code, "name_en": code, "description": "",
                "image_svg": f"https://pubchem.ncbi.nlm.nih.gov/images/ghs/{code}.svg",
                "image_gif": f"https://pubchem.ncbi.nlm.nih.gov/images/ghs/{code}.gif",
            })
    return result


def get_ghs_safety(smiles: str, cid: int | None = None) -> dict:
    """Alias for safety_lookup for backward compatibility."""
    return safety_lookup(smiles, cid)


# ═══════════════════════════════════════════════════════════════════════════════
# Experimental properties from PubChem PUG View
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_numeric(text: str) -> float | None:
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
    """Fetch experimental properties from PubChem PUG View."""
    result: dict[str, Any] = {
        "melting_point": None, "boiling_point": None, "density": None,
        "solubility": None, "flash_point": None, "vapor_pressure": None, "logp": None,
    }

    url = f"{PUBCHEM_VIEW_URL}/data/compound/{cid}/JSON?heading=Experimental+Properties"
    data = _get_json(url)
    if not data:
        return result

    try:
        sections = data["Record"]["Section"]
    except (KeyError, TypeError):
        return result

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

    density_texts = _walk_pug_view(sections, "Density")
    if density_texts:
        val = _extract_numeric(density_texts[0])
        if val is not None and 0 < val < 25:
            result["density"] = val

    sol_texts = _walk_pug_view(sections, "Solubility")
    if sol_texts:
        result["solubility"] = sol_texts[0]

    fp_texts = _walk_pug_view(sections, "Flash Point")
    if fp_texts:
        val = _extract_numeric(fp_texts[0])
        if val is not None:
            result["flash_point"] = val

    vp_texts = _walk_pug_view(sections, "Vapor Pressure")
    if vp_texts:
        result["vapor_pressure"] = vp_texts[0]

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
    """Fetch LD50 / acute toxicity data from PubChem."""
    result: dict[str, Any] = {"ld50_oral": None, "ld50_dermal": None, "ld50_inhalation": None}

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

    if all(v is None for v in result.values()) and all_texts:
        result["ld50_oral"] = all_texts[0]

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CAS number from PubChem synonyms
# ═══════════════════════════════════════════════════════════════════════════════

_CAS_PATTERN = re.compile(r"^\d{2,7}-\d{2}-\d$")


def get_cas_number(cid: int) -> str | None:
    """Extract CAS Registry Number from PubChem synonyms."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# Functions merged from src/services/pubchem.py
# ═══════════════════════════════════════════════════════════════════════════════

def _fahrenheit_to_celsius(f: float) -> float:
    return (f - 32) * 5 / 9


def _walk_sections(sections: list[dict[str, Any]], heading: str) -> dict[str, Any] | None:
    """Recursively find a section with the given TOCHeading."""
    for sec in sections:
        if sec.get("TOCHeading", "").lower() == heading.lower():
            return sec
        children = sec.get("Section", [])
        if children:
            result = _walk_sections(children, heading)
            if result is not None:
                return result
    return None


def _extract_temperature_celsius(section: dict[str, Any]) -> float | None:
    celsius_values: list[float] = []
    fahrenheit_values: list[float] = []

    for info in section.get("Information", []):
        val = info.get("Value", {})
        for swm in val.get("StringWithMarkup", []):
            text: str = swm.get("String", "")
            if not text:
                continue
            num = _extract_numeric(text)
            if num is None:
                continue
            text_lower = text.lower()
            if "\u00b0c" in text_lower or "deg c" in text_lower or "\u00b0 c" in text_lower:
                celsius_values.append(num)
            elif "\u00b0f" in text_lower or "deg f" in text_lower or "\u00b0 f" in text_lower:
                fahrenheit_values.append(num)

        nums = val.get("Number")
        unit = val.get("Unit", "")
        if nums is not None:
            ns = nums if isinstance(nums, list) else [nums]
            if "c" in unit.lower():
                celsius_values.append(float(ns[0]))
            elif "f" in unit.lower():
                fahrenheit_values.append(float(ns[0]))

    if celsius_values:
        return celsius_values[0]
    if fahrenheit_values:
        return round(_fahrenheit_to_celsius(fahrenheit_values[0]), 2)
    return None


def _extract_string_value(section: dict[str, Any]) -> str | None:
    for info in section.get("Information", []):
        val = info.get("Value", {})
        for swm in val.get("StringWithMarkup", []):
            text = swm.get("String", "")
            if text:
                return text
        num = val.get("Number")
        if num is not None:
            nums = num if isinstance(num, list) else [num]
            unit = val.get("Unit", "")
            return f"{nums[0]} {unit}".strip()
    return None


@lru_cache(maxsize=512)
def get_density(smiles: str) -> float | None:
    """Fetch density (g/mL) for the compound from PubChem PUG View."""
    cid = get_cid_by_smiles(smiles)
    if cid is None:
        return None
    url = f"{PUBCHEM_VIEW_URL}/data/compound/{cid}/JSON?heading=Density"
    data = _get_json(url)
    if data is None:
        return None
    try:
        sections = data["Record"]["Section"]
    except (KeyError, TypeError):
        return None
    density_sec = _walk_sections(sections, "Density")
    if density_sec is None:
        return None
    text = _extract_string_value(density_sec)
    if text is None:
        return None
    return _extract_numeric(text)


@lru_cache(maxsize=512)
def get_boiling_point(smiles: str) -> float | None:
    """Fetch boiling point (deg C) from PubChem."""
    cid = get_cid_by_smiles(smiles)
    if cid is None:
        return None
    url = f"{PUBCHEM_VIEW_URL}/data/compound/{cid}/JSON?heading=Boiling+Point"
    data = _get_json(url)
    if data is None:
        return None
    try:
        sections = data["Record"]["Section"]
    except (KeyError, TypeError):
        return None
    sec = _walk_sections(sections, "Boiling Point")
    if sec is None:
        return None
    return _extract_temperature_celsius(sec)


@lru_cache(maxsize=512)
def get_melting_point(smiles: str) -> float | None:
    """Fetch melting point (deg C) from PubChem."""
    cid = get_cid_by_smiles(smiles)
    if cid is None:
        return None
    url = f"{PUBCHEM_VIEW_URL}/data/compound/{cid}/JSON?heading=Melting+Point"
    data = _get_json(url)
    if data is None:
        return None
    try:
        sections = data["Record"]["Section"]
    except (KeyError, TypeError):
        return None
    sec = _walk_sections(sections, "Melting Point")
    if sec is None:
        return None
    return _extract_temperature_celsius(sec)


def estimate_physical_state(smiles: str) -> str:
    """Heuristic: solid / liquid / gas at ~25 C based on melting & boiling points."""
    mp = get_melting_point(smiles)
    bp = get_boiling_point(smiles)
    if mp is not None and mp > 25:
        return "solid"
    if bp is not None and bp < 25:
        return "gas"
    if mp is not None and mp <= 25:
        return "liquid"
    if bp is not None and bp >= 25:
        return "liquid"
    density = get_density(smiles)
    if density is not None and density > 0:
        return "liquid"
    return "unknown"


@lru_cache(maxsize=512)
def get_iupac_name(smiles: str) -> str:
    """Return IUPAC name from PubChem, or empty string on failure."""
    props = get_compound_properties(smiles)
    name = props.get("IUPACName", "")
    if name:
        return name
    cid = get_cid_by_smiles(smiles)
    if cid is None:
        return ""
    url = f"{PUBCHEM_BASE_URL}/compound/cid/{cid}/property/IUPACName/JSON"
    data = _get_json(url)
    if data is None:
        return ""
    try:
        return data["PropertyTable"]["Properties"][0].get("IUPACName", "")
    except (KeyError, IndexError, TypeError):
        return ""
