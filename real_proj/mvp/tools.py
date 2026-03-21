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


def safety_lookup(smiles: str) -> dict:
    """Fetch GHS safety data from PubChem for a SMILES string.

    Returns dict with: ghs_pictograms, h_phrases, p_phrases, ld50, flash_point.
    """
    result: dict[str, Any] = {
        "ghs_pictograms": [],
        "h_phrases": [],
        "p_phrases": [],
        "ld50": None,
        "flash_point": None,
    }

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
            heading = sec.get("TOCHeading", "").lower()

            for info in sec.get("Information", []):
                val = info.get("Value", {})
                for swm in val.get("StringWithMarkup", []):
                    text = swm.get("String", "")
                    if not text:
                        continue

                    # Pictograms
                    if "pictogram" in heading:
                        m = re.search(r"GHS\d{2}", text)
                        if m:
                            pictograms.append(m.group())
                    # Also check markup for pictogram URLs
                    for markup in swm.get("Markup", []):
                        extra = markup.get("Extra", "") or markup.get("URL", "")
                        m = re.search(r"GHS\d{2}", extra)
                        if m and m.group() not in pictograms:
                            pictograms.append(m.group())

                    # H/P phrases
                    if heading in ("hazard statements", "hazard statement",
                                   "ghs hazard statements"):
                        found_h = re.findall(r"H\d{3}[A-Za-z]?[^;]*", text)
                        h_phrases.extend(s.strip() for s in found_h if s.strip())
                    elif heading in ("precautionary statements", "precautionary statement",
                                     "precautionary statement codes"):
                        found_p = re.findall(r"P\d{3}[A-Za-z]?[^;]*", text)
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
# ppe_recommender — PPE based on H-phrases
# ═══════════════════════════════════════════════════════════════════════════════

_PPE_MAP: dict[str, list[str]] = {
    "H200": ["Explosion-proof equipment nearby"],
    "H201": ["Explosion-proof equipment nearby"],
    "H202": ["Explosion-proof equipment nearby"],
    "H220": ["Flame-resistant lab coat", "Explosion-proof equipment nearby"],
    "H221": ["Flame-resistant lab coat"],
    "H224": ["Flame-resistant lab coat", "Explosion-proof equipment nearby"],
    "H225": ["Flame-resistant lab coat"],
    "H226": ["Flame-resistant lab coat"],
    "H290": ["Chemical-resistant gloves"],
    "H300": ["Nitrile gloves", "Full face shield", "Fume hood required"],
    "H301": ["Nitrile gloves", "Fume hood required"],
    "H302": ["Nitrile gloves"],
    "H310": ["Chemical-resistant suit", "Nitrile gloves", "Full face shield"],
    "H311": ["Nitrile gloves", "Chemical-resistant apron"],
    "H312": ["Nitrile gloves"],
    "H314": ["Chemical splash goggles", "Face shield", "Chemical-resistant gloves"],
    "H315": ["Nitrile gloves"],
    "H317": ["Nitrile gloves"],
    "H318": ["Chemical splash goggles"],
    "H319": ["Safety goggles"],
    "H330": ["Self-contained breathing apparatus (SCBA)", "Fume hood required"],
    "H331": ["Fume hood required", "Respiratory protection"],
    "H332": ["Fume hood required"],
    "H334": ["Respiratory protection", "Fume hood required"],
    "H335": ["Fume hood required"],
    "H336": ["Fume hood required"],
    "H340": ["Fume hood required", "Nitrile gloves", "Minimize exposure"],
    "H341": ["Fume hood required", "Nitrile gloves"],
    "H350": ["Fume hood required", "Nitrile gloves", "Minimize exposure"],
    "H351": ["Fume hood required", "Nitrile gloves"],
    "H360": ["Full protective equipment", "Minimize exposure"],
    "H370": ["Full protective equipment", "Fume hood required"],
    "H372": ["Full protective equipment", "Fume hood required"],
    "H400": ["Contain spills", "Chemical-resistant gloves"],
    "H410": ["Contain spills", "Chemical-resistant gloves"],
}


def ppe_recommender(substances: str, h_phrases: str) -> list[str]:
    """Recommend PPE based on H-phrases.

    Args:
        substances: SMILES string (for logging only).
        h_phrases: comma-separated H-phrases (e.g. "H225,H319").

    Returns:
        Sorted deduplicated list of PPE recommendations.
    """
    ppe_set: set[str] = {"Lab coat", "Nitrile gloves", "Safety goggles"}

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
    url = f"{PUBCHEM_BASE_URL}/compound/cid/{cid}/property/CanonicalSMILES/JSON"
    data = _get_json(url)
    if not data:
        return None
    try:
        props = data["PropertyTable"]["Properties"][0]
        return props.get("CanonicalSMILES")
    except (KeyError, IndexError, TypeError):
        return None


def get_compound_properties(smiles: str) -> dict[str, Any]:
    encoded = quote(smiles, safe="")
    url = (
        f"{PUBCHEM_BASE_URL}/compound/smiles/{encoded}/property/"
        "MolecularWeight,MolecularFormula,IUPACName,IsomericSMILES/JSON"
    )
    data = _get_json(url)
    if not data:
        return {}
    try:
        return data["PropertyTable"]["Properties"][0]
    except (KeyError, IndexError, TypeError):
        return {}
