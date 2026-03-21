"""Validation node: detect input type (SMILES vs name), validate, resolve via PubChem.

If a name contains Cyrillic characters and PubChem lookup fails,
uses LLM to translate it to English before retrying.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from ..config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL
from ..tools import get_cid_by_name, get_cid_by_smiles, get_smiles_by_cid, get_compound_properties

logger = logging.getLogger(__name__)

_CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁ]")

_SMILES_PATTERN = re.compile(
    r"^[A-Za-z0-9@+\-\[\]\(\)\\/=#$%.:~]+$"
)


def _detect_input_type(user_input: str) -> str:
    """Heuristic: is this SMILES or a compound name?"""
    stripped = user_input.strip()
    if " " in stripped:
        return "name"
    if not _SMILES_PATTERN.match(stripped):
        return "name"

    smiles_chars = set("=()[]@/\\#%+")
    if smiles_chars & set(stripped):
        return "smiles"
    if any(ch.isdigit() for ch in stripped):
        return "smiles"
    if stripped.isalpha():
        if stripped.lower() == stripped:
            return "name"
        mol = Chem.MolFromSmiles(stripped)
        if mol is not None:
            return "smiles"
        return "name"

    return "smiles"


def validate_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: validate user query and resolve to canonical SMILES.

    Reads: state["query"]
    Writes: state["validation"], state["smiles"], state["error"]
    """
    query = state.get("query", "").strip()
    if not query:
        return {
            "validation": {
                "is_valid": False,
                "input_type": "name",
                "error": "Empty input",
            },
            "error": "Empty input — nothing to validate.",
        }

    input_type = _detect_input_type(query)
    logger.info("[validate] query=%r  detected_type=%s", query, input_type)

    if input_type == "smiles":
        return _validate_smiles(query)
    return _validate_name(query)


def _validate_smiles(smiles: str) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "validation": {
                "is_valid": False,
                "input_type": "smiles",
                "canonical_smiles": None,
                "error": "RDKit could not parse SMILES.",
            },
            "error": f"Invalid SMILES: {smiles}",
        }

    canon = Chem.MolToSmiles(mol, isomericSmiles=True)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    mw = round(Descriptors.MolWt(mol), 4)

    cid = get_cid_by_smiles(canon)
    iupac = None
    if cid:
        props = get_compound_properties(canon)
        iupac = props.get("IUPACName")

    logger.info("[validate] SMILES valid → canon=%s  CID=%s", canon, cid)

    return {
        "validation": {
            "is_valid": True,
            "input_type": "smiles",
            "canonical_smiles": canon,
            "iupac_name": iupac,
            "molecular_formula": formula,
            "molecular_weight": mw,
            "pubchem_cid": cid,
            "error": None,
        },
        "smiles": canon,
        "pubchem_cid": cid or 0,
    }


def _translate_name_via_llm(name_ru: str) -> str | None:
    """Translate a Russian chemical name to English using LLM."""
    if not OPENROUTER_API_KEY:
        return None
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            temperature=0,
            max_tokens=100,
        )
        resp = llm.invoke(
            f"Переведи название химического вещества на английский. "
            f"Ответь ТОЛЬКО английским названием, без пояснений.\n\n"
            f"Вещество: {name_ru}"
        )
        result = resp.content.strip().strip('"').strip("'").strip(".")
        if result and not _CYRILLIC_RE.search(result):
            logger.info("[validate] LLM translated %r → %r", name_ru, result)
            return result
    except Exception as e:
        logger.warning("[validate] LLM translation failed: %s", e)
    return None


def _validate_name(name: str) -> dict[str, Any]:
    # Шаг 1: name → CID через PubChem
    cid = get_cid_by_name(name)

    # Шаг 1.5: если не нашли и имя содержит кириллицу — переводим через LLM
    if cid is None and _CYRILLIC_RE.search(name):
        english_name = _translate_name_via_llm(name)
        if english_name:
            cid = get_cid_by_name(english_name)
            if cid:
                logger.info("[validate] Resolved via LLM: %r → %r → CID=%d", name, english_name, cid)

    if cid is None:
        return {
            "validation": {
                "is_valid": False,
                "input_type": "name",
                "canonical_smiles": None,
                "error": f"Вещество '{name}' не найдено в PubChem.",
            },
            "error": f"Вещество '{name}' не найдено в PubChem.",
        }

    # Шаг 2: CID → SMILES
    smiles = get_smiles_by_cid(cid)
    if not smiles:
        return {
            "validation": {
                "is_valid": False,
                "input_type": "name",
                "canonical_smiles": None,
                "error": f"PubChem CID {cid} найден, но SMILES недоступен.",
            },
            "error": f"PubChem CID {cid} — SMILES недоступен.",
        }

    # Шаг 3: валидация SMILES через RDKit
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "validation": {
                "is_valid": False,
                "input_type": "name",
                "canonical_smiles": None,
                "error": f"PubChem вернул невалидный SMILES: {smiles}",
            },
            "error": f"Невалидный SMILES из PubChem: {smiles}",
        }

    # Шаг 4: канонизация + обогащение
    canon = Chem.MolToSmiles(mol, isomericSmiles=True)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    mw = round(Descriptors.MolWt(mol), 4)

    props = get_compound_properties(canon)
    iupac = props.get("IUPACName")

    logger.info("[validate] name=%r → CID=%d → SMILES=%s", name, cid, canon)

    return {
        "validation": {
            "is_valid": True,
            "input_type": "name",
            "canonical_smiles": canon,
            "iupac_name": iupac,
            "molecular_formula": formula,
            "molecular_weight": mw,
            "pubchem_cid": cid,
            "error": None,
        },
        "smiles": canon,
        "pubchem_cid": cid,
    }
