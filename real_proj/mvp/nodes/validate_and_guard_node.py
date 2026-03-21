"""Combined validation + guard node.

Merges validate_node (PubChem resolution, SMILES canonicalization, LLM
translation) with guard tools (banlist_check, safety_lookup, ppe_recommender).

Three routing outcomes via state["validation"]["resolve_status"]:
  "found"     — molecule resolved and safe (or warning) -> molecule_info
  "banned"    — molecule in banlist -> END(error)
  "not_found" — PubChem could not resolve -> fallback to research_node
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from ..config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL
from ..tools import (
    get_cid_by_name,
    get_cid_by_smiles,
    get_smiles_by_cid,
    get_compound_properties,
    banlist_check,
    reaction_banlist_check,
    safety_lookup,
    ppe_recommender,
)

logger = logging.getLogger(__name__)

_CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁ]")
_SMILES_PATTERN = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)\\/=#$%.:~]+$")


def validate_and_guard_node(state: dict[str, Any]) -> dict[str, Any]:
    """Validate query, resolve via PubChem, then run safety checks.

    Reads:  state["query"]
    Writes: state["validation"], state["smiles"], state["pubchem_cid"],
            state["guard_result"], state["error"]
    """
    query = state.get("query", "").strip()
    if not query:
        return {
            "validation": {
                "is_valid": False,
                "resolve_status": "not_found",
                "error": "Empty input",
            },
            "error": "Пустой запрос — нечего валидировать.",
        }

    # ── Step 1: Resolve molecule ──
    resolve_result = _resolve_molecule(query)
    validation = resolve_result.get("validation", {})

    if not validation.get("is_valid", False):
        resolve_status = "not_found"
        validation["resolve_status"] = resolve_status
        logger.info("[validate_and_guard] query=%r -> not_found", query[:60])
        return resolve_result

    # ── Step 2: Run safety checks ──
    smiles = resolve_result.get("smiles", "")
    cid = resolve_result.get("pubchem_cid")

    guard_result = _run_safety_checks(
        smiles=smiles,
        cid=cid,
        reaction_description=state.get("reaction_description", ""),
    )

    overall = guard_result.get("overall_status", "SAFE")

    if overall == "CRITICAL_STOP":
        resolve_status = "banned"
        reason = (
            guard_result.get("molecule_check", {}).get("reason", "")
            or guard_result.get("reaction_check", {}).get("reason", "")
        )
        validation["resolve_status"] = resolve_status
        return {
            "validation": validation,
            "smiles": smiles,
            "pubchem_cid": cid,
            "guard_result": guard_result,
            "error": f"CRITICAL_STOP: {reason}",
        }

    resolve_status = "found"
    validation["resolve_status"] = resolve_status
    logger.info("[validate_and_guard] query=%r -> found, status=%s", query[:60], overall)

    return {
        "validation": validation,
        "smiles": smiles,
        "pubchem_cid": cid,
        "guard_result": guard_result,
    }


# ─── Molecule resolution (from validate_node) ───────────────────────────────

def _detect_input_type(user_input: str) -> str:
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


def _resolve_molecule(query: str) -> dict[str, Any]:
    input_type = _detect_input_type(query)
    logger.info("[validate_and_guard] query=%r  detected_type=%s", query, input_type)

    if input_type == "smiles":
        return _resolve_smiles(query)
    return _resolve_name(query)


def _resolve_smiles(smiles: str) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "validation": {
                "is_valid": False,
                "input_type": "smiles",
                "canonical_smiles": None,
                "error": "RDKit could not parse SMILES.",
            },
        }

    canon = Chem.MolToSmiles(mol, isomericSmiles=True)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    mw = round(Descriptors.MolWt(mol), 4)
    cid = get_cid_by_smiles(canon)
    iupac = None
    if cid:
        props = get_compound_properties(canon)
        iupac = props.get("IUPACName")

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
            logger.info("[validate_and_guard] LLM translated %r → %r", name_ru, result)
            return result
    except Exception as e:
        logger.warning("[validate_and_guard] LLM translation failed: %s", e)
    return None


def _resolve_name(name: str) -> dict[str, Any]:
    cid = get_cid_by_name(name)

    if cid is None and _CYRILLIC_RE.search(name):
        english_name = _translate_name_via_llm(name)
        if english_name:
            cid = get_cid_by_name(english_name)

    if cid is None:
        return {
            "validation": {
                "is_valid": False,
                "input_type": "name",
                "canonical_smiles": None,
                "error": f"Вещество '{name}' не найдено в PubChem.",
            },
        }

    smiles = get_smiles_by_cid(cid)
    if not smiles:
        return {
            "validation": {
                "is_valid": False,
                "input_type": "name",
                "canonical_smiles": None,
                "error": f"PubChem CID {cid} найден, но SMILES недоступен.",
            },
        }

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "validation": {
                "is_valid": False,
                "input_type": "name",
                "canonical_smiles": None,
                "error": f"PubChem вернул невалидный SMILES: {smiles}",
            },
        }

    canon = Chem.MolToSmiles(mol, isomericSmiles=True)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    mw = round(Descriptors.MolWt(mol), 4)
    props = get_compound_properties(canon)
    iupac = props.get("IUPACName")

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


# ─── Safety checks (from guard_node) ────────────────────────────────────────

def _determine_overall_status(
    mol_status: str,
    rxn_status: str,
) -> Literal["SAFE", "WARNING", "CRITICAL_STOP"]:
    critical = {"banned", "prohibited"}
    warning = {"restricted"}
    if mol_status in critical or rxn_status in critical:
        return "CRITICAL_STOP"
    if mol_status in warning or rxn_status in warning:
        return "WARNING"
    return "SAFE"


def _run_safety_checks(
    smiles: str,
    cid: int | None,
    reaction_description: str = "",
) -> dict[str, Any]:
    mol_check = banlist_check(smiles)
    rxn_check = reaction_banlist_check(reaction_description)
    safety = safety_lookup(smiles, cid=cid)

    h_phrases_str = ",".join(safety.get("h_phrases", []))
    ppe = ppe_recommender(smiles, h_phrases_str)

    overall = _determine_overall_status(
        mol_status=mol_check.get("status", "clear"),
        rxn_status=rxn_check.get("status", "allowed"),
    )

    return {
        "overall_status": overall,
        "molecule_check": mol_check,
        "reaction_check": rxn_check,
        "safety_data": safety,
        "ppe_recommendations": ppe,
    }
