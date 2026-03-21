"""Validation node: detect input type (SMILES vs name), validate, resolve via PubChem."""

from __future__ import annotations

import logging
import re
from typing import Any

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from ..tools import get_cid_by_name, get_cid_by_smiles, get_smiles_by_cid, get_compound_properties

logger = logging.getLogger(__name__)

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
    }


def _validate_name(name: str) -> dict[str, Any]:
    cid = get_cid_by_name(name)
    if cid is None:
        return {
            "validation": {
                "is_valid": False,
                "input_type": "name",
                "canonical_smiles": None,
                "error": f"Compound '{name}' not found in PubChem.",
            },
            "error": f"Compound '{name}' not found in PubChem.",
        }

    smiles = get_smiles_by_cid(cid)
    if not smiles:
        return {
            "validation": {
                "is_valid": False,
                "input_type": "name",
                "canonical_smiles": None,
                "error": f"PubChem CID {cid} found but SMILES unavailable.",
            },
            "error": f"PubChem CID {cid} — no SMILES available.",
        }

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "validation": {
                "is_valid": False,
                "input_type": "name",
                "canonical_smiles": None,
                "error": f"PubChem returned unparseable SMILES: {smiles}",
            },
            "error": f"Unparseable SMILES from PubChem: {smiles}",
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
    }
