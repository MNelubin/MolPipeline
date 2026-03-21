"""Quick validation of user-supplied molecule identifiers (SMILES or name)."""

from __future__ import annotations

import logging
import re
from typing import Literal

from ..models.validation import MoleculeValidationResult
from .rdkit_tools import (
    canonicalize,
    get_average_molecular_weight,
    get_molecular_formula,
    validate_smiles,
)
from .pubchem import (
    get_cid_by_name,
    get_cid_by_smiles,
    get_iupac_name,
    get_smiles_by_cid,
)
from .research import classify_user_input as _classify

logger = logging.getLogger(__name__)

_SMILES_PATTERN = re.compile(
    r"^[A-Za-z0-9"
    r"@+\-\[\]\(\)\\\/=#$%.:~]+"
    r"$"
)


def _detect_input_type(user_input: str) -> Literal["smiles", "name"]:
    stripped = user_input.strip()
    if " " in stripped:
        return "name"
    if not _SMILES_PATTERN.match(stripped):
        return "name"
    smiles_structural = set("=()[]@/\\#%+")
    if smiles_structural & set(stripped):
        return "smiles"
    if any(ch.isdigit() for ch in stripped):
        return "smiles"
    if stripped.isalpha():
        if stripped.lower() == stripped:
            return "name"
        if validate_smiles(stripped):
            return "smiles"
        return "name"
    return "smiles"


def _fail(
    user_input: str, input_type: Literal["smiles", "name"], error: str
) -> MoleculeValidationResult:
    return MoleculeValidationResult(
        is_valid=False,
        input_text=user_input,
        input_type=input_type,
        error=error,
    )


def _enrich(
    user_input: str,
    input_type: Literal["smiles", "name"],
    smiles: str,
    cid: int | None,
) -> MoleculeValidationResult:
    canonical = canonicalize(smiles)

    iupac: str | None = None
    if cid is not None:
        iupac = get_iupac_name(canonical) or None

    try:
        formula = get_molecular_formula(canonical)
    except ValueError:
        formula = None

    try:
        mw = round(get_average_molecular_weight(canonical), 4)
    except ValueError:
        mw = None

    return MoleculeValidationResult(
        is_valid=True,
        input_text=user_input,
        input_type=input_type,
        canonical_smiles=canonical,
        iupac_name=iupac,
        molecular_formula=formula,
        molecular_weight=mw,
        pubchem_cid=cid,
    )


def validate_molecule_input(user_input: str) -> MoleculeValidationResult:
    """Validate a molecule specified by SMILES or name and return structured info."""
    stripped = user_input.strip()
    if not stripped:
        return _fail(user_input, "name", "Empty input")

    if _classify(stripped) == "research_query":
        return MoleculeValidationResult(
            is_valid=False,
            input_text=user_input,
            input_type="research_query",
            error="Input is a research query, not a specific molecule. Use ResearchAgent.",
        )

    input_type = _detect_input_type(stripped)

    if input_type == "smiles":
        return _validate_smiles_input(stripped)
    return _validate_name_input(stripped)


def _validate_smiles_input(smiles: str) -> MoleculeValidationResult:
    if not validate_smiles(smiles):
        return _fail(smiles, "smiles", "RDKit could not parse SMILES")
    cid = get_cid_by_smiles(smiles)
    if cid is None:
        logger.info("SMILES '%s' is valid RDKit but not found in PubChem", smiles)
    return _enrich(smiles, "smiles", smiles, cid)


def _validate_name_input(name: str) -> MoleculeValidationResult:
    cid = get_cid_by_name(name)
    if cid is None:
        return _fail(name, "name", f"Compound '{name}' not found in PubChem")
    smiles = get_smiles_by_cid(cid)
    if smiles is None:
        return _fail(name, "name", f"PubChem CID {cid} found but SMILES unavailable")
    if not validate_smiles(smiles):
        return _fail(name, "name", f"PubChem returned unparseable SMILES: {smiles}")
    return _enrich(name, "name", smiles, cid)
