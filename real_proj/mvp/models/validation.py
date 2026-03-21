from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class MoleculeValidationResult(BaseModel):
    """Result of validating a user-supplied molecule identifier."""

    is_valid: bool
    input_text: str
    input_type: Literal["smiles", "name", "research_query"]
    canonical_smiles: str | None = None
    iupac_name: str | None = None
    molecular_formula: str | None = None
    molecular_weight: float | None = None
    pubchem_cid: int | None = None
    error: str | None = None
