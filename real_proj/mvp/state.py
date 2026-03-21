"""Unified state for the 4-node MVP graph."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class MoleculeInfo(TypedDict, total=False):
    """Structured molecule card produced by the molecule_info node."""
    name: str
    synonyms: list[str]
    smiles: str
    molecular_formula: str
    molecular_weight: float
    physical_description: str
    properties: dict[str, Any]
    description: str
    ghs_classification: list[str]
    pubchem_cid: int
    image_2d: str
    image_3d: str
    pubchem_url: str


class GuardResult(TypedDict, total=False):
    """Safety check result from the guard node."""
    overall_status: Literal["SAFE", "WARNING", "CRITICAL_STOP"]
    molecule_check: dict
    reaction_check: dict
    safety_data: dict
    ppe_recommendations: list[str]


class ValidationResult(TypedDict, total=False):
    """Result from the validation node."""
    is_valid: bool
    input_type: Literal["smiles", "name"]
    canonical_smiles: str | None
    iupac_name: str | None
    molecular_formula: str | None
    molecular_weight: float | None
    pubchem_cid: int | None
    error: str | None


class MVPState(TypedDict, total=False):
    """Top-level state flowing through the graph."""
    # Input
    query: str

    # After validation
    validation: ValidationResult
    smiles: str  # canonical SMILES (set by validate_node)
    pubchem_cid: int  # CID from PubChem (set by validate_node)

    # After guard
    guard_result: GuardResult

    # After molecule_info
    molecule_info: MoleculeInfo
    final_answer: str

    # After retrosynthesis
    retro_result: dict[str, Any]

    # Error / early exit
    error: str
