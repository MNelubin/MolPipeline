"""State definitions for the multi-phase agent graph.

Three phases with interrupt points:
  Phase 1: classify -> validate_and_guard -> molecule_info -> INTERRUPT
  Phase 2: retrosynthesis -> [guard_safety, reagent_check] -> aggregate -> INTERRUPT
  Phase 3: stoichiometry -> experiment_planner -> END
"""

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
    """Safety check result from the guard / validate_and_guard node."""
    overall_status: Literal["SAFE", "WARNING", "CRITICAL_STOP"]
    molecule_check: dict
    explosive_check: dict
    safety_taxonomy: dict
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


class ReagentReport(TypedDict, total=False):
    """Reagent availability report for synthesis pathways."""
    pathway_reports: list[dict[str, Any]]
    all_available: bool
    unavailable_reagents: list[str]


class SafetyReport(TypedDict, total=False):
    """Safety report for synthesis pathways (separate from initial guard check)."""
    pathway_reports: list[dict[str, Any]]
    has_critical: bool
    warnings: list[str]


class TargetAmount(TypedDict, total=False):
    """User-specified target amount for stoichiometry calculations."""
    amount_type: Literal["product_mass", "reagent_mass", "reagent_moles"]
    value: float
    unit: str


class MVPState(TypedDict, total=False):
    """Top-level state flowing through the graph."""

    # ── Input ──
    query: str

    # ── Phase 1: Classification ──
    input_type: Literal["molecule", "research", "invalid"]

    # ── Phase 1: Research (if input_type == "research" or validate fallback) ──
    research_result: dict[str, Any]

    # ── Phase 1: Validation + Guard ──
    validation: ValidationResult
    smiles: str
    pubchem_cid: int
    guard_result: GuardResult

    # ── Phase 1: Molecule info ──
    molecule_info: MoleculeInfo
    final_answer: str

    # ── Phase 2: Retrosynthesis ──
    retro_result: dict[str, Any]
    synthesis_pathways: list[dict[str, Any]]

    # ── Phase 2: Reagent check + Guard safety (parallel) ──
    reagent_report: ReagentReport
    safety_report: SafetyReport

    # ── Phase 2 -> 3: User selections ──
    selected_pathway: int | None

    # ── Phase 3: Stoichiometry ──
    target_amount: TargetAmount | None
    calculations: dict[str, Any] | None

    # ── Phase 3: Experiment protocol ──
    experiment_protocol: dict[str, Any] | None

    # ── Control ──
    current_phase: Literal["identification", "synthesis", "experiment"]
    cycle_counts: dict[str, int]
    error: str

    # ── Journal ──
    session_id: str

    # ── Model override ──
    llm_model: str | None
