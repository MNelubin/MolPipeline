"""Experiment protocol models."""

from pydantic import BaseModel, Field


class ReagentAmount(BaseModel):
    """Calculated amount of a reagent."""

    name: str
    smiles: str | None = None
    molecular_weight: float | None = None
    equivalents: float = 1.0
    moles: float | None = None
    mass_g: float | None = None
    volume_ml: float | None = None
    density: float | None = None
    notes: str | None = None  # e.g. "~3 drops", "catalytic"


class StepCalculation(BaseModel):
    """Stoichiometry calculations for a single reaction step."""

    step_number: int
    target_product_mass_g: float
    target_product_moles: float
    reagents: list[ReagentAmount] = Field(default_factory=list)
    theoretical_yield_g: float | None = None


class ExperimentCalculations(BaseModel):
    """All calculations for the experiment."""

    target_mass_g: float
    target_moles: float
    steps: list[StepCalculation] = Field(default_factory=list)


class ProtocolStep(BaseModel):
    """A single step in the experiment protocol."""

    step_number: int
    instruction: str
    duration: str | None = None
    temperature: str | None = None
    notes: str | None = None
    safety_note: str | None = None


class ExperimentProtocol(BaseModel):
    """Complete experiment protocol."""

    title: str
    target_molecule: str
    target_mass_g: float
    steps: list[ProtocolStep] = Field(default_factory=list)
    expected_yield: str | None = None
    safety_summary: str | None = None
    equipment_needed: list[str] = Field(default_factory=list)
    disposal_instructions: str | None = None
