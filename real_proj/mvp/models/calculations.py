from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class AmountType(str, Enum):
    PRODUCT_MASS = "product_mass"
    REAGENT_MASS = "reagent_mass"
    REAGENT_MOLES = "reagent_moles"


class PhysicalState(str, Enum):
    SOLID = "solid"
    LIQUID = "liquid"
    GAS = "gas"
    UNKNOWN = "unknown"


class ReagentInput(BaseModel):
    """Single reagent with its equivalents for equivalents_calc."""

    smiles: str
    name: str = ""
    equivalents: float = 1.0
    role: str = "reagent"


class StoichiometryRequest(BaseModel):
    """Input for stoichiometry_calc: reaction SMILES + desired product mass."""

    reaction_smiles: str = Field(
        ..., description='Reaction in "reactants>>products" SMILES format'
    )
    target_mass_g: float = Field(..., gt=0, description="Desired product mass in grams")
    target_product_smiles: str | None = Field(
        None, description="Specific product SMILES when reaction has multiple products"
    )


class EquivalentsRequest(BaseModel):
    """Input for equivalents_calc: reference reagent + list of reagents with equivalents."""

    reference_smiles: str = Field(
        ..., description="SMILES of the reference compound (product or limiting reagent)"
    )
    reference_amount: float = Field(..., gt=0)
    amount_type: AmountType = AmountType.PRODUCT_MASS
    reagents: list[ReagentInput]


class ReagentCalcResult(BaseModel):
    """Calculated quantities for a single reagent."""

    smiles: str
    name: str
    molecular_weight: float = Field(..., description="g/mol")
    equivalents: float
    moles: float
    mass_g: float
    density: float | None = Field(None, description="g/mL for liquids")
    volume_ml: float | None = Field(None, description="mL for liquids")
    state: PhysicalState = PhysicalState.UNKNOWN
    notes: str = ""


class CalculationResult(BaseModel):
    """Full result of a stoichiometry / equivalents calculation."""

    target_product_smiles: str
    target_mass_g: float
    target_moles: float
    reagents: list[ReagentCalcResult]
    warnings: list[str] = Field(default_factory=list)
