"""Reaction and synthesis pathway models."""

from pydantic import BaseModel, Field

from .molecule import MoleculeInfo


class ReactionConditions(BaseModel):
    """Conditions for a chemical reaction."""

    temperature: str | None = None
    solvent: str | None = None
    catalyst: str | None = None
    time: str | None = None
    pressure: str | None = None
    atmosphere: str | None = None  # N2, Ar, air
    procedure_description: str | None = None


class ReactionStep(BaseModel):
    """A single step in a synthesis pathway."""

    step_number: int
    reaction_smiles: str  # reactants>>products
    reaction_type: str | None = None  # acylation, reduction, etc.

    reagents: list[MoleculeInfo] = Field(default_factory=list)
    product: MoleculeInfo | None = None

    conditions: ReactionConditions = Field(default_factory=ReactionConditions)
    expected_yield: float | None = None  # 0.0 - 1.0

    source: str = "predicted"  # "ORD", "USPTO", "predicted"
    source_id: str | None = None  # reference to source dataset
    confidence: float = 0.5  # 0.0 - 1.0


class SynthesisPathway(BaseModel):
    """A complete synthesis pathway from starting materials to target."""

    pathway_id: str
    target_smiles: str
    steps: list[ReactionStep] = Field(default_factory=list)

    total_steps: int = 0
    overall_yield: float | None = None  # product of step yields
    safety_score: float | None = None  # 0-1, higher = safer
    cost_score: float | None = None  # 0-1, higher = cheaper
    confidence_score: float | None = None  # fraction of DB-confirmed steps

    def compute_scores(self) -> None:
        """Recompute derived scores from steps using retro_scorer."""
        self.total_steps = len(self.steps)
        if not self.steps:
            return

        # Overall yield = product of step yields
        yields = [
            s.expected_yield for s in self.steps if s.expected_yield is not None
        ]
        if yields:
            result = 1.0
            for y in yields:
                result *= y
            self.overall_yield = result

        # Confidence = fraction of DB-confirmed steps
        confirmed = sum(1 for s in self.steps if s.source != "predicted")
        self.confidence_score = confirmed / len(self.steps)

        # Use retro_scorer for richer pathway scoring
        try:
            from src.tools.retro_scorer import score_pathway

            steps_data = []
            for s in self.steps:
                steps_data.append({
                    "reaction_smiles": s.reaction_smiles,
                    "score": s.confidence,
                    "source": s.source,
                    "expected_yield": s.expected_yield,
                    "plausibility": s.confidence,
                })
            scoring = score_pathway(steps_data, self.target_smiles, self.safety_score)
            breakdown = scoring.get("breakdown", {})

            if self.safety_score is None:
                self.safety_score = breakdown.get("safety")
            if self.cost_score is None:
                self.cost_score = breakdown.get("buyability")
        except Exception:
            pass
