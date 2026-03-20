"""Molecule data models."""

from pydantic import BaseModel, Field


class MoleculeProperties(BaseModel):
    """Physical and chemical properties of a molecule."""

    melting_point: str | None = None
    boiling_point: str | None = None
    solubility: str | None = None
    density: float | None = None
    log_p: float | None = None
    appearance: str | None = None
    color: str | None = None
    odor: str | None = None
    state: str | None = None  # solid / liquid / gas


class GHSClassification(BaseModel):
    """GHS safety classification."""

    pictograms: list[str] = Field(default_factory=list)
    h_statements: list[str] = Field(default_factory=list)  # Hazard
    p_statements: list[str] = Field(default_factory=list)  # Precautionary
    signal_word: str | None = None  # Danger / Warning


class MoleculeInfo(BaseModel):
    """Complete information about a chemical compound."""

    name: str
    iupac_name: str | None = None
    synonyms: list[str] = Field(default_factory=list)
    smiles: str
    canonical_smiles: str | None = None
    molecular_formula: str | None = None
    molecular_weight: float | None = None
    pubchem_cid: int | None = None
    cas_number: str | None = None
    inchi: str | None = None

    properties: MoleculeProperties = Field(default_factory=MoleculeProperties)
    ghs: GHSClassification = Field(default_factory=GHSClassification)

    image_url: str | None = None
    is_commercially_available: bool | None = None

    def short_card(self) -> str:
        """Return a short text card for display."""
        lines = [f"**{self.name}**"]
        if self.molecular_formula:
            lines.append(f"Formula: {self.molecular_formula}")
        if self.molecular_weight:
            lines.append(f"MW: {self.molecular_weight:.2f} g/mol")
        if self.smiles:
            lines.append(f"SMILES: `{self.smiles}`")
        if self.properties.melting_point:
            lines.append(f"Tm: {self.properties.melting_point}")
        if self.properties.state:
            lines.append(f"State: {self.properties.state}")
        return "\n".join(lines)
