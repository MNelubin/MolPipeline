"""Safety report models."""

from pydantic import BaseModel, Field


class SafetyWarning(BaseModel):
    """A single safety warning."""

    severity: str  # "critical", "warning", "info"
    substance: str
    message: str
    ghs_codes: list[str] = Field(default_factory=list)


class SafetyReport(BaseModel):
    """Complete safety report for a synthesis pathway."""

    warnings: list[SafetyWarning] = Field(default_factory=list)
    required_ppe: list[str] = Field(default_factory=list)  # goggles, gloves, etc.
    incompatibilities: list[str] = Field(default_factory=list)
    disposal_notes: list[str] = Field(default_factory=list)
    storage_notes: list[str] = Field(default_factory=list)
    requires_fume_hood: bool = False
    requires_inert_atmosphere: bool = False
    overall_risk_level: str = "unknown"  # "low", "medium", "high", "critical"

    def has_critical_warnings(self) -> bool:
        return any(w.severity == "critical" for w in self.warnings)
