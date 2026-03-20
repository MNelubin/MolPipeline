"""GuardAgent — safety checker that runs in parallel with other agents."""

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import get_llm
from src.models.reaction import SynthesisPathway
from src.models.safety import SafetyReport, SafetyWarning
from src.models.state import AgentState
from src.tools.pubchem import pubchem_safety

GUARD_SYSTEM_PROMPT = """You are a chemical safety expert. Your job is to review
synthesis pathways and identify potential hazards.

For each substance and reaction step, check:
1. GHS classification of each reagent and product
2. Compatibility of reagents (could mixing them produce explosions, toxic gases, etc.)
3. Required personal protective equipment (PPE)
4. Storage requirements
5. Waste disposal requirements
6. Whether a fume hood is needed
7. Whether inert atmosphere is needed

Classify overall risk as: "low", "medium", "high", or "critical".

Be thorough but practical. Flag genuine dangers, not theoretical ones.
A standard organic chemistry lab has fume hoods, gloves, goggles, and basic equipment.

Return your analysis as JSON:
{
  "overall_risk_level": "medium",
  "requires_fume_hood": true,
  "requires_inert_atmosphere": false,
  "warnings": [
    {"severity": "warning", "substance": "acetic anhydride", "message": "Corrosive. Handle with care."}
  ],
  "required_ppe": ["safety goggles", "nitrile gloves", "lab coat"],
  "incompatibilities": [],
  "disposal_notes": ["Neutralize acid waste before disposal"],
  "storage_notes": ["Store acetic anhydride away from water"]
}"""


async def run_safety_check(
    state: AgentState, pathway: SynthesisPathway
) -> SafetyReport:
    """Run safety analysis on a synthesis pathway."""
    llm = get_llm(temperature=0.0)

    # Collect all substances in the pathway
    substances = set()
    for step in pathway.steps:
        if step.reaction_smiles:
            parts = step.reaction_smiles.replace(">>", ".").split(".")
            for p in parts:
                if p.strip():
                    substances.add(p.strip())

        for reagent in step.reagents:
            substances.add(reagent.name)

        if step.conditions.solvent:
            substances.add(step.conditions.solvent)
        if step.conditions.catalyst:
            substances.add(step.conditions.catalyst)

    # Get GHS data for substances with CIDs
    ghs_data = {}
    for step in pathway.steps:
        for reagent in step.reagents:
            if reagent.pubchem_cid:
                safety = pubchem_safety.invoke({"cid": reagent.pubchem_cid})
                if "error" not in safety:
                    ghs_data[reagent.name] = safety

    # Build context for LLM
    pathway_desc = f"Target: {pathway.target_smiles}\n"
    for step in pathway.steps:
        pathway_desc += (
            f"\nStep {step.step_number}: {step.reaction_smiles}\n"
            f"  Type: {step.reaction_type}\n"
            f"  Conditions: T={step.conditions.temperature}, "
            f"solvent={step.conditions.solvent}, "
            f"catalyst={step.conditions.catalyst}\n"
        )

    if ghs_data:
        pathway_desc += "\n\nGHS data from PubChem:\n"
        for name, data in ghs_data.items():
            pathway_desc += f"\n{name}:\n"
            if data.get("signal_word"):
                pathway_desc += f"  Signal: {data['signal_word']}\n"
            if data.get("h_statements"):
                pathway_desc += f"  Hazards: {'; '.join(data['h_statements'][:5])}\n"

    messages = [
        SystemMessage(content=GUARD_SYSTEM_PROMPT),
        HumanMessage(content=f"Review this synthesis for safety:\n{pathway_desc}"),
    ]

    response = await llm.ainvoke(messages)

    return _parse_safety_response(response.content)


def _parse_safety_response(response_text: str) -> SafetyReport:
    """Parse LLM safety response into a SafetyReport."""
    import json

    try:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            data = json.loads(response_text[json_start:json_end])

            warnings = []
            for w in data.get("warnings", []):
                warnings.append(
                    SafetyWarning(
                        severity=w.get("severity", "info"),
                        substance=w.get("substance", "unknown"),
                        message=w.get("message", ""),
                        ghs_codes=w.get("ghs_codes", []),
                    )
                )

            return SafetyReport(
                warnings=warnings,
                required_ppe=data.get("required_ppe", []),
                incompatibilities=data.get("incompatibilities", []),
                disposal_notes=data.get("disposal_notes", []),
                storage_notes=data.get("storage_notes", []),
                requires_fume_hood=data.get("requires_fume_hood", False),
                requires_inert_atmosphere=data.get(
                    "requires_inert_atmosphere", False
                ),
                overall_risk_level=data.get("overall_risk_level", "unknown"),
            )
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return SafetyReport(overall_risk_level="unknown")
