"""ProtocolAgent — generates step-by-step experiment protocols."""

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import get_llm
from src.models.protocol import (
    ExperimentCalculations,
    ExperimentProtocol,
    ProtocolStep,
)
from src.models.reaction import SynthesisPathway
from src.models.safety import SafetyReport
from src.models.state import AgentState

PROTOCOL_SYSTEM_PROMPT = """You are an experienced organic chemistry lab instructor
writing a detailed experimental protocol.

Given a synthesis pathway with calculated amounts, generate a step-by-step protocol
that a trained chemist can follow in the laboratory.

For each step, include:
- Exact masses/volumes of reagents (from the calculations provided)
- Specific instructions (what to add, how to add it, in what order)
- Temperature and time parameters
- What to observe (color changes, precipitate formation, etc.)
- Workup procedures (filtration, extraction, washing, drying, purification)
- Safety notes where relevant

Use professional chemistry lab language. Be specific with amounts.
Include practical tips (e.g., "add dropwise", "stir vigorously", "cool in ice bath").

Return as JSON:
{
  "title": "Synthesis of [compound]",
  "steps": [
    {
      "step_number": 1,
      "instruction": "Weigh 2.76 g of salicylic acid...",
      "duration": "5 min",
      "temperature": "room temperature",
      "notes": "Use analytical balance",
      "safety_note": "Wear gloves"
    }
  ],
  "expected_yield": "~3.6 g (70%)",
  "equipment_needed": ["round-bottom flask 100 mL", "..."],
  "disposal_instructions": "Collect organic waste in designated container"
}"""


async def run_protocol_generation(
    state: AgentState,
    pathway: SynthesisPathway,
    calculations: ExperimentCalculations,
    safety_report: SafetyReport | None = None,
) -> ExperimentProtocol:
    """Generate experiment protocol from pathway and calculations."""
    llm = get_llm(temperature=0.1)

    # Build detailed context
    context = f"Target: {pathway.target_smiles}\n"
    context += f"Target mass: {calculations.target_mass_g} g\n"
    context += f"Target moles: {calculations.target_moles} mol\n\n"

    for i, step in enumerate(pathway.steps):
        context += f"--- Reaction Step {step.step_number} ---\n"
        context += f"Reaction: {step.reaction_smiles}\n"
        context += f"Type: {step.reaction_type}\n"
        context += (
            f"Conditions: T={step.conditions.temperature}, "
            f"solvent={step.conditions.solvent}, "
            f"catalyst={step.conditions.catalyst}, "
            f"time={step.conditions.time}\n"
        )
        context += f"Expected yield: {step.expected_yield}\n"

        # Add calculated amounts
        if i < len(calculations.steps):
            calc_step = calculations.steps[i]
            context += f"\nCalculated amounts (for {calc_step.target_product_mass_g} g product):\n"
            for r in calc_step.reagents:
                line = f"  - {r.name}: {r.mass_g} g ({r.moles} mol, {r.equivalents} eq)"
                if r.volume_ml:
                    line += f" = {r.volume_ml} mL"
                context += line + "\n"

        context += "\n"

    # Add safety context
    if safety_report:
        context += "--- Safety Notes ---\n"
        context += f"Risk level: {safety_report.overall_risk_level}\n"
        if safety_report.requires_fume_hood:
            context += "REQUIRES FUME HOOD\n"
        if safety_report.requires_inert_atmosphere:
            context += "REQUIRES INERT ATMOSPHERE\n"
        for w in safety_report.warnings:
            context += f"WARNING [{w.severity}]: {w.substance} - {w.message}\n"
        context += f"PPE: {', '.join(safety_report.required_ppe)}\n"

    # TODO: Add RAG context from procedure_search_rag and technique_lookup

    messages = [
        SystemMessage(content=PROTOCOL_SYSTEM_PROMPT),
        HumanMessage(content=f"Generate protocol for:\n{context}"),
    ]

    response = await llm.ainvoke(messages)

    return _parse_protocol_response(
        response.content,
        pathway.target_smiles,
        calculations.target_mass_g,
    )


def _parse_protocol_response(
    response_text: str,
    target_smiles: str,
    target_mass_g: float,
) -> ExperimentProtocol:
    """Parse LLM protocol response into an ExperimentProtocol."""
    import json

    try:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            data = json.loads(response_text[json_start:json_end])

            steps = []
            for s in data.get("steps", []):
                steps.append(
                    ProtocolStep(
                        step_number=s.get("step_number", len(steps) + 1),
                        instruction=s.get("instruction", ""),
                        duration=s.get("duration"),
                        temperature=s.get("temperature"),
                        notes=s.get("notes"),
                        safety_note=s.get("safety_note"),
                    )
                )

            return ExperimentProtocol(
                title=data.get("title", f"Synthesis of {target_smiles}"),
                target_molecule=target_smiles,
                target_mass_g=target_mass_g,
                steps=steps,
                expected_yield=data.get("expected_yield"),
                equipment_needed=data.get("equipment_needed", []),
                disposal_instructions=data.get("disposal_instructions"),
            )
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return ExperimentProtocol(
        title=f"Synthesis of {target_smiles}",
        target_molecule=target_smiles,
        target_mass_g=target_mass_g,
        steps=[],
    )
