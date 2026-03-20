"""RetrosynthesisAgent — plans synthesis pathways using LLM + RAG validation."""

import uuid

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import MAX_RETRO_DEPTH, RETRO_TOP_N
from src.llm import get_llm
from src.models.molecule import MoleculeInfo
from src.models.reaction import ReactionConditions, ReactionStep, SynthesisPathway
from src.models.state import AgentState
from src.tools.scoring import reagent_availability

RETRO_SYSTEM_PROMPT = """You are an expert organic chemistry retrosynthesis planner.
Given a target molecule (SMILES), propose synthesis pathways by working backwards
from the target to commercially available starting materials.

For each proposed pathway, provide:
1. The reaction SMILES for each step (reactants>>products)
2. The reaction type (e.g., acylation, reduction, Grignard, Suzuki coupling, etc.)
3. Suggested conditions (temperature, solvent, catalyst, time)
4. Expected yield estimate (as a fraction 0.0-1.0)
5. Whether the starting materials are commercially available

Propose 2-3 different pathways when possible. Prefer:
- Fewer steps over more steps
- Well-known reactions over exotic ones
- Commercially available reagents
- Higher yielding routes

IMPORTANT: Only propose reactions that are chemically valid. Do not invent
reactions that violate chemical principles.

Return your analysis as structured JSON with the following format:
{
  "pathways": [
    {
      "pathway_id": "path_1",
      "description": "Brief description of the approach",
      "steps": [
        {
          "step_number": 1,
          "reaction_smiles": "reactant1.reactant2>>product",
          "reaction_type": "type of reaction",
          "reagent_names": ["reagent1 name", "reagent2 name"],
          "conditions": {
            "temperature": "80°C",
            "solvent": "THF",
            "catalyst": "none",
            "time": "2 hours"
          },
          "expected_yield": 0.75,
          "starting_materials_available": true
        }
      ]
    }
  ]
}"""


def create_retrosynthesis_agent():
    """Create the RetrosynthesisAgent LLM."""
    return get_llm(temperature=0.2)


async def run_retrosynthesis(
    state: AgentState, target_molecule: MoleculeInfo
) -> list[SynthesisPathway]:
    """Run retrosynthesis analysis for the target molecule.

    Uses LLM-based retrosynthesis (MVP approach) with RAG validation planned.
    """
    llm = create_retrosynthesis_agent()

    # Build context about the target
    context = f"""Target molecule: {target_molecule.name}
SMILES: {target_molecule.smiles}
Molecular formula: {target_molecule.molecular_formula}
Molecular weight: {target_molecule.molecular_weight}"""

    if target_molecule.properties.state:
        context += f"\nPhysical state: {target_molecule.properties.state}"
    if target_molecule.properties.melting_point:
        context += f"\nMelting point: {target_molecule.properties.melting_point}"

    # TODO: Add RAG context from reaction database search
    # rag_results = reaction_search_rag.invoke({"query": target_molecule.smiles})
    # if rag_results and not any("error" in r for r in rag_results):
    #     context += "\n\nKnown reactions from database:\n"
    #     for r in rag_results[:3]:
    #         context += f"- {r['content'][:200]}\n"

    messages = [
        SystemMessage(content=RETRO_SYSTEM_PROMPT),
        HumanMessage(content=f"Plan retrosynthesis for:\n{context}"),
    ]

    response = await llm.ainvoke(messages)

    # Parse the LLM response into SynthesisPathway objects
    pathways = _parse_retro_response(response.content, target_molecule.smiles)

    return pathways


def _parse_retro_response(
    response_text: str, target_smiles: str
) -> list[SynthesisPathway]:
    """Parse LLM retrosynthesis response into structured pathways."""
    import json

    pathways = []

    # Try to extract JSON from the response
    try:
        # Find JSON block in response
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)

            for p_data in data.get("pathways", []):
                steps = []
                for s_data in p_data.get("steps", []):
                    conditions = s_data.get("conditions", {})
                    step = ReactionStep(
                        step_number=s_data.get("step_number", len(steps) + 1),
                        reaction_smiles=s_data.get("reaction_smiles", ""),
                        reaction_type=s_data.get("reaction_type"),
                        conditions=ReactionConditions(
                            temperature=conditions.get("temperature"),
                            solvent=conditions.get("solvent"),
                            catalyst=conditions.get("catalyst"),
                            time=conditions.get("time"),
                        ),
                        expected_yield=s_data.get("expected_yield"),
                        source="predicted",
                        confidence=0.5,
                    )
                    steps.append(step)

                pathway = SynthesisPathway(
                    pathway_id=p_data.get(
                        "pathway_id", f"path_{uuid.uuid4().hex[:8]}"
                    ),
                    target_smiles=target_smiles,
                    steps=steps,
                )
                pathway.compute_scores()
                pathways.append(pathway)

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        # If JSON parsing fails, create a single "unparsed" pathway
        pathways.append(
            SynthesisPathway(
                pathway_id=f"path_{uuid.uuid4().hex[:8]}",
                target_smiles=target_smiles,
                steps=[],
                confidence_score=0.0,
            )
        )

    return pathways
