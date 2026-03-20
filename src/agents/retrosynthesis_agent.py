"""RetrosynthesisAgent — plans synthesis pathways using real reaction databases.

Priority order:
1. IBM RXN — automatic retrosynthesis from 2.5M real reactions
2. ASKCOS — template-based retrosynthesis from MIT
3. ORD — search for known published reactions
4. LLM — fallback for evaluation, gap-filling, conditions enrichment
"""

import asyncio
import logging
import uuid

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import MAX_RETRO_DEPTH, RXN_API_KEY, ASKCOS_BASE_URL
from src.llm import get_llm
from src.models.molecule import MoleculeInfo
from src.models.reaction import ReactionConditions, ReactionStep, SynthesisPathway
from src.models.state import AgentState

logger = logging.getLogger(__name__)

# LLM prompt for enriching/evaluating pathways (NOT for generating them)
ENRICHMENT_PROMPT = """You are an expert organic chemist reviewing synthesis pathways
obtained from reaction databases (IBM RXN, ASKCOS, Open Reaction Database).

Given the pathways below, for each reaction step:
1. Identify the reaction type (e.g., acylation, Grignard, Suzuki coupling)
2. Suggest realistic conditions if missing (temperature, solvent, catalyst, time)
3. Estimate yield if not provided (as fraction 0.0-1.0)
4. Name the reagents in plain English

Do NOT change the reaction SMILES — they come from validated databases.
Only add missing metadata.

Return as JSON:
{
  "enriched_steps": [
    {
      "step_number": 1,
      "reaction_type": "esterification",
      "reagent_names": ["salicylic acid", "acetic anhydride"],
      "conditions": {
        "temperature": "85°C",
        "solvent": "none",
        "catalyst": "H3PO4",
        "time": "15 min"
      },
      "expected_yield": 0.85
    }
  ]
}"""


async def run_retrosynthesis(
    state: AgentState, target_molecule: MoleculeInfo
) -> list[SynthesisPathway]:
    """Run retrosynthesis using real data sources, LLM only for enrichment.

    Queries IBM RXN and ASKCOS in parallel, merges results with ORD data,
    then uses LLM to enrich missing conditions/yields.
    """
    smiles = target_molecule.smiles
    all_pathways: list[SynthesisPathway] = []

    # --- Phase 1: Query real databases in parallel ---
    results = await _query_all_sources(smiles)

    for source_name, source_data in results.items():
        if "error" in source_data:
            logger.warning(f"{source_name} failed: {source_data['error']}")
            continue
        pathways = _convert_to_pathways(source_data, smiles, source_name)
        all_pathways.extend(pathways)

    # --- Phase 2: If no real data, fall back to LLM ---
    if not all_pathways:
        logger.info("No results from databases, falling back to LLM")
        llm_pathways = await _llm_retrosynthesis(target_molecule)
        all_pathways.extend(llm_pathways)

    # --- Phase 3: Enrich pathways with LLM (conditions, names, yields) ---
    if all_pathways:
        all_pathways = await _enrich_pathways_with_llm(all_pathways)

    # --- Phase 4: Score and sort ---
    for p in all_pathways:
        p.compute_scores()
    all_pathways.sort(
        key=lambda p: (p.confidence_score or 0, p.overall_yield or 0),
        reverse=True,
    )

    return all_pathways[:5]  # top 5


async def _query_all_sources(smiles: str) -> dict[str, dict]:
    """Query IBM RXN, ASKCOS, and ORD in parallel with timeouts."""
    loop = asyncio.get_event_loop()
    results = {}

    # Define tasks with individual timeouts
    api_tasks = {}
    if RXN_API_KEY:
        api_tasks["ibm_rxn"] = (
            loop.run_in_executor(None, _call_rxn, smiles),
            60,  # 60s timeout for RXN (it polls internally)
        )

    api_tasks["askcos"] = (
        loop.run_in_executor(None, _call_askcos, smiles),
        120,  # 120s for ASKCOS tree search
    )

    api_tasks["ord"] = (
        loop.run_in_executor(None, _call_ord, smiles),
        15,  # 15s for ORD search
    )

    # Run all with individual timeouts
    async def _run_with_timeout(name, future, timeout):
        try:
            return name, await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            return name, {"error": f"{name} timed out after {timeout}s"}
        except Exception as e:
            return name, {"error": str(e)}

    gathered = await asyncio.gather(
        *[_run_with_timeout(n, f, t) for n, (f, t) in api_tasks.items()]
    )

    for name, result in gathered:
        results[name] = result

    return results


def _call_rxn(smiles: str) -> dict:
    """Call IBM RXN retrosynthesis (synchronous)."""
    from src.tools.rxn_api import rxn_retrosynthesis
    return rxn_retrosynthesis.invoke({"smiles": smiles})


def _call_askcos(smiles: str) -> dict:
    """Call ASKCOS retrosynthesis (synchronous)."""
    from src.tools.askcos_api import askcos_retrosynthesis
    return askcos_retrosynthesis.invoke({"smiles": smiles})


def _call_ord(smiles: str) -> dict:
    """Call ORD product search (synchronous)."""
    from src.tools.ord_api import ord_search_by_product
    return ord_search_by_product.invoke({"smiles": smiles})


def _convert_to_pathways(
    data: dict, target_smiles: str, source: str
) -> list[SynthesisPathway]:
    """Convert raw API results to SynthesisPathway objects."""
    pathways = []

    # Handle pathway-based results (IBM RXN, ASKCOS)
    for p_data in data.get("pathways", []):
        steps = []
        for s_data in p_data.get("steps", []):
            step = ReactionStep(
                step_number=s_data.get("step_number", len(steps) + 1),
                reaction_smiles=s_data.get("reaction_smiles", ""),
                conditions=ReactionConditions(),
                confidence=s_data.get("confidence", s_data.get("score", 0.5)),
                source=source,
            )
            steps.append(step)

        pathway = SynthesisPathway(
            pathway_id=p_data.get("pathway_id", f"{source}_{uuid.uuid4().hex[:8]}"),
            target_smiles=target_smiles,
            steps=steps,
            confidence_score=p_data.get("confidence", 0.5),
        )
        pathways.append(pathway)

    # Handle reaction-list results (ORD)
    for rxn in data.get("reactions", []):
        reaction_smiles = rxn.get("reaction_smiles", "")
        if not reaction_smiles:
            continue

        conditions = ReactionConditions(
            temperature=rxn.get("temperature"),
            solvent=rxn.get("solvents", [None])[0] if rxn.get("solvents") else None,
            catalyst=rxn.get("catalysts", [None])[0] if rxn.get("catalysts") else None,
        )

        step = ReactionStep(
            step_number=1,
            reaction_smiles=reaction_smiles,
            conditions=conditions,
            expected_yield=rxn.get("yield", None),
            source="ord",
            source_id=rxn.get("doi", ""),
            confidence=0.9,  # published data = high confidence
        )

        pathway = SynthesisPathway(
            pathway_id=f"ord_{uuid.uuid4().hex[:8]}",
            target_smiles=target_smiles,
            steps=[step],
        )
        pathways.append(pathway)

    return pathways


async def _enrich_pathways_with_llm(
    pathways: list[SynthesisPathway],
) -> list[SynthesisPathway]:
    """Use LLM to add missing conditions, reaction types, reagent names."""
    llm = get_llm(temperature=0.1)

    # Build context from all pathways
    context = ""
    for p in pathways:
        context += f"\n--- {p.pathway_id} (source: {p.steps[0].source if p.steps else 'unknown'}) ---\n"
        for step in p.steps:
            context += f"Step {step.step_number}: {step.reaction_smiles}\n"
            if step.conditions.temperature:
                context += f"  Temperature: {step.conditions.temperature}\n"
            if step.conditions.solvent:
                context += f"  Solvent: {step.conditions.solvent}\n"
            if step.expected_yield:
                context += f"  Yield: {step.expected_yield}\n"

    messages = [
        SystemMessage(content=ENRICHMENT_PROMPT),
        HumanMessage(content=f"Enrich these pathways:\n{context}"),
    ]

    try:
        response = await llm.ainvoke(messages)
        _apply_enrichment(pathways, response.content)
    except Exception as e:
        logger.warning(f"LLM enrichment failed: {e}")

    return pathways


def _apply_enrichment(pathways: list[SynthesisPathway], response_text: str) -> None:
    """Apply LLM enrichment data to pathways."""
    import json

    try:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start < 0 or json_end <= json_start:
            return

        data = json.loads(response_text[json_start:json_end])
        enriched = data.get("enriched_steps", [])

        # Build a flat list of all steps across all pathways
        all_steps = []
        for p in pathways:
            all_steps.extend(p.steps)

        for enriched_step in enriched:
            step_num = enriched_step.get("step_number", 0)
            # Find matching step
            for step in all_steps:
                if step.step_number == step_num:
                    if enriched_step.get("reaction_type"):
                        step.reaction_type = enriched_step["reaction_type"]
                    conds = enriched_step.get("conditions", {})
                    if conds.get("temperature") and not step.conditions.temperature:
                        step.conditions.temperature = conds["temperature"]
                    if conds.get("solvent") and not step.conditions.solvent:
                        step.conditions.solvent = conds["solvent"]
                    if conds.get("catalyst") and not step.conditions.catalyst:
                        step.conditions.catalyst = conds["catalyst"]
                    if conds.get("time") and not step.conditions.time:
                        step.conditions.time = conds["time"]
                    if enriched_step.get("expected_yield") and not step.expected_yield:
                        step.expected_yield = enriched_step["expected_yield"]
                    break

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Failed to parse enrichment: {e}")


# --- LLM fallback (only when databases return nothing) ---

LLM_RETRO_PROMPT = """You are an expert organic chemist. The retrosynthesis databases
returned no results for this molecule, so you need to propose pathways yourself.

IMPORTANT: Only propose well-known, validated reactions. Be conservative.

Return JSON:
{
  "pathways": [
    {
      "pathway_id": "llm_path_1",
      "steps": [
        {
          "step_number": 1,
          "reaction_smiles": "reactants>>product",
          "reaction_type": "type",
          "reagent_names": ["name1", "name2"],
          "conditions": {"temperature": "80°C", "solvent": "THF", "catalyst": "none", "time": "2h"},
          "expected_yield": 0.75
        }
      ]
    }
  ]
}"""


async def _llm_retrosynthesis(target_molecule: MoleculeInfo) -> list[SynthesisPathway]:
    """Fallback: use LLM when no database results available."""
    llm = get_llm(temperature=0.2)

    context = f"""Target: {target_molecule.name}
SMILES: {target_molecule.smiles}
Formula: {target_molecule.molecular_formula}
MW: {target_molecule.molecular_weight}"""

    messages = [
        SystemMessage(content=LLM_RETRO_PROMPT),
        HumanMessage(content=f"Plan retrosynthesis for:\n{context}"),
    ]

    response = await llm.ainvoke(messages)
    return _parse_llm_response(response.content, target_molecule.smiles)


def _parse_llm_response(
    response_text: str, target_smiles: str
) -> list[SynthesisPathway]:
    """Parse LLM retrosynthesis response."""
    import json

    pathways = []
    try:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            data = json.loads(response_text[json_start:json_end])

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
                        source="llm_fallback",
                        confidence=0.3,  # low confidence for LLM-only
                    )
                    steps.append(step)

                pathway = SynthesisPathway(
                    pathway_id=p_data.get("pathway_id", f"llm_{uuid.uuid4().hex[:8]}"),
                    target_smiles=target_smiles,
                    steps=steps,
                )
                pathways.append(pathway)

    except (json.JSONDecodeError, KeyError, TypeError):
        pathways.append(
            SynthesisPathway(
                pathway_id=f"llm_{uuid.uuid4().hex[:8]}",
                target_smiles=target_smiles,
                steps=[],
                confidence_score=0.0,
            )
        )

    return pathways
