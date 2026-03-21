"""Molecule info node: gather data + LLM synthesis via OpenRouter."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from ..config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL, LLM_TEMPERATURE
from ..tools import pubchem_lookup, rdkit_properties

logger = logging.getLogger(__name__)

_MOLECULE_CARD_PROMPT = PromptTemplate.from_template("""
You are a chemistry expert. Fill in a molecule card based on the data provided.

Input data:
1. User query: {query}
2. PubChem data: {pubchem_data}
3. RDKit data: {rdkit_data}
4. Safety data: {safety_data}

Return a JSON object with these fields. Use your knowledge to fill in gaps.

Fields:
- "name": IUPAC name (English)
- "synonyms": list of common synonyms (list of strings)
- "smiles": SMILES string
- "molecular_formula": molecular formula
- "molecular_weight": molar mass (number)
- "properties": dict with keys: "melting_point", "boiling_point", "solubility", "density", "logP", "physical_state"
- "ghs_classification": list of GHS hazard classes (list of strings)
- "spectral_notes": brief note about spectral data (IR, NMR)
- "description": brief description in Russian
- "pubchem_cid": CID number (0 if unknown)

IMPORTANT: RDKit data (weight, logP) takes priority over PubChem computed values.
Return ONLY valid JSON, no markdown code fences.
""")


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def molecule_info_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: gather molecule info and produce a structured card.

    Reads: state["query"], state["smiles"], state["guard_result"]
    Writes: state["molecule_info"], state["final_answer"]
    """
    query = state.get("query", "")
    smiles = state.get("smiles", "")
    guard_result = state.get("guard_result", {})

    logger.info("[molecule_info] query=%r smiles=%r", query, smiles)

    # 1. Get data from PubChem and RDKit
    pubchem_result = {}
    rdkit_result = {}

    if smiles:
        rdkit_result = rdkit_properties(smiles)
        pubchem_data = pubchem_lookup(smiles)
        if "error" not in pubchem_data:
            pubchem_result = pubchem_data
        else:
            # Try by original query (might be a name)
            pubchem_data = pubchem_lookup(query)
            if "error" not in pubchem_data:
                pubchem_result = pubchem_data

    cid = pubchem_result.get("cid")

    # 2. Structure URL
    structure_url = ""
    if cid:
        structure_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG"
    elif smiles:
        structure_url = "Generate from SMILES"

    # 3. Safety data summary for the prompt
    safety_data = guard_result.get("safety_data", {})

    # 4. LLM synthesis
    llm = _get_llm()
    prompt_value = _MOLECULE_CARD_PROMPT.format(
        query=query,
        pubchem_data=json.dumps(pubchem_result, ensure_ascii=False),
        rdkit_data=json.dumps(rdkit_result, ensure_ascii=False),
        safety_data=json.dumps(safety_data, ensure_ascii=False),
    )

    try:
        llm_response = llm.invoke([HumanMessage(content=prompt_value)])
        text = llm_response.content
    except Exception as e:
        logger.error("[molecule_info] LLM call failed: %s", e)
        text = "{}"

    # Parse JSON response
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        parsed = json.loads(text)
    except Exception as e:
        logger.warning("[molecule_info] JSON parse error: %s", e)
        parsed = {}

    # 5. Build molecule_info
    props = parsed.get("properties", {})

    rdkit_weight = rdkit_result.get("molecular_weight")
    parsed_weight = parsed.get("molecular_weight")
    final_weight = _safe_float(rdkit_weight if rdkit_weight else parsed_weight)

    parsed_cid = parsed.get("pubchem_cid")
    final_cid = _safe_int(parsed_cid if parsed_cid not in ("", None) else cid)

    molecule_info = {
        "name": parsed.get("name", pubchem_result.get("iupac", "Unknown")),
        "synonyms": parsed.get("synonyms", pubchem_result.get("synonyms", [])),
        "smiles": parsed.get("smiles", smiles or ""),
        "molecular_formula": parsed.get("molecular_formula", pubchem_result.get("formula", "")),
        "molecular_weight": final_weight,
        "properties": {
            "melting_point": props.get("melting_point", "N/A"),
            "boiling_point": props.get("boiling_point", "N/A"),
            "solubility": props.get("solubility", "N/A"),
            "density": props.get("density", "N/A"),
            "logP": props.get("logP", rdkit_result.get("logp")),
            "physical_state": props.get("physical_state", "N/A"),
            "spectral_notes": parsed.get("spectral_notes", "N/A"),
        },
        "description": parsed.get("description", ""),
        "ghs_classification": parsed.get("ghs_classification", []),
        "pubchem_cid": final_cid,
        "structure_url": structure_url,
    }

    # 6. Build final text answer
    guard_status = guard_result.get("overall_status", "UNKNOWN")
    ppe_list = guard_result.get("ppe_recommendations", [])
    h_phrases = safety_data.get("h_phrases", [])
    ghs_pics = safety_data.get("ghs_pictograms", [])

    final_text = (
        f"{'='*60}\n"
        f"  MOLECULE CARD: {molecule_info['name']}\n"
        f"{'='*60}\n"
        f"  SMILES:    {molecule_info['smiles']}\n"
        f"  Formula:   {molecule_info['molecular_formula']}\n"
        f"  Weight:    {molecule_info['molecular_weight']:.2f} g/mol\n"
        f"  PubChem:   CID {molecule_info['pubchem_cid']}\n"
        f"  Structure: {molecule_info['structure_url']}\n"
        f"\n"
        f"  Properties:\n"
        f"    Melting point:  {molecule_info['properties']['melting_point']}\n"
        f"    Boiling point:  {molecule_info['properties']['boiling_point']}\n"
        f"    Solubility:     {molecule_info['properties']['solubility']}\n"
        f"    Density:        {molecule_info['properties']['density']}\n"
        f"    LogP:           {molecule_info['properties']['logP']}\n"
        f"    State:          {molecule_info['properties']['physical_state']}\n"
        f"\n"
        f"  Description: {molecule_info['description']}\n"
        f"\n"
        f"{'='*60}\n"
        f"  SAFETY REPORT\n"
        f"{'='*60}\n"
        f"  Status:       {guard_status}\n"
        f"  GHS Pictograms: {', '.join(ghs_pics) if ghs_pics else 'None'}\n"
        f"  H-phrases:    {'; '.join(h_phrases[:5]) if h_phrases else 'None'}\n"
        f"  PPE:          {', '.join(ppe_list) if ppe_list else 'Standard lab equipment'}\n"
        f"{'='*60}\n"
    )

    return {
        "molecule_info": molecule_info,
        "final_answer": final_text,
    }
