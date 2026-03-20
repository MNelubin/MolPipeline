"""MoleculeInfoAgent — retrieves information about any chemical compound."""

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import get_llm
from src.models.molecule import GHSClassification, MoleculeInfo, MoleculeProperties
from src.models.state import AgentState
from src.tools.pubchem import (
    pubchem_description,
    pubchem_image_url,
    pubchem_lookup,
    pubchem_safety,
)
from src.tools.rdkit_tools import rdkit_properties

SYSTEM_PROMPT = """You are a chemistry information specialist. Your job is to gather
complete information about a chemical compound using the available tools.

Steps:
1. Use pubchem_lookup to find the compound and get its CID and basic data
2. Use rdkit_properties to calculate molecular properties from SMILES
3. Use pubchem_safety to get GHS safety classification
4. Use pubchem_description to get a textual description
5. Use pubchem_image_url to get the structure image

If pubchem_lookup fails, try different identifier types (name, SMILES, formula).
Always return structured data about the compound.

IMPORTANT: If a tool returns an error, note it but continue with other tools.
Do not hallucinate properties — only report what the tools return."""


def create_molecule_info_agent():
    """Create the MoleculeInfoAgent with its tools."""
    llm = get_llm()

    tools = [
        pubchem_lookup,
        rdkit_properties,
        pubchem_safety,
        pubchem_description,
        pubchem_image_url,
    ]

    return llm.bind_tools(tools), tools


def build_molecule_info(tool_results: dict) -> MoleculeInfo:
    """Build a MoleculeInfo object from collected tool results."""
    pubchem_data = tool_results.get("pubchem", {})
    rdkit_data = tool_results.get("rdkit", {})
    safety_data = tool_results.get("safety", {})
    desc_data = tool_results.get("description", {})

    properties = MoleculeProperties(
        melting_point=pubchem_data.get("melting_point"),
        boiling_point=pubchem_data.get("boiling_point"),
        density=pubchem_data.get("density"),
        log_p=rdkit_data.get("logp"),
    )

    ghs = GHSClassification(
        pictograms=safety_data.get("pictograms", []),
        h_statements=safety_data.get("h_statements", []),
        p_statements=safety_data.get("p_statements", []),
        signal_word=safety_data.get("signal_word"),
    )

    return MoleculeInfo(
        name=pubchem_data.get("iupac_name", "Unknown"),
        iupac_name=pubchem_data.get("iupac_name"),
        smiles=pubchem_data.get("smiles", rdkit_data.get("canonical_smiles", "")),
        canonical_smiles=rdkit_data.get("canonical_smiles"),
        molecular_formula=(
            pubchem_data.get("molecular_formula")
            or rdkit_data.get("molecular_formula")
        ),
        molecular_weight=(
            pubchem_data.get("molecular_weight") or rdkit_data.get("molecular_weight")
        ),
        pubchem_cid=pubchem_data.get("cid"),
        inchi=pubchem_data.get("inchi"),
        properties=properties,
        ghs=ghs,
        image_url=tool_results.get("image_url"),
    )


async def run_molecule_info(state: AgentState, query: str) -> MoleculeInfo:
    """Run the molecule info pipeline for a given query.

    This is a simplified direct-call version that doesn't rely on the LLM
    for tool orchestration — it calls tools directly in sequence.
    """
    tool_results = {}

    # Step 1: PubChem lookup
    pubchem_result = pubchem_lookup.invoke({"query": query})
    if "error" not in pubchem_result:
        tool_results["pubchem"] = pubchem_result
    else:
        tool_results["pubchem"] = {}

    smiles = pubchem_result.get("smiles", "")
    cid = pubchem_result.get("cid")

    # Step 2: RDKit properties (if we have SMILES)
    if smiles:
        rdkit_result = rdkit_properties.invoke({"smiles": smiles})
        if "error" not in rdkit_result:
            tool_results["rdkit"] = rdkit_result
        else:
            tool_results["rdkit"] = {}
    else:
        tool_results["rdkit"] = {}

    # Step 3: Safety data
    if cid:
        safety_result = pubchem_safety.invoke({"cid": cid})
        if "error" not in safety_result:
            tool_results["safety"] = safety_result
        else:
            tool_results["safety"] = {}
    else:
        tool_results["safety"] = {}

    # Step 4: Description
    if cid:
        desc_result = pubchem_description.invoke({"cid": cid})
        tool_results["description"] = desc_result
    else:
        tool_results["description"] = {}

    # Step 5: Image URL
    if cid:
        tool_results["image_url"] = pubchem_image_url.invoke({"cid": cid})

    return build_molecule_info(tool_results)
