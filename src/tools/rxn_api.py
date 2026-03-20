"""IBM RXN for Chemistry API — real retrosynthesis predictions.

Note: IBM RXN uses CloudFront which may geo-block certain regions.
If blocked, the tool returns an error and the retrosynthesis agent
falls back to ASKCOS / LLM.
"""

import logging
import time

from langchain_core.tools import tool

import os

from src.config import RXN_API_KEY, RXN_PROJECT_NAME, SOCKS_PROXY

logger = logging.getLogger(__name__)

_wrapper = None
_init_failed = False


def _setup_proxy():
    """Configure SOCKS proxy for IBM RXN if set."""
    if SOCKS_PROXY:
        os.environ.setdefault("HTTP_PROXY", SOCKS_PROXY)
        os.environ.setdefault("HTTPS_PROXY", SOCKS_PROXY)
        logger.info(f"Using SOCKS proxy for IBM RXN: {SOCKS_PROXY[:30]}...")


def _get_rxn():
    """Lazy-init the RXN wrapper."""
    global _wrapper, _init_failed
    if _init_failed:
        raise RuntimeError("IBM RXN previously failed to initialize (geo-blocked?)")
    if _wrapper is None:
        _setup_proxy()
        from rxn4chemistry import RXN4ChemistryWrapper

        _wrapper = RXN4ChemistryWrapper(api_key=RXN_API_KEY)
        resp = _wrapper.create_project(RXN_PROJECT_NAME)
        # Detect CloudFront geo-block
        resp_str = str(resp.get("response", ""))
        if "403 ERROR" in resp_str or "CloudFront" in resp_str:
            _init_failed = True
            _wrapper = None
            raise RuntimeError("IBM RXN geo-blocked from this server")
        if not _wrapper.project_id:
            _init_failed = True
            _wrapper = None
            raise RuntimeError(f"IBM RXN project creation failed: {resp_str[:200]}")
        time.sleep(2)
    return _wrapper


@tool
def rxn_retrosynthesis(smiles: str, max_steps: int = 5) -> dict:
    """Run automatic retrosynthesis prediction via IBM RXN.

    Takes a target molecule SMILES and returns retrosynthetic pathways
    predicted by a model trained on 2.5M real reactions.

    Args:
        smiles: Target molecule SMILES string
        max_steps: Maximum retrosynthesis depth (default 5)
    """
    try:
        rxn = _get_rxn()
    except Exception as e:
        return {"error": f"Failed to initialize RXN client: {e}"}

    try:
        response = rxn.predict_automatic_retrosynthesis(
            smiles, max_steps=max_steps
        )
    except Exception as e:
        return {"error": f"RXN retrosynthesis request failed: {e}"}

    prediction_id = response.get("prediction_id")
    if not prediction_id:
        return {"error": f"No prediction_id returned: {response}"}

    # Poll for results — short timeout since the caller has its own timeout
    for attempt in range(6):
        time.sleep(8)
        try:
            results = rxn.get_predict_automatic_retrosynthesis_results(
                prediction_id
            )
        except Exception as e:
            return {"error": f"Failed to get results: {e}"}

        status = results.get("status", "")
        if status == "SUCCESS":
            return _parse_rxn_results(results, smiles)
        elif status in ("FAILED", "ERROR"):
            return {"error": f"RXN prediction failed: {results}"}
        # else still RUNNING, continue polling

    return {"error": "RXN prediction still processing (caller will timeout)"}


def _parse_rxn_results(results: dict, target_smiles: str) -> dict:
    """Parse IBM RXN retrosynthesis results into our format."""
    paths = results.get("retrosynthetic_paths", [])

    pathways = []
    for i, path in enumerate(paths):
        steps = []
        _extract_steps(path, steps, step_num=1)

        pathways.append({
            "pathway_id": f"rxn_path_{i + 1}",
            "source": "ibm_rxn",
            "confidence": path.get("confidence", 0.0),
            "steps": steps,
        })

    return {
        "target_smiles": target_smiles,
        "num_pathways": len(pathways),
        "pathways": pathways,
    }


def _extract_steps(node: dict, steps: list, step_num: int) -> int:
    """Recursively extract reaction steps from RXN tree structure."""
    children = node.get("children", [])
    if not children:
        return step_num

    # This node has children = it's a reaction
    reactant_smiles = []
    for child in children:
        smiles = child.get("smiles", "")
        if smiles:
            reactant_smiles.append(smiles)
        # Recurse into children
        step_num = _extract_steps(child, steps, step_num)

    product_smiles = node.get("smiles", "")
    if reactant_smiles and product_smiles:
        reaction_smiles = ".".join(reactant_smiles) + ">>" + product_smiles
        steps.append({
            "step_number": step_num,
            "reaction_smiles": reaction_smiles,
            "reactants": reactant_smiles,
            "product": product_smiles,
            "confidence": node.get("confidence", 0.0),
        })
        step_num += 1

    return step_num


@tool
def rxn_predict_reaction(reactants_smiles: str) -> dict:
    """Predict the product of a forward reaction using IBM RXN.

    Args:
        reactants_smiles: Reactants as SMILES, separated by dots (e.g., 'CC(=O)O.OC1=CC=CC=C1')
    """
    try:
        rxn = _get_rxn()
        response = rxn.predict_reaction(reactants_smiles)
    except Exception as e:
        return {"error": f"RXN forward prediction failed: {e}"}

    prediction_id = response.get("prediction_id")
    if not prediction_id:
        return {"error": f"No prediction_id: {response}"}

    time.sleep(5)

    try:
        results = rxn.get_predict_reaction_results(prediction_id)
    except Exception as e:
        return {"error": f"Failed to get results: {e}"}

    return {
        "reactants": reactants_smiles,
        "predicted_product": results.get("response", {}).get(
            "payload", {}).get("attempts", [{}])[0].get(
            "smiles", "unknown"),
        "confidence": results.get("response", {}).get(
            "payload", {}).get("attempts", [{}])[0].get(
            "confidence", 0.0),
    }
