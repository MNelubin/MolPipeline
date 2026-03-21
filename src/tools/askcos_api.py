"""ASKCOS API — self-hosted retrosynthesis planning via MIT ASKCOS."""

import httpx
from langchain_core.tools import tool

from src.config import ASKCOS_BASE_URL

_client = httpx.Client(timeout=300.0)  # retro can be slow


@tool
def askcos_retrosynthesis(smiles: str) -> dict:
    """Run tree-search retrosynthesis via ASKCOS v2.

    Performs MCTS-based retrosynthesis expanding a full tree
    of possible synthesis routes from commercially available starting materials.

    Args:
        smiles: Target molecule SMILES
    """
    url = f"{ASKCOS_BASE_URL}/api/tree-search/mcts/call-sync-without-token"

    payload = {
        "smiles": smiles,
    }

    try:
        resp = _client.post(url, json=payload)
        resp.raise_for_status()
    except httpx.ConnectError:
        return {"error": f"ASKCOS not reachable at {ASKCOS_BASE_URL}"}
    except httpx.HTTPStatusError as e:
        return {"error": f"ASKCOS API error: {e.response.status_code} {e.response.text[:200]}"}
    except httpx.RequestError as e:
        return {"error": f"ASKCOS request failed: {e}"}

    data = resp.json()

    if not data.get("result"):
        return {"error": "ASKCOS returned no results", "raw": str(data)[:500]}

    return _parse_askcos_results(data, smiles)


@tool
def askcos_one_step(smiles: str, top_n: int = 10) -> dict:
    """Run one-step retrosynthesis via ASKCOS v2.

    Returns possible one-step disconnections for the target molecule
    using template-relevance model.

    Args:
        smiles: Target molecule SMILES
        top_n: Number of top results to return (default 10)
    """
    url = f"{ASKCOS_BASE_URL}/api/retro/template-relevance/call-sync"

    payload = {
        "smiles": [smiles],
    }

    try:
        resp = _client.post(url, json=payload)
        resp.raise_for_status()
    except httpx.ConnectError:
        return {"error": f"ASKCOS not reachable at {ASKCOS_BASE_URL}"}
    except httpx.HTTPStatusError as e:
        return {"error": f"ASKCOS API error: {e.response.status_code}"}
    except httpx.RequestError as e:
        return {"error": f"ASKCOS request failed: {e}"}

    data = resp.json()

    # v2 returns list of results per input SMILES
    all_results = data if isinstance(data, list) else [data]
    results = all_results[0] if all_results else []
    if isinstance(results, dict):
        results = results.get("result", results.get("outcomes", []))

    parsed = []
    for r in (results if isinstance(results, list) else [])[:top_n]:
        reactants = r.get("smiles", r.get("precursors", ""))
        if isinstance(reactants, list):
            reactants = ".".join(reactants)
        parsed.append({
            "reactants": reactants,
            "template": r.get("template", r.get("tforms", "")),
            "score": r.get("score", r.get("prob", 0.0)),
            "num_examples": r.get("num_examples", 0),
            "source": "askcos",
        })

    return {
        "target": smiles,
        "num_results": len(parsed),
        "disconnections": parsed,
    }


def _parse_askcos_results(data: dict, target_smiles: str) -> dict:
    """Parse ASKCOS tree-search results into our format."""
    trees = data.get("result", {}).get("trees", [])
    if not trees and isinstance(data.get("result"), list):
        trees = data["result"]

    pathways = []
    for i, tree in enumerate(trees[:5]):  # top 5 trees
        steps = []
        _extract_askcos_steps(tree, steps, step_num=1)

        score = tree.get("score", 0.0)
        if isinstance(score, dict):
            score = score.get("plausibility", 0.0)

        pathways.append({
            "pathway_id": f"askcos_path_{i + 1}",
            "source": "askcos",
            "confidence": score,
            "steps": steps,
        })

    return {
        "target_smiles": target_smiles,
        "num_pathways": len(pathways),
        "pathways": pathways,
    }


def _extract_askcos_steps(node: dict, steps: list, step_num: int) -> int:
    """Recursively extract steps from ASKCOS tree."""
    children = node.get("children", [])
    if not children:
        return step_num

    # Process children first (bottom-up)
    reactant_smiles = []
    for child in children:
        child_smiles = child.get("smiles", child.get("chemical", {}).get("smiles", ""))
        if child_smiles:
            reactant_smiles.append(child_smiles)
        step_num = _extract_askcos_steps(child, steps, step_num)

    product_smiles = node.get("smiles", node.get("chemical", {}).get("smiles", ""))

    if reactant_smiles and product_smiles:
        reaction_smiles = ".".join(reactant_smiles) + ">>" + product_smiles

        # Extract reaction metadata if present
        reaction_data = node.get("reaction", {})

        steps.append({
            "step_number": step_num,
            "reaction_smiles": reaction_smiles,
            "reactants": reactant_smiles,
            "product": product_smiles,
            "template": reaction_data.get("template", ""),
            "score": reaction_data.get("score", node.get("score", 0.0)),
            "num_examples": reaction_data.get("num_examples", 0),
        })
        step_num += 1

    return step_num
