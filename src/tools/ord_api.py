"""Open Reaction Database (ORD) — search for known reactions by product."""

import httpx
from langchain_core.tools import tool

ORD_SEARCH_URL = "https://client.open-reaction-database.org/api"

_client = httpx.Client(timeout=30.0)


@tool
def ord_search_by_product(smiles: str, limit: int = 10) -> dict:
    """Search the Open Reaction Database for reactions that produce a given molecule.

    Uses SMILES substructure search against ORD's public API to find
    real, published reactions with experimental conditions and yields.

    Args:
        smiles: Product molecule SMILES to search for
        limit: Maximum number of results (default 10)
    """
    # ORD search endpoint
    url = f"{ORD_SEARCH_URL}/query"

    payload = {
        "useStereochemistry": False,
        "similarity": 0.6,
        "component": [
            {
                "smiles": smiles,
                "source": "output",
                "mode": "substructure",
            }
        ],
        "limit": limit,
    }

    try:
        resp = _client.post(url, json=payload)
        resp.raise_for_status()
    except httpx.ConnectError:
        # Try alternative URL
        return _fallback_ord_search(smiles, limit)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return _fallback_ord_search(smiles, limit)
        return {"error": f"ORD API error: {e.response.status_code}"}
    except httpx.RequestError as e:
        return _fallback_ord_search(smiles, limit)

    return _parse_ord_results(resp.json(), smiles)


def _fallback_ord_search(smiles: str, limit: int) -> dict:
    """Fallback: try the ORD browse/search endpoint."""
    try:
        url = "https://open-reaction-database.org/api/search"
        resp = _client.get(url, params={"product": smiles, "limit": limit})
        resp.raise_for_status()
        return _parse_ord_results(resp.json(), smiles)
    except Exception:
        pass

    # Second fallback: try the client API with GET
    try:
        url = "https://client.open-reaction-database.org/api/fetch_reactions"
        resp = _client.get(url, params={"product_smiles": smiles, "limit": limit})
        resp.raise_for_status()
        return _parse_ord_results(resp.json(), smiles)
    except Exception as e:
        return {
            "error": f"ORD search unavailable: {e}",
            "target": smiles,
            "reactions": [],
        }


@tool
def ord_search_by_reactant(smiles: str, limit: int = 10) -> dict:
    """Search ORD for reactions that use a given molecule as a reactant.

    Args:
        smiles: Reactant molecule SMILES
        limit: Maximum number of results (default 10)
    """
    url = f"{ORD_SEARCH_URL}/query"

    payload = {
        "useStereochemistry": False,
        "similarity": 0.6,
        "component": [
            {
                "smiles": smiles,
                "source": "input",
                "mode": "substructure",
            }
        ],
        "limit": limit,
    }

    try:
        resp = _client.post(url, json=payload)
        resp.raise_for_status()
    except Exception as e:
        return {"error": f"ORD search failed: {e}", "reactions": []}

    return _parse_ord_results(resp.json(), smiles)


def _parse_ord_results(data: dict | list, query_smiles: str) -> dict:
    """Parse ORD API response into structured reactions."""
    reactions = []

    # Handle various response formats
    reaction_list = data if isinstance(data, list) else data.get("reactions", data.get("results", []))

    for rxn in reaction_list[:20]:
        parsed = _parse_single_reaction(rxn)
        if parsed:
            reactions.append(parsed)

    return {
        "query_smiles": query_smiles,
        "num_results": len(reactions),
        "reactions": reactions,
    }


def _parse_single_reaction(rxn: dict) -> dict | None:
    """Parse a single ORD reaction record."""
    if not rxn:
        return None

    # Extract reaction SMILES
    identifiers = rxn.get("identifiers", [])
    reaction_smiles = ""
    for ident in identifiers:
        if ident.get("type") == "REACTION_SMILES" or "smiles" in ident.get("type", "").lower():
            reaction_smiles = ident.get("value", "")
            break

    # Extract conditions
    conditions = rxn.get("conditions", {})
    temperature = conditions.get("temperature", {})
    temp_val = temperature.get("setpoint", {}).get("value")
    temp_units = temperature.get("setpoint", {}).get("units")

    # Extract yield from outcomes
    outcomes = rxn.get("outcomes", [])
    yield_val = None
    for outcome in outcomes:
        for product in outcome.get("products", []):
            for measurement in product.get("measurements", []):
                if measurement.get("type") == "YIELD":
                    yield_val = measurement.get("percentage", {}).get("value")
                    break

    # Extract inputs (reactants/reagents)
    inputs_data = rxn.get("inputs", {})
    reactants = []
    solvents = []
    catalysts = []
    for input_name, input_val in inputs_data.items() if isinstance(inputs_data, dict) else []:
        for component in input_val.get("components", []):
            role = component.get("reaction_role", "")
            smiles = ""
            for ident in component.get("identifiers", []):
                if "smiles" in ident.get("type", "").lower():
                    smiles = ident.get("value", "")
                    break

            if not smiles:
                continue

            if role == "REACTANT":
                reactants.append(smiles)
            elif role == "SOLVENT":
                solvents.append(smiles)
            elif role == "CATALYST":
                catalysts.append(smiles)

    # Extract provenance (DOI, source)
    provenance = rxn.get("provenance", {})
    doi = provenance.get("doi", "")

    result = {
        "reaction_smiles": reaction_smiles,
        "reactants": reactants,
        "solvents": solvents,
        "catalysts": catalysts,
        "source": "ord",
        "doi": doi,
    }

    if temp_val is not None:
        result["temperature"] = f"{temp_val} {temp_units}" if temp_units else str(temp_val)
    if yield_val is not None:
        result["yield"] = yield_val

    return result
