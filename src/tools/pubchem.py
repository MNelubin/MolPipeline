"""PubChem API tools for molecule lookup, safety data, and images."""

import time

import httpx
from langchain_core.tools import tool

from src.config import PUBCHEM_BASE_URL, PUBCHEM_RATE_LIMIT_DELAY, PUBCHEM_VIEW_URL

_client = httpx.Client(timeout=30.0)
_last_request_time = 0.0


def _rate_limit() -> None:
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < PUBCHEM_RATE_LIMIT_DELAY:
        time.sleep(PUBCHEM_RATE_LIMIT_DELAY - elapsed)
    _last_request_time = time.time()


def _detect_identifier_type(query: str) -> tuple[str, str]:
    """Detect the type of chemical identifier and return (namespace, identifier)."""
    query = query.strip()

    # CAS number pattern: digits-digits-digit
    if all(c in "0123456789-" for c in query) and query.count("-") == 2:
        return "name", query  # PubChem resolves CAS via name search

    # SMILES: contains special chars like = ( ) [ ] # / but not spaces
    smiles_chars = set("=()[]#/\\@+-.%")
    if " " not in query and any(c in smiles_chars for c in query):
        return "smiles", query

    # Molecular formula: starts with uppercase, contains only letters and digits
    if query[0].isupper() and all(c.isalnum() for c in query.replace(" ", "")):
        # Could be a formula or a name — try formula first
        if any(c.isdigit() for c in query) and len(query) < 20:
            return "formula", query

    # Default: treat as name
    return "name", query


@tool
def pubchem_lookup(query: str) -> dict:
    """Look up a chemical compound in PubChem by name, SMILES, CAS number, or formula.

    Returns basic properties: CID, IUPAC name, molecular formula, weight, SMILES, etc.
    """
    namespace, identifier = _detect_identifier_type(query)
    _rate_limit()

    url = f"{PUBCHEM_BASE_URL}/compound/{namespace}/{identifier}/JSON"
    try:
        resp = _client.get(url)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return {"error": f"Compound not found: {query}"}
        return {"error": f"PubChem API error: {e.response.status_code}"}
    except httpx.RequestError as e:
        return {"error": f"Network error: {str(e)}"}

    data = resp.json()
    compounds = data.get("PC_Compounds", [])
    if not compounds:
        return {"error": f"No compounds returned for: {query}"}

    compound = compounds[0]
    cid = compound.get("id", {}).get("id", {}).get("cid")

    # Extract properties from the compound record
    props = {}
    for prop in compound.get("props", []):
        urn = prop.get("urn", {})
        label = urn.get("label", "")
        name = urn.get("name", "")
        value = prop.get("value", {})

        # Get the actual value (could be sval, fval, ival, etc.)
        val = (
            value.get("sval")
            or value.get("fval")
            or value.get("ival")
            or value.get("binary")
        )

        key = f"{label}_{name}" if name else label
        props[key.strip()] = val

    result = {
        "cid": cid,
        "iupac_name": props.get("IUPAC Name_Preferred", props.get("IUPAC Name_CAS-like Style")),
        "molecular_formula": props.get("Molecular Formula"),
        "molecular_weight": props.get("Molecular Weight"),
        "smiles": props.get("SMILES_Canonical", props.get("SMILES_Absolute")),
        "inchi": props.get("InChI_Standard"),
        "charge": props.get("Charge"),
    }

    return {k: v for k, v in result.items() if v is not None}


@tool
def pubchem_safety(cid: int) -> dict:
    """Get GHS safety classification data for a compound from PubChem.

    Args:
        cid: PubChem Compound ID
    """
    _rate_limit()

    url = f"{PUBCHEM_VIEW_URL}/data/compound/{cid}/JSON"
    params = {"heading": "GHS Classification"}
    try:
        resp = _client.get(url, params=params)
        resp.raise_for_status()
    except httpx.HTTPStatusError:
        return {"error": f"No safety data found for CID {cid}"}
    except httpx.RequestError as e:
        return {"error": f"Network error: {str(e)}"}

    data = resp.json()

    result = {
        "cid": cid,
        "pictograms": [],
        "h_statements": [],
        "p_statements": [],
        "signal_word": None,
    }

    # Navigate the nested PubChem View structure
    try:
        record = data.get("Record", {})
        sections = record.get("Section", [])
        for section in sections:
            for subsection in section.get("Section", []):
                heading = subsection.get("TOCHeading", "")

                for subsubsection in subsection.get("Section", []):
                    sub_heading = subsubsection.get("TOCHeading", "")
                    infos = subsubsection.get("Information", [])

                    for info in infos:
                        val = info.get("Value", {})
                        strings = val.get("StringWithMarkup", [])

                        if "Pictogram" in sub_heading:
                            for s in strings:
                                text = s.get("String", "")
                                if text:
                                    result["pictograms"].append(text)

                        elif "Signal" in sub_heading:
                            for s in strings:
                                text = s.get("String", "")
                                if text:
                                    result["signal_word"] = text

                        elif sub_heading.startswith("GHS Hazard"):
                            for s in strings:
                                text = s.get("String", "")
                                if text:
                                    result["h_statements"].append(text)

                        elif sub_heading.startswith("Precautionary"):
                            for s in strings:
                                text = s.get("String", "")
                                if text:
                                    result["p_statements"].append(text)

    except (KeyError, IndexError):
        pass

    return result


@tool
def pubchem_description(cid: int) -> dict:
    """Get the textual description of a compound from PubChem.

    Args:
        cid: PubChem Compound ID
    """
    _rate_limit()

    url = f"{PUBCHEM_VIEW_URL}/data/compound/{cid}/JSON"
    params = {"heading": "Record Description"}
    try:
        resp = _client.get(url, params=params)
        resp.raise_for_status()
    except httpx.HTTPStatusError:
        return {"cid": cid, "description": "No description available"}
    except httpx.RequestError as e:
        return {"error": f"Network error: {str(e)}"}

    data = resp.json()
    descriptions = []

    try:
        record = data.get("Record", {})
        for section in record.get("Section", []):
            for subsection in section.get("Section", []):
                for info in subsection.get("Information", []):
                    val = info.get("Value", {})
                    for s in val.get("StringWithMarkup", []):
                        text = s.get("String", "")
                        if text and len(text) > 20:
                            descriptions.append(text)
    except (KeyError, IndexError):
        pass

    return {"cid": cid, "description": "\n".join(descriptions[:3])}


@tool
def pubchem_image_url(cid: int) -> str:
    """Get the 2D structure image URL for a compound from PubChem.

    Args:
        cid: PubChem Compound ID
    """
    return f"{PUBCHEM_BASE_URL}/compound/cid/{cid}/PNG?image_size=300x300"
