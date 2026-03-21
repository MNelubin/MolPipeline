"""Open Reaction Database (ORD) — local SQLite search + remote API fallback.

Primary: search local SQLite index built from ORD protobuf data (~2.3M reactions).
Fallback: query ORD public API at open-reaction-database.org.
"""

import logging
import sqlite3
from pathlib import Path

import httpx
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

ORD_DB_PATH = Path(__file__).parent.parent.parent / "data" / "ord_reactions.db"
ORD_API_URL = "https://open-reaction-database.org/api"

_client = httpx.Client(timeout=30.0)

# RDKit for canonical SMILES matching
try:
    from rdkit import Chem

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def _get_db() -> sqlite3.Connection | None:
    """Get SQLite connection to local ORD index."""
    if not ORD_DB_PATH.exists():
        return None
    return sqlite3.connect(str(ORD_DB_PATH))


def _local_search_by_product(smiles: str, limit: int = 10) -> list[dict]:
    """Search local ORD SQLite index for reactions producing the target."""
    conn = _get_db()
    if conn is None:
        return []

    results = []

    # Try exact canonical match first
    if HAS_RDKIT:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)

            cursor = conn.execute(
                """
                SELECT r.id, r.reaction_smiles, r.yield_pct,
                       r.temperature, r.solvent, r.catalyst
                FROM product_index pi
                JOIN reactions r ON r.id = pi.reaction_id
                WHERE pi.canonical_smiles = ?
                LIMIT ?
                """,
                (canonical, limit),
            )
            results = _rows_to_dicts(cursor)

    # If no exact match, try substring match on product components
    if not results:
        cursor = conn.execute(
            """
            SELECT r.id, r.reaction_smiles, r.yield_pct,
                   r.temperature, r.solvent, r.catalyst
            FROM components c
            JOIN reactions r ON r.id = c.reaction_id
            WHERE c.role = 'product' AND c.smiles = ?
            LIMIT ?
            """,
            (smiles, limit),
        )
        results = _rows_to_dicts(cursor)

    # Substructure search via RDKit if exact match found too few
    if len(results) < limit and HAS_RDKIT:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Get more candidates and filter by substructure
            cursor = conn.execute(
                """
                SELECT DISTINCT c.smiles, c.reaction_id
                FROM components c
                WHERE c.role = 'product'
                LIMIT 50000
                """
            )
            sub_ids = []
            for prod_smi, rxn_id in cursor:
                if rxn_id in {r["reaction_id"] for r in results}:
                    continue
                prod_mol = Chem.MolFromSmiles(prod_smi)
                if prod_mol and prod_mol.HasSubstructMatch(mol):
                    sub_ids.append(rxn_id)
                    if len(sub_ids) + len(results) >= limit:
                        break

            if sub_ids:
                placeholders = ",".join("?" for _ in sub_ids)
                cursor = conn.execute(
                    f"""
                    SELECT id, reaction_smiles, yield_pct,
                           temperature, solvent, catalyst
                    FROM reactions
                    WHERE id IN ({placeholders})
                    """,
                    sub_ids,
                )
                results.extend(_rows_to_dicts(cursor))

    conn.close()
    return results[:limit]


def _local_search_by_reactant(smiles: str, limit: int = 10) -> list[dict]:
    """Search local ORD index for reactions using the given reactant."""
    conn = _get_db()
    if conn is None:
        return []

    # Canonicalize
    search_smiles = smiles
    if HAS_RDKIT:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            search_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

    cursor = conn.execute(
        """
        SELECT r.id, r.reaction_smiles, r.yield_pct,
               r.temperature, r.solvent, r.catalyst
        FROM components c
        JOIN reactions r ON r.id = c.reaction_id
        WHERE c.role = 'reactant' AND c.smiles = ?
        LIMIT ?
        """,
        (search_smiles, limit),
    )
    results = _rows_to_dicts(cursor)
    conn.close()
    return results[:limit]


def _rows_to_dicts(cursor) -> list[dict]:
    """Convert SQLite rows to reaction dicts."""
    results = []
    for row in cursor:
        rxn_id, rxn_smi, yield_pct, temp, solvent, catalyst = row
        if not rxn_smi:
            continue

        reactant_str = rxn_smi.split(">>")[0] if ">>" in rxn_smi else ""

        result = {
            "reaction_id": rxn_id,
            "reaction_smiles": rxn_smi,
            "reactants": reactant_str.split(".") if reactant_str else [],
            "source": "ord",
        }
        if yield_pct is not None:
            result["yield"] = yield_pct
        if temp:
            result["temperature"] = temp
        if solvent:
            result["solvents"] = [solvent]
        if catalyst:
            result["catalysts"] = [catalyst]
        results.append(result)
    return results


@tool
def ord_search_by_product(smiles: str, limit: int = 10) -> dict:
    """Search the Open Reaction Database for reactions that produce a given molecule.

    Searches local ORD index (~2.3M reactions from USPTO patents) first,
    falls back to ORD public API if local index is not available.

    Args:
        smiles: Product molecule SMILES to search for
        limit: Maximum number of results (default 10)
    """
    # Try local SQLite first
    local_results = _local_search_by_product(smiles, limit)
    if local_results:
        logger.info(f"ORD local: found {len(local_results)} reactions for {smiles}")
        return {
            "query_smiles": smiles,
            "num_results": len(local_results),
            "reactions": local_results,
            "source": "ord_local",
        }

    # Fallback to remote API
    logger.info(f"ORD local: no results, trying remote API for {smiles}")
    return _remote_search(smiles, "product", limit)


@tool
def ord_search_by_reactant(smiles: str, limit: int = 10) -> dict:
    """Search ORD for reactions that use a given molecule as a reactant.

    Args:
        smiles: Reactant molecule SMILES
        limit: Maximum number of results (default 10)
    """
    local_results = _local_search_by_reactant(smiles, limit)
    if local_results:
        return {
            "query_smiles": smiles,
            "num_results": len(local_results),
            "reactions": local_results,
            "source": "ord_local",
        }
    return _remote_search(smiles, "reactant", limit)


def _remote_search(smiles: str, role: str, limit: int) -> dict:
    """Query ORD remote API as fallback."""
    url = f"{ORD_API_URL}/query"
    params = {
        "component": smiles,
        "limit": limit,
    }

    try:
        resp = _client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        reactions = data if isinstance(data, list) else data.get("reactions", [])
        parsed = []
        for rxn in reactions[:limit]:
            p = _parse_remote_reaction(rxn)
            if p:
                parsed.append(p)

        return {
            "query_smiles": smiles,
            "num_results": len(parsed),
            "reactions": parsed,
            "source": "ord_remote",
        }
    except Exception as e:
        logger.warning(f"ORD remote search failed: {e}")
        return {
            "query_smiles": smiles,
            "num_results": 0,
            "reactions": [],
            "error": f"ORD unavailable: {e}",
        }


def _parse_remote_reaction(rxn: dict) -> dict | None:
    """Parse a single remote ORD API reaction."""
    if not rxn:
        return None

    identifiers = rxn.get("identifiers", [])
    rxn_smiles = ""
    for ident in identifiers:
        if ident.get("type") == "REACTION_SMILES" or "smiles" in ident.get("type", "").lower():
            rxn_smiles = ident.get("value", "")
            break

    conditions = rxn.get("conditions", {})
    temp = conditions.get("temperature", {})
    temp_val = temp.get("setpoint", {}).get("value")
    temp_units = temp.get("setpoint", {}).get("units")

    yield_val = None
    for outcome in rxn.get("outcomes", []):
        for product in outcome.get("products", []):
            for meas in product.get("measurements", []):
                if meas.get("type") == "YIELD":
                    yield_val = meas.get("percentage", {}).get("value")
                    break

    inputs_data = rxn.get("inputs", {})
    reactants = []
    solvents = []
    catalysts = []
    if isinstance(inputs_data, dict):
        for input_name, input_val in inputs_data.items():
            for comp in input_val.get("components", []):
                role = comp.get("reaction_role", "")
                smi = ""
                for ident in comp.get("identifiers", []):
                    if "smiles" in ident.get("type", "").lower():
                        smi = ident.get("value", "")
                        break
                if not smi:
                    continue
                if role == "REACTANT":
                    reactants.append(smi)
                elif role == "SOLVENT":
                    solvents.append(smi)
                elif role == "CATALYST":
                    catalysts.append(smi)

    result = {
        "reaction_smiles": rxn_smiles,
        "reactants": reactants,
        "solvents": solvents,
        "catalysts": catalysts,
        "source": "ord",
    }
    if temp_val is not None:
        result["temperature"] = f"{temp_val} {temp_units}" if temp_units else str(temp_val)
    if yield_val is not None:
        result["yield"] = yield_val

    return result
