"""Banned chemicals and reactions filter.

Autonomous filtering system inspired by ASKCOS expand_one_controller.
Checks retrosynthesis results against known dangerous/illegal chemicals
and reaction patterns before presenting them to users.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"


@lru_cache(maxsize=1)
def _load_banned_chemicals() -> dict[str, dict]:
    """Load banned chemicals list and index by canonical SMILES."""
    path = DATA_DIR / "banned_chemicals.json"
    if not path.exists():
        logger.warning("banned_chemicals.json not found")
        return {}

    with open(path) as f:
        data = json.load(f)

    index = {}
    try:
        from rdkit import Chem
    except ImportError:
        # Without RDKit, use raw SMILES as keys
        for entry in data.get("chemicals", data.get("banned", [])):
            index[entry["smiles"]] = entry
        return index

    for entry in data.get("chemicals", data.get("banned", [])):
        mol = Chem.MolFromSmiles(entry["smiles"])
        if mol:
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
            index[canonical] = entry
        else:
            index[entry["smiles"]] = entry

    return index


@lru_cache(maxsize=1)
def _load_banned_reactions() -> list[dict]:
    """Load banned reaction SMARTS patterns."""
    path = DATA_DIR / "banned_reactions.json"
    if not path.exists():
        logger.warning("banned_reactions.json not found")
        return []

    with open(path) as f:
        data = json.load(f)

    patterns = []
    try:
        from rdkit import Chem
    except ImportError:
        return data.get("reactions", data.get("banned", []))

    for entry in data.get("reactions", data.get("banned", [])):
        smarts = entry.get("smarts", "")
        # Validate that the SMARTS pattern is parseable
        if ">>" in smarts:
            parts = smarts.split(">>")
            reactant_pat = Chem.MolFromSmarts(parts[0])
            product_pat = Chem.MolFromSmarts(parts[1])
            if reactant_pat or product_pat:
                entry["_parsed"] = True
                patterns.append(entry)
                continue
        else:
            pat = Chem.MolFromSmarts(smarts)
            if pat:
                entry["_parsed"] = True
                patterns.append(entry)
                continue

        # Keep unparsed patterns for substring matching
        entry["_parsed"] = False
        patterns.append(entry)

    return patterns


def check_smiles_banned(smiles: str) -> dict | None:
    """Check if a single SMILES is in the banned list.

    Args:
        smiles: SMILES string to check.

    Returns:
        Banned entry dict if found, None otherwise.
    """
    banned = _load_banned_chemicals()
    if not banned:
        return None

    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    except ImportError:
        canonical = smiles

    return banned.get(canonical)


def check_reaction_banned(reaction_smiles: str) -> dict | None:
    """Check if a reaction matches any banned reaction pattern.

    Checks both exact SMILES match and substructure SMARTS match.

    Args:
        reaction_smiles: Reaction SMILES in format 'reactants>>products'.

    Returns:
        Banned entry dict if matched, None otherwise.
    """
    patterns = _load_banned_reactions()
    if not patterns:
        return None

    try:
        from rdkit import Chem
    except ImportError:
        return None

    if ">>" not in reaction_smiles:
        return None

    reactant_str, product_str = reaction_smiles.split(">>", 1)
    product_mols = [
        Chem.MolFromSmiles(s.strip())
        for s in product_str.split(".")
        if s.strip()
    ]
    product_mols = [m for m in product_mols if m is not None]

    for entry in patterns:
        smarts = entry.get("smarts", "")
        if not smarts or not entry.get("_parsed", False):
            continue

        if ">>" in smarts:
            # Check product side of SMARTS against products
            _, prod_smarts = smarts.split(">>", 1)
            prod_pat = Chem.MolFromSmarts(prod_smarts)
            if prod_pat:
                for pmol in product_mols:
                    if pmol.HasSubstructMatch(prod_pat):
                        return entry
        else:
            # Check pattern against all product molecules
            pat = Chem.MolFromSmarts(smarts)
            if pat:
                for pmol in product_mols:
                    if pmol.HasSubstructMatch(pat):
                        return entry

    return None


def filter_retro_results(results: list[dict], target_smiles: str) -> list[dict]:
    """Filter retrosynthesis results, removing banned chemicals and reactions.

    Mirrors ASKCOS expand_one_controller deduplication logic:
    1. Check each precursor SMILES against banned chemicals
    2. Check each reaction SMILES against banned reactions
    3. Skip trivial results (precursor == target)

    Args:
        results: List of retro result dicts with 'reactants' key (SMILES).
        target_smiles: Target molecule SMILES.

    Returns:
        Filtered list with banned entries removed.
    """
    try:
        from rdkit import Chem
        target_mol = Chem.MolFromSmiles(target_smiles)
        cano_target = Chem.MolToSmiles(target_mol, isomericSmiles=True) if target_mol else target_smiles
    except ImportError:
        cano_target = target_smiles

    filtered = []
    for result in results:
        reactants_str = result.get("reactants", "")
        if not reactants_str:
            continue

        # Canonicalize
        try:
            from rdkit import Chem
            parts = []
            for smi in reactants_str.split("."):
                mol = Chem.MolFromSmiles(smi.strip())
                if mol:
                    parts.append(Chem.MolToSmiles(mol, isomericSmiles=True))
                else:
                    parts.append(smi.strip())
            cano_reactants = ".".join(parts)
        except ImportError:
            cano_reactants = reactants_str

        # Check 1: trivial result (precursor == target)
        if cano_reactants == cano_target:
            continue

        # Check 2: banned chemicals in precursors
        is_banned = False
        for smi in cano_reactants.split("."):
            ban = check_smiles_banned(smi)
            if ban:
                logger.warning(
                    f"Filtered banned chemical: {ban['name']} "
                    f"({ban['category']}) from retro results"
                )
                is_banned = True
                break

        if is_banned:
            continue

        # Check 3: banned reaction pattern
        rxn_smi = cano_reactants + ">>" + cano_target
        ban = check_reaction_banned(rxn_smi)
        if ban:
            logger.warning(
                f"Filtered banned reaction: {ban['name']} "
                f"({ban['category']}) from retro results"
            )
            continue

        filtered.append(result)

    return filtered


@tool
def check_chemical_safety(smiles: str) -> dict:
    """Check if a chemical is on the banned/restricted list.

    Checks against CWC schedules, explosive precursors, narcotics precursors,
    and environmentally banned substances.

    Args:
        smiles: SMILES string of the chemical to check.
    """
    ban = check_smiles_banned(smiles)
    if ban:
        return {
            "banned": True,
            "name": ban.get("name", "unknown"),
            "category": ban.get("category", "unknown"),
            "reason": ban.get("reason", ""),
            "smiles": smiles,
        }
    return {"banned": False, "smiles": smiles}


@tool
def check_reaction_safety(reaction_smiles: str) -> dict:
    """Check if a reaction matches known banned/dangerous reaction patterns.

    Checks against CWC prohibited synthesis routes, explosive synthesis,
    and controlled substance synthesis patterns.

    Args:
        reaction_smiles: Reaction SMILES in format 'reactants>>products'.
    """
    ban = check_reaction_banned(reaction_smiles)
    if ban:
        return {
            "banned": True,
            "name": ban.get("name", "unknown"),
            "category": ban.get("category", "unknown"),
            "reason": ban.get("reason", ""),
            "reaction_smiles": reaction_smiles,
        }
    return {"banned": False, "reaction_smiles": reaction_smiles}
