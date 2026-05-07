"""Shared SMILES normalization helpers.

AiZynthFinder and template planners often emit atom-mapped SMILES like
``[CH3:1][CH2:2]...``. These are useful for reaction tracing but poison
catalog lookups, PubChem resolving, and recursive route search. Runtime tools
should therefore translate planner output back to ordinary canonical SMILES
before using it as molecule identity.
"""

from __future__ import annotations

from rdkit import Chem


def clear_atom_maps(mol: Chem.Mol) -> Chem.Mol:
    """Remove atom-map annotations from an RDKit molecule in-place."""
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    return mol


def canonicalize_smiles(smiles: str) -> str | None:
    """Return canonical, unmapped SMILES or None when parsing fails."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    clear_atom_maps(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def canonicalize_smiles_list(smiles: str) -> str | None:
    """Canonicalize a dot-separated list of molecules, preserving order."""
    parts: list[str] = []
    for part in smiles.split("."):
        part = part.strip()
        if not part:
            continue
        canonical = canonicalize_smiles(part)
        if canonical is None:
            return None
        parts.append(canonical)
    return ".".join(parts)
