"""RDKit tools: molecular properties, SMILES validation, reaction parsing."""

from __future__ import annotations

from collections import Counter

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


def rdkit_properties(smiles: str) -> dict:
    """Calculate molecular properties from SMILES using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: {smiles}"}

    return {
        "molecular_weight": round(Descriptors.MolWt(mol), 4),
        "logp": round(Descriptors.MolLogP(mol), 4),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "h_bond_acceptors": Lipinski.NumHAcceptors(mol),
        "h_bond_donors": Lipinski.NumHDonors(mol),
        "heavy_atoms": Descriptors.HeavyAtomCount(mol),
        "ring_count": Lipinski.RingCount(mol),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Functions merged from src/services/molecular.py
# ═══════════════════════════════════════════════════════════════════════════════

def validate_smiles(smiles: str) -> bool:
    """Return True if RDKit can parse the SMILES string."""
    if not smiles or not smiles.strip():
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def get_molecular_weight(smiles: str) -> float:
    """Exact molecular weight (g/mol) from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Descriptors.ExactMolWt(mol)


def get_average_molecular_weight(smiles: str) -> float:
    """Average molecular weight (uses natural isotope distribution)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Descriptors.MolWt(mol)


def get_molecular_formula(smiles: str) -> str:
    """Molecular formula string (e.g. 'C9H8O4') from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return rdMolDescriptors.CalcMolFormula(mol)


def canonicalize(smiles: str) -> str:
    """Return canonical SMILES. Returns the input unchanged on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol)


def parse_reaction_smiles(
    reaction_smiles: str,
) -> tuple[list[str], list[str]]:
    """Split 'A.B>>C.D' into ([reactant_smiles, ...], [product_smiles, ...])."""
    if ">>" in reaction_smiles:
        parts = reaction_smiles.split(">>")
        if len(parts) != 2:
            raise ValueError(
                f"Expected exactly one '>>' in reaction SMILES: {reaction_smiles}"
            )
        reactants_str, products_str = parts
    elif ">" in reaction_smiles:
        parts = reaction_smiles.split(">")
        if len(parts) != 3:
            raise ValueError(
                f"Expected 'reactants>agents>products' format: {reaction_smiles}"
            )
        reactants_str, _, products_str = parts
    else:
        raise ValueError(
            f"No '>>' or '>' separator found in reaction SMILES: {reaction_smiles}"
        )

    reactants = [s.strip() for s in reactants_str.split(".") if s.strip()]
    products = [s.strip() for s in products_str.split(".") if s.strip()]

    if not reactants:
        raise ValueError("No reactants found in reaction SMILES")
    if not products:
        raise ValueError("No products found in reaction SMILES")

    return reactants, products


def count_stoichiometric_coefficients(smiles_list: list[str]) -> dict[str, int]:
    """Given a list of (possibly repeated) SMILES, return canonical SMILES -> count."""
    canonical = [canonicalize(s) for s in smiles_list]
    return dict(Counter(canonical))
