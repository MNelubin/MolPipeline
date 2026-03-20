"""RDKit-based tools for molecular property calculations."""

from langchain_core.tools import tool


@tool
def rdkit_properties(smiles: str) -> dict:
    """Calculate molecular properties from a SMILES string using RDKit.

    Returns: molecular weight, LogP, H-bond donors/acceptors, TPSA,
    rotatable bonds, canonical SMILES, and validity check.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
    except ImportError:
        return {"error": "RDKit is not installed"}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: {smiles}", "valid": False}

    return {
        "valid": True,
        "canonical_smiles": Chem.MolToSmiles(mol),
        "molecular_weight": round(Descriptors.ExactMolWt(mol), 4),
        "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
        "logp": round(Descriptors.MolLogP(mol), 2),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "heavy_atom_count": mol.GetNumHeavyAtoms(),
        "ring_count": Descriptors.RingCount(mol),
    }


@tool
def rdkit_validate_smiles(smiles: str) -> dict:
    """Validate a SMILES string and return canonical form if valid."""
    try:
        from rdkit import Chem
    except ImportError:
        return {"error": "RDKit is not installed"}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid": False, "input": smiles}
    return {"valid": True, "canonical_smiles": Chem.MolToSmiles(mol), "input": smiles}


@tool
def rdkit_reaction_balance(reaction_smiles: str) -> dict:
    """Check if a reaction SMILES is balanced (atom conservation).

    Args:
        reaction_smiles: Reaction in format 'reactants>>products'
    """
    try:
        from rdkit import Chem
    except ImportError:
        return {"error": "RDKit is not installed"}

    if ">>" not in reaction_smiles:
        return {"error": "Invalid reaction SMILES: must contain '>>'"}

    parts = reaction_smiles.split(">>")
    if len(parts) != 2:
        return {"error": "Invalid reaction SMILES format"}

    reactant_smiles, product_smiles = parts

    def count_atoms(smiles_str: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for smi in smiles_str.split("."):
            mol = Chem.MolFromSmiles(smi.strip())
            if mol is None:
                continue
            mol = Chem.AddHs(mol)
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                counts[symbol] = counts.get(symbol, 0) + 1
        return counts

    reactant_atoms = count_atoms(reactant_smiles)
    product_atoms = count_atoms(product_smiles)

    all_elements = set(reactant_atoms.keys()) | set(product_atoms.keys())
    imbalanced = {}
    for elem in all_elements:
        r_count = reactant_atoms.get(elem, 0)
        p_count = product_atoms.get(elem, 0)
        if r_count != p_count:
            imbalanced[elem] = {"reactants": r_count, "products": p_count}

    return {
        "balanced": len(imbalanced) == 0,
        "reactant_atoms": reactant_atoms,
        "product_atoms": product_atoms,
        "imbalanced_elements": imbalanced,
    }
