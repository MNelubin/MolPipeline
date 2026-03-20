"""Stoichiometry and equivalents calculation tools."""

from langchain_core.tools import tool


@tool
def stoichiometry_calc(
    reaction_smiles: str,
    target_product_mass_g: float,
    product_index: int = 0,
) -> dict:
    """Calculate stoichiometry for a reaction given a target product mass.

    Args:
        reaction_smiles: Reaction SMILES in format 'reactants>>products'
        target_product_mass_g: Desired mass of product in grams
        product_index: Index of the target product (if multiple products)
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError:
        return {"error": "RDKit is not installed"}

    if ">>" not in reaction_smiles:
        return {"error": "Invalid reaction SMILES: must contain '>>'"}

    reactant_str, product_str = reaction_smiles.split(">>")
    reactants = [s.strip() for s in reactant_str.split(".") if s.strip()]
    products = [s.strip() for s in product_str.split(".") if s.strip()]

    if not products or product_index >= len(products):
        return {"error": "No products found or invalid product_index"}

    # Calculate product moles
    prod_mol = Chem.MolFromSmiles(products[product_index])
    if prod_mol is None:
        return {"error": f"Invalid product SMILES: {products[product_index]}"}

    prod_mw = Descriptors.ExactMolWt(prod_mol)
    target_moles = target_product_mass_g / prod_mw

    # Calculate required mass for each reagent (1:1 stoichiometry assumed)
    reagent_data = []
    for smi in reactants:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            reagent_data.append({"smiles": smi, "error": "Invalid SMILES"})
            continue

        mw = Descriptors.ExactMolWt(mol)
        mass = target_moles * mw

        reagent_data.append({
            "smiles": smi,
            "molecular_formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
            "molecular_weight": round(mw, 4),
            "moles": round(target_moles, 6),
            "mass_g": round(mass, 4),
            "equivalents": 1.0,
        })

    return {
        "target_product_smiles": products[product_index],
        "target_product_mw": round(prod_mw, 4),
        "target_mass_g": target_product_mass_g,
        "target_moles": round(target_moles, 6),
        "reagents": reagent_data,
    }


@tool
def equivalents_calc(
    reagents: list[dict],
    reference_amount: float,
    amount_type: str = "product_mass",
    product_mw: float | None = None,
) -> dict:
    """Calculate masses and volumes for reagents given equivalents.

    This is the key calculator tool. It takes a list of reagents with their
    equivalents and a reference amount, then calculates moles, masses, and volumes.

    Args:
        reagents: List of dicts with keys: name, smiles, equivalents, and optionally density
        reference_amount: Amount in grams (mass) or moles
        amount_type: "product_mass" | "reagent_mass" | "reagent_moles"
        product_mw: Molecular weight of product (required if amount_type is "product_mass")
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError:
        return {"error": "RDKit is not installed"}

    # Determine reference moles
    if amount_type == "product_mass":
        if product_mw is None or product_mw <= 0:
            return {"error": "product_mw required for amount_type='product_mass'"}
        reference_moles = reference_amount / product_mw
    elif amount_type == "reagent_moles":
        reference_moles = reference_amount
    elif amount_type == "reagent_mass":
        # Need the first reagent's MW to convert
        if not reagents:
            return {"error": "No reagents provided"}
        first_smi = reagents[0].get("smiles", "")
        first_mol = Chem.MolFromSmiles(first_smi)
        if first_mol is None:
            return {"error": f"Invalid SMILES for first reagent: {first_smi}"}
        first_mw = Descriptors.ExactMolWt(first_mol)
        reference_moles = reference_amount / first_mw
    else:
        return {"error": f"Unknown amount_type: {amount_type}"}

    results = []
    for r in reagents:
        smiles = r.get("smiles", "")
        name = r.get("name", smiles)
        equiv = r.get("equivalents", 1.0)
        density = r.get("density")

        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            results.append({"name": name, "error": f"Invalid SMILES: {smiles}"})
            continue

        mw = Descriptors.ExactMolWt(mol)
        moles = reference_moles * equiv
        mass = moles * mw
        volume = (mass / density) if density and density > 0 else None

        entry = {
            "name": name,
            "smiles": smiles,
            "molecular_weight": round(mw, 4),
            "equivalents": equiv,
            "moles": round(moles, 6),
            "mass_g": round(mass, 4),
        }
        if volume is not None:
            entry["volume_ml"] = round(volume, 3)
        if density:
            entry["density"] = density

        # Add helpful notes for small amounts
        if mass < 0.01:
            entry["notes"] = "Very small amount — consider using solution"
        elif mass < 0.1 and density:
            drops = volume * 20 if volume else None  # ~20 drops per mL
            if drops and drops < 10:
                entry["notes"] = f"~{int(drops)} drops"

        results.append(entry)

    return {
        "reference_moles": round(reference_moles, 6),
        "amount_type": amount_type,
        "reference_amount": reference_amount,
        "reagents": results,
    }
