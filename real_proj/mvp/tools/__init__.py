"""Unified tools package — re-exports all public functions for backward compatibility.

Existing imports like ``from ..tools import get_cid_by_name`` continue to work.
"""

from .pubchem import (  # noqa: F401
    _get_json,
    enrich_ghs_pictograms,
    estimate_physical_state,
    get_boiling_point,
    get_cas_number,
    get_cid_by_name,
    get_cid_by_smiles,
    get_compound_properties,
    get_density,
    get_experimental_properties,
    get_ghs_pictogram_info,
    get_ghs_safety,
    get_iupac_name,
    get_ld50,
    get_melting_point,
    get_molecule_images,
    get_physical_description,
    get_smiles_by_cid,
    GHS_PICTOGRAMS,
    pubchem_lookup,
    safety_lookup,
)

from .rdkit_tools import (  # noqa: F401
    canonicalize,
    count_stoichiometric_coefficients,
    get_average_molecular_weight,
    get_molecular_formula,
    get_molecular_weight,
    parse_reaction_smiles,
    rdkit_properties,
    validate_smiles,
)

from .safety import (  # noqa: F401
    banlist_check,
    ppe_recommender,
    reaction_banlist_check,
)
