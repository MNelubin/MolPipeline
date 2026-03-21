"""Calculation tools: stoichiometry_calc and equivalents_calc."""

from __future__ import annotations

import logging
from collections import Counter

from ..models.calculations import (
    AmountType,
    CalculationResult,
    EquivalentsRequest,
    PhysicalState,
    ReagentCalcResult,
    StoichiometryRequest,
)
from .rdkit_tools import (
    canonicalize,
    get_average_molecular_weight,
    parse_reaction_smiles,
    validate_smiles,
)
from .pubchem import (
    estimate_physical_state,
    get_density,
    get_iupac_name,
)

logger = logging.getLogger(__name__)

DROP_VOLUME_ML = 0.05


def _build_reagent_result(
    smiles: str,
    *,
    equivalents: float,
    moles: float,
    name: str = "",
    warnings: list[str],
) -> ReagentCalcResult:
    mw = get_average_molecular_weight(smiles)
    mass_g = moles * mw

    if not name:
        name = get_iupac_name(smiles)

    state_str = estimate_physical_state(smiles)
    state = PhysicalState(state_str) if state_str in PhysicalState.__members__.values() else PhysicalState.UNKNOWN

    density: float | None = None
    volume_ml: float | None = None
    notes = ""

    if state == PhysicalState.LIQUID:
        density = get_density(smiles)
        if density is not None and density > 0:
            volume_ml = round(mass_g / density, 4)
            if volume_ml < 0.1:
                drops = round(volume_ml / DROP_VOLUME_ML, 1)
                if drops == 1:
                    word = "капля"
                elif 2 <= drops <= 4:
                    word = "капли"
                else:
                    word = "капель"
                notes = f"~{drops} {word}"
        else:
            warnings.append(
                f"Плотность не найдена для {name or smiles}; объём не рассчитан"
            )

    return ReagentCalcResult(
        smiles=smiles,
        name=name,
        molecular_weight=round(mw, 4),
        equivalents=round(equivalents, 4),
        moles=round(moles, 6),
        mass_g=round(mass_g, 4),
        density=round(density, 4) if density is not None else None,
        volume_ml=round(volume_ml, 4) if volume_ml is not None else None,
        state=state,
        notes=notes,
    )


def stoichiometry_calc(request: StoichiometryRequest) -> CalculationResult:
    """Calculate masses / volumes of every reagent for a given reaction."""
    warnings: list[str] = []

    reactant_list, product_list = parse_reaction_smiles(request.reaction_smiles)

    if request.target_product_smiles:
        target_smiles = canonicalize(request.target_product_smiles)
        canonical_products = [canonicalize(p) for p in product_list]
        if target_smiles not in canonical_products:
            raise ValueError(
                f"target_product_smiles '{request.target_product_smiles}' "
                f"not found among products: {product_list}"
            )
    else:
        target_smiles = canonicalize(product_list[0])
        if len(product_list) > 1:
            warnings.append(
                "Несколько продуктов в реакции; расчёт ведётся по первому. "
                "Укажите target_product_smiles для уточнения."
            )

    if not validate_smiles(target_smiles):
        raise ValueError(f"Invalid product SMILES: {target_smiles}")

    target_mw = get_average_molecular_weight(target_smiles)
    target_moles = request.target_mass_g / target_mw

    canonical_reactants = [canonicalize(s) for s in reactant_list]
    coeff_map: dict[str, int] = dict(Counter(canonical_reactants))

    canonical_products_all = [canonicalize(s) for s in product_list]
    product_coeff = Counter(canonical_products_all).get(target_smiles, 1)

    reagent_results: list[ReagentCalcResult] = []
    for smiles, coeff in coeff_map.items():
        equiv = coeff / product_coeff
        moles = target_moles * equiv
        result = _build_reagent_result(
            smiles, equivalents=equiv, moles=moles, warnings=warnings,
        )
        reagent_results.append(result)

    return CalculationResult(
        target_product_smiles=target_smiles,
        target_mass_g=round(request.target_mass_g, 4),
        target_moles=round(target_moles, 6),
        reagents=reagent_results,
        warnings=warnings,
    )


def equivalents_calc(request: EquivalentsRequest) -> CalculationResult:
    """Calculate masses / volumes for each reagent based on equivalents."""
    warnings: list[str] = []

    ref_smiles = canonicalize(request.reference_smiles)
    if not validate_smiles(ref_smiles):
        raise ValueError(f"Invalid reference SMILES: {request.reference_smiles}")

    ref_mw = get_average_molecular_weight(ref_smiles)

    if request.amount_type == AmountType.PRODUCT_MASS:
        reference_moles = request.reference_amount / ref_mw
    elif request.amount_type == AmountType.REAGENT_MASS:
        reference_moles = request.reference_amount / ref_mw
    elif request.amount_type == AmountType.REAGENT_MOLES:
        reference_moles = request.reference_amount
    else:
        raise ValueError(f"Unknown amount_type: {request.amount_type}")

    reagent_results: list[ReagentCalcResult] = []
    for reagent in request.reagents:
        r_smiles = canonicalize(reagent.smiles)
        if not validate_smiles(r_smiles):
            warnings.append(f"Невалидный SMILES пропущен: {reagent.smiles}")
            continue

        moles = reference_moles * reagent.equivalents
        result = _build_reagent_result(
            r_smiles, equivalents=reagent.equivalents, moles=moles,
            name=reagent.name, warnings=warnings,
        )
        reagent_results.append(result)

    return CalculationResult(
        target_product_smiles=ref_smiles,
        target_mass_g=round(
            request.reference_amount
            if request.amount_type == AmountType.PRODUCT_MASS
            else reference_moles * ref_mw,
            4,
        ),
        target_moles=round(reference_moles, 6),
        reagents=reagent_results,
        warnings=warnings,
    )
