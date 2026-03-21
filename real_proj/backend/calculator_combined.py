"""Stoichiometry & equivalents calculator — all-in-one module.

Merged from: config.py, molecular.py, pubchem.py, calculator.py

Usage:
    from calculator_combined import calculator_agent

    # Stoichiometry calc
    result = calculator_agent({
        "reaction_smiles": "CC(=O)Oc1ccccc1C(O)=O>>CC(=O)O.Oc1ccccc1C(O)=O",
        "target_mass_g": 1.0,
    })

    # Equivalents calc
    result = calculator_agent({
        "reference_smiles": "CCO",
        "reference_amount": 1.0,
        "amount_type": "reagent_moles",
        "reagents": [
            {"smiles": "CCO", "equivalents": 1.0, "name": "Ethanol"},
            {"smiles": "CC(=O)O", "equivalents": 1.2, "name": "Acetic acid"},
        ],
    })
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# config.py — Pydantic models & enums
# ═══════════════════════════════════════════════════════════════════════════════

import logging
import re
import time
from collections import Counter
from enum import Enum
from functools import lru_cache
from typing import Any, Dict
from urllib.parse import quote

import requests
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

logger = logging.getLogger(__name__)


class AmountType(str, Enum):
    PRODUCT_MASS  = "product_mass"
    REAGENT_MASS  = "reagent_mass"
    REAGENT_MOLES = "reagent_moles"


class PhysicalState(str, Enum):
    SOLID   = "solid"
    LIQUID  = "liquid"
    GAS     = "gas"
    UNKNOWN = "unknown"


class ReagentInput(BaseModel):
    smiles:      str
    name:        str   = ""
    equivalents: float = 1.0
    role:        str   = "reagent"


class StoichiometryRequest(BaseModel):
    reaction_smiles:       str
    target_mass_g:         float = Field(..., gt=0)
    target_product_smiles: str | None = None


class EquivalentsRequest(BaseModel):
    reference_smiles:  str
    reference_amount:  float = Field(..., gt=0)
    amount_type:       AmountType = AmountType.PRODUCT_MASS
    reagents:          list[ReagentInput]


class ReagentCalcResult(BaseModel):
    smiles:           str
    name:             str
    molecular_weight: float
    equivalents:      float
    moles:            float
    mass_g:           float
    density:          float | None = None
    volume_ml:        float | None = None
    state:            PhysicalState = PhysicalState.UNKNOWN
    notes:            str = ""


class CalculationResult(BaseModel):
    target_product_smiles: str
    target_mass_g:         float
    target_moles:          float
    reagents:              list[ReagentCalcResult]
    warnings:              list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# molecular.py — RDKit helpers
# ═══════════════════════════════════════════════════════════════════════════════

def validate_smiles(smiles: str) -> bool:
    if not smiles or not smiles.strip():
        return False
    return Chem.MolFromSmiles(smiles) is not None


def get_average_molecular_weight(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Descriptors.MolWt(mol)


def get_molecular_formula(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return rdMolDescriptors.CalcMolFormula(mol)


def canonicalize(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol)


def parse_reaction_smiles(reaction_smiles: str) -> tuple[list[str], list[str]]:
    if ">>" in reaction_smiles:
        parts = reaction_smiles.split(">>")
        if len(parts) != 2:
            raise ValueError(f"Expected exactly one '>>' in: {reaction_smiles}")
        reactants_str, products_str = parts
    elif ">" in reaction_smiles:
        parts = reaction_smiles.split(">")
        if len(parts) != 3:
            raise ValueError(f"Expected 'reactants>agents>products': {reaction_smiles}")
        reactants_str, _, products_str = parts
    else:
        raise ValueError(f"No '>>' or '>' separator found in: {reaction_smiles}")

    reactants = [s.strip() for s in reactants_str.split(".") if s.strip()]
    products  = [s.strip() for s in products_str.split(".")  if s.strip()]
    if not reactants:
        raise ValueError("No reactants found")
    if not products:
        raise ValueError("No products found")
    return reactants, products


# ═══════════════════════════════════════════════════════════════════════════════
# pubchem.py — PUG REST client
# ═══════════════════════════════════════════════════════════════════════════════

_BASE_URL     = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_PUG_VIEW_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
_TIMEOUT      = 15
_RETRY_DELAY  = 0.3


def _get_json(url: str, *, retries: int = 2) -> dict | None:
    for attempt in range(1, retries + 2):
        try:
            resp = requests.get(url, timeout=_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                return None
            if resp.status_code == 503 and attempt <= retries:
                time.sleep(_RETRY_DELAY * attempt)
                continue
            return None
        except requests.RequestException:
            if attempt <= retries:
                time.sleep(_RETRY_DELAY * attempt)
    return None


@lru_cache(maxsize=512)
def get_cid_by_smiles(smiles: str) -> int | None:
    data = _get_json(f"{_BASE_URL}/compound/smiles/{quote(smiles, safe='')}/cids/JSON")
    try:
        return data["IdentifierList"]["CID"][0]
    except (KeyError, IndexError, TypeError):
        return None


@lru_cache(maxsize=512)
def get_compound_properties(smiles: str) -> dict:
    url = (f"{_BASE_URL}/compound/smiles/{quote(smiles, safe='')}/property/"
           "MolecularWeight,MolecularFormula,IUPACName,IsomericSMILES/JSON")
    data = _get_json(url)
    try:
        return data["PropertyTable"]["Properties"][0]
    except (KeyError, IndexError, TypeError):
        return {}


def _extract_numeric(text: str) -> float | None:
    m = re.search(r"(-?\d+\.?\d*)", text)
    return float(m.group(1)) if m else None


def _walk_sections(sections: list, heading: str) -> dict | None:
    for sec in sections:
        if sec.get("TOCHeading", "").lower() == heading.lower():
            return sec
        found = _walk_sections(sec.get("Section", []), heading)
        if found:
            return found
    return None


def _extract_temperature_celsius(section: dict) -> float | None:
    celsius, fahrenheit = [], []
    for info in section.get("Information", []):
        for swm in info.get("Value", {}).get("StringWithMarkup", []):
            text = swm.get("String", "")
            num = _extract_numeric(text)
            if num is None:
                continue
            tl = text.lower()
            if "°c" in tl or "deg c" in tl:
                celsius.append(num)
            elif "°f" in tl or "deg f" in tl:
                fahrenheit.append(num)
    if celsius:
        return celsius[0]
    if fahrenheit:
        return round((fahrenheit[0] - 32) * 5 / 9, 2)
    return None


def _extract_string_value(section: dict) -> str | None:
    for info in section.get("Information", []):
        val = info.get("Value", {})
        for swm in val.get("StringWithMarkup", []):
            text = swm.get("String", "")
            if text:
                return text
        num = val.get("Number")
        if num is not None:
            nums = num if isinstance(num, list) else [num]
            return f"{nums[0]} {val.get('Unit', '')}".strip()
    return None


@lru_cache(maxsize=512)
def get_density(smiles: str) -> float | None:
    cid = get_cid_by_smiles(smiles)
    if not cid:
        return None
    data = _get_json(f"{_PUG_VIEW_URL}/data/compound/{cid}/JSON?heading=Density")
    if not data:
        return None
    try:
        sec = _walk_sections(data["Record"]["Section"], "Density")
        return _extract_numeric(_extract_string_value(sec) or "") if sec else None
    except (KeyError, TypeError):
        return None


@lru_cache(maxsize=512)
def _get_temperature(smiles: str, heading: str) -> float | None:
    cid = get_cid_by_smiles(smiles)
    if not cid:
        return None
    safe_heading = heading.replace(" ", "+")
    data = _get_json(f"{_PUG_VIEW_URL}/data/compound/{cid}/JSON?heading={safe_heading}")
    if not data:
        return None
    try:
        sec = _walk_sections(data["Record"]["Section"], heading)
        return _extract_temperature_celsius(sec) if sec else None
    except (KeyError, TypeError):
        return None


def get_boiling_point(smiles: str) -> float | None:
    return _get_temperature(smiles, "Boiling Point")


def get_melting_point(smiles: str) -> float | None:
    return _get_temperature(smiles, "Melting Point")


def estimate_physical_state(smiles: str) -> str:
    mp = get_melting_point(smiles)
    bp = get_boiling_point(smiles)
    if mp is not None and mp > 25:
        return "solid"
    if bp is not None and bp < 25:
        return "gas"
    if mp is not None and mp <= 25:
        return "liquid"
    if bp is not None and bp >= 25:
        return "liquid"
    return "liquid" if get_density(smiles) else "unknown"


def get_iupac_name(smiles: str) -> str:
    return get_compound_properties(smiles).get("IUPACName", "")


# ═══════════════════════════════════════════════════════════════════════════════
# calculator.py — business logic
# ═══════════════════════════════════════════════════════════════════════════════

DROP_VOLUME_ML = 0.05


def _build_reagent_result(smiles: str, *, equivalents: float, moles: float,
                          name: str = "", warnings: list[str]) -> ReagentCalcResult:
    mw     = get_average_molecular_weight(smiles)
    mass_g = moles * mw
    if not name:
        name = get_iupac_name(smiles)

    state_str = estimate_physical_state(smiles)
    state = PhysicalState(state_str) if state_str in PhysicalState._value2member_map_ else PhysicalState.UNKNOWN

    density = volume_ml = None
    notes = ""

    if state == PhysicalState.LIQUID:
        density = get_density(smiles)
        if density and density > 0:
            volume_ml = round(mass_g / density, 4)
            if volume_ml < 0.1:
                drops = round(volume_ml / DROP_VOLUME_ML, 1)
                word = "капля" if drops == 1 else ("капли" if 2 <= drops <= 4 else "капель")
                notes = f"~{drops} {word}"
        else:
            warnings.append(f"Плотность не найдена для {name or smiles}; объём не рассчитан")

    return ReagentCalcResult(
        smiles=smiles, name=name,
        molecular_weight=round(mw, 4),
        equivalents=round(equivalents, 4),
        moles=round(moles, 6),
        mass_g=round(mass_g, 4),
        density=round(density, 4) if density is not None else None,
        volume_ml=round(volume_ml, 4) if volume_ml is not None else None,
        state=state, notes=notes,
    )


def stoichiometry_calc(request: StoichiometryRequest) -> CalculationResult:
    warnings: list[str] = []
    reactant_list, product_list = parse_reaction_smiles(request.reaction_smiles)

    if request.target_product_smiles:
        target_smiles = canonicalize(request.target_product_smiles)
        if target_smiles not in [canonicalize(p) for p in product_list]:
            raise ValueError(f"target_product_smiles not found among products: {product_list}")
    else:
        target_smiles = canonicalize(product_list[0])
        if len(product_list) > 1:
            warnings.append("Несколько продуктов; расчёт по первому. Укажите target_product_smiles.")

    target_mw    = get_average_molecular_weight(target_smiles)
    target_moles = request.target_mass_g / target_mw

    coeff_map     = dict(Counter([canonicalize(s) for s in reactant_list]))
    product_coeff = Counter([canonicalize(s) for s in product_list]).get(target_smiles, 1)

    reagents = [
        _build_reagent_result(smi, equivalents=coeff / product_coeff,
                              moles=target_moles * coeff / product_coeff, warnings=warnings)
        for smi, coeff in coeff_map.items()
    ]

    return CalculationResult(
        target_product_smiles=target_smiles,
        target_mass_g=round(request.target_mass_g, 4),
        target_moles=round(target_moles, 6),
        reagents=reagents, warnings=warnings,
    )


def equivalents_calc(request: EquivalentsRequest) -> CalculationResult:
    warnings: list[str] = []
    ref_smiles = canonicalize(request.reference_smiles)
    if not validate_smiles(ref_smiles):
        raise ValueError(f"Invalid reference SMILES: {request.reference_smiles}")

    ref_mw = get_average_molecular_weight(ref_smiles)
    if request.amount_type == AmountType.REAGENT_MOLES:
        reference_moles = request.reference_amount
    else:
        reference_moles = request.reference_amount / ref_mw

    reagents = []
    for r in request.reagents:
        r_smiles = canonicalize(r.smiles)
        if not validate_smiles(r_smiles):
            warnings.append(f"Невалидный SMILES пропущен: {r.smiles}")
            continue
        reagents.append(_build_reagent_result(
            r_smiles, equivalents=r.equivalents,
            moles=reference_moles * r.equivalents,
            name=r.name, warnings=warnings,
        ))

    return CalculationResult(
        target_product_smiles=ref_smiles,
        target_mass_g=round(reference_moles * ref_mw, 4),
        target_moles=round(reference_moles, 6),
        reagents=reagents, warnings=warnings,
    )


def calculator_agent(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Universal entry point — auto-detects calc type from input keys."""
    if "reaction_smiles" in request_data and "target_mass_g" in request_data:
        req = StoichiometryRequest(
            reaction_smiles=request_data["reaction_smiles"],
            target_mass_g=request_data["target_mass_g"],
            target_product_smiles=request_data.get("target_product_smiles"),
        )
        return stoichiometry_calc(req).model_dump()

    if "reference_smiles" in request_data and "reagents" in request_data:
        reagents = [
            ReagentInput(smiles=r["smiles"], name=r.get("name", ""),
                         equivalents=r.get("equivalents", 1.0), role=r.get("role", "reagent"))
            for r in request_data["reagents"]
        ]
        req = EquivalentsRequest(
            reference_smiles=request_data["reference_smiles"],
            reference_amount=request_data["reference_amount"],
            amount_type=AmountType(request_data["amount_type"]),
            reagents=reagents,
        )
        return equivalents_calc(req).model_dump()

    raise ValueError(
        "Не удалось определить тип расчёта. "
        "Для stoichiometry_calc: reaction_smiles + target_mass_g. "
        "Для equivalents_calc: reference_smiles + reference_amount + amount_type + reagents."
    )
