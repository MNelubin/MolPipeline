from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class AmountType(str, Enum):
    PRODUCT_MASS = "product_mass"
    REAGENT_MASS = "reagent_mass"
    REAGENT_MOLES = "reagent_moles"


class PhysicalState(str, Enum):
    SOLID = "solid"
    LIQUID = "liquid"
    GAS = "gas"
    UNKNOWN = "unknown"


class ReagentInput(BaseModel):
    """Single reagent with its equivalents for equivalents_calc."""

    smiles: str
    name: str = ""
    equivalents: float = 1.0
    role: str = "reagent"


class StoichiometryRequest(BaseModel):
    """Input for stoichiometry_calc: reaction SMILES + desired product mass."""

    reaction_smiles: str = Field(
        ..., description='Reaction in "reactants>>products" SMILES format'
    )
    target_mass_g: float = Field(..., gt=0, description="Desired product mass in grams")
    target_product_smiles: str | None = Field(
        None, description="Specific product SMILES when reaction has multiple products"
    )


class EquivalentsRequest(BaseModel):
    """Input for equivalents_calc: reference reagent + list of reagents with equivalents."""

    reference_smiles: str = Field(
        ..., description="SMILES of the reference compound (product or limiting reagent)"
    )
    reference_amount: float = Field(..., gt=0)
    amount_type: AmountType = AmountType.PRODUCT_MASS
    reagents: list[ReagentInput]


class ReagentCalcResult(BaseModel):
    """Calculated quantities for a single reagent."""

    smiles: str
    name: str
    molecular_weight: float = Field(..., description="g/mol")
    equivalents: float
    moles: float
    mass_g: float
    density: float | None = Field(None, description="g/mL for liquids")
    volume_ml: float | None = Field(None, description="mL for liquids")
    state: PhysicalState = PhysicalState.UNKNOWN
    notes: str = ""


class CalculationResult(BaseModel):
    """Full result of a stoichiometry / equivalents calculation."""

    target_product_smiles: str
    target_mass_g: float
    target_moles: float
    reagents: list[ReagentCalcResult]
    warnings: list[str] = Field(default_factory=list)

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class MoleculeValidationResult(BaseModel):
    """Result of validating a user-supplied molecule identifier."""

    is_valid: bool
    input_text: str
    input_type: Literal["smiles", "name"]
    canonical_smiles: str | None = None
    iupac_name: str | None = None
    molecular_formula: str | None = None
    molecular_weight: float | None = None
    pubchem_cid: int | None = None
    error: str | None = None

"""RDKit wrappers for molecular weight, formula, SMILES validation, and reaction parsing."""

from __future__ import annotations

from collections import Counter

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


def validate_smiles(smiles: str) -> bool:
    """Return True if RDKit can parse the SMILES string."""
    if not smiles or not smiles.strip():
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def get_molecular_weight(smiles: str) -> float:
    """Exact molecular weight (g/mol) from SMILES.

    Raises ValueError when the SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Descriptors.ExactMolWt(mol)


def get_average_molecular_weight(smiles: str) -> float:
    """Average molecular weight (uses natural isotope distribution).

    This is what you would use to weigh a reagent on a balance.
    Raises ValueError when the SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Descriptors.MolWt(mol)


def get_molecular_formula(smiles: str) -> str:
    """Molecular formula string (e.g. 'C9H8O4') from SMILES.

    Raises ValueError when the SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return rdMolDescriptors.CalcMolFormula(mol)


def canonicalize(smiles: str) -> str:
    """Return canonical SMILES.  Returns the input unchanged on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol)


def parse_reaction_smiles(
    reaction_smiles: str,
) -> tuple[list[str], list[str]]:
    """Split 'A.B>>C.D' into ([reactant_smiles, ...], [product_smiles, ...]).

    Supports both '>>' and '>' separators.  Agent-separated components ('>A>')
    are ignored — only the left and right sides are returned.

    Raises ValueError for malformed strings.
    """
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
    """Given a list of (possibly repeated) SMILES, return canonical SMILES -> count.

    In SMILES notation, stoichiometric coefficients are expressed by repeating
    the component: 2 equivalents of A appear as ['A', 'A'].
    """
    canonical = [canonicalize(s) for s in smiles_list]
    return dict(Counter(canonical))

"""PubChem PUG REST client for fetching compound physical properties (density, melting/boiling points)."""

from __future__ import annotations

import logging
import re
import time
from functools import lru_cache
from typing import Any
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_PUG_VIEW_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"

_REQUEST_TIMEOUT = 15
_RETRY_DELAY = 0.3


def _get_json(url: str, *, retries: int = 2) -> dict[str, Any] | None:
    """GET *url* and return parsed JSON, or None on any failure."""
    for attempt in range(1, retries + 2):
        try:
            resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                return None
            if resp.status_code == 503 and attempt <= retries:
                time.sleep(_RETRY_DELAY * attempt)
                continue
            logger.warning("PubChem %s returned %s", url, resp.status_code)
            return None
        except requests.RequestException as exc:
            logger.warning("PubChem request failed (%s): %s", url, exc)
            if attempt <= retries:
                time.sleep(_RETRY_DELAY * attempt)
    return None


@lru_cache(maxsize=512)
def get_cid_by_smiles(smiles: str) -> int | None:
    """Resolve a SMILES string to a PubChem CID."""
    encoded = quote(smiles, safe="")
    url = f"{_BASE_URL}/compound/smiles/{encoded}/cids/JSON"
    data = _get_json(url)
    if data is None:
        return None
    try:
        return data["IdentifierList"]["CID"][0]
    except (KeyError, IndexError, TypeError):
        return None


@lru_cache(maxsize=512)
def get_cid_by_name(name: str) -> int | None:
    """Resolve a compound name (IUPAC, trivial, CAS, etc.) to a PubChem CID."""
    encoded = quote(name, safe="")
    url = f"{_BASE_URL}/compound/name/{encoded}/cids/JSON"
    data = _get_json(url)
    if data is None:
        return None
    try:
        return data["IdentifierList"]["CID"][0]
    except (KeyError, IndexError, TypeError):
        return None


@lru_cache(maxsize=512)
def get_smiles_by_cid(cid: int) -> str | None:
    """Fetch canonical SMILES for a PubChem CID."""
    url = f"{_BASE_URL}/compound/cid/{cid}/property/CanonicalSMILES/JSON"
    data = _get_json(url)
    if data is None:
        return None
    try:
        props = data["PropertyTable"]["Properties"][0]
        return (
            props.get("CanonicalSMILES")
            or props.get("ConnectivitySMILES")
            or props.get("SMILES")
        )
    except (KeyError, IndexError, TypeError):
        return None


@lru_cache(maxsize=512)
def get_compound_properties(smiles: str) -> dict[str, Any]:
    """Fetch a subset of computed properties from PubChem for *smiles*.

    Returns a dict that may contain: MolecularWeight, IUPACName, etc.
    Always returns a dict (empty on failure).
    """
    encoded = quote(smiles, safe="")
    url = (
        f"{_BASE_URL}/compound/smiles/{encoded}/property/"
        "MolecularWeight,MolecularFormula,IUPACName,IsomericSMILES/JSON"
    )
    data = _get_json(url)
    if data is None:
        return {}
    try:
        return data["PropertyTable"]["Properties"][0]
    except (KeyError, IndexError, TypeError):
        return {}


def _extract_numeric(text: str) -> float | None:
    """Pull the first signed decimal / integer number out of *text*."""
    m = re.search(r"(-?\d+\.?\d*)", text)
    if m:
        return float(m.group(1))
    return None


def _fahrenheit_to_celsius(f: float) -> float:
    return (f - 32) * 5 / 9


def _walk_sections(
    sections: list[dict[str, Any]], heading: str
) -> dict[str, Any] | None:
    """Recursively find a section with the given TOCHeading."""
    for sec in sections:
        if sec.get("TOCHeading", "").lower() == heading.lower():
            return sec
        children = sec.get("Section", [])
        if children:
            result = _walk_sections(children, heading)
            if result is not None:
                return result
    return None


def _extract_temperature_celsius(section: dict[str, Any]) -> float | None:
    """Extract a temperature in deg C from a PUG View section.

    PubChem stores multiple values from different sources.  Strategy:
      1. Prefer entries explicitly marked as deg C.
      2. Fall back to converting deg F.
      3. Ignore entries with no recognisable unit.
    """
    celsius_values: list[float] = []
    fahrenheit_values: list[float] = []

    for info in section.get("Information", []):
        val = info.get("Value", {})
        for swm in val.get("StringWithMarkup", []):
            text: str = swm.get("String", "")
            if not text:
                continue
            num = _extract_numeric(text)
            if num is None:
                continue
            text_lower = text.lower()
            if "\u00b0c" in text_lower or "deg c" in text_lower or "\u00b0 c" in text_lower:
                celsius_values.append(num)
            elif "\u00b0f" in text_lower or "deg f" in text_lower or "\u00b0 f" in text_lower:
                fahrenheit_values.append(num)

        nums = val.get("Number")
        unit = val.get("Unit", "")
        if nums is not None:
            ns = nums if isinstance(nums, list) else [nums]
            if "c" in unit.lower():
                celsius_values.append(float(ns[0]))
            elif "f" in unit.lower():
                fahrenheit_values.append(float(ns[0]))

    if celsius_values:
        return celsius_values[0]
    if fahrenheit_values:
        return round(_fahrenheit_to_celsius(fahrenheit_values[0]), 2)
    return None


def _extract_string_value(section: dict[str, Any]) -> str | None:
    """Extract the first StringWithMarkup value from a section's Information list."""
    for info in section.get("Information", []):
        val = info.get("Value", {})
        for swm in val.get("StringWithMarkup", []):
            text = swm.get("String", "")
            if text:
                return text
        num = val.get("Number")
        if num is not None:
            nums = num if isinstance(num, list) else [num]
            unit = val.get("Unit", "")
            return f"{nums[0]} {unit}".strip()
    return None


@lru_cache(maxsize=512)
def get_density(smiles: str) -> float | None:
    """Fetch density (g/mL) for the compound from PubChem PUG View.

    Returns None if density is unavailable or the compound is not found.
    """
    cid = get_cid_by_smiles(smiles)
    if cid is None:
        return None

    url = f"{_PUG_VIEW_URL}/data/compound/{cid}/JSON?heading=Density"
    data = _get_json(url)
    if data is None:
        return None

    try:
        sections = data["Record"]["Section"]
    except (KeyError, TypeError):
        return None

    density_sec = _walk_sections(sections, "Density")
    if density_sec is None:
        return None

    text = _extract_string_value(density_sec)
    if text is None:
        return None

    return _extract_numeric(text)


@lru_cache(maxsize=512)
def get_boiling_point(smiles: str) -> float | None:
    """Fetch boiling point (deg C) from PubChem. Returns None if unavailable."""
    cid = get_cid_by_smiles(smiles)
    if cid is None:
        return None

    url = f"{_PUG_VIEW_URL}/data/compound/{cid}/JSON?heading=Boiling+Point"
    data = _get_json(url)
    if data is None:
        return None

    try:
        sections = data["Record"]["Section"]
    except (KeyError, TypeError):
        return None

    sec = _walk_sections(sections, "Boiling Point")
    if sec is None:
        return None

    return _extract_temperature_celsius(sec)


@lru_cache(maxsize=512)
def get_melting_point(smiles: str) -> float | None:
    """Fetch melting point (deg C) from PubChem. Returns None if unavailable."""
    cid = get_cid_by_smiles(smiles)
    if cid is None:
        return None

    url = f"{_PUG_VIEW_URL}/data/compound/{cid}/JSON?heading=Melting+Point"
    data = _get_json(url)
    if data is None:
        return None

    try:
        sections = data["Record"]["Section"]
    except (KeyError, TypeError):
        return None

    sec = _walk_sections(sections, "Melting Point")
    if sec is None:
        return None

    return _extract_temperature_celsius(sec)


def estimate_physical_state(smiles: str) -> str:
    """Heuristic: solid / liquid / gas at ~25 C based on melting & boiling points.

    Falls back to checking density availability when temperature data is missing.
    """
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

    density = get_density(smiles)
    if density is not None and density > 0:
        return "liquid"

    return "unknown"


def get_iupac_name(smiles: str) -> str:
    """Return IUPAC name from PubChem, or empty string on failure.

    Tries SMILES-based query first, falls back to CID-based lookup when the
    PUG REST endpoint rejects the percent-encoded SMILES.
    """
    props = get_compound_properties(smiles)
    name = props.get("IUPACName", "")
    if name:
        return name

    cid = get_cid_by_smiles(smiles)
    if cid is None:
        return ""
    url = (
        f"{_BASE_URL}/compound/cid/{cid}/property/"
        "IUPACName/JSON"
    )
    data = _get_json(url)
    if data is None:
        return ""
    try:
        return data["PropertyTable"]["Properties"][0].get("IUPACName", "")
    except (KeyError, IndexError, TypeError):
        return ""

"""Quick validation of user-supplied molecule identifiers (SMILES or name)."""

from __future__ import annotations

import logging
import re
from typing import Literal

from src.models.validation import MoleculeValidationResult
from src.services.molecular import (
    canonicalize,
    get_average_molecular_weight,
    get_molecular_formula,
    validate_smiles,
)
from src.services.pubchem import (
    get_cid_by_name,
    get_cid_by_smiles,
    get_iupac_name,
    get_smiles_by_cid,
)

logger = logging.getLogger(__name__)

_SMILES_PATTERN = re.compile(
    r"^[A-Za-z0-9"
    r"@+\-\[\]\(\)\\\/=#$%.:~]+"
    r"$"
)


def _detect_input_type(user_input: str) -> Literal["smiles", "name"]:
    """Heuristic to decide whether *user_input* looks like SMILES or a name.

    SMILES strings are single-token (no spaces) and typically contain
    characters like ``=``, ``#``, ``(``, ``)``, ``[``, ``]``, ``@``, ``/``,
    ``\\``.  Purely alphabetic tokens that look like English words (e.g.
    "benzene") are treated as names.  Ambiguous cases (e.g. "CCO") are
    resolved by attempting RDKit parsing.
    """
    stripped = user_input.strip()

    if " " in stripped:
        return "name"

    if not _SMILES_PATTERN.match(stripped):
        return "name"

    smiles_structural = set("=()[]@/\\#%+")
    if smiles_structural & set(stripped):
        return "smiles"

    if any(ch.isdigit() for ch in stripped):
        return "smiles"

    if stripped.isalpha():
        if stripped.lower() == stripped:
            return "name"
        if validate_smiles(stripped):
            return "smiles"
        return "name"

    return "smiles"


def _fail(
    user_input: str, input_type: Literal["smiles", "name"], error: str
) -> MoleculeValidationResult:
    return MoleculeValidationResult(
        is_valid=False,
        input_text=user_input,
        input_type=input_type,
        error=error,
    )


def _enrich(
    user_input: str,
    input_type: Literal["smiles", "name"],
    smiles: str,
    cid: int | None,
) -> MoleculeValidationResult:
    """Build a successful result enriched with RDKit / PubChem data."""
    canonical = canonicalize(smiles)

    iupac: str | None = None
    if cid is not None:
        iupac = get_iupac_name(canonical) or None

    try:
        formula = get_molecular_formula(canonical)
    except ValueError:
        formula = None

    try:
        mw = round(get_average_molecular_weight(canonical), 4)
    except ValueError:
        mw = None

    return MoleculeValidationResult(
        is_valid=True,
        input_text=user_input,
        input_type=input_type,
        canonical_smiles=canonical,
        iupac_name=iupac,
        molecular_formula=formula,
        molecular_weight=mw,
        pubchem_cid=cid,
    )


def validate_molecule_input(user_input: str) -> MoleculeValidationResult:
    """Validate a molecule specified by SMILES or name and return structured info.

    Detects the input type automatically.  For SMILES, validates with RDKit
    first, then optionally resolves in PubChem.  For names/CAS numbers,
    resolves via PubChem and retrieves the canonical SMILES.
    """
    stripped = user_input.strip()
    if not stripped:
        return _fail(user_input, "name", "Empty input")

    input_type = _detect_input_type(stripped)

    if input_type == "smiles":
        return _validate_smiles_input(stripped)
    return _validate_name_input(stripped)


def _validate_smiles_input(smiles: str) -> MoleculeValidationResult:
    if not validate_smiles(smiles):
        return _fail(smiles, "smiles", "RDKit could not parse SMILES")

    cid = get_cid_by_smiles(smiles)
    if cid is None:
        logger.info("SMILES '%s' is valid RDKit but not found in PubChem", smiles)

    return _enrich(smiles, "smiles", smiles, cid)


def _validate_name_input(name: str) -> MoleculeValidationResult:
    cid = get_cid_by_name(name)
    if cid is None:
        return _fail(name, "name", f"Compound '{name}' not found in PubChem")

    smiles = get_smiles_by_cid(cid)
    if smiles is None:
        return _fail(
            name, "name", f"PubChem CID {cid} found but SMILES unavailable"
        )

    if not validate_smiles(smiles):
        return _fail(
            name, "name", f"PubChem returned unparseable SMILES: {smiles}"
        )

    return _enrich(name, "name", smiles, cid)

"""Calculation tools: stoichiometry_calc and equivalents_calc.

These are standalone Python functions.  They will be wrapped as LangChain
tools later.
"""

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
from ..services.molecular import (
    canonicalize,
    get_average_molecular_weight,
    parse_reaction_smiles,
    validate_smiles,
)
from ..services.pubchem import (
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
    """Common helper: given SMILES, equivalents and moles, compute mass, volume, etc."""
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


# ---------------------------------------------------------------------------
# stoichiometry_calc
# ---------------------------------------------------------------------------

def stoichiometry_calc(request: StoichiometryRequest) -> CalculationResult:
    """Calculate masses / volumes of every reagent for a given reaction and
    desired product mass.

    Stoichiometric coefficients are inferred from repeated SMILES components
    (e.g. ``A.A.B>>C`` means 2 eq of A and 1 eq of B).
    """
    warnings: list[str] = []

    reactant_list, product_list = parse_reaction_smiles(request.reaction_smiles)

    # --- determine target product ---
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

    # --- stoichiometric coefficients from repeated SMILES ---
    canonical_reactants = [canonicalize(s) for s in reactant_list]
    coeff_map: dict[str, int] = dict(Counter(canonical_reactants))

    # product coefficient (to normalise)
    canonical_products_all = [canonicalize(s) for s in product_list]
    product_coeff = Counter(canonical_products_all).get(target_smiles, 1)

    reagent_results: list[ReagentCalcResult] = []
    for smiles, coeff in coeff_map.items():
        equiv = coeff / product_coeff
        moles = target_moles * equiv
        result = _build_reagent_result(
            smiles,
            equivalents=equiv,
            moles=moles,
            warnings=warnings,
        )
        reagent_results.append(result)

    return CalculationResult(
        target_product_smiles=target_smiles,
        target_mass_g=round(request.target_mass_g, 4),
        target_moles=round(target_moles, 6),
        reagents=reagent_results,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# equivalents_calc
# ---------------------------------------------------------------------------

def equivalents_calc(request: EquivalentsRequest) -> CalculationResult:
    """Calculate masses / volumes for each reagent based on equivalents and a
    reference amount.

    ``amount_type`` controls how ``reference_amount`` is interpreted:

    * ``product_mass``  – grams of target product
    * ``reagent_mass``  – grams of the reference reagent (first in the list, or
      the one whose SMILES matches ``reference_smiles``)
    * ``reagent_moles`` – moles directly
    """
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
            r_smiles,
            equivalents=reagent.equivalents,
            moles=moles,
            name=reagent.name,
            warnings=warnings,
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



r"""
guard_node.py — GuardAgent как нода LangGraph.

════════════════════════════════════════════════════════════════════
БЫСТРЫЙ СТАРТ
════════════════════════════════════════════════════════════════════

Нода принимает из стейта:
  • smiles: str               — каноническая SMILES-строка вещества (обязательно)
  • reaction_description: str — текстовое описание реакции (опционально, "" по умолчанию)

Нода добавляет / перезаписывает в стейте:
  • guard_result: GuardResult — итоговый отчёт проверки (см. схему ниже)

Минимальный пример подключения к графу:

    from langgraph.graph import StateGraph
    from guard_node import guard_node, GuardStateProtocol

    graph = StateGraph(YourState)           # YourState должен включать поля из GuardStateProtocol
    graph.add_node("guard", guard_node)
    graph.add_edge("some_upstream_node", "guard")

════════════════════════════════════════════════════════════════════
СХЕМА ВОЗВРАЩАЕМОГО guard_result
════════════════════════════════════════════════════════════════════

GuardResult (TypedDict):
  overall_status:    "SAFE" | "WARNING" | "CRITICAL_STOP"
  molecule_check:    dict  — см. MoleculeCheckResult (models.py)
  reaction_check:    dict  — см. ReactionCheckResult (models.py)
  safety_data:       dict  — см. SafetyData (models.py): GHS, H/P-фразы
  ppe_recommendations: list[str] — рекомендованные СИЗ

Логика overall_status:
  CRITICAL_STOP — молекула или реакция помечена "banned" / "prohibited"
  WARNING       — молекула или реакция "restricted"
  SAFE          — оба чека "clear" / "allowed"

════════════════════════════════════════════════════════════════════
ОШИБКИ, НАЙДЕННЫЕ В ИСХОДНОМ КОДЕ (и исправленные здесь)
════════════════════════════════════════════════════════════════════

tools.py / safety_lookup:
  БЫЛО:  re.match(r"GHS\d{2}", p)
  СТАЛО: re.search(r"GHS\d{2}", p)
  ПРИЧИНА: PubChem возвращает строки вида
    "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS02.svg"
    re.match проверяет только начало строки → все пиктограммы терялись.
  ТАКЖЕ: добавлен отдельный парсинг pictogram-секции вместо смешивания
    с H/P-фразами, чтобы избежать ложных срабатываний _parse_codes.

rag.py / _get_or_create_collection:
  БЫЛО:  existing._collection.count()
  СТАЛО: len(existing.get()["ids"])
  ПРИЧИНА: _collection — приватный атрибут обёртки LangChain Chroma;
    в chromadb >= 0.4 его имя и интерфейс менялись.
    Публичный метод .get() надёжнее.

models.py / MoleculeCheckResult:
  БЫЛО:  status: Literal["clear", "restricted", "banned"]
  СТАЛО: добавлен валидатор — smiles не может быть пустой строкой.
  ПРИЧИНА: пустой SMILES пропускался в инструменты и вызывал
    молчаливые ошибки RDKit.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, TypedDict

from tools import banlist_check, ppe_recommender, reaction_banlist_check, safety_lookup

logger = logging.getLogger(__name__)


# ─── Типизация ────────────────────────────────────────────────────────────────

class GuardResult(TypedDict):
    """Итоговый результат, который нода записывает в стейт."""
    overall_status: Literal["SAFE", "WARNING", "CRITICAL_STOP"]
    molecule_check: dict
    reaction_check: dict
    safety_data: dict
    ppe_recommendations: list[str]


class GuardStateProtocol(TypedDict, total=False):
    """
    Минимальный контракт стейта, с которым работает нода.

    Включите эти поля в свой StateGraph-стейт.
    total=False означает, что все поля опциональны при объявлении,
    но smiles обязан присутствовать в рантайме.

    Пример:
        class MyState(GuardStateProtocol):
            messages: list[BaseMessage]
            some_other_field: str
    """
    smiles: str               # каноническая SMILES — обязательное поле рантайма
    reaction_description: str # описание реакции — опционально
    guard_result: GuardResult # сюда нода пишет результат


# ─── Вспомогательная функция ──────────────────────────────────────────────────

def _determine_overall_status(
    mol_status: str,
    rxn_status: str,
) -> Literal["SAFE", "WARNING", "CRITICAL_STOP"]:
    """
    Агрегирует статусы молекулярной и реакционной проверок.

    Приоритет: CRITICAL_STOP > WARNING > SAFE
    """
    critical = {"banned", "prohibited"}
    warning = {"restricted"}

    if mol_status in critical or rxn_status in critical:
        return "CRITICAL_STOP"
    if mol_status in warning or rxn_status in warning:
        return "WARNING"
    return "SAFE"


# ─── Нода ─────────────────────────────────────────────────────────────────────

def guard_node(state: GuardStateProtocol) -> dict[str, Any]:
    """
    LangGraph-нода безопасности.

    Получает из стейта каноническую SMILES-строку и опциональное
    описание реакции; возвращает частичное обновление стейта с
    ключом ``guard_result``.

    Шаги:
      1. banlist_check         — точное совпадение + SMARTS-подструктуры
      2. reaction_banlist_check — семантический поиск запрещённых реакций
      3. safety_lookup         — GHS-данные из PubChem (пиктограммы, H/P-фразы)
      4. ppe_recommender       — рекомендации СИЗ на основе H-фраз
      5. Агрегация overall_status

    Args:
        state: стейт графа, содержащий как минимум ``smiles``.

    Returns:
        Словарь ``{"guard_result": GuardResult}`` для частичного
        обновления стейта LangGraph.

    Raises:
        ValueError: если ``smiles`` отсутствует или пуст в стейте.
    """
    smiles: str = state.get("smiles", "").strip()
    if not smiles:
        raise ValueError(
            "guard_node: поле 'smiles' отсутствует или пусто в стейте."
        )

    reaction_description: str = state.get("reaction_description", "")

    logger.info("[guard_node] Старт проверки: smiles=%r", smiles)

    # ── 1. Молекулярный бан-лист ───────────────────────────────────────────────
    mol_check: dict = banlist_check.invoke({"smiles": smiles})
    logger.info("[guard_node] banlist_check → %s", mol_check.get("status"))

    # ── 2. Реакционный бан-лист ───────────────────────────────────────────────
    rxn_check: dict = reaction_banlist_check.invoke(
        {"reaction_description": reaction_description}
    )
    logger.info("[guard_node] reaction_banlist_check → %s", rxn_check.get("status"))

    # ── 3. GHS / PubChem ──────────────────────────────────────────────────────
    safety: dict = safety_lookup.invoke({"smiles": smiles})
    logger.info(
        "[guard_node] safety_lookup → %d H-фраз, %d P-фраз, пиктограммы: %s",
        len(safety.get("h_phrases", [])),
        len(safety.get("p_phrases", [])),
        safety.get("ghs_pictograms", []),
    )

    # ── 4. СИЗ ────────────────────────────────────────────────────────────────
    h_phrases_str: str = ",".join(safety.get("h_phrases", []))
    ppe: list[str] = ppe_recommender.invoke(
        {"substances": smiles, "h_phrases": h_phrases_str}
    )
    logger.info("[guard_node] ppe_recommender → %s", ppe)

    # ── 5. Агрегация ──────────────────────────────────────────────────────────
    overall = _determine_overall_status(
        mol_status=mol_check.get("status", "clear"),
        rxn_status=rxn_check.get("status", "allowed"),
    )
    logger.info("[guard_node] overall_status=%s", overall)

    guard_result: GuardResult = {
        "overall_status": overall,
        "molecule_check": mol_check,
        "reaction_check": rxn_check,
        "safety_data": safety,
        "ppe_recommendations": ppe,
    }

    # Возвращаем только изменение стейта — LangGraph сам сделает merge
    return {"guard_result": guard_result}



import json
import requests
from typing import TypedDict, Optional, List, Dict, Any, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_gigachat import GigaChat
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import json
from rdkit import Chem
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate

# -------------------------------------------------------------------
# 1. Настройка модели (используем LangChain-обёртку GigaChat)
# -------------------------------------------------------------------
SBER_AUTH = "MDE5Y2U4ZDAtMzEwMC03NjQyLWJkN2UtNTliOTc5Yzc1YjVkOmRkNzgyYmQwLTQ4MzYtNGY4OC1hZDM0LWI3M2JmODJmYTA4YQ=="
SBER_ID = "019ce8d0-3100-7642-bd7e-59b979c75b5d"
SCOPE='GIGACHAT_API_PERS'

llm = GigaChat(
    credentials=SBER_AUTH,
    verify_ssl_certs=False,
    profanity_check=False,
    scope=SCOPE
)

# -------------------------------------------------------------------
# 2. Определение состояния
# -------------------------------------------------------------------
class MoleculeInfo(TypedDict):
    name: str
    synonyms: List[str]
    smiles: str
    molecular_formula: str
    molecular_weight: float
    properties: Dict[str, Any]
    description: str
    ghs_classification: List[str]
    pubchem_cid: int

class State(TypedDict):
    query: str
    messages: Annotated[List, add_messages]
    pubchem_result: Optional[dict]
    rdkit_result: Optional[dict]
    target_molecule: MoleculeInfo
    need_clarify: Optional[bool]
    clarification_cnt: int
    final_answer: str

# -------------------------------------------------------------------
# 3. Инструменты
# -------------------------------------------------------------------
def pubchem_lookup(name: str) -> dict:
    """Получить данные молекулы из PubChem: свойства + синонимы."""
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
    
    # 1. Запрашиваем свойства
    props_url = f"{base_url}/{name}/property/MolecularFormula,MolecularWeight,IUPACName,IsomericSMILES,CanonicalSMILES,XLogP,TPSA/JSON"
    
    try:
        response = requests.get(props_url)
        if response.status_code != 200:
            return {"error": f"Молекула не найдена: {name}"}
        
        props_data = response.json()["PropertyTable"]["Properties"][0]
        
        # 2. Запрашиваем синонимы и CID (отдельный запрос, так как они в другом формате)
        cid = props_data.get("CID")
        synonyms = []
        if cid:
            syn_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
            syn_resp = requests.get(syn_url)
            if syn_resp.status_code == 200:
                # Берем первые 5 синонимов для краткости
                all_syns = syn_resp.json().get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])
                synonyms = all_syns[:5] # Топ-5 синонимов

        smiles = props_data.get("IsomericSMILES") or props_data.get("CanonicalSMILES")

        return {
            "cid": cid,
            "formula": props_data.get("MolecularFormula"),
            "weight": props_data.get("MolecularWeight"),
            "iupac": props_data.get("IUPACName"),
            "smiles": smiles,
            "logp": props_data.get("XLogP"),
            "tpsa": props_data.get("TPSA"),
            "synonyms": synonyms
        }
    except Exception as e:
        return {"error": f"Ошибка запроса: {str(e)}"}

def rdkit_properties(smiles: str) -> dict:
    """Рассчитывает свойства молекулы (MW, logP, TPSA, rotatable bonds) по SMILES с помощью RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Некорректная SMILES: {smiles}"}

    # Основные физико-химические свойства
    mw = Descriptors.MolWt(mol)                     # молекулярная масса
    logp = Descriptors.MolLogP(mol)                 # коэффициент распределения (logP)
    tpsa = Descriptors.TPSA(mol)                    # полярная поверхность
    rot_bonds = Lipinski.NumRotatableBonds(mol)     # количество ротируемых связей

    # Дополнительные (по желанию)
    hba = Lipinski.NumHAcceptors(mol)               # акцепторы водородных связей
    hbd = Lipinski.NumHDonors(mol)                  # доноры водородных связей
    heavy_atoms = Descriptors.HeavyAtomCount(mol)   # количество тяжёлых атомов
    ring_count = Lipinski.RingCount(mol)            # количество колец

    return {
        "molecular_weight": mw,
        "logp": logp,
        "tpsa": tpsa,
        "rotatable_bonds": rot_bonds,
        "h_bond_acceptors": hba,
        "h_bond_donors": hbd,
        "heavy_atoms": heavy_atoms,
        "ring_count": ring_count,
    }


# -------------------------------------------------------------------
# 4. Узлы графа
# -------------------------------------------------------------------
from langchain_core.messages import HumanMessage

def setup_node(state: State) -> dict:
    """Инициализация состояния."""
    molecule_info = state.get("target_molecule", {})
    
    updated_molecule = {
        "name": molecule_info.get("name", ""),
        "synonyms": molecule_info.get("synonyms", []),
        "smiles": molecule_info.get("smiles", ""),
        "molecular_formula": molecule_info.get("molecular_formula", ""),
        "molecular_weight": molecule_info.get("molecular_weight", 0.0),
        "properties": molecule_info.get("properties", {}),
        "description": molecule_info.get("description", ""),
        "ghs_classification": molecule_info.get("ghs_classification", []),
        "pubchem_cid": molecule_info.get("pubchem_cid", 0),
    }
    
    return {
        "target_molecule": updated_molecule,
        "messages": [HumanMessage(content=state["query"])],
        "need_clarify": False,
        "clarification_cnt": 0,
        "final_answer": "",
    }

def MoleculeInfoAgent(state: State) -> dict:
    """
    Агент для сбора информации о молекуле.
    Автоматически определяет, является ли query названием или SMILES.
    """
    query = state["query"]
    
    pubchem_result = {}
    rdkit_result = {}
    smiles = None
    molecule_name = None
    cid = None

    # ====================================================================
    # 1. Определение типа входа: SMILES или Название
    # ====================================================================
    if Chem.MolFromSmiles(query):
        # --- СЛУЧАЙ 1: Входные данные - это SMILES ---
        smiles = query
        rdkit_result = rdkit_properties(smiles)
        pubchem_data = pubchem_lookup(smiles)
        
        if "error" not in pubchem_data:
            pubchem_result = pubchem_data
            molecule_name = pubchem_result.get("iupac")
            cid = pubchem_result.get("cid")
        else:
            name_prompt = f"Определи название молекулы по её SMILES нотации. Напиши только название на английском. SMILES: {smiles}"
            resp = llm.invoke([HumanMessage(content=name_prompt)])
            molecule_name = resp.content.strip()
    else:
        # --- СЛУЧАЙ 2: Входные данные - это Название ---
        response = llm.invoke([HumanMessage(content=f"Извлеки название молекулы: {query}")])
        molecule_name = response.content.strip() or query
        
        pubchem_data = pubchem_lookup(molecule_name)
        if "error" in pubchem_data:
            pubchem_data = pubchem_lookup(query)
            
        pubchem_result = pubchem_data
        smiles = pubchem_result.get("smiles")
        cid = pubchem_result.get("cid")
        
        if not smiles:
            response = llm.invoke([HumanMessage(content=f"Сгенерируй или извлеки SMILES для молекулы: {molecule_name}. Ответ должен содержать только одну строку SMILES.")])
            candidate = response.content.strip()
            if Chem.MolFromSmiles(candidate):
                smiles = candidate
            else:
                smiles = None

        if smiles:
            rdkit_result = rdkit_properties(smiles)

    # ====================================================================
    # 2. Формирование ссылки на структуру
    # ====================================================================
    structure_url = ""
    if cid:
        structure_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG"
    elif smiles:
        structure_url = "Требуется генерация из SMILES"

    # ====================================================================
    # 3. Объединение результатов (LLM Synthesis)
    # ====================================================================
    prompt = PromptTemplate.from_template("""
    Ты — химик-эксперт. Заполни карточку молекулы.
    
    Входные данные:
    1. Запрос пользователя: {query}
    2. Данные PubChem: {pubchem_data}
    3. Данные RDKit: {rdkit_data}
    
    Твоя задача — вернуть JSON со следующими полями. Если данные отсутствуют, используй свои знания.
    
    Поля JSON:
    - "name": IUPAC название (англ).
    - "synonyms": список основных синонимов (список строк).
    - "smiles": SMILES строка.
    - "molecular_formula": Брутто-формула.
    - "molecular_weight": Молярная масса (число).
    - "properties": Словарь, содержащий ключи: "melting_point", "boiling_point", "solubility", "density", "logP", "physical_state".
    - "ghs_classification": Список классов опасности GHS (список строк).
    - "spectral_notes": Краткая заметка о спектральных данных (ИК, ЯМР).
    - "description": Краткое описание на русском.
    - "pubchem_cid": CID число (если нет, то 0).
    
    ВАЖНО: Данные из RDKit (вес, logP) приоритетны.
    """)

    prompt_value = prompt.format(
        query=query,
        pubchem_data=json.dumps(pubchem_result, ensure_ascii=False),
        rdkit_data=json.dumps(rdkit_result, ensure_ascii=False)
    )

    llm_response = llm.invoke([HumanMessage(content=prompt_value)])
    text = llm_response.content

    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        parsed = json.loads(text)
    except Exception as e:
        print(f"Ошибка парсинга JSON: {e}")
        parsed = {}

    # ====================================================================
    # 4. Безопасное извлечение данных
    # ====================================================================
    
    # Вспомогательные функции для безопасного преобразования типов
    def safe_int(val, default=0):
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    def safe_float(val, default=0.0):
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    current = state.get("target_molecule", {})
    props = parsed.get("properties", {})

    # Извлечение CID: сначала из парсинга, если там пусто/ошибка -> берем из переменной cid, иначе 0
    parsed_cid = parsed.get("pubchem_cid")
    final_cid = safe_int(parsed_cid if parsed_cid not in ["", None] else cid, 0)

    # Извлечение веса: приоритет RDKit, затем LLM
    parsed_weight = parsed.get("molecular_weight")
    rdkit_weight = rdkit_result.get("molecular_weight")
    final_weight = safe_float(rdkit_weight if rdkit_weight else parsed_weight, 0.0)

    updated_molecule = {
        "name": parsed.get("name", molecule_name or "Unknown"),
        "synonyms": parsed.get("synonyms", pubchem_result.get("synonyms", [])),
        "smiles": parsed.get("smiles", smiles or ""),
        "molecular_formula": parsed.get("molecular_formula", ""),
        "molecular_weight": final_weight,
        "properties": {
            "melting_point": props.get("melting_point", "N/A"),
            "boiling_point": props.get("boiling_point", "N/A"),
            "solubility": props.get("solubility", "N/A"),
            "density": props.get("density", "N/A"),
            "logP": props.get("logP", rdkit_result.get("logp")),
            "physical_state": props.get("physical_state", "N/A"),
            "spectral_notes": parsed.get("spectral_notes", "N/A")
        },
        "description": parsed.get("description", ""),
        "ghs_classification": parsed.get("ghs_classification", []),
        "pubchem_cid": final_cid,
        "structure_url": structure_url
    }

    final_text = (
        f"Молекула: {updated_molecule['name']} (CID: {updated_molecule['pubchem_cid']})\n"
        f"SMILES: {updated_molecule['smiles']}\n"
        f"Формула: {updated_molecule['molecular_formula']}\n"
        f"Вес: {updated_molecule['molecular_weight']}\n"
        f"Т пл.: {updated_molecule['properties']['melting_point']} | Т кип.: {updated_molecule['properties']['boiling_point']}\n"
        f"Растворимость: {updated_molecule['properties']['solubility']}\n"
        f"GHS: {', '.join(updated_molecule['ghs_classification'])}\n"
        f"Структура: {updated_molecule['structure_url']}\n"
        f"Описание: {updated_molecule['description']}"
    )

    return {
        "target_molecule": updated_molecule,
        "pubchem_result": pubchem_result,
        "rdkit_result": rdkit_result,
        "final_answer": final_text,
        "messages": [AIMessage(content=final_text)]
    }