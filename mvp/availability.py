"""Commercial availability checks for reagents and starting materials."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote_plus

from .tools import retro_tools as _retro


VENDOR_NAMES = {
    "EM": "eMolecules",
    "MC": "Mcule",
    "LN": "LabNetwork",
    "CB": "ChemBridge",
    "CS": "ChemSpace",
    "SA": "Sigma-Aldrich",
}


def _canonicalize_smiles(smiles: str) -> tuple[str | None, dict[str, Any]]:
    """Return canonical SMILES and lightweight descriptors."""
    smiles = smiles.strip()
    if not smiles:
        return None, {}

    if not _retro.HAS_RDKIT:
        return smiles, {"rdkit_available": False}

    mol = _retro.Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, {}

    canonical = _retro.Chem.MolToSmiles(mol, isomericSmiles=True)
    descriptors = {
        "rdkit_available": True,
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "rings": _retro.Descriptors.RingCount(mol),
        "chiral_centers": len(_retro.Chem.FindMolChiralCenters(mol)),
    }
    return canonical, descriptors


def _normalize_source(source: Any) -> str | None:
    if source is None:
        return None
    text = str(source).strip()
    if not text:
        return None
    parts = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    if not parts:
        parts = [text]
    return ", ".join(VENDOR_NAMES.get(part, part) for part in parts)


def _supplier_search_links(query: str, canonical_smiles: str) -> list[dict[str, str]]:
    term = quote_plus(query or canonical_smiles)
    smiles_term = quote_plus(canonical_smiles)
    return [
        {
            "label": "Sigma-Aldrich",
            "url": f"https://www.sigmaaldrich.com/US/en/search/{term}?focus=products&page=1&perpage=30&sort=relevance&term={term}&type=product",
        },
        {"label": "eMolecules", "url": f"https://www.emolecules.com/search/#?q={smiles_term}"},
        {"label": "ChemSpace", "url": f"https://chem-space.com/search?q={smiles_term}"},
        {"label": "MolPort", "url": f"https://www.molport.com/shop/find-chemicals?search={smiles_term}"},
    ]


def _pack_estimates(ppg: Any) -> list[dict[str, Any]]:
    try:
        price_per_gram = float(ppg)
    except (TypeError, ValueError):
        return []

    if price_per_gram <= 0:
        return []

    return [
        {"size_g": grams, "estimated_usd": round(price_per_gram * grams, 2)}
        for grams in (0.25, 1, 5, 10)
    ]


def check_reagent_availability(
    smiles: str,
    *,
    label: str | None = None,
    input_value: str | None = None,
    resolution: str | None = None,
) -> dict[str, Any]:
    """Classify reagent availability using local catalogs plus safe fallbacks."""
    canonical, descriptors = _canonicalize_smiles(smiles)
    raw_input = input_value or label or smiles

    if canonical is None:
        return {
            "input": raw_input,
            "label": label or raw_input,
            "smiles": smiles,
            "canonical_smiles": None,
            "resolution": resolution or "unresolved",
            "available": False,
            "availability_level": "invalid",
            "basis": "invalid_smiles",
            "confidence": "high",
            "ppg": None,
            "source": None,
            "source_label": None,
            "estimated_pack_prices": [],
            "supplier_search_links": [],
            "descriptors": {},
            "warnings": ["Could not parse this item as a valid molecule."],
        }

    catalog = _retro._buyables_lookup(canonical)
    cheap = canonical in _retro._get_cheap_canonical()
    heuristic = False if (catalog or cheap) else _retro._is_buyable(canonical)

    if catalog:
        level = "catalog"
        basis = "local_buyables_db"
        confidence = "high"
        available = True
        ppg = catalog.get("ppg")
        source = catalog.get("source")
    elif cheap:
        level = "common_lab_reagent"
        basis = "curated_common_reagents"
        confidence = "high"
        available = True
        ppg = None
        source = "common"
    elif heuristic:
        level = "heuristic_likely"
        basis = "small_simple_structure_heuristic"
        confidence = "low"
        available = True
        ppg = None
        source = None
    else:
        level = "not_found"
        basis = "not_found_in_local_catalog_or_heuristic"
        confidence = "medium"
        available = False
        ppg = None
        source = None

    warnings = [
        "Catalog data is local and can be stale; confirm live stock and price with the supplier.",
    ]
    if level == "heuristic_likely":
        warnings.append("This item was marked likely available by structure only, not by catalog match.")
    if level == "not_found":
        warnings.append("No local catalog match was found; supplier links are manual search hints.")

    return {
        "input": raw_input,
        "label": label or raw_input,
        "smiles": smiles,
        "canonical_smiles": canonical,
        "resolution": resolution or "smiles",
        "available": available,
        "availability_level": level,
        "basis": basis,
        "confidence": confidence,
        "ppg": float(ppg) if ppg is not None else None,
        "source": source,
        "source_label": _normalize_source(source),
        "estimated_pack_prices": _pack_estimates(ppg),
        "supplier_search_links": _supplier_search_links(label or raw_input, canonical),
        "descriptors": descriptors,
        "warnings": warnings,
    }


def summarize_availability(items: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(items)
    levels = {
        "catalog": 0,
        "common_lab_reagent": 0,
        "heuristic_likely": 0,
        "not_found": 0,
        "invalid": 0,
    }
    ppg_values: list[float] = []

    for item in items:
        level = item.get("availability_level") or "not_found"
        levels[level] = levels.get(level, 0) + 1
        if item.get("ppg") is not None:
            ppg_values.append(float(item["ppg"]))

    available_count = sum(1 for item in items if item.get("available"))
    summary = {
        "total": total,
        "available_count": available_count,
        "unavailable_count": total - available_count,
        "availability_ratio": round(available_count / total, 4) if total else 0,
        "catalog_count": levels.get("catalog", 0),
        "common_count": levels.get("common_lab_reagent", 0),
        "heuristic_count": levels.get("heuristic_likely", 0),
        "not_found_count": levels.get("not_found", 0),
        "invalid_count": levels.get("invalid", 0),
        "priced_count": len(ppg_values),
    }
    if ppg_values:
        summary.update({
            "min_ppg": round(min(ppg_values), 4),
            "max_ppg": round(max(ppg_values), 4),
            "avg_ppg": round(sum(ppg_values) / len(ppg_values), 4),
            "estimated_total_1g_usd": round(sum(ppg_values), 2),
        })
    return summary
