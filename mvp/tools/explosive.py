"""Explosive hazard detection as a separate safety channel."""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from typing import Any

from rdkit import Chem

from ..config import DATA_DIR

logger = logging.getLogger(__name__)

_H_CODE_RE = re.compile(r"H20[0-5][A-Za-z]?")


@lru_cache(maxsize=1)
def _explosive_hazards() -> dict[str, Any]:
    path = DATA_DIR / "explosive_hazards.json"
    if not path.exists():
        logger.warning("Explosive hazard data file not found: %s", path)
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _canonical(smiles: str) -> tuple[str | None, Any | None]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    return Chem.MolToSmiles(mol, isomericSmiles=True), mol


@lru_cache(maxsize=1)
def _exact_compounds() -> list[dict[str, Any]]:
    compounds: list[dict[str, Any]] = []
    for entry in _explosive_hazards().get("exact_compounds", []):
        canon, _ = _canonical(entry.get("smiles", ""))
        if not canon:
            continue
        normalized = dict(entry)
        normalized["canonical_smiles"] = canon
        normalized["aliases_normalized"] = [str(alias).casefold() for alias in entry.get("aliases", [])]
        compounds.append(normalized)
    return compounds


@lru_cache(maxsize=1)
def _motifs() -> list[tuple[dict[str, Any], Any]]:
    motifs: list[tuple[dict[str, Any], Any]] = []
    for entry in _explosive_hazards().get("motifs", []):
        pattern = Chem.MolFromSmarts(entry.get("smarts", ""))
        if pattern is not None:
            motifs.append((entry, pattern))
    return motifs


def explosive_alias_check(query: str) -> dict[str, Any]:
    """Check common names/aliases without resolving a molecule first."""
    text = (query or "").casefold()
    for entry in _exact_compounds():
        if any(alias and alias in text for alias in entry.get("aliases_normalized", [])):
            return _result(
                status="blocked",
                name=entry.get("name"),
                danger_level=entry.get("danger_level", "high"),
                hazard_family=entry.get("hazard_family"),
                reason=f"Explosive hazard alias detected: {entry.get('name')}.",
                basis="alias",
            )
    return _clear_result(reason="No explosive hazard alias detected.")


def explosive_hazard_check(
    smiles: str,
    *,
    safety_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect explosive hazard independently from controlled-substance banlists."""
    canon, mol = _canonical(smiles)
    if mol is None or canon is None:
        return _clear_result(reason="Invalid SMILES; explosive hazard check skipped.")

    for entry in _exact_compounds():
        if entry.get("canonical_smiles") == canon:
            return _result(
                status="blocked",
                name=entry.get("name"),
                danger_level=entry.get("danger_level", "high"),
                hazard_family=entry.get("hazard_family"),
                reason=f"Exact explosive hazard match: {entry.get('name')}.",
                basis="exact",
                smiles=canon,
            )

    for entry, pattern in _motifs():
        matches = mol.GetSubstructMatches(pattern)
        min_matches = int(entry.get("min_matches") or 1)
        if len(matches) >= min_matches:
            danger_level = entry.get("danger_level", "medium")
            return _result(
                status="blocked" if danger_level in {"critical", "high"} else "warning",
                name=entry.get("name"),
                danger_level=danger_level,
                hazard_family=entry.get("hazard_family"),
                reason=entry.get("reason") or f"Explosive hazard motif detected: {entry.get('name')}.",
                basis="smarts",
                smiles=canon,
                matches=len(matches),
            )

    h_codes = _extract_h_codes(safety_data or {})
    explosive_codes = set(_explosive_hazards().get("_meta", {}).get("ghs_explosive_codes", []))
    matched_codes = sorted(code for code in h_codes if code in explosive_codes)
    if matched_codes:
        return _result(
            status="blocked",
            name="GHS explosive hazard",
            danger_level="high",
            hazard_family="ghs_explosive",
            reason=f"GHS explosive hazard statements detected: {', '.join(matched_codes)}.",
            basis="ghs",
            smiles=canon,
            h_codes=matched_codes,
        )

    return _clear_result(smiles=canon, reason="No explosive hazard detected.")


def _extract_h_codes(safety_data: dict[str, Any]) -> set[str]:
    codes: set[str] = set()
    for phrase in safety_data.get("h_phrases", []) or []:
        codes.update(code[:4] for code in _H_CODE_RE.findall(str(phrase)))
    for code in safety_data.get("ghs_codes", []) or []:
        codes.update(match[:4] for match in _H_CODE_RE.findall(str(code)))
    return codes


def _result(**kwargs: Any) -> dict[str, Any]:
    return {
        "hazard_type": "explosive",
        "status": kwargs.pop("status"),
        "name": kwargs.pop("name", None),
        "danger_level": kwargs.pop("danger_level", None),
        "hazard_family": kwargs.pop("hazard_family", None),
        "reason": kwargs.pop("reason", ""),
        **kwargs,
    }


def _clear_result(**kwargs: Any) -> dict[str, Any]:
    return _result(
        status="clear",
        name=None,
        danger_level=None,
        hazard_family=None,
        **kwargs,
    )
