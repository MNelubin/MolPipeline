"""Shared fixtures and pytest configuration for MVP pipeline tests."""

from __future__ import annotations

import pytest


# ── Markers ───────────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line("markers", "integration: tests that hit PubChem or ORD (need network/DB)")
    config.addinivalue_line("markers", "slow: tests that load the 192 MB retro model")
    config.addinivalue_line("markers", "llm: tests that call the LLM API")


# ── SMILES fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def aspirin_smiles():
    return "CC(=O)Oc1ccccc1C(=O)O"


@pytest.fixture
def aspirin_canonical():
    return "CC(=O)Oc1ccccc1C(=O)O"


@pytest.fixture
def caffeine_smiles():
    return "Cn1cnc2c1c(=O)n(c(=O)n2C)C"


@pytest.fixture
def ethanol_smiles():
    return "CCO"


@pytest.fixture
def fentanyl_smiles():
    # DEA Schedule II - should be blocked by banlist
    return "CCC(=O)N(c1ccccc1)C1CCN(CCc2ccccc2)CC1"


@pytest.fixture
def invalid_smiles():
    return "NOTASMILES!!!"


# ── State fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def aspirin_validated_state(aspirin_smiles):
    return {
        "query": "aspirin",
        "smiles": aspirin_smiles,
        "pubchem_cid": 2244,
        "validation": {
            "is_valid": True,
            "input_type": "name",
            "canonical_smiles": aspirin_smiles,
            "pubchem_cid": 2244,
            "error": None,
        },
    }


@pytest.fixture
def aspirin_guarded_state(aspirin_validated_state):
    return {
        **aspirin_validated_state,
        "guard_result": {
            "overall_status": "SAFE",
            "molecule_check": {"status": "clear"},
            "reaction_check": {"status": "allowed"},
            "safety_data": {"h_phrases": ["H302"], "ghs_pictograms": []},
            "ppe_recommendations": ["Перчатки", "Очки"],
        },
    }


@pytest.fixture
def fentanyl_validated_state(fentanyl_smiles):
    return {
        "query": "fentanyl",
        "smiles": fentanyl_smiles,
        "pubchem_cid": 3345,
        "validation": {
            "is_valid": True,
            "input_type": "name",
            "canonical_smiles": fentanyl_smiles,
            "pubchem_cid": 3345,
            "error": None,
        },
    }


@pytest.fixture
def mock_molecule_info():
    """Minimal molecule_info dict for retrosynthesis tests."""
    return {
        "name": "Аспирин (2-ацетилоксибензойная кислота)",
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "molecular_formula": "C9H8O4",
        "molecular_weight": 180.16,
    }


@pytest.fixture
def mock_ord_route():
    """Minimal ORD route dict."""
    return {
        "reaction_id": "ord-test-001",
        "reaction_smiles": "OC(=O)c1ccccc1O.CC(=O)OC(C)=O>>CC(=O)Oc1ccccc1C(=O)O",
        "reactants": "CC(=O)OC(C)=O.OC(=O)c1ccccc1O",
        "source": "ord",
        "score": 0.85,
        "plausibility": 0.90,
        "expected_yield": 0.80,
        "temperature": "25°C",
        "solvent": "CC(=O)O",
        "procedure_details": "Mix salicylic acid with acetic anhydride. Stir for 1 hour.",
    }
