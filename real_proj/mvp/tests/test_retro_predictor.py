"""Tests for retro_predictor: ASKCOS template-relevance model.

Slow tests (model loading ~192 MB) are marked with @pytest.mark.slow.
Run with: pytest -m slow
Skip with: pytest -m "not slow"
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from ..retro_predictor import (
    _fingerprint,
    _canonicalize_reactants,
    _apply_template,
    _ensure_loaded,
    predict_retro,
)


# ═════════════════════════════════════════════════════════════════════════════
# _fingerprint
# ═════════════════════════════════════════════════════════════════════════════

class TestFingerprint:
    def test_valid_smiles_returns_array(self, aspirin_smiles):
        fp = _fingerprint(aspirin_smiles)
        assert fp is not None
        assert isinstance(fp, np.ndarray)

    def test_fingerprint_shape(self, aspirin_smiles):
        fp = _fingerprint(aspirin_smiles)
        assert fp.shape == (2048,)

    def test_fingerprint_dtype(self, aspirin_smiles):
        fp = _fingerprint(aspirin_smiles)
        assert fp.dtype == np.float32

    def test_fingerprint_binary_values(self, aspirin_smiles):
        fp = _fingerprint(aspirin_smiles)
        # Morgan bit fingerprint: values should be 0 or 1
        assert set(fp.tolist()).issubset({0.0, 1.0})

    def test_invalid_smiles_returns_none(self):
        fp = _fingerprint("NOTASMILES!!!")
        assert fp is None

    def test_empty_smiles_returns_none(self):
        fp = _fingerprint("")
        assert fp is None

    def test_ethanol_fingerprint(self, ethanol_smiles):
        fp = _fingerprint(ethanol_smiles)
        assert fp is not None
        assert fp.shape == (2048,)

    def test_different_molecules_different_fingerprints(self, aspirin_smiles, ethanol_smiles):
        fp_asp = _fingerprint(aspirin_smiles)
        fp_eth = _fingerprint(ethanol_smiles)
        assert not np.array_equal(fp_asp, fp_eth)

    def test_same_molecule_same_fingerprint(self, aspirin_smiles):
        fp1 = _fingerprint(aspirin_smiles)
        fp2 = _fingerprint(aspirin_smiles)
        assert np.array_equal(fp1, fp2)

    def test_removes_atom_mapping(self):
        # Mapped SMILES should produce same FP as unmapped
        mapped = "[CH3:1][C:2](=[O:3])[O:4][c:5]1[cH:6][cH:7][cH:8][cH:9][c:10]1[C:11](=[O:12])[OH:13]"
        unmapped = "CC(=O)Oc1ccccc1C(=O)O"
        fp_mapped = _fingerprint(mapped)
        fp_unmapped = _fingerprint(unmapped)
        if fp_mapped is not None:
            assert np.array_equal(fp_mapped, fp_unmapped)


# ═════════════════════════════════════════════════════════════════════════════
# _canonicalize_reactants
# ═════════════════════════════════════════════════════════════════════════════

class TestCanonicalizeReactants:
    def test_single_valid_smiles(self):
        result = _canonicalize_reactants("CCO")
        assert result == "CCO"

    def test_multi_reactant_sorted(self):
        result = _canonicalize_reactants("CC(=O)O.CCO")
        parts = result.split(".")
        assert parts == sorted(parts)

    def test_invalid_smiles_returns_none(self):
        result = _canonicalize_reactants("INVALID")
        assert result is None

    def test_non_canonical_normalized(self):
        result = _canonicalize_reactants("OCC")
        assert result == "CCO"

    def test_empty_parts_skipped(self):
        # Double dot (empty SMILES part)
        result = _canonicalize_reactants("CCO..CC(=O)O")
        assert result is not None
        assert "CCO" in result


# ═════════════════════════════════════════════════════════════════════════════
# _apply_template
# ═════════════════════════════════════════════════════════════════════════════

class TestApplyTemplate:
    def test_returns_list(self, aspirin_smiles):
        # Simple ester hydrolysis template
        smarts = "[C:1](=[O:2])[O:3]>>[C:1](=[O:2])[OH:2].[O:3]"
        result = _apply_template(smarts, aspirin_smiles)
        assert isinstance(result, list)

    def test_invalid_smarts_returns_empty(self, aspirin_smiles):
        result = _apply_template("NOTVALIDSMARTS", aspirin_smiles)
        assert result == []

    def test_invalid_smiles_returns_empty(self):
        smarts = "[C:1](=[O:2])[O:3]>>[C:1](=[O:2])[OH:2].[O:3]"
        result = _apply_template(smarts, "NOTVALID")
        assert result == []


# ═════════════════════════════════════════════════════════════════════════════
# _ensure_loaded
# ═════════════════════════════════════════════════════════════════════════════

class TestEnsureLoaded:
    def test_returns_bool(self):
        result = _ensure_loaded()
        assert isinstance(result, bool)

    def test_returns_false_when_model_missing(self):
        import real_proj.mvp.retro_predictor as predictor
        # Reset loaded state
        original_loaded = predictor._loaded
        original_model = predictor._model
        predictor._loaded = False
        predictor._model = None

        with patch("real_proj.mvp.retro_predictor.MODEL_DIR") as mock_dir:
            mock_path = MagicMock()
            mock_path.exists.return_value = False
            mock_dir.__truediv__ = lambda self, other: mock_path

            result = _ensure_loaded()

        # Restore
        predictor._loaded = original_loaded
        predictor._model = original_model

        assert result is False


# ═════════════════════════════════════════════════════════════════════════════
# predict_retro (requires loaded model — marked slow)
# ═════════════════════════════════════════════════════════════════════════════

class TestPredictRetro:
    @pytest.mark.slow
    def test_aspirin_returns_list(self, aspirin_smiles):
        results = predict_retro(aspirin_smiles, top_n=5)
        assert isinstance(results, list)

    @pytest.mark.slow
    def test_results_have_required_keys(self, aspirin_smiles):
        results = predict_retro(aspirin_smiles, top_n=5)
        for r in results:
            for key in ("reactants", "reaction_smiles", "score", "source"):
                assert key in r, f"Missing key: {key}"

    @pytest.mark.slow
    def test_source_is_retro_model(self, aspirin_smiles):
        results = predict_retro(aspirin_smiles, top_n=5)
        for r in results:
            assert r["source"] == "retro_model"

    @pytest.mark.slow
    def test_score_in_range(self, aspirin_smiles):
        results = predict_retro(aspirin_smiles, top_n=5)
        for r in results:
            assert 0.0 < r["score"] <= 1.0

    @pytest.mark.slow
    def test_top_n_respected(self, aspirin_smiles):
        results = predict_retro(aspirin_smiles, top_n=3)
        assert len(results) <= 3

    @pytest.mark.slow
    def test_no_self_loop_in_reactants(self, aspirin_smiles):
        """Product SMILES should not appear in reactants (no-op reaction)."""
        results = predict_retro(aspirin_smiles, top_n=10)
        for r in results:
            reactant_parts = r["reactants"].split(".")
            assert aspirin_smiles not in reactant_parts, \
                f"Product found in reactants: {r['reactants']}"

    @pytest.mark.slow
    def test_reaction_smiles_format(self, aspirin_smiles):
        results = predict_retro(aspirin_smiles, top_n=5)
        for r in results:
            assert ">>" in r["reaction_smiles"]
            parts = r["reaction_smiles"].split(">>")
            assert len(parts) == 2

    @pytest.mark.slow
    def test_deduplicated_results(self, aspirin_smiles):
        """No two results should have identical canonical reactant sets."""
        results = predict_retro(aspirin_smiles, top_n=10)
        seen = set()
        for r in results:
            reactants = r["reactants"]
            assert reactants not in seen, f"Duplicate reactants: {reactants}"
            seen.add(reactants)

    @pytest.mark.slow
    def test_caffeine_returns_something(self, caffeine_smiles):
        results = predict_retro(caffeine_smiles, top_n=5)
        assert isinstance(results, list)
        # May return 0 if no templates apply, but should not crash

    def test_returns_empty_when_model_not_loaded(self, aspirin_smiles):
        with patch("real_proj.mvp.retro_predictor._ensure_loaded", return_value=False):
            results = predict_retro(aspirin_smiles)
            assert results == []

    def test_invalid_smiles_returns_empty(self):
        results = predict_retro("NOTASMILES!!!", top_n=5)
        assert results == []

    @pytest.mark.slow
    def test_num_examples_is_int(self, aspirin_smiles):
        results = predict_retro(aspirin_smiles, top_n=5)
        for r in results:
            if "num_examples" in r:
                assert isinstance(r["num_examples"], int)

    @pytest.mark.slow
    def test_plausibility_in_range(self, aspirin_smiles):
        results = predict_retro(aspirin_smiles, top_n=5)
        for r in results:
            assert 0.0 <= r["plausibility"] <= 1.0
