"""Standalone retrosynthesis predictor extracted from ASKCOS v2.

Uses the template-relevance model (reaxys, 163K templates) to predict
one-step retrosynthetic disconnections. No ASKCOS stack needed.

Model: Morgan fingerprint (2048-bit) → 5×Dense(300) → softmax(163723)
Templates applied via rdchiral to generate actual precursor SMILES.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Model directory
MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "retro_model"

# Lazy-loaded globals
_model = None
_templates: list[dict] | None = None
_loaded = False


def _ensure_loaded() -> bool:
    """Lazy-load model and templates on first use."""
    global _model, _templates, _loaded

    if _loaded:
        return _model is not None

    _loaded = True

    model_path = MODEL_DIR / "model_latest.pt"
    templates_path = MODEL_DIR / "templates.jsonl"

    if not model_path.exists() or not templates_path.exists():
        logger.warning("Retro model not found at %s", MODEL_DIR)
        return False

    try:
        import torch
        import torch.nn as nn

        # ── Load templates ──
        _templates = []
        with open(templates_path, "r") as f:
            for line in f:
                t = json.loads(line.strip())
                t.pop("references", None)
                _templates.append(t)
        logger.info("Loaded %d templates", len(_templates))

        # ── Build model from checkpoint ──
        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
        args = checkpoint["args"]

        # Build TemplRel architecture
        if isinstance(args.hidden_sizes, str):
            args.hidden_sizes = [int(s) for s in args.hidden_sizes.split(",")]

        class Dense(nn.Module):
            def __init__(self, in_f, out_f):
                super().__init__()
                self.linear = nn.Linear(in_f, out_f)
                self.act = nn.ReLU()
            def forward(self, x):
                return self.act(self.linear(x))

        class TemplRel(nn.Module):
            def __init__(self, a):
                super().__init__()
                layers = [Dense(a.fp_size, a.hidden_sizes[0])]
                for i in range(len(a.hidden_sizes) - 1):
                    layers.append(Dense(a.hidden_sizes[i], a.hidden_sizes[i + 1]))
                self.layers = nn.ModuleList(layers)
                self.output_layer = nn.Linear(a.hidden_sizes[-1], a.n_templates)
                self.dropout = nn.Dropout(a.dropout)

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                    x = self.dropout(x)
                return self.output_layer(x)

        _model = TemplRel(args)
        state_dict = {k.replace("module.", ""): v
                      for k, v in checkpoint["state_dict"].items()}
        _model.load_state_dict(state_dict)
        _model.eval()
        logger.info(
            "Loaded retro model: %d params, %d templates",
            sum(p.numel() for p in _model.parameters()),
            len(_templates),
        )
        return True

    except Exception as e:
        logger.error("Failed to load retro model: %s", e)
        _model = None
        _templates = None
        return False


def _fingerprint(smiles: str) -> np.ndarray:
    """Compute Morgan fingerprint (radius=2, 2048 bits) as numpy array."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Remove atom mapping
    for a in mol.GetAtoms():
        a.ClearProp("molAtomMapNumber")
    smiles_clean = Chem.MolToSmiles(mol, isomericSmiles=True)
    mol = Chem.MolFromSmiles(smiles_clean)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=2048, useChirality=True
    )
    arr = np.zeros((2048,), dtype="float32")
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _apply_template(smarts: str, product_smiles: str) -> list[str]:
    """Apply a retrosynthetic template to produce precursor sets."""
    try:
        from rdchiral.main import rdchiralRun
        from rdchiral.initialization import rdchiralReaction, rdchiralReactants

        # Wrap SMARTS in rdchiral format
        wrapped = "(" + smarts.replace(">>", ")>>(") + ")"
        rxn = rdchiralReaction(wrapped)
        prod = rdchiralReactants(product_smiles)
        results = rdchiralRun(rxn, prod, return_mapped=False)
        return results if results else []
    except Exception:
        return []


def _canonicalize_reactants(reactant_smi: str) -> str | None:
    """Canonicalize a dot-separated reactant SMILES string."""
    from rdkit import Chem

    parts = reactant_smi.split(".")
    canonical = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        mol = Chem.MolFromSmiles(p)
        if mol is None:
            return None
        canonical.append(Chem.MolToSmiles(mol, isomericSmiles=True))
    canonical.sort()
    return ".".join(canonical)


def predict_retro(
    smiles: str,
    top_n: int = 10,
    max_templates: int = 200,
) -> list[dict[str, Any]]:
    """Predict one-step retrosynthetic disconnections.

    Returns list of dicts with: reactants, reaction_smiles, score,
    num_examples, template, source.
    Deduplicated by canonical reactant set.
    """
    if not _ensure_loaded():
        return []

    import torch

    # 1. Fingerprint
    fp = _fingerprint(smiles)
    if fp is None:
        logger.warning("Cannot fingerprint: %s", smiles)
        return []

    # 2. Forward pass
    input_tensor = torch.tensor(fp).unsqueeze(0).float()
    with torch.no_grad():
        logits = _model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().numpy()

    # 3. Sort by probability
    top_indices = np.argsort(-probs)[:max_templates]

    # 4. Apply templates and collect results
    seen: set[str] = set()
    results: list[dict[str, Any]] = []

    for idx in top_indices:
        if len(results) >= top_n:
            break

        idx = int(idx)
        score = float(probs[idx])
        if score < 0.001:
            break

        template = _templates[idx]
        smarts = template.get("reaction_smarts", "")
        if not smarts:
            continue

        # Apply template
        precursor_sets = _apply_template(smarts, smiles)
        if not precursor_sets:
            continue

        for reactant_smi in precursor_sets:
            canonical = _canonicalize_reactants(reactant_smi)
            if canonical is None or canonical in seen:
                continue
            # Skip if product is in reactants (no transformation)
            if smiles in canonical.split("."):
                continue

            seen.add(canonical)
            results.append({
                "reactants": canonical,
                "reaction_smiles": f"{canonical}>>{smiles}",
                "score": round(score, 4),
                "plausibility": round(min(score * 1.5, 1.0), 4),
                "num_examples": template.get("count", template.get("num_examples", 0)),
                "template": smarts,
                "source": "retro_model",
            })

            if len(results) >= top_n:
                break

    logger.info("Retro predictor: %d results for %s", len(results), smiles[:30])
    return results
