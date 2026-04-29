"""Tests for the optional RetroCast bridge."""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

from ..services.retrocast_bridge import (
    adapt_aizynth_payload_with_retrocast,
    get_retrocast_runtime_info,
)


class _FakeReactionStep:
    def __init__(self, reactants):
        self.reactants = reactants


class _FakeMolecule:
    def __init__(self, smiles, synthesis_step=None):
        self.smiles = smiles
        self.synthesis_step = synthesis_step

    def __hash__(self):
        return hash(self.smiles)


class _FakeRoute:
    def __init__(self):
        self.target = _FakeMolecule(
            "CCO",
            synthesis_step=_FakeReactionStep([
                _FakeMolecule("CC=O"),
                _FakeMolecule("O"),
            ]),
        )
        self.length = 3
        self.leaves = {_FakeMolecule("CC=O"), _FakeMolecule("O")}
        self.has_convergent_reaction = False
        self.content_hash = "hash-1"
        self.signature = "sig-1"
        self.retrocast_version = "0.5.3"

    def model_dump(self, mode="json"):
        return {"target": {"smiles": "CCO"}, "mode": mode}


class _FakeTargetInput:
    def __init__(self, id, smiles):
        self.id = id
        self.smiles = smiles


def _fake_retrocast_module():
    module = types.ModuleType("retrocast")
    module.__version__ = "0.5.3"
    module.ADAPTER_MAP = {"aizynth": object(), "retrostar": object()}
    module.TargetInput = _FakeTargetInput
    module.adapt_routes = lambda raw_routes, target, adapter_name, max_routes=None: [_FakeRoute()]
    return module


class TestRetroCastRuntimeInfo:
    def test_reports_unavailable_without_package(self):
        with patch.dict(sys.modules, {"retrocast": None}):
            info = get_retrocast_runtime_info()
        assert info["available"] is False
        assert info["adapters"] == []

    def test_reports_version_and_adapters_when_package_exists(self):
        fake = _fake_retrocast_module()
        with patch.dict(sys.modules, {"retrocast": fake}):
            info = get_retrocast_runtime_info()
        assert info["available"] is True
        assert info["version"] == "0.5.3"
        assert "aizynth" in info["adapters"]


class TestRetroCastAizynthBridge:
    def test_adapts_aizynth_payload_to_route_summary(self):
        payload = {
            "smiles": "CCO",
            "routes": [{"type": "mol", "smiles": "CCO", "children": []}],
        }
        fake = _fake_retrocast_module()
        with patch.dict(sys.modules, {"retrocast": fake}), \
             patch("mvp.services.aizynth_client.extract_route_trees", return_value=payload["routes"]):
            summaries = adapt_aizynth_payload_with_retrocast(payload)

        assert len(summaries) == 1
        summary = summaries[0]
        assert summary["target_smiles"] == "CCO"
        assert summary["reactants"] == "CC=O.O"
        assert summary["reaction_smiles"] == "CC=O.O>>CCO"
        assert summary["num_steps"] == 3
        assert summary["leaf_count"] == 2
        assert summary["retrocast_version"] == "0.5.3"
