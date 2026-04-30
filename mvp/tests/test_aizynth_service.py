"""Tests for the local AiZynthFinder service wrapper."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from .. import aizynth_service as svc


class TestAiZynthServiceHelpers:
    def test_named_section_returns_sorted_keys(self):
        config = {"stock": {"zinc": {}, "emolecules": {}}}
        assert svc._named_section(config, "stock") == ["emolecules", "zinc"]

    def test_runtime_config_overrides_search_limits(self):
        request = svc.AiZynthRunRequest(
            smiles="CCO",
            max_transforms=8,
            time_limit=25,
            iterations=300,
            expansion_model="uspto",
            stock="zinc",
        )
        config = svc._runtime_config({"search": {"max_transforms": 2}}, request)
        assert config["search"]["max_transforms"] == 8
        assert config["search"]["time_limit"] == 25
        assert config["search"]["iteration_limit"] == 300

    def test_extract_route_dicts_prefers_stats_trees(self):
        stats = {"trees": [{"id": 1}, {"id": 2}]}
        finder = object()
        assert svc._extract_route_dicts(stats, finder, 1) == [{"id": 1}]

    def test_extract_route_dicts_falls_back_to_reaction_trees(self):
        route_a = SimpleNamespace(to_dict=lambda include_metadata=True: {"route": "a"})
        route_b = SimpleNamespace(to_dict=lambda include_metadata=True: {"route": "b"})
        finder = SimpleNamespace(routes=SimpleNamespace(reaction_trees=[route_a, route_b]))
        assert svc._extract_route_dicts({}, finder, 2) == [{"route": "a"}, {"route": "b"}]


class TestAiZynthServiceApi:
    def test_resources_endpoint_uses_loaded_config(self):
        client = TestClient(svc.app)
        fake_config = {
            "stock": {"zinc": {}},
            "expansion": {"uspto": {}},
            "filter": {"uspto": {}},
        }
        with patch("mvp.aizynth_service._load_base_config", return_value=fake_config):
            response = client.get("/api/v1/resources")
        assert response.status_code == 200
        payload = response.json()
        assert payload["stocks"] == ["zinc"]
        assert payload["expansion_models"] == ["uspto"]

    def test_run_endpoint_returns_normalized_payload(self):
        client = TestClient(svc.app)
        with patch("mvp.aizynth_service._run_search", return_value={"routes": [{"id": 1}], "statistics": {}}):
            response = client.post(
                "/api/v1/run",
                json={"smiles": "CCO", "stock": "zinc", "expansion_model": "uspto"},
            )
        assert response.status_code == 200
        assert response.json()["routes"] == [{"id": 1}]
