"""Tests for AiZynthFinder service payload normalization."""

from __future__ import annotations

from ..services.aizynth_client import (
    _extract_first_disconnection,
    extract_route_trees,
    normalize_aizynth_routes,
)


class TestAiZynthClientHelpers:
    def test_extract_first_disconnection_from_root_metadata(self):
        tree = {
            "metadata": {"mapped_reaction_smiles": "CCO>>CC=O.O"},
            "children": [],
        }
        assert _extract_first_disconnection(tree) == ("CC=O.O", "CCO")

    def test_extract_first_disconnection_recurses_into_children(self):
        tree = {
            "metadata": {},
            "children": [
                {"metadata": {"mapped_reaction_smiles": "CCN>>CC.Cl"}, "children": []},
            ],
        }
        assert _extract_first_disconnection(tree) == ("CC.Cl", "CCN")

    def test_extract_route_trees_supports_list_payload(self):
        payload = {"routes": [{"metadata": {}, "children": []}]}
        trees = extract_route_trees(payload)
        assert len(trees) == 1


class TestNormalizeAizynthRoutes:
    def test_normalizes_single_route(self):
        payload = {
            "smiles": "CCO",
            "statistics": {"is_solved": True, "number_of_solved_routes": 1},
            "stock_info": {"price": "n/a"},
            "parameters": {"stock": "zinc"},
            "routes": [
                {
                    "metadata": {"mapped_reaction_smiles": "CCO>>CC=O.O"},
                    "children": [],
                }
            ],
        }
        routes = normalize_aizynth_routes(payload)
        assert len(routes) == 1
        route = routes[0]
        assert route["source"] == "aizynthfinder"
        assert route["reactants"] == "CC=O.O"
        assert route["reaction_smiles"] == "CC=O.O>>CCO"
        assert route["target_smiles"] == "CCO"
        assert route["provenance"]["provider"] == "aizynthfinder"
        assert route["provenance"]["statistics"]["is_solved"] is True

    def test_skips_routes_without_mapped_reaction(self):
        payload = {
            "smiles": "CCO",
            "routes": [
                {"metadata": {}, "children": []},
            ],
        }
        assert normalize_aizynth_routes(payload) == []
