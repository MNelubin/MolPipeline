"""Tests for retrosynthesis_node: route search, formatting, procedure steps."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from ..nodes.retrosynthesis_node import retrosynthesis_node, _format_retro_text


# ═════════════════════════════════════════════════════════════════════════════
# _format_retro_text
# ═════════════════════════════════════════════════════════════════════════════

class TestFormatRetroText:
    def test_no_routes_message(self):
        text = _format_retro_text("Аспирин", [], [], 0)
        assert "не найдены" in text

    def test_header_contains_molecule_name(self):
        text = _format_retro_text("Аспирин", [], [], 0)
        assert "Аспирин" in text

    def test_source_ord_label(self, mock_ord_route):
        mock_ord_route["final_score"] = 0.85
        mock_ord_route["scoring"] = {
            "model_score": 0.85, "plausibility": 0.9,
            "buyability": 1.0, "simplicity": 0.5,
        }
        mock_ord_route["procedure_steps_ru"] = []
        text = _format_retro_text("Test", [mock_ord_route], ["ord"], 1)
        assert "ORD" in text

    def test_source_model_label(self):
        route = {
            "reactants": "CCO.CC(=O)O",
            "reaction_smiles": "CCO.CC(=O)O>>CC(=O)OCC",
            "source": "retro_model",
            "final_score": 0.70,
            "scoring": {
                "model_score": 0.70, "plausibility": 0.8,
                "buyability": 0.9, "simplicity": 0.6,
            },
            "procedure_steps_ru": [],
        }
        text = _format_retro_text("Test", [route], ["retro_model"], 1)
        assert "МОДЕЛЬ" in text

    def test_score_displayed(self, mock_ord_route):
        mock_ord_route["final_score"] = 0.876
        mock_ord_route["scoring"] = {
            "model_score": 0.85, "plausibility": 0.9,
            "buyability": 1.0, "simplicity": 0.5,
        }
        mock_ord_route["procedure_steps_ru"] = []
        text = _format_retro_text("Test", [mock_ord_route], ["ord"], 1)
        assert "0.876" in text

    def test_procedure_steps_displayed(self, mock_ord_route):
        mock_ord_route["final_score"] = 0.8
        mock_ord_route["scoring"] = {
            "model_score": 0.85, "plausibility": 0.9,
            "buyability": 1.0, "simplicity": 0.5,
        }
        mock_ord_route["procedure_steps_ru"] = [
            {"step": "1", "description": "Смешайте реагенты", "reason": "ORD процедура"},
            {"step": "2", "description": "Нагрейте до 60°C", "reason": "ORD процедура"},
        ]
        text = _format_retro_text("Test", [mock_ord_route], ["ord"], 1)
        assert "ПРОЦЕДУРА СИНТЕЗА" in text
        assert "Смешайте реагенты" in text

    def test_long_smiles_truncated(self, mock_ord_route):
        mock_ord_route["reactants"] = "C" * 100  # very long
        mock_ord_route["final_score"] = 0.8
        mock_ord_route["scoring"] = {
            "model_score": 0.85, "plausibility": 0.9,
            "buyability": 1.0, "simplicity": 0.5,
        }
        mock_ord_route["procedure_steps_ru"] = []
        text = _format_retro_text("Test", [mock_ord_route], ["ord"], 1)
        assert "..." in text

    def test_multiple_routes_numbered(self):
        routes = []
        for i in range(3):
            routes.append({
                "reactants": f"CCO{i}",
                "source": "ord",
                "final_score": 0.8 - i * 0.1,
                "scoring": {"model_score": 0.8, "plausibility": 0.9,
                            "buyability": 1.0, "simplicity": 0.5},
                "procedure_steps_ru": [],
            })
        text = _format_retro_text("Test", routes, ["ord"], 3)
        assert "Путь #1" in text
        assert "Путь #2" in text
        assert "Путь #3" in text


# ═════════════════════════════════════════════════════════════════════════════
# retrosynthesis_node
# ═════════════════════════════════════════════════════════════════════════════

class TestFormatRetroTextExtra:
    def test_source_counts_and_provenance_displayed(self):
        route = {
            "reactants": "CC=O.O",
            "reaction_smiles": "CC=O.O>>CCO",
            "source": "aizynthfinder",
            "num_steps": 4,
            "final_score": 0.81,
            "scoring": {
                "model_score": 0.70,
                "plausibility": 0.75,
                "buyability": 0.8,
                "simplicity": 0.6,
            },
            "provenance": {
                "provider": "aizynthfinder",
                "retrieval_mode": "service_tree_search",
            },
            "procedure_steps_ru": [],
        }
        text = _format_retro_text(
            "Test",
            [route],
            ["ord", "aizynthfinder"],
            2,
            source_counts={"ord": 1, "aizynthfinder": 1},
        )
        assert "AiZynthFinder=1" in text
        assert "Шагов в маршруте: 4" in text
        assert "service_tree_search" in text


class TestRetrosynthesisNode:
    def _mock_search_and_rank(self, routes=None, sources=None, total=0):
        return {
            "routes": routes or [],
            "best_route": routes[0] if routes else None,
            "sources_used": sources or [],
            "total_found": total,
        }

    def test_empty_smiles_returns_empty_routes(self):
        result = retrosynthesis_node({"smiles": "", "molecule_info": {}})
        assert result["retro_result"]["routes"] == []
        assert "error" in result["retro_result"]

    def test_missing_smiles_returns_empty_routes(self):
        result = retrosynthesis_node({"molecule_info": {}})
        assert result["retro_result"]["routes"] == []

    def test_output_has_retro_result_key(self, aspirin_smiles, mock_molecule_info, mock_ord_route):
        mock_ord_route["final_score"] = 0.85
        mock_ord_route["scoring"] = {
            "model_score": 0.85, "plausibility": 0.9,
            "buyability": 1.0, "simplicity": 0.5,
        }
        state = {"smiles": aspirin_smiles, "molecule_info": mock_molecule_info}
        mock_result = self._mock_search_and_rank([mock_ord_route], ["ord"], 1)

        with patch("mvp.nodes.retrosynthesis_node.search_and_rank",
                   return_value=mock_result), \
             patch("mvp.nodes.retrosynthesis_node.format_procedure_russian",
                   return_value=[]):
            result = retrosynthesis_node(state)

        assert "retro_result" in result

    def test_output_has_final_answer_key(self, aspirin_smiles, mock_molecule_info, mock_ord_route):
        mock_ord_route["final_score"] = 0.85
        mock_ord_route["scoring"] = {
            "model_score": 0.85, "plausibility": 0.9,
            "buyability": 1.0, "simplicity": 0.5,
        }
        state = {"smiles": aspirin_smiles, "molecule_info": mock_molecule_info}
        mock_result = self._mock_search_and_rank([mock_ord_route], ["ord"], 1)

        with patch("mvp.nodes.retrosynthesis_node.search_and_rank",
                   return_value=mock_result), \
             patch("mvp.nodes.retrosynthesis_node.format_procedure_russian",
                   return_value=[]):
            result = retrosynthesis_node(state)

        assert "final_answer" in result
        assert len(result["final_answer"]) > 10

    def test_appends_to_existing_final_answer(self, aspirin_smiles, mock_molecule_info):
        state = {
            "smiles": aspirin_smiles,
            "molecule_info": mock_molecule_info,
            "final_answer": "EXISTING_TEXT\n",
        }
        mock_result = self._mock_search_and_rank([], [], 0)

        with patch("mvp.nodes.retrosynthesis_node.search_and_rank",
                   return_value=mock_result), \
             patch("mvp.nodes.retrosynthesis_node.format_procedure_russian",
                   return_value=[]):
            result = retrosynthesis_node(state)

        assert "EXISTING_TEXT" in result["final_answer"]

    def test_procedure_steps_added_to_routes(self, aspirin_smiles, mock_molecule_info, mock_ord_route):
        mock_ord_route["final_score"] = 0.85
        mock_ord_route["scoring"] = {
            "model_score": 0.85, "plausibility": 0.9,
            "buyability": 1.0, "simplicity": 0.5,
        }
        state = {"smiles": aspirin_smiles, "molecule_info": mock_molecule_info}
        mock_result = self._mock_search_and_rank([mock_ord_route], ["ord"], 1)
        mock_steps = [{"step": "1", "description": "Тест", "reason": "ORD"}]

        with patch("mvp.nodes.retrosynthesis_node.search_and_rank",
                   return_value=mock_result), \
             patch("mvp.nodes.retrosynthesis_node.format_procedure_russian",
                   return_value=mock_steps) as mock_fmt:
            result = retrosynthesis_node(state)

        mock_fmt.assert_called_once()
        routes = result["retro_result"]["routes"]
        assert routes[0]["procedure_steps_ru"] == mock_steps

    def test_no_routes_message_in_text(self, aspirin_smiles, mock_molecule_info):
        state = {"smiles": aspirin_smiles, "molecule_info": mock_molecule_info}
        mock_result = self._mock_search_and_rank([], [], 0)

        with patch("mvp.nodes.retrosynthesis_node.search_and_rank",
                   return_value=mock_result), \
             patch("mvp.nodes.retrosynthesis_node.format_procedure_russian",
                   return_value=[]):
            result = retrosynthesis_node(state)

        assert "не найдены" in result["final_answer"]

    def test_retro_result_sources_passed_through(self, aspirin_smiles, mock_molecule_info):
        state = {"smiles": aspirin_smiles, "molecule_info": mock_molecule_info}
        mock_result = self._mock_search_and_rank([], ["ord"], 0)

        with patch("mvp.nodes.retrosynthesis_node.search_and_rank",
                   return_value=mock_result), \
             patch("mvp.nodes.retrosynthesis_node.format_procedure_russian",
                   return_value=[]):
            result = retrosynthesis_node(state)

        assert result["retro_result"]["sources_used"] == ["ord"]

    @pytest.mark.integration
    def test_real_search_aspirin(self, aspirin_smiles, mock_molecule_info):
        """Integration: actual ORD + model search for aspirin."""
        state = {"smiles": aspirin_smiles, "molecule_info": mock_molecule_info}
        result = retrosynthesis_node(state)
        assert "retro_result" in result
        routes = result["retro_result"]["routes"]
        assert len(routes) > 0
        assert routes[0]["final_score"] > 0

