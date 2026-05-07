"""Tests for planner-specific retrosynthesis tree adapters."""

from __future__ import annotations

from unittest.mock import patch

from ..services.planner_tree_adapter import adapt_aizynth_tree_to_runtime


def test_adapts_nested_aizynth_tree_to_runtime_schema():
    raw_tree = {
        "type": "mol",
        "smiles": "[CH3:1][CH2:2][OH:3]",
        "children": [
            {
                "type": "reaction",
                "metadata": {"mapped_reaction_smiles": "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3].[OH2:4]"},
                "children": [
                    {
                        "type": "mol",
                        "smiles": "[CH3:1][CH:2]=[O:3]",
                        "children": [
                            {
                                "type": "reaction",
                                "metadata": {"mapped_reaction_smiles": "[CH3:1][CH:2]=[O:3]>>[CH3:1][OH:2]"},
                                "children": [
                                    {"type": "mol", "smiles": "[CH3:1][OH:2]", "in_stock": True, "children": []},
                                ],
                            }
                        ],
                    },
                    {"type": "mol", "smiles": "[OH2:4]", "in_stock": True, "children": []},
                ],
            }
        ],
    }

    with patch("mvp.services.planner_tree_adapter.banlist_check", return_value={"status": "clear"}), \
         patch("mvp.services.planner_tree_adapter._is_buyable", return_value=False):
        result = adapt_aizynth_tree_to_runtime(raw_tree, target_smiles="CCO")

    tree = result["tree"]
    assert tree["smiles"] == "CCO"
    assert tree["route"]["reactants"] == "CC=O.O"
    assert tree["children"][0]["smiles"] == "CC=O"
    assert tree["children"][0]["children"][0]["smiles"] == "CO"
    assert result["stats"]["max_depth_reached"] == 2
