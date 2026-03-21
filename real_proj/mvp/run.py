#!/usr/bin/env python3
"""Entry point for the MVP pipeline.

Usage:
    python -m real_proj.mvp.run "aspirin"
    python -m real_proj.mvp.run "CC(=O)Oc1ccccc1C(O)=O"
    python -m real_proj.mvp.run  # interactive mode
"""

from __future__ import annotations

import sys
import logging

# Ensure config is loaded first (sets env vars for LangSmith)
from . import config as _cfg  # noqa: F401
from .graph import build_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("mvp")


def run(query: str) -> dict:
    """Run the MVP graph on a single query and return the final state."""
    logger.info("Building graph...")
    app = build_graph()

    logger.info("Running query: %r", query)
    result = app.invoke({"query": query})

    # Print result
    if result.get("error"):
        print(f"\n{'!'*60}")
        print(f"  ERROR: {result['error']}")
        print(f"{'!'*60}")

        guard = result.get("guard_result", {})
        if guard:
            mol_check = guard.get("molecule_check", {})
            rxn_check = guard.get("reaction_check", {})
            if mol_check.get("status") in ("banned", "restricted"):
                print(f"\n  Molecule: {mol_check.get('name', 'Unknown')}")
                print(f"  Status:   {mol_check.get('status')}")
                print(f"  Category: {mol_check.get('category')}")
                print(f"  Reason:   {mol_check.get('reason')}")
            if rxn_check.get("status") in ("prohibited", "restricted"):
                print(f"\n  Reaction: {rxn_check.get('reason')}")

        validation = result.get("validation", {})
        if validation and not validation.get("is_valid"):
            print(f"\n  Validation failed: {validation.get('error')}")

    elif result.get("final_answer"):
        print(f"\n{result['final_answer']}")

    else:
        print("\n  No result produced.")

    return result


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("MVP Pipeline: validate → guard → molecule_info")
        print("=" * 50)
        query = input("Enter molecule (name or SMILES): ").strip()
        if not query:
            print("No input. Exiting.")
            sys.exit(0)

    run(query)


if __name__ == "__main__":
    main()
