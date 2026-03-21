#!/usr/bin/env python3
"""Test script: retrosynthesis scoring pipeline demo.

Demonstrates the scoring system working INDEPENDENTLY — no ASKCOS needed.
Sources: ORD (if available) + realistic example data.
Runs scoring, banned filtering, and ranks top 3.
"""

import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx

ORD_URL = "https://client.open-reaction-database.org/api/query"

# Test molecules
TEST_MOLECULES = {
    "aspirin": "CC(=O)Oc1ccccc1C(O)=O",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(O)=O",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
}

# Realistic retrosynthesis data for testing (from literature / known routes)
EXAMPLE_RETRO_DATA = {
    "aspirin": [
        {
            "reactants": "OC(=O)c1ccccc1O.CC(=O)OC(C)=O",
            "reaction_smiles": "OC(=O)c1ccccc1O.CC(=O)OC(C)=O>>CC(=O)Oc1ccccc1C(O)=O",
            "score": 0.95,
            "source": "literature",
            "plausibility": 0.98,
            "expected_yield": 0.85,
            "description": "Classic aspirin synthesis: salicylic acid + acetic anhydride",
        },
        {
            "reactants": "OC(=O)c1ccccc1O.CC(=O)Cl",
            "reaction_smiles": "OC(=O)c1ccccc1O.CC(=O)Cl>>CC(=O)Oc1ccccc1C(O)=O",
            "score": 0.80,
            "source": "literature",
            "plausibility": 0.90,
            "expected_yield": 0.70,
            "description": "Aspirin via acetyl chloride (harsher conditions)",
        },
        {
            "reactants": "OC(=O)c1ccccc1O.CC(=O)OC(=O)C.O=P(O)(O)O",
            "reaction_smiles": "OC(=O)c1ccccc1O.CC(=O)OC(=O)C.O=P(O)(O)O>>CC(=O)Oc1ccccc1C(O)=O",
            "score": 0.88,
            "source": "literature",
            "plausibility": 0.92,
            "expected_yield": 0.90,
            "description": "Aspirin with phosphoric acid catalyst (improved yield)",
        },
        {
            "reactants": "Oc1ccccc1.CC(=O)OC(C)=O.OC(=O)C",
            "reaction_smiles": "Oc1ccccc1.CC(=O)OC(C)=O.OC(=O)C>>CC(=O)Oc1ccccc1C(O)=O",
            "score": 0.30,
            "source": "predicted",
            "plausibility": 0.40,
            "expected_yield": 0.20,
            "description": "Bad route: phenol + acetic anhydride (wrong substrate, no COOH)",
        },
        {
            "reactants": "OC(=O)c1ccccc1OCCC.CC(=O)OC(C)=O",
            "reaction_smiles": "OC(=O)c1ccccc1OCCC.CC(=O)OC(C)=O>>CC(=O)Oc1ccccc1C(O)=O",
            "score": 0.15,
            "source": "predicted",
            "plausibility": 0.25,
            "expected_yield": None,
            "description": "Bad route: complex starting material, low plausibility",
        },
        {
            "reactants": "c1ccc2c(c1)oc(=O)o2.CC(=O)OC(C)=O.O",
            "reaction_smiles": "c1ccc2c(c1)oc(=O)o2.CC(=O)OC(C)=O.O>>CC(=O)Oc1ccccc1C(O)=O",
            "score": 0.45,
            "source": "predicted",
            "plausibility": 0.50,
            "expected_yield": 0.35,
            "description": "Via salicylaldehyde cyclic carbonate intermediate",
        },
    ],
    "paracetamol": [
        {
            "reactants": "Nc1ccc(O)cc1.CC(=O)OC(C)=O",
            "reaction_smiles": "Nc1ccc(O)cc1.CC(=O)OC(C)=O>>CC(=O)Nc1ccc(O)cc1",
            "score": 0.92,
            "source": "literature",
            "plausibility": 0.95,
            "expected_yield": 0.88,
            "description": "Classic: 4-aminophenol + acetic anhydride",
        },
        {
            "reactants": "Nc1ccc(O)cc1.CC(=O)Cl",
            "reaction_smiles": "Nc1ccc(O)cc1.CC(=O)Cl>>CC(=O)Nc1ccc(O)cc1",
            "score": 0.78,
            "source": "literature",
            "plausibility": 0.85,
            "expected_yield": 0.72,
            "description": "4-aminophenol + acetyl chloride",
        },
        {
            "reactants": "Nc1ccc(O)cc1.CC(O)=O",
            "reaction_smiles": "Nc1ccc(O)cc1.CC(O)=O>>CC(=O)Nc1ccc(O)cc1",
            "score": 0.60,
            "source": "literature",
            "plausibility": 0.70,
            "expected_yield": 0.45,
            "description": "4-aminophenol + acetic acid (needs activation)",
        },
        {
            "reactants": "Oc1ccc([N+](=O)[O-])cc1.CC(=O)OC(C)=O.[Fe]",
            "reaction_smiles": "Oc1ccc([N+](=O)[O-])cc1.CC(=O)OC(C)=O.[Fe]>>CC(=O)Nc1ccc(O)cc1",
            "score": 0.70,
            "source": "literature",
            "plausibility": 0.75,
            "expected_yield": 0.55,
            "description": "One-pot: 4-nitrophenol reduction + acetylation",
        },
        {
            "reactants": "c1ccc(cc1)NC(=O)C.BrBr.O",
            "reaction_smiles": "c1ccc(cc1)NC(=O)C.BrBr.O>>CC(=O)Nc1ccc(O)cc1",
            "score": 0.25,
            "source": "predicted",
            "plausibility": 0.30,
            "expected_yield": None,
            "description": "Poor: acetanilide bromination (wrong reaction type)",
        },
    ],
    "ibuprofen": [
        {
            "reactants": "CC(C)Cc1ccc(cc1)C(C)=O.O=C=O",
            "reaction_smiles": "CC(C)Cc1ccc(cc1)C(C)=O>>CC(C)Cc1ccc(cc1)C(C)C(O)=O",
            "score": 0.85,
            "source": "literature",
            "plausibility": 0.88,
            "expected_yield": 0.77,
            "description": "BHC process: 4-isobutylacetophenone carboxylation",
        },
        {
            "reactants": "CC(C)Cc1ccc(cc1)CC#N.O",
            "reaction_smiles": "CC(C)Cc1ccc(cc1)CC#N.O>>CC(C)Cc1ccc(cc1)C(C)C(O)=O",
            "score": 0.55,
            "source": "predicted",
            "plausibility": 0.60,
            "expected_yield": 0.50,
            "description": "Via nitrile hydrolysis (longer route)",
        },
        {
            "reactants": "CC(C)Cc1ccc(cc1)CBr.[Mg].O=C=O",
            "reaction_smiles": "CC(C)Cc1ccc(cc1)CBr.[Mg].O=C=O>>CC(C)Cc1ccc(cc1)C(C)C(O)=O",
            "score": 0.65,
            "source": "literature",
            "plausibility": 0.72,
            "expected_yield": 0.60,
            "description": "Grignard route: ArCH2Br + Mg → ArCH2MgBr + CO2",
        },
        {
            "reactants": "CC(Cl)c1ccc(CC(C)C)cc1.[Ag]C#N.O",
            "reaction_smiles": "CC(Cl)c1ccc(CC(C)C)cc1.[Ag]C#N.O>>CC(C)Cc1ccc(cc1)C(C)C(O)=O",
            "score": 0.35,
            "source": "predicted",
            "plausibility": 0.40,
            "expected_yield": None,
            "description": "Via benzyl chloride + silver cyanide (expensive, poor route)",
        },
    ],
    "caffeine": [
        {
            "reactants": "Cn1c(=O)c2c(nc(Cl)n2C)n(C)c1=O",
            "reaction_smiles": "Cn1c(=O)c2c(nc(Cl)n2C)n(C)c1=O.[H][H]>>Cn1c(=O)c2c(ncn2C)n(C)c1=O",
            "score": 0.75,
            "source": "literature",
            "plausibility": 0.80,
            "expected_yield": 0.70,
            "description": "Chlorocaffeine hydrogenolysis",
        },
        {
            "reactants": "Cn1c(=O)c2c([nH]cn2)n(C)c1=O.CI",
            "reaction_smiles": "Cn1c(=O)c2c([nH]cn2)n(C)c1=O.CI>>Cn1c(=O)c2c(ncn2C)n(C)c1=O",
            "score": 0.82,
            "source": "literature",
            "plausibility": 0.85,
            "expected_yield": 0.65,
            "description": "Theophylline + methyl iodide N-methylation",
        },
        {
            "reactants": "CN.OC(=O)c1ccccc1.CC(=O)OC(C)=O.O=C=O",
            "reaction_smiles": "CN.OC(=O)c1ccccc1.CC(=O)OC(C)=O.O=C=O>>Cn1c(=O)c2c(ncn2C)n(C)c1=O",
            "score": 0.10,
            "source": "predicted",
            "plausibility": 0.15,
            "expected_yield": None,
            "description": "Nonsense: random small molecules (bad prediction)",
        },
    ],
}


def query_ord(smiles: str, limit: int = 10) -> list[dict]:
    """Query Open Reaction Database for reactions producing the target."""
    client = httpx.Client(timeout=15.0)
    payload = {
        "useStereochemistry": False,
        "similarity": 0.6,
        "component": [{"smiles": smiles, "source": "output", "mode": "substructure"}],
        "limit": limit,
    }

    try:
        resp = client.post(ORD_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [!] ORD not reachable: {e}")
        return []

    reactions = data if isinstance(data, list) else data.get("reactions", data.get("results", []))
    parsed = []
    for rxn in reactions[:limit]:
        rxn_smi = ""
        for ident in rxn.get("identifiers", []):
            if "smiles" in ident.get("type", "").lower() or ident.get("type") == "REACTION_SMILES":
                rxn_smi = ident.get("value", "")
                break
        if not rxn_smi or ">>" not in rxn_smi:
            continue
        reactant_str = rxn_smi.split(">>")[0]

        yield_val = None
        for outcome in rxn.get("outcomes", []):
            for product in outcome.get("products", []):
                for m in product.get("measurements", []):
                    if m.get("type") == "YIELD":
                        yield_val = m.get("percentage", {}).get("value")

        parsed.append({
            "reactants": reactant_str,
            "reaction_smiles": rxn_smi,
            "score": 0.8,
            "source": "ord",
            "expected_yield": yield_val / 100.0 if yield_val else None,
            "plausibility": 0.9,
            "description": f"ORD published reaction",
        })

    return parsed


def run_test(name: str, smiles: str):
    """Run full scoring pipeline for a molecule."""
    print(f"\n{'='*70}")
    print(f"  TARGET: {name.upper()}")
    print(f"  SMILES: {smiles}")
    print(f"{'='*70}")

    # --- Step 1: Collect retro results ---
    all_results = []

    # Try ORD
    print("\n[1] Querying Open Reaction Database...")
    ord_results = query_ord(smiles, limit=10)
    print(f"    Got {len(ord_results)} reactions from ORD")
    all_results.extend(ord_results)

    # Add example data
    examples = EXAMPLE_RETRO_DATA.get(name, [])
    if examples:
        print(f"[2] Loading {len(examples)} example retrosynthesis routes...")
        all_results.extend(examples)

    if not all_results:
        print("\n  [!] No results. Skipping.")
        return

    print(f"\n    Total candidates: {len(all_results)}")

    # --- Step 2: Filter banned ---
    print(f"\n[3] Filtering banned chemicals/reactions...")
    from src.tools.banned_filter import check_smiles_banned, check_reaction_banned

    filtered = []
    banned_count = 0
    warned = []
    for r in all_results:
        reactants_str = r.get("reactants", "")
        is_banned = False

        for smi in reactants_str.split("."):
            smi = smi.strip()
            if not smi:
                continue
            ban = check_smiles_banned(smi)
            if ban:
                danger = ban.get("danger_level", "unknown")
                if danger in ("critical", "high"):
                    print(f"    BLOCKED: {ban.get('name')} ({danger}) in: {r.get('description', '')}")
                    is_banned = True
                    banned_count += 1
                    break
                else:
                    warned.append(f"{ban.get('name')} ({danger})")

        rxn_smi = r.get("reaction_smiles", "")
        if not is_banned and rxn_smi:
            ban = check_reaction_banned(rxn_smi)
            if ban:
                print(f"    BLOCKED rxn: {ban.get('name')} ({ban.get('danger_level')})")
                is_banned = True
                banned_count += 1

        if not is_banned:
            filtered.append(r)

    if warned:
        for w in set(warned):
            print(f"    WARNING: {w}")

    print(f"    Passed: {len(filtered)}, Blocked: {banned_count}")

    if not filtered:
        print("  [!] All results were filtered out!")
        return

    # --- Step 3: Score and rank ---
    print(f"\n[4] Scoring {len(filtered)} results with retro_scorer...")
    from src.tools.retro_scorer import score_precursor_set

    scored = []
    for r in filtered:
        reactants = r.get("reactants", "")
        model_score = r.get("score", 0.5)
        plausibility = r.get("plausibility", 1.0)

        scoring = score_precursor_set(reactants, model_score, plausibility)
        r["scoring"] = scoring
        r["final_score"] = scoring["precursor_score"]
        scored.append(r)

    scored.sort(key=lambda x: x["final_score"], reverse=True)

    # --- Step 4: Display top 3 ---
    top_n = min(3, len(scored))
    print(f"\n{'='*70}")
    print(f"  TOP {top_n} RETROSYNTHESIS ROUTES FOR {name.upper()}")
    print(f"{'='*70}")

    for i, r in enumerate(scored[:top_n]):
        s = r["scoring"]
        bd = s.get("breakdown", {})
        print(f"\n  #{i+1}  [{r.get('source', '?').upper()}]  {r.get('description', '')}")
        print(f"  {'─'*60}")
        print(f"  Reactants:      {r.get('reactants', '?')}")
        if r.get("reaction_smiles"):
            rxn = r["reaction_smiles"]
            if len(rxn) > 80:
                rxn = rxn[:77] + "..."
            print(f"  Reaction:       {rxn}")

        buyable_count = sum(1 for x in s.get("reactants", []) if x.get("buyable"))
        print(f"  Atoms / React:  {s['total_heavy_atoms']} atoms, {s['num_reactants']} reactants")
        if s.get("rms_molecular_weight"):
            print(f"  RMS mol.wt:     {s['rms_molecular_weight']:.1f}")

        if r.get("expected_yield"):
            print(f"  Exp. yield:     {r['expected_yield']:.0%}")

        # Score breakdown
        print(f"  Breakdown:")
        print(f"    Model conf:   {bd.get('model_score', 0):.2f}  (x0.30)")
        print(f"    Plausibility: {bd.get('plausibility', 0):.2f}  (x0.25)")
        print(f"    Buyability:   {bd.get('buyability', 0):.0%} ({buyable_count}/{s['num_reactants']})  (x0.20)")
        print(f"    Simplicity:   {bd.get('simplicity', 0):.2f}  (x0.15)")
        print(f"    Efficiency:   {bd.get('efficiency', 0):.2f}  (x0.10)")
        print(f"  ═══ FINAL SCORE: {s['precursor_score']:.4f} / 1.00 ═══")

    # Comparison table
    print(f"\n  {'─'*66}")
    print(f"  COMPARISON TABLE (all {len(scored)} candidates)")
    print(f"  {'─'*66}")
    print(f"  {'#':<4}{'Source':<12}{'Score':>8}{'Buy%':>7}{'Atoms':>7}{'#React':>7}  Description")
    print(f"  {'─'*66}")
    for i, r in enumerate(scored):
        s = r["scoring"]
        desc = r.get("description", "")
        if len(desc) > 30:
            desc = desc[:27] + "..."
        marker = " <<<" if i < top_n else ""
        print(
            f"  {i+1:<4}"
            f"{r.get('source','?'):<12}"
            f"{s['precursor_score']:>8.4f}"
            f"{s['buyability_ratio']:>6.0%} "
            f"{s['total_heavy_atoms']:>6}"
            f"{s['num_reactants']:>7}"
            f"  {desc}{marker}"
        )


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--all":
            molecules = TEST_MOLECULES
        elif arg in TEST_MOLECULES:
            molecules = {arg: TEST_MOLECULES[arg]}
        else:
            molecules = {"custom": arg}
    else:
        molecules = {"aspirin": TEST_MOLECULES["aspirin"]}

    for name, smiles in molecules.items():
        run_test(name, smiles)

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
