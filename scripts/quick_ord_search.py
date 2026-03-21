#!/usr/bin/env python3
"""Quick ORD search: scan protobuf files directly, no index needed.
Then score results with retro_scorer.
"""
import gzip
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rdkit import Chem
from ord_schema.proto import dataset_pb2

ORD_DATA = Path("/opt/projects/ord-data/data")

# Molecules to test
TARGETS = {
    "aspirin": "CC(=O)Oc1ccccc1C(O)=O",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(O)=O",
}


def search_ord_product(target_smiles, max_files=200, max_results=15):
    """Scan ORD protobuf files for reactions producing the target."""
    mol = Chem.MolFromSmiles(target_smiles)
    if not mol:
        return []
    canon = Chem.MolToSmiles(mol, isomericSmiles=True)

    pb_files = sorted(ORD_DATA.rglob("*.pb.gz"))
    found = []
    seen = set()

    for pf in pb_files[:max_files]:
        try:
            with gzip.open(pf, "rb") as f:
                ds = dataset_pb2.Dataset()
                ds.ParseFromString(f.read())
        except Exception:
            continue

        for rxn in ds.reactions:
            for outcome in rxn.outcomes:
                for prod in outcome.products:
                    for ident in prod.identifiers:
                        if ident.type != ident.SMILES:
                            continue
                        pmol = Chem.MolFromSmiles(ident.value)
                        if not pmol:
                            continue
                        pcan = Chem.MolToSmiles(pmol, isomericSmiles=True)
                        if pcan != canon:
                            continue

                        # Found a match — extract reactants
                        reactants = []
                        solvents = []
                        catalysts = []
                        for key, inp in rxn.inputs.items():
                            for comp in inp.components:
                                smi = ""
                                for ci in comp.identifiers:
                                    if ci.type == ci.SMILES:
                                        smi = ci.value
                                        break
                                if not smi:
                                    continue
                                if comp.reaction_role in (1, 2):  # REACTANT, REAGENT
                                    reactants.append(smi)
                                elif comp.reaction_role == 3:  # SOLVENT
                                    solvents.append(smi)
                                elif comp.reaction_role == 4:  # CATALYST
                                    catalysts.append(smi)

                        # Yield
                        yld = None
                        for m in prod.measurements:
                            if m.type == m.YIELD:
                                yld = m.percentage.value

                        if not reactants:
                            continue

                        rxn_smi = ".".join(reactants) + ">>" + pcan
                        # Deduplicate by canonical reactant set
                        r_key = tuple(sorted(
                            Chem.MolToSmiles(Chem.MolFromSmiles(r), isomericSmiles=True)
                            for r in reactants if Chem.MolFromSmiles(r)
                        ))
                        if r_key in seen:
                            continue
                        seen.add(r_key)

                        found.append({
                            "reactants": ".".join(reactants),
                            "reaction_smiles": rxn_smi,
                            "score": 0.85,
                            "source": "ord",
                            "plausibility": 0.90,
                            "expected_yield": yld / 100.0 if yld else None,
                            "description": f"ORD {rxn.reaction_id[:20]}",
                            "solvent": solvents[0] if solvents else None,
                            "catalyst": catalysts[0] if catalysts else None,
                        })
                        if len(found) >= max_results:
                            return found
    return found


def run(name, smiles):
    from src.tools.banned_filter import check_smiles_banned, check_reaction_banned
    from src.tools.retro_scorer import score_precursor_set

    print(f"\n{'='*70}")
    print(f"  {name.upper()}: {smiles}")
    print(f"{'='*70}")

    print("\n[1] Scanning ORD protobuf files (may take 30-60s)...")
    t0 = time.time()
    results = search_ord_product(smiles, max_files=546, max_results=20)
    elapsed = time.time() - t0
    print(f"    Found {len(results)} unique routes in {elapsed:.1f}s")

    if not results:
        print("    No ORD data found for this molecule.")
        return

    # Filter banned
    print(f"\n[2] Filtering banned...")
    filtered = []
    for r in results:
        banned = False
        for smi in r["reactants"].split("."):
            smi = smi.strip()
            if not smi:
                continue
            ban = check_smiles_banned(smi)
            if ban and ban.get("danger_level") in ("critical", "high"):
                print(f"    BLOCKED: {ban.get('name')}")
                banned = True
                break
        rxn = r.get("reaction_smiles", "")
        if not banned and rxn:
            ban = check_reaction_banned(rxn)
            if ban:
                print(f"    BLOCKED rxn: {ban.get('name')}")
                banned = True
        if not banned:
            filtered.append(r)

    print(f"    {len(filtered)} passed, {len(results) - len(filtered)} blocked")

    # Score
    print(f"\n[3] Scoring...")
    for r in filtered:
        s = score_precursor_set(r["reactants"], r["score"], r["plausibility"])
        r["scoring"] = s
        r["final_score"] = s["precursor_score"]

    filtered.sort(key=lambda x: x["final_score"], reverse=True)

    # Display top 3
    top = min(3, len(filtered))
    print(f"\n{'='*70}")
    print(f"  TOP {top} ORD ROUTES FOR {name.upper()}")
    print(f"{'='*70}")

    for i, r in enumerate(filtered[:top]):
        s = r["scoring"]
        bd = s.get("breakdown", {})
        yld = r.get("expected_yield")
        yld_str = f"{yld:.0%}" if yld else "N/A"

        print(f"\n  #{i+1}  [ORD]  yield={yld_str}  {r.get('description','')}")
        print(f"  {'─'*60}")
        print(f"  Reactants: {r['reactants'][:80]}")
        rxn = r.get("reaction_smiles", "")
        if len(rxn) > 80:
            rxn = rxn[:77] + "..."
        print(f"  Reaction:  {rxn}")
        if r.get("solvent"):
            print(f"  Solvent:   {r['solvent']}")
        if r.get("catalyst"):
            print(f"  Catalyst:  {r['catalyst']}")

        print(f"  Score breakdown:")
        print(f"    Model:       {bd.get('model_score',0):.2f} (x0.30)")
        print(f"    Plausibil:   {bd.get('plausibility',0):.2f} (x0.25)")
        print(f"    Buyability:  {bd.get('buyability',0):.0%}  (x0.20)")
        print(f"    Simplicity:  {bd.get('simplicity',0):.2f} (x0.15)")
        print(f"    Efficiency:  {bd.get('efficiency',0):.2f} (x0.10)")
        print(f"  ═══ SCORE: {s['precursor_score']:.4f} / 1.00 ═══")

    # Full table
    print(f"\n  {'─'*70}")
    print(f"  ALL {len(filtered)} ROUTES")
    print(f"  {'─'*70}")
    print(f"  {'#':<4}{'Score':>7}{'Yield':>7}{'Buy%':>6}{'Atoms':>6}{'#R':>4}  Reactants")
    print(f"  {'─'*70}")
    for i, r in enumerate(filtered):
        s = r["scoring"]
        yld = r.get("expected_yield")
        yld_s = f"{yld:.0%}" if yld else " N/A"
        reactants_short = r["reactants"][:40]
        if len(r["reactants"]) > 40:
            reactants_short += "..."
        mark = " <<<" if i < top else ""
        print(
            f"  {i+1:<4}"
            f"{s['precursor_score']:>7.4f}"
            f"{yld_s:>7}"
            f"{s['buyability_ratio']:>5.0%} "
            f"{s['total_heavy_atoms']:>5}"
            f"{s['num_reactants']:>4}"
            f"  {reactants_short}{mark}"
        )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg in TARGETS:
            run(arg, TARGETS[arg])
        elif arg == "--all":
            for name, smi in TARGETS.items():
                run(name, smi)
        else:
            run("custom", arg)
    else:
        run("aspirin", TARGETS["aspirin"])
