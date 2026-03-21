#!/usr/bin/env python3
"""Build SQLite index from ORD protobuf data for fast local search.

Parses all ORD .pb.gz files, extracts reaction SMILES, reactants,
products, yields, conditions, and stores them in a SQLite database.
"""

import gzip
import sqlite3
import sys
import os
import time
from pathlib import Path

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ord_schema.proto import dataset_pb2, reaction_pb2

ORD_DATA_DIR = Path("/opt/projects/ord-data/data")
DB_PATH = Path("/opt/projects/chemist-agent/data/ord_reactions.db")


def create_db(conn: sqlite3.Connection):
    """Create tables."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS reactions (
            id TEXT PRIMARY KEY,
            dataset_id TEXT,
            reaction_smiles TEXT,
            yield_pct REAL,
            temperature TEXT,
            solvent TEXT,
            catalyst TEXT
        );

        CREATE TABLE IF NOT EXISTS components (
            reaction_id TEXT,
            smiles TEXT,
            role TEXT,
            FOREIGN KEY (reaction_id) REFERENCES reactions(id)
        );

        CREATE INDEX IF NOT EXISTS idx_comp_smiles ON components(smiles);
        CREATE INDEX IF NOT EXISTS idx_comp_role ON components(role);
        CREATE INDEX IF NOT EXISTS idx_rxn_smiles ON reactions(reaction_smiles);
    """)


def extract_reaction(rxn, dataset_id: str) -> tuple[dict, list[dict]] | None:
    """Extract reaction data from protobuf message."""
    rxn_id = rxn.reaction_id
    if not rxn_id:
        return None

    reactants = []
    products = []
    solvents = []
    catalysts = []

    for key, inp in rxn.inputs.items():
        for comp in inp.components:
            role_num = comp.reaction_role
            smiles = ""
            for ident in comp.identifiers:
                if ident.type == ident.SMILES:
                    smiles = ident.value
                    break
            if not smiles:
                continue

            # Role types: 0=UNSPECIFIED, 1=REACTANT, 2=REAGENT, 3=SOLVENT, 4=CATALYST,
            # 5=WORKUP, 6=INTERNAL_STANDARD, 7=AUTHENTIC_STANDARD, 8=PRODUCT
            if role_num == 1:
                reactants.append(smiles)
            elif role_num == 2:
                reactants.append(smiles)  # treat reagent as reactant
            elif role_num == 3:
                solvents.append(smiles)
            elif role_num == 4:
                catalysts.append(smiles)

    for outcome in rxn.outcomes:
        for prod in outcome.products:
            for ident in prod.identifiers:
                if ident.type == ident.SMILES:
                    products.append(ident.value)
                    break

    if not reactants or not products:
        return None

    # Yield
    yield_pct = None
    for outcome in rxn.outcomes:
        for prod in outcome.products:
            for meas in prod.measurements:
                if meas.type == meas.YIELD:
                    yield_pct = meas.percentage.value
                    break
            if yield_pct is not None:
                break

    # Temperature
    temp_str = None
    if rxn.conditions.HasField("temperature"):
        tc = rxn.conditions.temperature
        if tc.HasField("setpoint"):
            val = tc.setpoint.value
            units = tc.setpoint.units
            unit_name = reaction_pb2.Temperature.TemperatureUnit.Name(units) if units else ""
            temp_str = f"{val} {unit_name}".strip()

    # Reaction SMILES
    rxn_smiles = ".".join(reactants) + ">>" + ".".join(products)

    reaction_data = {
        "id": rxn_id,
        "dataset_id": dataset_id,
        "reaction_smiles": rxn_smiles,
        "yield_pct": yield_pct,
        "temperature": temp_str,
        "solvent": solvents[0] if solvents else None,
        "catalyst": catalysts[0] if catalysts else None,
    }

    components = []
    for smi in reactants:
        components.append({"reaction_id": rxn_id, "smiles": smi, "role": "reactant"})
    for smi in products:
        components.append({"reaction_id": rxn_id, "smiles": smi, "role": "product"})
    for smi in solvents:
        components.append({"reaction_id": rxn_id, "smiles": smi, "role": "solvent"})
    for smi in catalysts:
        components.append({"reaction_id": rxn_id, "smiles": smi, "role": "catalyst"})

    return reaction_data, components


def build_index():
    """Build the SQLite index from all ORD protobuf files."""
    pb_files = sorted(ORD_DATA_DIR.rglob("*.pb.gz"))
    print(f"Found {len(pb_files)} ORD dataset files")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    create_db(conn)

    total_rxns = 0
    total_components = 0
    start = time.time()

    for i, pb_file in enumerate(pb_files):
        dataset_id = pb_file.stem  # e.g. ord_dataset-bf316bf...

        try:
            with gzip.open(pb_file, "rb") as f:
                ds = dataset_pb2.Dataset()
                ds.ParseFromString(f.read())
        except Exception as e:
            print(f"  [!] Error reading {pb_file.name}: {e}")
            continue

        batch_rxns = []
        batch_comps = []

        for rxn in ds.reactions:
            result = extract_reaction(rxn, dataset_id)
            if result is None:
                continue
            rxn_data, comp_data = result
            batch_rxns.append(rxn_data)
            batch_comps.extend(comp_data)

        # Bulk insert
        if batch_rxns:
            conn.executemany(
                "INSERT OR IGNORE INTO reactions VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(r["id"], r["dataset_id"], r["reaction_smiles"],
                  r["yield_pct"], r["temperature"], r["solvent"], r["catalyst"])
                 for r in batch_rxns],
            )
            conn.executemany(
                "INSERT INTO components VALUES (?, ?, ?)",
                [(c["reaction_id"], c["smiles"], c["role"]) for c in batch_comps],
            )
            conn.commit()

        total_rxns += len(batch_rxns)
        total_components += len(batch_comps)

        if (i + 1) % 50 == 0 or (i + 1) == len(pb_files):
            elapsed = time.time() - start
            rate = total_rxns / elapsed if elapsed > 0 else 0
            print(
                f"  [{i+1}/{len(pb_files)}] "
                f"{total_rxns:,} reactions, {total_components:,} components "
                f"({rate:.0f} rxn/s)"
            )

    # Create canonical SMILES index for product search
    print("\nBuilding canonical product index...")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS product_index (
            canonical_smiles TEXT,
            reaction_id TEXT,
            FOREIGN KEY (reaction_id) REFERENCES reactions(id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prod_canon ON product_index(canonical_smiles)")

    # Populate using RDKit canonical SMILES
    try:
        from rdkit import Chem
        cursor = conn.execute(
            "SELECT reaction_id, smiles FROM components WHERE role = 'product'"
        )
        batch = []
        canon_count = 0
        for rxn_id, smi in cursor:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                canon = Chem.MolToSmiles(mol, isomericSmiles=True)
                batch.append((canon, rxn_id))
                canon_count += 1
                if len(batch) >= 10000:
                    conn.executemany(
                        "INSERT INTO product_index VALUES (?, ?)", batch
                    )
                    conn.commit()
                    batch = []
        if batch:
            conn.executemany("INSERT INTO product_index VALUES (?, ?)", batch)
            conn.commit()
        print(f"  Indexed {canon_count:,} canonical product SMILES")
    except ImportError:
        print("  [!] RDKit not available, skipping canonical index")

    conn.close()
    elapsed = time.time() - start
    db_size = DB_PATH.stat().st_size / (1024 * 1024)
    print(f"\nDone! {total_rxns:,} reactions indexed in {elapsed:.1f}s")
    print(f"Database: {DB_PATH} ({db_size:.1f} MB)")


if __name__ == "__main__":
    build_index()
