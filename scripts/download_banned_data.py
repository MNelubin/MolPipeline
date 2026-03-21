#!/usr/bin/env python3
"""Download and unify banned/dangerous chemicals and reactions from open sources.

Sources for chemicals:
    1. Costanzi Research — CWC Schedules (nerve agents, vesicants, precursors)
    2. Costanzi Research — Australia Group precursors (novichok precursors)
    3. EU Regulation 2019/1148 — Explosive precursors (Annex I + II)
    4. DEA List I / List II — Narcotics precursors
    5. PubChem — Additional hazardous chemicals via GHS classification

Sources for reactions:
    1. Known dangerous reaction SMARTS patterns (CWC, explosive, narcotics)
    2. RDKit PAINS/BRENK filters (structural alerts)

Output format (JSON):
    chemicals -> data/banned_chemicals.json
    reactions -> data/banned_reactions.json

Each chemical entry:
    {
        "smiles": "...",
        "canonical_smiles": "...",
        "name": "...",
        "cas": "...",
        "category": "cwc_schedule_1 | explosive_precursor | narcotics_precursor | ...",
        "danger_level": "critical | high | medium | low",
        "source": "costanzi_cwc | eu_2019_1148 | dea_list_1 | ...",
        "description": "reason why banned or dangerous",
        "schedule": "CWC 1A | CWC 2B | DEA List I | EU Annex I | ...",
        "pubchem_cid": 12345 or null
    }

danger_level mapping:
    critical — Chemical weapons, nerve agents, Schedule 1 CWC -> always block
    high     — Direct precursors to weapons/explosives/narcotics -> block by default
    medium   — Dual-use chemicals, Schedule 3 CWC, DEA List II -> warn
    low      — Reportable/monitored substances -> inform
"""

import csv
import io
import json
import logging
import sys
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# RDKit helpers (optional — graceful degradation)
# ---------------------------------------------------------------------------

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    logger.warning("RDKit not installed — SMILES canonicalization disabled")


def canonicalize(smiles: str) -> str | None:
    """Canonicalize SMILES. Returns None if invalid."""
    if not smiles or smiles == "---":
        return None
    if not HAS_RDKIT:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


# ---------------------------------------------------------------------------
# Danger level assignment
# ---------------------------------------------------------------------------

CATEGORY_DANGER = {
    # CWC
    "cwc_schedule_1": "critical",
    "cwc_schedule_1_example": "critical",
    "cwc_schedule_1_precursor": "critical",
    "cwc_schedule_2": "high",
    "cwc_schedule_2_precursor": "high",
    "cwc_schedule_3": "medium",
    "cwc_schedule_3_precursor": "medium",
    # Australia Group
    "australia_group": "high",
    # Explosives
    "eu_annex_i_explosive": "high",
    "eu_annex_ii_reportable": "medium",
    "explosive_precursor": "high",
    # Narcotics
    "dea_list_1": "high",
    "dea_list_2": "medium",
    "narcotics_precursor": "high",
    # Environment
    "stockholm_convention": "medium",
    # Highly toxic
    "highly_toxic": "high",
}


def assign_danger(category: str) -> str:
    return CATEGORY_DANGER.get(category, "medium")


# ---------------------------------------------------------------------------
# Source 1: Costanzi Research — CWC Schedules
# ---------------------------------------------------------------------------

CWC_CSV_URL = "https://costanziresearch.com/wp-content/uploads/costanzi_research_cwc_schedules.csv"


def _cwc_entry_to_category(entry_number: str, entry_type: str, cwc_category: str) -> str:
    """Map CWC entry to our category system."""
    entry_upper = entry_number.upper()

    if "1A" in entry_upper or "1B" in entry_upper:
        if entry_type in ("Family", "Individual", "Protein"):
            if "precursor" in cwc_category.lower():
                return "cwc_schedule_1_precursor"
            return "cwc_schedule_1"
        elif entry_type == "Family_Example":
            if "precursor" in cwc_category.lower():
                return "cwc_schedule_1_precursor"
            return "cwc_schedule_1_example"
    elif "2A" in entry_upper or "2B" in entry_upper:
        if "precursor" in cwc_category.lower():
            return "cwc_schedule_2_precursor"
        return "cwc_schedule_2"
    elif "3A" in entry_upper or "3B" in entry_upper:
        if "precursor" in cwc_category.lower():
            return "cwc_schedule_3_precursor"
        return "cwc_schedule_3"

    # Fallback based on category text
    cat_lower = cwc_category.lower()
    if any(x in cat_lower for x in ("nerve", "vesicant", "choking", "incapacitant", "toxin", "novichok", "carbamate")):
        return "cwc_schedule_1"
    if "precursor" in cat_lower:
        return "cwc_schedule_2_precursor"
    return "cwc_schedule_2"


def download_cwc_chemicals() -> list[dict]:
    """Download CWC scheduled chemicals from Costanzi Research."""
    logger.info("Downloading CWC Schedules from Costanzi Research...")
    resp = httpx.get(CWC_CSV_URL, timeout=30.0, follow_redirects=True)
    resp.raise_for_status()

    reader = csv.DictReader(io.StringIO(resp.text))
    chemicals = []

    for row in reader:
        smiles = row.get("SMILES", "").strip()
        if not smiles or smiles == "---":
            continue

        canonical = canonicalize(smiles)
        if canonical is None:
            logger.debug(f"Skipping invalid SMILES: {smiles[:50]}")
            continue

        entry_number = row.get("Entry_Number", "")
        entry_type = row.get("Entry_Type", "")
        cwc_category = row.get("Category", "")
        category = _cwc_entry_to_category(entry_number, entry_type, cwc_category)

        cid_raw = row.get("CID", "")
        try:
            cid = int(cid_raw) if cid_raw and cid_raw != "---" else None
        except ValueError:
            cid = None

        chemicals.append({
            "smiles": smiles,
            "canonical_smiles": canonical,
            "name": row.get("Name", "").strip(),
            "cas": row.get("CAS", "").strip() or None,
            "category": category,
            "danger_level": assign_danger(category),
            "source": "costanzi_cwc",
            "description": f"CWC {cwc_category} — {entry_number} ({entry_type})",
            "schedule": entry_number,
            "pubchem_cid": cid,
        })

    logger.info(f"  CWC: {len(chemicals)} chemicals loaded")
    return chemicals


# ---------------------------------------------------------------------------
# Source 2: Costanzi Research — Australia Group Precursors
# ---------------------------------------------------------------------------

AG_CSV_URL = "https://costanziresearch.com/wp-content/uploads/costanzi_research_ag_precursors_update.csv"


def download_ag_chemicals() -> list[dict]:
    """Download Australia Group novichok precursors from Costanzi Research."""
    logger.info("Downloading Australia Group precursors from Costanzi Research...")
    try:
        resp = httpx.get(AG_CSV_URL, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning(f"  AG download failed: {e}")
        return []

    reader = csv.DictReader(io.StringIO(resp.text))
    chemicals = []

    for row in reader:
        smiles = row.get("SMILES", "").strip()
        if not smiles or smiles == "---":
            continue

        canonical = canonicalize(smiles)
        if canonical is None:
            continue

        cid_raw = row.get("CID", "")
        try:
            cid = int(cid_raw) if cid_raw and cid_raw != "---" else None
        except ValueError:
            cid = None

        chemicals.append({
            "smiles": smiles,
            "canonical_smiles": canonical,
            "name": row.get("Name", "").strip(),
            "cas": row.get("CAS", "").strip() or None,
            "category": "australia_group",
            "danger_level": "high",
            "source": "costanzi_ag",
            "description": f"Australia Group precursor — {row.get('Entry_Number', '')}",
            "schedule": row.get("Entry_Number", ""),
            "pubchem_cid": cid,
        })

    logger.info(f"  AG: {len(chemicals)} chemicals loaded")
    return chemicals


# ---------------------------------------------------------------------------
# Source 3: EU Regulation 2019/1148 — Explosive Precursors
# ---------------------------------------------------------------------------

# Annex I: restricted (cannot be sold to general public above thresholds)
# Annex II: reportable (suspicious transactions must be reported)
EU_EXPLOSIVE_PRECURSORS = [
    # Annex I — Restricted
    {"smiles": "OO", "name": "Hydrogen peroxide", "cas": "7722-84-1",
     "category": "eu_annex_i_explosive", "schedule": "EU Annex I",
     "description": "Restricted above 12% w/w. Used in TATP/HMTD synthesis."},
    {"smiles": "C[N+](=O)[O-]", "name": "Nitromethane", "cas": "75-52-5",
     "category": "eu_annex_i_explosive", "schedule": "EU Annex I",
     "description": "Restricted above 16% w/w. Explosive fuel component."},
    {"smiles": "O=[N+]([O-])O", "name": "Nitric acid", "cas": "7697-37-2",
     "category": "eu_annex_i_explosive", "schedule": "EU Annex I",
     "description": "Restricted above 3% w/w. Used in nitroglycerin/nitrocellulose synthesis."},
    {"smiles": "[K+].[O-]Cl(=O)=O", "name": "Potassium chlorate", "cas": "3811-04-9",
     "category": "eu_annex_i_explosive", "schedule": "EU Annex I",
     "description": "Restricted above 40% w/w. Oxidizer for improvised explosives."},
    {"smiles": "[K+].[O-]Cl(=O)(=O)=O", "name": "Potassium perchlorate", "cas": "7778-74-7",
     "category": "eu_annex_i_explosive", "schedule": "EU Annex I",
     "description": "Restricted above 40% w/w. Powerful oxidizer for pyrotechnics/explosives."},
    {"smiles": "[Na+].[O-]Cl(=O)=O", "name": "Sodium chlorate", "cas": "7775-09-9",
     "category": "eu_annex_i_explosive", "schedule": "EU Annex I",
     "description": "Restricted above 40% w/w. Oxidizer."},
    {"smiles": "[Na+].[O-]Cl(=O)(=O)=O", "name": "Sodium perchlorate", "cas": "7601-89-0",
     "category": "eu_annex_i_explosive", "schedule": "EU Annex I",
     "description": "Restricted above 40% w/w. Oxidizer."},
    {"smiles": "[NH4+].[O-][N+]([O-])=O", "name": "Ammonium nitrate", "cas": "6484-52-2",
     "category": "eu_annex_i_explosive", "schedule": "EU Annex I",
     "description": "Restricted above 16% w/w nitrogen. ANFO explosive component."},
    # Annex II — Reportable
    {"smiles": "C1CN2CN3CN(C1)C2C3", "name": "Hexamethylenetetramine (hexamine)", "cas": "100-97-0",
     "category": "eu_annex_ii_reportable", "schedule": "EU Annex II",
     "description": "Reportable. Precursor to RDX (cyclotrimethylenetrinitramine)."},
    {"smiles": "CC(C)=O", "name": "Acetone", "cas": "67-64-1",
     "category": "eu_annex_ii_reportable", "schedule": "EU Annex II",
     "description": "Reportable. Used in TATP synthesis and narcotics manufacturing."},
    {"smiles": "[K+].[O-][N+]([O-])=O", "name": "Potassium nitrate", "cas": "7757-79-1",
     "category": "eu_annex_ii_reportable", "schedule": "EU Annex II",
     "description": "Reportable. Black powder component."},
    {"smiles": "[Na+].[O-][N+]([O-])=O", "name": "Sodium nitrate", "cas": "7631-99-4",
     "category": "eu_annex_ii_reportable", "schedule": "EU Annex II",
     "description": "Reportable. Oxidizer for explosives."},
    {"smiles": "[Ca+2].[O-][N+]([O-])=O.[O-][N+]([O-])=O", "name": "Calcium nitrate", "cas": "10124-37-5",
     "category": "eu_annex_ii_reportable", "schedule": "EU Annex II",
     "description": "Reportable. Fertilizer/oxidizer."},
    {"smiles": "OS(O)(=O)=O", "name": "Sulfuric acid", "cas": "7664-93-9",
     "category": "eu_annex_ii_reportable", "schedule": "EU Annex II",
     "description": "Reportable. Used in nitration reactions for explosives."},
    {"smiles": "[Mg+2].[O-][N+]([O-])=O.[O-][N+]([O-])=O", "name": "Magnesium nitrate hexahydrate",
     "cas": "13446-18-9", "category": "eu_annex_ii_reportable", "schedule": "EU Annex II",
     "description": "Reportable. Oxidizer."},
]


def get_eu_explosive_precursors() -> list[dict]:
    """Return EU 2019/1148 explosive precursors (hardcoded from regulation)."""
    logger.info("Loading EU explosive precursors (Regulation 2019/1148)...")
    chemicals = []
    for entry in EU_EXPLOSIVE_PRECURSORS:
        canonical = canonicalize(entry["smiles"])
        chemicals.append({
            "smiles": entry["smiles"],
            "canonical_smiles": canonical or entry["smiles"],
            "name": entry["name"],
            "cas": entry["cas"],
            "category": entry["category"],
            "danger_level": assign_danger(entry["category"]),
            "source": "eu_2019_1148",
            "description": entry["description"],
            "schedule": entry["schedule"],
            "pubchem_cid": None,
        })
    logger.info(f"  EU: {len(chemicals)} chemicals loaded")
    return chemicals


# ---------------------------------------------------------------------------
# Source 4: DEA List I / List II
# ---------------------------------------------------------------------------

DEA_CHEMICALS = [
    # List I — Important precursors to controlled substances
    {"smiles": "C[C@@H](O)c1ccc(O)cc1", "name": "Synephrine / 4-Hydroxyephedrine",
     "cas": "94-07-5", "list": "1"},
    {"smiles": "OC(c1ccccc1)C(NC)C", "name": "Pseudoephedrine",
     "cas": "90-82-4", "list": "1"},
    {"smiles": "OC(c1ccccc1)C(N)C", "name": "Phenylpropanolamine (norpseudoephedrine)",
     "cas": "14838-15-4", "list": "1"},
    {"smiles": "CC(=O)Cc1ccccc1", "name": "Phenylacetone (P2P)", "cas": "103-79-7", "list": "1",
     "description": "Key precursor to amphetamine and methamphetamine."},
    {"smiles": "OC(=O)Cc1ccccc1", "name": "Phenylacetic acid", "cas": "103-82-2", "list": "1",
     "description": "Precursor to phenylacetone and amphetamines."},
    {"smiles": "N#CCc1ccccc1", "name": "Phenylacetonitrile (benzyl cyanide)",
     "cas": "140-29-4", "list": "1",
     "description": "Precursor to phenylacetic acid and amphetamines."},
    {"smiles": "NCCc1ccc2[nH]c3ccccc3c2c1", "name": "Tryptamine",
     "cas": "61-54-1", "list": "1",
     "description": "Precursor to DMT and psilocybin analogues."},
    {"smiles": "CC(=O)c1ccc2c(c1)OCO2", "name": "3,4-MDP2P (PMK)",
     "cas": "4676-39-5", "list": "1",
     "description": "Direct precursor to MDMA (ecstasy)."},
    {"smiles": "OC(=O)c1ccc2c(c1)OCO2", "name": "Piperonylic acid",
     "cas": "94-53-1", "list": "1",
     "description": "Precursor to MDMA synthesis."},
    {"smiles": "C=CCc1ccc2c(c1)OCO2", "name": "Safrole",
     "cas": "94-59-7", "list": "1",
     "description": "Natural precursor to MDMA. Found in sassafras oil."},
    {"smiles": "C=CCc1ccc(OC)c(OC)c1", "name": "Methyleugenol",
     "cas": "93-15-2", "list": "1",
     "description": "Precursor to amphetamine-type stimulants."},
    {"smiles": "C/C=C/c1ccc(OC)c(OC)c1", "name": "Isomethyleugenol",
     "cas": "93-16-3", "list": "1",
     "description": "Precursor in MDMA-type synthesis."},
    {"smiles": "C/C=C\\c1ccc2c(c1)OCO2", "name": "Isosafrole",
     "cas": "120-58-1", "list": "1",
     "description": "Isomer of safrole, precursor to MDMA."},
    {"smiles": "COc1cc(CC=O)ccc1OC", "name": "Homoveratraldehyde",
     "cas": "120-14-9", "list": "1",
     "description": "Precursor to mescaline analogues."},
    {"smiles": "OC(=O)/C=C/c1ccccc1", "name": "trans-Cinnamic acid",
     "cas": "140-10-3", "list": "1",
     "description": "Precursor to phenylacetone via decarboxylation."},
    {"smiles": "C1C2CC3CC(CC1C3)C2", "name": "Adamantane",
     "cas": "281-23-2", "list": "1",
     "description": "Precursor to amantadine and related drugs."},
    {"smiles": "OCC1OC(O)C(O)C(O)C1O", "name": "D-Glucose / Dextrose (precursor context)",
     "cas": "50-99-7", "list": "1",
     "description": "List I when used for illicit drug dilution."},
    {"smiles": "CC(=O)OC(CC(=O)[O-])C(=O)[O-]", "name": "Citric acid derivative context",
     "cas": "77-92-9", "list": "1",
     "description": "List I chemical for heroin preparation."},
    {"smiles": "C(#N)c1ccccc1", "name": "Benzonitrile (context)",
     "cas": "100-47-0", "list": "1",
     "description": "Fentanyl synthesis precursor pathway."},
    {"smiles": "O=CC1=CC2=CC=CC=C2N1C", "name": "N-Methylisatoic anhydride",
     "cas": "10328-92-4", "list": "1",
     "description": "Methaqualone precursor."},
    {"smiles": "OC(c1ccccc1)(c1ccccc1)C1CCNCC1", "name": "4-Piperidinol, 4,4-diphenyl- (azacyclonol context)",
     "cas": "115-46-8", "list": "1",
     "description": "Fentanyl family synthesis intermediate."},
    {"smiles": "O=C(Cl)c1ccccc1", "name": "Benzoyl chloride",
     "cas": "98-88-4", "list": "1",
     "description": "Precursor in fentanyl synthesis (benzoylation step)."},
    {"smiles": "ClCCN1CCCC1", "name": "1-(2-Chloroethyl)pyrrolidine",
     "cas": "5765-40-2", "list": "1",
     "description": "Fentanyl precursor (Janssen synthesis)."},
    {"smiles": "C1CCNCC1", "name": "Piperidine", "cas": "110-89-4", "list": "1",
     "description": "Fentanyl family precursor. CWC Schedule 3 / DEA List I."},
    # List II — Reagents/solvents used in illicit manufacturing
    {"smiles": "CC(C)=O", "name": "Acetone", "cas": "67-64-1", "list": "2",
     "description": "Solvent in cocaine/heroin processing. Also EU reportable."},
    {"smiles": "CCOCC", "name": "Diethyl ether", "cas": "60-29-7", "list": "2",
     "description": "Extraction solvent in narcotics manufacturing."},
    {"smiles": "Cl", "name": "Hydrochloric acid (HCl gas)", "cas": "7647-01-0", "list": "2",
     "description": "Salt formation in drug manufacturing."},
    {"smiles": "OS(O)(=O)=O", "name": "Sulfuric acid", "cas": "7664-93-9", "list": "2",
     "description": "Cocaine/heroin processing. Also EU reportable."},
    {"smiles": "Cc1ccccc1", "name": "Toluene", "cas": "108-88-3", "list": "2",
     "description": "Solvent in cocaine/narcotics processing."},
    {"smiles": "CCC(C)=O", "name": "Methyl ethyl ketone (MEK)", "cas": "78-93-3", "list": "2",
     "description": "Solvent in narcotics manufacturing."},
    {"smiles": "ClC(Cl)Cl", "name": "Chloroform", "cas": "67-66-3", "list": "2",
     "description": "Extraction solvent in drug processing."},
    {"smiles": "ClCCl", "name": "Dichloromethane (DCM)", "cas": "75-09-2", "list": "2",
     "description": "Extraction solvent. Common lab reagent."},
    {"smiles": "CC(=O)OCC", "name": "Ethyl acetate", "cas": "141-78-6", "list": "2",
     "description": "Extraction solvent in narcotics processing."},
    {"smiles": "CC(O)C", "name": "Isopropanol", "cas": "67-63-0", "list": "2",
     "description": "Solvent and reagent in drug manufacturing."},
    {"smiles": "CCCCCC", "name": "n-Hexane", "cas": "110-54-3", "list": "2",
     "description": "Extraction solvent in cocaine processing."},
]


def get_dea_chemicals() -> list[dict]:
    """Return DEA List I/II precursor chemicals."""
    logger.info("Loading DEA List I/II chemicals...")
    chemicals = []
    for entry in DEA_CHEMICALS:
        canonical = canonicalize(entry["smiles"])
        dea_list = entry["list"]
        category = f"dea_list_{dea_list}"

        chemicals.append({
            "smiles": entry["smiles"],
            "canonical_smiles": canonical or entry["smiles"],
            "name": entry["name"],
            "cas": entry.get("cas"),
            "category": category,
            "danger_level": assign_danger(category),
            "source": f"dea_list_{dea_list}",
            "description": entry.get("description", f"DEA List {dea_list} regulated chemical"),
            "schedule": f"DEA List {dea_list}",
            "pubchem_cid": None,
        })
    logger.info(f"  DEA: {len(chemicals)} chemicals loaded")
    return chemicals


# ---------------------------------------------------------------------------
# Source 5: Additional highly toxic / environmental bans
# ---------------------------------------------------------------------------

ADDITIONAL_HAZARDOUS = [
    # Extremely toxic — no legitimate small-scale synthesis use
    {"smiles": "O=C(Cl)Cl", "name": "Phosgene", "cas": "75-44-5",
     "category": "highly_toxic", "schedule": "CWC Schedule 3",
     "description": "Chemical weapon. Lethal choking agent. CWC Schedule 3."},
    {"smiles": "C(#N)Cl", "name": "Cyanogen chloride", "cas": "506-77-4",
     "category": "highly_toxic", "schedule": "CWC Schedule 3",
     "description": "Blood agent. CWC Schedule 3."},
    {"smiles": "C(#N)Br", "name": "Cyanogen bromide", "cas": "506-68-3",
     "category": "highly_toxic", "schedule": "N/A",
     "description": "Extremely toxic. Used as chemical weapon historically."},
    {"smiles": "C(=O)(F)F", "name": "Carbonyl difluoride", "cas": "353-50-4",
     "category": "highly_toxic", "schedule": "N/A",
     "description": "Extremely toxic gas. Phosgene analogue."},
    {"smiles": "CN(C)C(=O)Cl", "name": "Dimethylcarbamoyl chloride", "cas": "79-44-7",
     "category": "highly_toxic", "schedule": "CWC Schedule 3",
     "description": "CWC Schedule 3. Potent carcinogen (IARC Group 2A)."},
    {"smiles": "[Tl]O", "name": "Thallium(I) oxide", "cas": "1314-32-5",
     "category": "highly_toxic", "schedule": "N/A",
     "description": "Extreme systemic poison. No antidote."},
    {"smiles": "O=[As]O[As]=O", "name": "Arsenic trioxide", "cas": "1327-53-3",
     "category": "highly_toxic", "schedule": "N/A",
     "description": "Classic poison. IARC Group 1 carcinogen."},
    {"smiles": "[O-][N+](=O)OO[N+](=O)[O-]", "name": "Dinitrogen tetroxide", "cas": "10544-72-6",
     "category": "highly_toxic", "schedule": "N/A",
     "description": "Powerful oxidizer. Extremely toxic and corrosive gas."},
    # Stockholm Convention POPs
    {"smiles": "Clc1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl", "name": "Hexachlorobenzene", "cas": "118-74-1",
     "category": "stockholm_convention", "schedule": "Stockholm Annex A",
     "description": "Persistent organic pollutant. Banned under Stockholm Convention."},
    {"smiles": "ClC(Cl)(Cl)C(Cl)(Cl)Cl", "name": "Hexachloroethane", "cas": "67-72-1",
     "category": "stockholm_convention", "schedule": "Stockholm Annex A",
     "description": "Persistent organic pollutant. Toxic smoke agent."},
    {"smiles": "ClC(=C(c1ccc(Cl)cc1)c1ccc(Cl)cc1)Cl", "name": "DDT", "cas": "50-29-3",
     "category": "stockholm_convention", "schedule": "Stockholm Annex B",
     "description": "Persistent organic pollutant. Globally restricted."},
]


def get_additional_hazardous() -> list[dict]:
    """Return additional hazardous chemicals."""
    logger.info("Loading additional hazardous chemicals...")
    chemicals = []
    for entry in ADDITIONAL_HAZARDOUS:
        canonical = canonicalize(entry["smiles"])
        chemicals.append({
            "smiles": entry["smiles"],
            "canonical_smiles": canonical or entry["smiles"],
            "name": entry["name"],
            "cas": entry.get("cas"),
            "category": entry["category"],
            "danger_level": assign_danger(entry["category"]),
            "source": "curated",
            "description": entry.get("description", ""),
            "schedule": entry.get("schedule", ""),
            "pubchem_cid": None,
        })
    logger.info(f"  Additional: {len(chemicals)} chemicals loaded")
    return chemicals


# ---------------------------------------------------------------------------
# Merge and deduplicate chemicals
# ---------------------------------------------------------------------------

def merge_chemicals(all_sources: list[list[dict]]) -> list[dict]:
    """Merge chemicals from all sources, dedup by canonical SMILES.

    When duplicates found, keep the one with higher danger_level.
    """
    DANGER_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    seen: dict[str, dict] = {}

    for source in all_sources:
        for chem in source:
            key = chem["canonical_smiles"]
            if key is None:
                continue

            if key in seen:
                existing = seen[key]
                # Keep higher danger level
                if DANGER_ORDER.get(chem["danger_level"], 0) > DANGER_ORDER.get(existing["danger_level"], 0):
                    # Merge: keep new but append source info
                    old_desc = existing.get("description", "")
                    new_desc = chem.get("description", "")
                    if old_desc and old_desc not in new_desc:
                        chem["description"] = f"{new_desc} | Also: {old_desc}"
                    chem["also_listed_in"] = existing.get("source", "")
                    seen[key] = chem
                else:
                    # Keep existing but note the additional source
                    old_sources = existing.get("also_listed_in", "")
                    new_source = chem.get("source", "")
                    if new_source and new_source not in old_sources:
                        existing["also_listed_in"] = f"{old_sources}, {new_source}".strip(", ")
            else:
                seen[key] = chem

    return list(seen.values())


# ---------------------------------------------------------------------------
# Banned reaction patterns
# ---------------------------------------------------------------------------

def get_banned_reactions() -> list[dict]:
    """Return banned/dangerous reaction SMARTS patterns."""
    logger.info("Building banned reaction patterns...")

    reactions = [
        # === CRITICAL: Chemical weapons synthesis ===
        {
            "smarts": "[P:1](=O)([F:2])([O:3])[C:4]",
            "name": "Phosphonofluoridate formation (G-series nerve agents)",
            "category": "cwc_synthesis",
            "danger_level": "critical",
            "description": "Product pattern for G-series nerve agents (sarin, soman, tabun). "
                           "Formation of P-F bond in organophosphorus compound.",
        },
        {
            "smarts": "[P:1](=O)([C:2])([F:3])",
            "name": "Alkylphosphonofluoridate",
            "category": "cwc_synthesis",
            "danger_level": "critical",
            "description": "Core sarin/soman structural motif with P-F bond.",
        },
        {
            "smarts": "[P:1](=O)([C:2])([S:3][C:4][C:5][N:6])",
            "name": "V-series nerve agent formation",
            "category": "cwc_synthesis",
            "danger_level": "critical",
            "description": "VX-type nerve agent pattern: P-S-C-C-N chain.",
        },
        {
            "smarts": "[P:1](=[O:2])([N:3]=[C:4]([N:5])[C:6])[F:7]",
            "name": "Novichok-type agent formation",
            "category": "cwc_synthesis",
            "danger_level": "critical",
            "description": "A-series (Novichok) nerve agent pattern: phosphoramidofluoridate "
                           "with amidine group.",
        },
        {
            "smarts": "[S:1]([C:2][C:3][Cl:4])([C:5][C:6][Cl:7])",
            "name": "Sulfur mustard synthesis",
            "category": "cwc_synthesis",
            "danger_level": "critical",
            "description": "Mustard gas (yperite) pattern: bis(2-chloroethyl) sulfide.",
        },
        {
            "smarts": "[N:1]([C:2][C:3][Cl:4])([C:5][C:6][Cl:7])[C:8][C:9][Cl:10]",
            "name": "Nitrogen mustard HN3 synthesis",
            "category": "cwc_synthesis",
            "danger_level": "critical",
            "description": "Tris(2-chloroethyl)amine — nitrogen mustard chemical weapon.",
        },
        {
            "smarts": "[As:1]([Cl:2])([Cl:3])/[CH:4]=[CH:5]/[Cl:6]",
            "name": "Lewisite synthesis",
            "category": "cwc_synthesis",
            "danger_level": "critical",
            "description": "Chlorovinyldichloroarsine (Lewisite) — arsenic-based vesicant.",
        },
        # === HIGH: Explosive synthesis ===
        {
            "smarts": "[C:1]([O:2][N+](=O)[O-])([O:3][N+](=O)[O-])[O:4][N+](=O)[O-]",
            "name": "Trinitroglycerin-type formation",
            "category": "explosive_synthesis",
            "danger_level": "high",
            "description": "Polyol nitrate ester formation. Nitroglycerin pattern.",
        },
        {
            "smarts": "[c:1]([N+](=O)[O-])[c:2]([N+](=O)[O-])[c:3]([N+](=O)[O-])",
            "name": "Trinitroaromatic formation",
            "category": "explosive_synthesis",
            "danger_level": "high",
            "description": "TNT-class explosive: adjacent nitro groups on aromatic ring.",
        },
        {
            "smarts": "[N:1]1([N+](=O)[O-])[C:2][N:3]([N+](=O)[O-])[C:4][N:5]([N+](=O)[O-])[C:6]1",
            "name": "RDX/HMX formation",
            "category": "explosive_synthesis",
            "danger_level": "high",
            "description": "Cyclic nitramine explosive (RDX/HMX) pattern.",
        },
        {
            "smarts": "[O:1]1[O:2][C:3]([C:4])([C:5])[O:6][O:7][C:8]([C:9])([C:10])[O:11][O:12][C:13]([C:14])([C:15])1",
            "name": "TATP (triacetone triperoxide) formation",
            "category": "explosive_synthesis",
            "danger_level": "high",
            "description": "TATP peroxide explosive — cyclic triperoxide.",
        },
        {
            "smarts": "[C:1]([N+](=O)[O-])([N+](=O)[O-])[N+](=O)[O-]",
            "name": "Trinitromethyl formation",
            "category": "explosive_synthesis",
            "danger_level": "high",
            "description": "Gem-trinitro group — extremely powerful explosive moiety.",
        },
        {
            "smarts": "[C:1][N:2]=[N+:3]=[N-:4]",
            "name": "Organic azide formation (bulk)",
            "category": "explosive_synthesis",
            "danger_level": "high",
            "description": "Organic azides: shock-sensitive when low MW or high N/C ratio. "
                           "Note: some azides are legitimate reagents (click chemistry). "
                           "Flag for review rather than auto-block.",
        },
        {
            "smarts": "[C:1](=O)([Cl:2])[Cl:3]",
            "name": "Phosgene formation",
            "category": "cwc_synthesis",
            "danger_level": "critical",
            "description": "Phosgene synthesis — CWC Schedule 3 choking agent.",
        },
        # === HIGH: Narcotics synthesis ===
        {
            "smarts": "[C:1](=[O:2])[C:3]([NH:4])[c:5]1[c:6][c:7][c:8][c:9][c:10]1",
            "name": "Cathinone/methcathinone scaffold",
            "category": "narcotics_synthesis",
            "danger_level": "high",
            "description": "Aminophenone (cathinone class). Schedule I controlled substance scaffold.",
        },
        {
            "smarts": "[c:1]1[c:2][c:3]2[c:4]([c:5][c:6]1)OCO2",
            "name": "Methylenedioxy ring formation",
            "category": "narcotics_synthesis",
            "danger_level": "high",
            "description": "Formation of 3,4-methylenedioxy group — key pharmacophore in MDMA/MDA.",
        },
        {
            "smarts": "[N:1]1([C:2](=O)[c:3])[C:4][C:5][C:6]([c:7])[C:8][C:9]1",
            "name": "4-Anilidopiperidine (fentanyl core)",
            "category": "narcotics_synthesis",
            "danger_level": "critical",
            "description": "Fentanyl core scaffold: N-acyl-4-anilinopiperidine. "
                           "All fentanyl analogues are Schedule I/II. Lethal in microgram doses.",
        },
        # === MEDIUM: Dual-use patterns (warn, don't block) ===
        {
            "smarts": "[C:1][O:2][N+:3](=O)[O-:4]",
            "name": "Nitrate ester (general)",
            "category": "dual_use",
            "danger_level": "medium",
            "description": "Single nitrate ester — used in medicine (GTN) and explosives. "
                           "Flag for review. Block only if multiple nitrate esters on same molecule.",
        },
        {
            "smarts": "[O:1]=[Cl:2](=O)(=O)[O-:3]",
            "name": "Perchlorate formation",
            "category": "dual_use",
            "danger_level": "medium",
            "description": "Perchlorate salts — powerful oxidizer. "
                           "Used legitimately in analytical chemistry but also in explosives.",
        },
    ]

    # Validate SMARTS if RDKit available
    if HAS_RDKIT:
        valid = []
        for rxn in reactions:
            pat = Chem.MolFromSmarts(rxn["smarts"])
            if pat is not None:
                rxn["valid_smarts"] = True
                valid.append(rxn)
            else:
                logger.warning(f"  Invalid SMARTS, keeping anyway: {rxn['name']}")
                rxn["valid_smarts"] = False
                valid.append(rxn)
        reactions = valid

    logger.info(f"  Reactions: {len(reactions)} patterns built")
    return reactions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("Downloading and unifying banned/dangerous chemicals data")
    logger.info("=" * 60)

    # Download / load all chemical sources
    cwc = download_cwc_chemicals()
    ag = download_ag_chemicals()
    eu = get_eu_explosive_precursors()
    dea = get_dea_chemicals()
    additional = get_additional_hazardous()

    # Merge and deduplicate
    all_chemicals = merge_chemicals([cwc, ag, eu, dea, additional])

    # Sort: critical first, then high, medium, low
    danger_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    all_chemicals.sort(key=lambda c: (danger_order.get(c["danger_level"], 9), c["name"]))

    # Stats
    stats = {}
    for c in all_chemicals:
        dl = c["danger_level"]
        stats[dl] = stats.get(dl, 0) + 1

    logger.info("")
    logger.info(f"Total unique chemicals: {len(all_chemicals)}")
    for level in ("critical", "high", "medium", "low"):
        logger.info(f"  {level}: {stats.get(level, 0)}")

    # Save chemicals
    chem_path = DATA_DIR / "banned_chemicals.json"
    chem_output = {
        "_meta": {
            "total": len(all_chemicals),
            "stats": stats,
            "sources": [
                "Costanzi Research CWC Schedules (costanziresearch.com)",
                "Costanzi Research Australia Group Precursors",
                "EU Regulation 2019/1148 (explosive precursors)",
                "DEA List I / List II (narcotics precursors)",
                "Stockholm Convention (persistent organic pollutants)",
                "Curated: highly toxic / CWC Schedule 3",
            ],
            "danger_levels": {
                "critical": "Always block. Chemical weapons, nerve agents, CWC Schedule 1.",
                "high": "Block by default. Direct weapon/explosive/narcotics precursors.",
                "medium": "Warn user. Dual-use, DEA List II, CWC Schedule 3, reportable.",
                "low": "Inform user. Monitored substances, low standalone risk.",
            },
            "format_version": "2.0",
        },
        "chemicals": all_chemicals,
    }

    with open(chem_path, "w", encoding="utf-8") as f:
        json.dump(chem_output, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved chemicals to {chem_path}")

    # Build and save reaction patterns
    reactions = get_banned_reactions()
    rxn_stats = {}
    for r in reactions:
        dl = r["danger_level"]
        rxn_stats[dl] = rxn_stats.get(dl, 0) + 1

    logger.info(f"Total reaction patterns: {len(reactions)}")
    for level in ("critical", "high", "medium"):
        logger.info(f"  {level}: {rxn_stats.get(level, 0)}")

    rxn_path = DATA_DIR / "banned_reactions.json"
    rxn_output = {
        "_meta": {
            "total": len(reactions),
            "stats": rxn_stats,
            "sources": [
                "CWC prohibited synthesis routes",
                "Known explosive synthesis patterns",
                "Narcotics synthesis scaffolds (DEA/UN)",
                "Dual-use reaction patterns",
            ],
            "danger_levels": {
                "critical": "Always block. CWC weapon synthesis, phosgene, fentanyl core.",
                "high": "Block by default. Explosive synthesis, narcotics scaffolds.",
                "medium": "Warn user. Dual-use patterns with legitimate applications.",
            },
            "format_version": "2.0",
        },
        "reactions": reactions,
    }

    with open(rxn_path, "w", encoding="utf-8") as f:
        json.dump(rxn_output, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved reactions to {rxn_path}")

    logger.info("")
    logger.info("Done!")


if __name__ == "__main__":
    main()
