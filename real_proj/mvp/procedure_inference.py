"""Rule-based procedure inference and procedure formatting.

When ORD procedure_details is missing, infers synthesis steps from
reaction conditions (temperature, solvent, catalyst, reactant/product
properties) using chemical heuristics.

Also formats raw English procedures into structured Russian step-by-step.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# ═════════════════════════════════════════════════════════════════════════════
# Solvent database (bp, polarity class)
# ═════════════════════════════════════════════════════════════════════════════

SOLVENTS: dict[str, dict[str, Any]] = {
    # SMILES → {name_ru, bp (°C), polarity}
    "ClCCl": {"name_ru": "дихлорметан", "bp": 40, "polarity": "средняя"},
    "C(Cl)(Cl)Cl": {"name_ru": "хлороформ", "bp": 61, "polarity": "средняя"},
    "CCOCC": {"name_ru": "диэтиловый эфир", "bp": 35, "polarity": "низкая"},
    "C1CCOC1": {"name_ru": "ТГФ", "bp": 66, "polarity": "средняя"},
    "CC(C)=O": {"name_ru": "ацетон", "bp": 56, "polarity": "высокая"},
    "CCO": {"name_ru": "этанол", "bp": 78, "polarity": "высокая"},
    "CO": {"name_ru": "метанол", "bp": 65, "polarity": "высокая"},
    "O": {"name_ru": "вода", "bp": 100, "polarity": "высокая"},
    "Cc1ccccc1": {"name_ru": "толуол", "bp": 111, "polarity": "низкая"},
    "c1ccccc1": {"name_ru": "бензол", "bp": 80, "polarity": "низкая"},
    "CC#N": {"name_ru": "ацетонитрил", "bp": 82, "polarity": "высокая"},
    "CS(C)=O": {"name_ru": "ДМСО", "bp": 189, "polarity": "высокая"},
    "CN(C)C=O": {"name_ru": "ДМФА", "bp": 153, "polarity": "высокая"},
    "CC(=O)OCC": {"name_ru": "этилацетат", "bp": 77, "polarity": "средняя"},
    "CCCCCC": {"name_ru": "гексан", "bp": 69, "polarity": "низкая"},
    "C(C)(=O)O": {"name_ru": "уксусная кислота", "bp": 118, "polarity": "высокая"},
    "C1COCCO1": {"name_ru": "диоксан", "bp": 101, "polarity": "средняя"},
}

# Common catalyst patterns
HETEROGENEOUS_CATALYSTS = {
    "[Pd]", "Pd/C", "[Pd]/C", "[Pt]", "[Ni]", "Pd(PPh3)",
}

# ═════════════════════════════════════════════════════════════════════════════
# Rule-based procedure inference
# ═════════════════════════════════════════════════════════════════════════════


def infer_procedure(route: dict[str, Any]) -> list[dict[str, str]]:
    """Infer synthesis procedure steps from reaction conditions.

    Uses heuristics based on temperature, solvent, catalyst, and
    reactant/product properties.

    Returns list of step dicts: {step, description, reason}
    """
    steps: list[dict[str, str]] = []
    temp = route.get("temperature")
    solvent = route.get("solvent")
    catalyst = route.get("catalyst")
    reactants_str = route.get("reactants", "")
    product_smiles = route.get("reaction_smiles", "").split(">>")[-1] if ">>" in route.get("reaction_smiles", "") else ""

    # Parse temperature
    temp_c = _parse_temp(temp)

    # Get solvent info
    solvent_info = _get_solvent_info(solvent)
    solvent_name = solvent_info.get("name_ru", solvent) if solvent_info else (solvent or "")
    solvent_bp = solvent_info.get("bp") if solvent_info else None

    # Product properties
    product_props = _get_mol_props(product_smiles) if product_smiles else {}

    step_num = 1

    # ── Step 1: Atmosphere / sensitivity ──
    if _is_sensitive_reaction(catalyst, reactants_str):
        steps.append({
            "step": str(step_num),
            "description": "Подготовить инертную атмосферу (N₂ или Ar). Высушить стеклянную посуду.",
            "reason": "Реакция чувствительна к воде/воздуху",
        })
        step_num += 1

    # ── Step 2: Dissolution / mixing ──
    if solvent_name:
        steps.append({
            "step": str(step_num),
            "description": f"Растворить реагенты в {solvent_name}.",
            "reason": f"Растворитель: {solvent_name}",
        })
    else:
        steps.append({
            "step": str(step_num),
            "description": "Смешать реагенты.",
            "reason": "Растворитель не указан",
        })
    step_num += 1

    # ── Step 3: Catalyst ──
    if catalyst:
        cat_name = _translate_catalyst(catalyst)
        steps.append({
            "step": str(step_num),
            "description": f"Добавить катализатор: {cat_name}.",
            "reason": f"Катализатор: {catalyst}",
        })
        step_num += 1

    # ── Step 4: Temperature control ──
    if temp_c is not None:
        if temp_c < 0:
            steps.append({
                "step": str(step_num),
                "description": f"Охладить реакционную смесь до {temp_c}°C (ледяная баня / криостат).",
                "reason": f"Температура реакции ({temp_c}°C) ниже 0°C",
            })
            step_num += 1
        elif solvent_bp is not None and temp_c > solvent_bp - 5:
            steps.append({
                "step": str(step_num),
                "description": f"Нагреть до кипения ({temp_c}°C) с обратным холодильником (рефлюкс).",
                "reason": f"T реакции ({temp_c}°C) ≥ T кипения растворителя ({solvent_bp}°C)",
            })
            step_num += 1
        elif temp_c > 25:
            steps.append({
                "step": str(step_num),
                "description": f"Нагреть реакционную смесь до {temp_c}°C.",
                "reason": f"Требуется нагрев до {temp_c}°C",
            })
            step_num += 1

    # ── Step 5: Reaction time ──
    reaction_time = _estimate_reaction_time(temp_c, catalyst)
    steps.append({
        "step": str(step_num),
        "description": f"Перемешивать при заданной температуре {reaction_time}.",
        "reason": "Время реакции (оценка по условиям)",
    })
    step_num += 1

    # ── Step 6: Catalyst removal ──
    if _is_heterogeneous_catalyst(catalyst):
        steps.append({
            "step": str(step_num),
            "description": "Отфильтровать катализатор через целит (Celite).",
            "reason": "Гетерогенный катализатор требует фильтрации",
        })
        step_num += 1

    # ── Step 7: Workup — determined by product properties ──
    workup_steps = _infer_workup(product_props, solvent_info, solvent_name)
    for ws in workup_steps:
        ws["step"] = str(step_num)
        steps.append(ws)
        step_num += 1

    return steps


def _parse_temp(temp: Any) -> float | None:
    """Parse temperature string to °C value."""
    if temp is None:
        return None
    if isinstance(temp, (int, float)):
        return float(temp)
    temp_str = str(temp)
    # Try to extract number
    m = re.search(r"(-?\d+\.?\d*)", temp_str)
    if not m:
        return None
    val = float(m.group(1))
    # Convert if Kelvin
    if "KELVIN" in temp_str.upper() or "K" in temp_str.upper().split()[-1:]:
        val -= 273.15
    elif "FAHRENHEIT" in temp_str.upper():
        val = (val - 32) * 5 / 9
    return val


def _get_solvent_info(solvent: str | None) -> dict[str, Any] | None:
    """Look up solvent info by SMILES."""
    if not solvent:
        return None
    # Direct match
    if solvent in SOLVENTS:
        return SOLVENTS[solvent]
    # Try canonical match
    if HAS_RDKIT:
        mol = Chem.MolFromSmiles(solvent)
        if mol:
            canon = Chem.MolToSmiles(mol, isomericSmiles=True)
            if canon in SOLVENTS:
                return SOLVENTS[canon]
    return None


def _get_mol_props(smiles: str) -> dict[str, Any]:
    """Get basic molecular properties for workup inference."""
    if not HAS_RDKIT or not smiles:
        return {}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rings = Descriptors.RingCount(mol)
    # Rough melting point heuristic: high MW + many rings → likely solid
    likely_solid = (mw > 150 and rings >= 1 and logp < 3) or mw > 300
    return {
        "mw": mw,
        "logp": logp,
        "tpsa": tpsa,
        "hba": hba,
        "rings": rings,
        "likely_solid": likely_solid,
        "lipophilic": logp > 2,
    }


def _is_sensitive_reaction(catalyst: str | None, reactants: str) -> bool:
    """Check if reaction likely requires inert atmosphere."""
    sensitive_markers = {
        "[Li]", "[Na]", "MgBr", "MgCl", "MgI",  # organometallics
        "BuLi", "[Pd(", "Pd(PPh3", "Pd(dppf",    # Pd-catalyzed
        "NaH", "KH", "LiAlH",                     # hydrides
        "B(O", "boronic",                          # boronic acids (Suzuki)
    }
    combined = (catalyst or "") + " " + reactants
    return any(m in combined for m in sensitive_markers)


def _is_heterogeneous_catalyst(catalyst: str | None) -> bool:
    if not catalyst:
        return False
    markers = ["Pd/C", "Pt/C", "Ni/", "Raney", "целит", "Celite", "Al2O3"]
    return any(m.lower() in catalyst.lower() for m in markers)


def _translate_catalyst(catalyst: str) -> str:
    """Basic translation of catalyst names."""
    translations = {
        "Pd/C": "палладий на угле (Pd/C)",
        "Pt/C": "платина на угле (Pt/C)",
        "Raney Ni": "никель Ренея",
        "NaOH": "гидроксид натрия (NaOH)",
        "KOH": "гидроксид калия (KOH)",
        "H2SO4": "серная кислота (H₂SO₄)",
        "HCl": "соляная кислота (HCl)",
        "BF3": "трифторид бора (BF₃)",
        "AlCl3": "хлорид алюминия (AlCl₃)",
    }
    for eng, rus in translations.items():
        if eng.lower() in catalyst.lower():
            return rus
    return catalyst


def _estimate_reaction_time(temp_c: float | None, catalyst: str | None) -> str:
    """Rough estimate of reaction time based on conditions."""
    if temp_c is not None and temp_c > 100:
        return "2-4 часа"
    if temp_c is not None and temp_c < 0:
        return "1-2 часа"
    if catalyst and _is_heterogeneous_catalyst(catalyst):
        return "4-12 часов"
    return "2-6 часов"


def _infer_workup(
    product_props: dict[str, Any],
    solvent_info: dict | None,
    solvent_name: str,
) -> list[dict[str, str]]:
    """Infer workup procedure steps from product properties."""
    steps: list[dict[str, str]] = []

    likely_solid = product_props.get("likely_solid", False)
    lipophilic = product_props.get("lipophilic", False)
    solvent_polarity = (solvent_info or {}).get("polarity", "средняя")

    if likely_solid:
        # Solid product → filtration + recrystallization
        steps.append({
            "step": "",
            "description": "Отфильтровать выпавший осадок (вакуумная фильтрация). Промыть холодным растворителем.",
            "reason": "Продукт — твёрдое вещество, нерастворимое в реакционной среде",
        })
        steps.append({
            "step": "",
            "description": "Перекристаллизовать для очистки.",
            "reason": "Твёрдый продукт с примесями → перекристаллизация",
        })
    else:
        # Liquid/dissolved product → extraction
        if solvent_polarity == "низкая" or lipophilic:
            steps.append({
                "step": "",
                "description": "Промыть реакционную смесь водой, затем насыщенным раствором NaCl. Разделить фазы.",
                "reason": "Продукт растворён в органической фазе → экстракция",
            })
            steps.append({
                "step": "",
                "description": "Высушить органическую фазу над безводным MgSO₄ или Na₂SO₄. Отфильтровать осушитель.",
                "reason": "Удаление остаточной воды из органической фазы",
            })
        else:
            steps.append({
                "step": "",
                "description": "Экстрагировать продукт органическим растворителем (этилацетат или ДХМ). Промыть водой.",
                "reason": "Продукт в полярном растворителе → экстракция",
            })
            steps.append({
                "step": "",
                "description": "Высушить органическую фазу над безводным Na₂SO₄.",
                "reason": "Удаление воды",
            })

    # Always: remove solvent
    steps.append({
        "step": "",
        "description": "Упарить растворитель на роторном испарителе.",
        "reason": "Удаление растворителя",
    })

    # Purification
    mw = product_props.get("mw", 0)
    if mw > 200 and not likely_solid:
        steps.append({
            "step": "",
            "description": "Очистить колоночной хроматографией (силикагель).",
            "reason": "Сложная молекула — требуется хроматографическая очистка",
        })

    return steps


# ═════════════════════════════════════════════════════════════════════════════
# Procedure formatting (ORD English → structured Russian)
# ═════════════════════════════════════════════════════════════════════════════

# Key English→Russian term translations for procedures
_TERM_MAP = {
    "was added": "добавлен",
    "were added": "добавлены",
    "was dissolved": "растворён",
    "were dissolved": "растворены",
    "was stirred": "перемешивали",
    "was heated": "нагревали",
    "was cooled": "охлаждали",
    "was filtered": "отфильтровали",
    "was washed": "промыли",
    "was dried": "высушили",
    "was concentrated": "упарили",
    "was purified": "очистили",
    "was extracted": "экстрагировали",
    "the mixture": "смесь",
    "the solution": "раствор",
    "the reaction": "реакцию",
    "the product": "продукт",
    "the residue": "остаток",
    "room temperature": "комнатной температуре",
    "ice bath": "ледяной бане",
    "under nitrogen": "в атмосфере азота",
    "under argon": "в атмосфере аргона",
    "under vacuum": "под вакуумом",
    "overnight": "в течение ночи (12-16 ч)",
    "dropwise": "по каплям",
    "column chromatography": "колоночной хроматографией",
    "silica gel": "силикагеле",
    "flash chromatography": "флэш-хроматографией",
    "recrystallization": "перекристаллизацией",
    "recrystallized": "перекристаллизовали",
    "evaporated": "упарили",
    "concentrated in vacuo": "упарили при пониженном давлении",
    "rotary evaporator": "роторном испарителе",
    "anhydrous": "безводном",
    "saturated": "насыщенным",
    "aqueous": "водным",
    "organic layer": "органическую фазу",
    "aqueous layer": "водную фазу",
    "white solid": "белое твёрдое вещество",
    "yellow oil": "жёлтое масло",
    "colorless oil": "бесцветное масло",
    "yield": "выход",
}


def format_procedure_russian(
    route: dict[str, Any],
    use_inference: bool = True,
) -> list[dict[str, str]]:
    """Format a route's procedure as structured Russian steps.

    If ORD procedure_details exists, parses and translates it.
    Otherwise, infers from conditions using rules.

    Returns list of step dicts: {step, description, reason}
    """
    procedure = route.get("procedure_details", "")

    if procedure and len(procedure) > 50:
        # Parse ORD procedure into steps
        return _parse_english_procedure(procedure)

    if use_inference:
        return infer_procedure(route)

    return [{
        "step": "1",
        "description": "Процедура синтеза не найдена.",
        "reason": "Нет данных в ORD",
    }]


def _parse_english_procedure(text: str) -> list[dict[str, str]]:
    """Parse English procedure text into numbered Russian steps.

    Splits by sentences, groups into logical steps, applies term translation.
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    if not sentences:
        return [{"step": "1", "description": text[:500], "reason": "ORD (оригинал)"}]

    steps: list[dict[str, str]] = []
    current_group: list[str] = []
    step_num = 1

    # Group sentences into logical steps by action keywords
    action_markers = [
        "was added", "were added", "was dissolved", "was stirred",
        "was heated", "was cooled", "was filtered", "was washed",
        "was extracted", "was purified", "was concentrated",
        "the mixture was", "the reaction was", "the product was",
        "added to", "poured into",
    ]

    for sent in sentences:
        sent_lower = sent.lower()
        is_new_action = any(m in sent_lower for m in action_markers)

        if is_new_action and current_group:
            # Save current group as a step
            combined = " ".join(current_group)
            translated = _light_translate(combined)
            steps.append({
                "step": str(step_num),
                "description": translated,
                "reason": "ORD процедура",
            })
            step_num += 1
            current_group = [sent]
        else:
            current_group.append(sent)

    # Last group
    if current_group:
        combined = " ".join(current_group)
        translated = _light_translate(combined)
        steps.append({
            "step": str(step_num),
            "description": translated,
            "reason": "ORD процедура",
        })

    return steps


def _light_translate(text: str) -> str:
    """Apply dictionary-based translation of common chemistry terms."""
    result = text
    for eng, rus in _TERM_MAP.items():
        result = re.sub(re.escape(eng), rus, result, flags=re.IGNORECASE)
    return result
