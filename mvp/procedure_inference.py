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


def _classify_molecule(smiles: str) -> str:
    """Classify molecule as 'organic' | 'inorganic' | 'elemental' | 'unknown'.

    elemental — all atoms are the same element (I2, Cl2, S8, etc.)
    inorganic  — no carbon atoms but mixed elements (NaCl, H2SO4, etc.)
    organic    — contains carbon
    """
    if not smiles:
        return "unknown"
    if not HAS_RDKIT:
        # Simple heuristic without RDKit
        has_c = "C" in smiles or "c" in smiles
        return "organic" if has_c else "inorganic"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "unknown"
    atomic_nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
    if not atomic_nums:
        return "unknown"
    has_carbon = any(n == 6 for n in atomic_nums)
    if has_carbon:
        return "organic"
    unique_elements = set(atomic_nums)
    if len(unique_elements) == 1:
        return "elemental"
    return "inorganic"


def _is_nonsensical_reaction(reaction_smiles: str) -> str | None:
    """Check if a reaction SMILES is chemically nonsensical.

    Returns a human-readable warning string, or None if the reaction seems valid.
    """
    if not reaction_smiles or ">>" not in reaction_smiles:
        return None
    parts = reaction_smiles.split(">>")
    reactant_str = parts[0].strip()
    product_str = parts[-1].strip()

    if not reactant_str or not product_str:
        return None

    if not HAS_RDKIT:
        return None

    rmol = Chem.MolFromSmiles(reactant_str)
    pmol = Chem.MolFromSmiles(product_str)
    if rmol is None or pmol is None:
        return f"Невалидный SMILES в уравнении реакции: {reaction_smiles[:80]}"

    r_can = Chem.MolToSmiles(rmol)
    p_can = Chem.MolToSmiles(pmol)

    # Reactant == Product (trivial / identity)
    if r_can == p_can:
        return f"Уравнение реакции тривиально: реагент и продукт идентичны ({r_can})"

    # Atom count check (without reagents after '>') — simple sanity
    r_atoms_by_element: dict[int, int] = {}
    for a in rmol.GetAtoms():
        r_atoms_by_element[a.GetAtomicNum()] = r_atoms_by_element.get(a.GetAtomicNum(), 0) + 1
    p_atoms_by_element: dict[int, int] = {}
    for a in pmol.GetAtoms():
        p_atoms_by_element[a.GetAtomicNum()] = p_atoms_by_element.get(a.GetAtomicNum(), 0) + 1

    # For single-component reaction (no '.'), check conservation
    if "." not in reactant_str:
        for elem_num, count in p_atoms_by_element.items():
            r_count = r_atoms_by_element.get(elem_num, 0)
            if count > r_count * 2:  # allow some flexibility for stoichiometry errors
                elem_sym = Chem.GetPeriodicTable().GetElementSymbol(elem_num)
                return (
                    f"Нарушение атомного баланса: реагент содержит {r_count} атомов {elem_sym}, "
                    f"а продукт — {count}. Маршрут может быть некорректным."
                )

    return None


def _infer_elemental_procedure(product_smiles: str) -> list[dict[str, str]]:
    """Generate appropriate procedure for elemental substances (I2, S8, etc.)."""
    if not HAS_RDKIT or not product_smiles:
        return []
    mol = Chem.MolFromSmiles(product_smiles)
    if mol is None:
        return []
    # Get element symbol
    atoms = list(mol.GetAtoms())
    if not atoms:
        return []
    elem_num = atoms[0].GetAtomicNum()
    elem_sym = Chem.GetPeriodicTable().GetElementSymbol(elem_num)
    n_atoms = len(atoms)
    elem_name = _ELEMENT_NAMES_RU.get(elem_sym, elem_sym)

    # Halogens: Cl2, Br2, I2 — oxidation of halide salt
    if elem_sym in ("I", "Br", "Cl", "F"):
        if elem_sym == "I":
            return [
                {"step": "1", "description": f"{elem_name} ({elem_sym}₂) — коммерчески доступный реактив. Рекомендуется закупить у химического поставщика (квалификация ч.д.а. или х.ч.).", "reason": "Промышленный способ"},
                {"step": "2", "description": "При необходимости лабораторного синтеза: растворить 2 г KI в 5 мл воды, медленно добавить раствор Cl₂ в воде (или HCl + H₂O₂). Выпавший осадок I₂ отфильтровать.", "reason": "Окисление KI хлором"},
                {"step": "3", "description": "Промыть осадок холодной водой для удаления KCl. Высушить между листами фильтровальной бумаги.", "reason": "Промывка"},
                {"step": "4", "description": "Для дополнительной очистки провести возгонку (сублимацию): нагреть I₂ в фарфоровой чашке, закрытой воронкой с холодной водой. I₂ возгоняется при 113°C и кристаллизуется на охлаждённой поверхности.", "reason": "Сублимация — стандартный метод очистки I₂"},
                {"step": "5", "description": "Хранить в плотно закрытом тёмном флаконе. ВНИМАНИЕ: I₂ — раздражающее вещество, работать в вытяжном шкафу.", "reason": "Техника безопасности"},
            ]
        elif elem_sym == "Br":
            return [
                {"step": "1", "description": f"{elem_name} ({elem_sym}₂) — коммерчески доступная жидкость. Закупить у поставщика.", "reason": "Промышленный реактив"},
                {"step": "2", "description": "ВНИМАНИЕ: Br₂ — сильнодействующий яд и сильный окислитель. Работать только в вытяжном шкафу, СИЗ: перчатки, защитные очки, лабораторный халат.", "reason": "Техника безопасности"},
            ]
        else:
            return [
                {"step": "1", "description": f"{elem_name} ({elem_sym}₂) — промышленный газ, поставляется в баллонах. Синтез в лаборатории нецелесообразен.", "reason": "Промышленный реактив"},
            ]

    # Sulfur
    if elem_sym == "S":
        return [
            {"step": "1", "description": "Сера (S₈) — коммерчески доступный реактив, закупить у поставщика (ч.д.а.).", "reason": "Промышленный реактив"},
            {"step": "2", "description": "При необходимости очистки: растворить в CS₂, отфильтровать нерастворимые примеси, упарить CS₂ — получится ромбическая сера.", "reason": "Очистка сублимацией или растворением в CS₂"},
        ]

    # Generic elemental
    return [
        {"step": "1", "description": f"Элементарный {elem_name} ({elem_sym}{n_atoms if n_atoms > 1 else ''}) — как правило, коммерчески доступен. Рекомендуется закупить у химического поставщика.", "reason": "Элементарное вещество"},
        {"step": "2", "description": "Если требуется синтез: обратитесь к специализированной неорганической химической литературе — методика зависит от конкретного элемента.", "reason": "Специфика неорганического синтеза"},
    ]


def _infer_inorganic_procedure(route: dict[str, Any], product_smiles: str) -> list[dict[str, str]]:
    """Generate a basic inorganic synthesis procedure.

    Unlike organic synthesis, inorganic reactions:
    - Often don't need organic solvents
    - May use aqueous media
    - Purification is via filtration/recrystallization/precipitation, not chromatography
    """
    temp = route.get("temperature")
    solvent = route.get("solvent")
    catalyst = route.get("catalyst")
    temp_c = _parse_temp(temp)
    solvent_info = _get_solvent_info(solvent)
    solvent_name = solvent_info.get("name_ru", solvent) if solvent_info else (solvent or "вода")

    steps: list[dict[str, str]] = []
    step_num = 1

    steps.append({
        "step": str(step_num),
        "description": f"Подготовить реагенты. Взвесить необходимые количества согласно таблице реагентов.",
        "reason": "Неорганический синтез",
    })
    step_num += 1

    if solvent_name:
        steps.append({
            "step": str(step_num),
            "description": f"Растворить реагенты в {solvent_name} при перемешивании.",
            "reason": f"Растворитель: {solvent_name}",
        })
        step_num += 1

    if catalyst:
        steps.append({
            "step": str(step_num),
            "description": f"Добавить катализатор/активатор: {catalyst}.",
            "reason": f"Катализатор: {catalyst}",
        })
        step_num += 1

    if temp_c is not None:
        if temp_c < 0:
            steps.append({
                "step": str(step_num),
                "description": f"Охладить до {temp_c}°C (ледяная баня).",
                "reason": f"T реакции {temp_c}°C",
            })
        elif temp_c > 25:
            steps.append({
                "step": str(step_num),
                "description": f"Нагреть реакционную смесь до {temp_c}°C при перемешивании.",
                "reason": f"T реакции {temp_c}°C",
            })
        step_num += 1

    steps.append({
        "step": str(step_num),
        "description": "Перемешивать 1-3 часа до завершения реакции (контроль по изменению цвета/осадку).",
        "reason": "Время реакции (оценка)",
    })
    step_num += 1

    # Product properties — solid or dissolved?
    product_props = _get_mol_props(product_smiles) if product_smiles else {}
    likely_solid = product_props.get("likely_solid", True)  # inorganics often solid

    if likely_solid:
        steps.append({
            "step": str(step_num),
            "description": "Отфильтровать выпавший осадок (вакуумная фильтрация через бумажный фильтр). Промыть дистиллированной водой.",
            "reason": "Осаждение неорганического продукта",
        })
        step_num += 1
        steps.append({
            "step": str(step_num),
            "description": "Высушить продукт в сушильном шкафу (60-120°C, 2-4 ч). При необходимости перекристаллизовать из воды.",
            "reason": "Сушка и очистка",
        })
        step_num += 1
    else:
        steps.append({
            "step": str(step_num),
            "description": "Упарить раствор досуха (роторный испаритель или водяная баня). Перекристаллизовать из подходящего растворителя.",
            "reason": "Выделение продукта из раствора",
        })
        step_num += 1

    return steps


_ELEMENT_NAMES_RU = {
    "H": "водород", "He": "гелий", "Li": "литий", "Be": "бериллий",
    "B": "бор", "C": "углерод", "N": "азот", "O": "кислород",
    "F": "фтор", "Ne": "неон", "Na": "натрий", "Mg": "магний",
    "Al": "алюминий", "Si": "кремний", "P": "фосфор", "S": "сера",
    "Cl": "хлор", "Ar": "аргон", "K": "калий", "Ca": "кальций",
    "Fe": "железо", "Cu": "медь", "Zn": "цинк", "Br": "бром",
    "I": "йод", "Pt": "платина", "Au": "золото", "Hg": "ртуть",
    "Pb": "свинец", "Ag": "серебро", "Cr": "хром", "Mn": "марганец",
    "Ni": "никель", "Co": "кобальт", "Mo": "молибден", "W": "вольфрам",
}


def infer_procedure(route: dict[str, Any]) -> list[dict[str, str]]:
    """Infer synthesis procedure steps from reaction conditions.

    Uses heuristics based on temperature, solvent, catalyst, and
    reactant/product properties. Handles inorganic/elemental molecules
    with appropriate non-organic procedures.

    Returns list of step dicts: {step, description, reason}
    """
    steps: list[dict[str, str]] = []
    temp = route.get("temperature")
    solvent = route.get("solvent")
    catalyst = route.get("catalyst")
    reactants_str = route.get("reactants", "")
    reaction_smiles = route.get("reaction_smiles", "")
    product_smiles = reaction_smiles.split(">>")[-1] if ">>" in reaction_smiles else ""

    # ── Early exit: check for nonsensical reaction ──
    nonsensical_warning = _is_nonsensical_reaction(reaction_smiles)

    # ── Early exit: elemental molecules (I2, S8, Cl2, etc.) ──
    mol_class = _classify_molecule(product_smiles) if product_smiles else "unknown"
    if mol_class == "elemental":
        elemental_steps = _infer_elemental_procedure(product_smiles)
        if nonsensical_warning:
            elemental_steps.insert(0, {
                "step": "!",
                "description": f"⚠ {nonsensical_warning}",
                "reason": "Проверка уравнения реакции",
            })
        if elemental_steps:
            # Re-number steps
            for i, s in enumerate(elemental_steps, 1):
                if s["step"] != "!":
                    s["step"] = str(i)
            return elemental_steps

    # ── Early exit: inorganic compounds (no carbon) ──
    if mol_class == "inorganic":
        inorganic_steps = _infer_inorganic_procedure(route, product_smiles)
        if nonsensical_warning:
            inorganic_steps.insert(0, {
                "step": "!",
                "description": f"⚠ {nonsensical_warning}",
                "reason": "Проверка уравнения реакции",
            })
        return inorganic_steps

    # ── Non-fatal warning for organic reactions with odd stoichiometry ──
    warning_step = None
    if nonsensical_warning:
        warning_step = {
            "step": "!",
            "description": f"⚠ {nonsensical_warning}",
            "reason": "Проверка уравнения реакции",
        }

    # Parse temperature
    temp_c = _parse_temp(temp)

    # Get solvent info
    solvent_info = _get_solvent_info(solvent)
    solvent_name = solvent_info.get("name_ru", solvent) if solvent_info else (solvent or "")
    solvent_bp = solvent_info.get("bp") if solvent_info else None

    # Product properties
    product_props = _get_mol_props(product_smiles) if product_smiles else {}

    step_num = 1

    if warning_step:
        steps.append(warning_step)

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
# Procedure formatting (ORD English → structured Russian via LLM)
# ═════════════════════════════════════════════════════════════════════════════

_PROCEDURE_SYSTEM_PROMPT = """\
Ты — ассистент-химик. Тебе дан текст процедуры синтеза на английском языке из базы Open Reaction Database.

Задача: перевести процедуру на русский язык и разбить на пронумерованные шаги.

Правила:
1. Каждый шаг — отдельное действие (добавление реагента, нагрев, перемешивание, фильтрация и т.д.)
2. Сохраняй все количества, температуры, времена, названия реагентов
3. Используй профессиональную химическую терминологию на русском
4. Названия реагентов оставляй на английском в скобках, если нет устоявшегося русского названия
5. Отвечай ТОЛЬКО в формате JSON — массив объектов

Формат ответа (ТОЛЬКО JSON, без markdown):
[
  {"step": "1", "description": "Описание шага на русском", "reason": "ORD процедура"},
  {"step": "2", "description": "Описание шага на русском", "reason": "ORD процедура"}
]"""


def _translate_procedure_via_llm(text: str) -> list[dict[str, str]] | None:
    """Translate English procedure to structured Russian steps via LLM."""
    try:
        from .config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL
    except ImportError:
        logger.debug("LLM procedure: config import failed")
        return None

    if not OPENROUTER_API_KEY:
        logger.debug("LLM procedure: no API key")
        return None

    try:
        import json as _json
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            temperature=0.05,
            max_tokens=2000,
        )

        # Truncate very long procedures
        proc_text = text[:3000] if len(text) > 3000 else text

        resp = llm.invoke([
            SystemMessage(content=_PROCEDURE_SYSTEM_PROMPT),
            HumanMessage(content=proc_text),
        ])

        raw = resp.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        steps = _json.loads(raw)
        if isinstance(steps, list) and steps:
            # Validate format
            for s in steps:
                if not isinstance(s, dict) or "description" not in s:
                    logger.warning("LLM procedure: invalid step format: %s", s)
                    return None
                s.setdefault("step", "?")
                s.setdefault("reason", "ORD процедура")
            logger.info("LLM procedure translation: %d steps", len(steps))
            return steps
        logger.warning("LLM procedure: empty or invalid response: %s", raw[:200])
    except Exception as e:
        logger.warning("LLM procedure translation failed: %s", e)

    return None


def format_procedure_russian(
    route: dict[str, Any],
    use_inference: bool = True,
) -> list[dict[str, str]]:
    """Format a route's procedure as structured Russian steps.

    If ORD procedure_details exists, translates via LLM.
    Otherwise, infers from conditions using rules.

    Returns list of step dicts: {step, description, reason}
    """
    procedure = route.get("procedure_details", "")

    if procedure and len(procedure) > 50:
        # Try LLM translation first
        llm_result = _translate_procedure_via_llm(procedure)
        if llm_result:
            return llm_result
        # Fallback: simple split (no translation)
        logger.warning("LLM fallback: returning raw procedure split")
        return _split_procedure_raw(procedure)

    if use_inference:
        return infer_procedure(route)

    return [{
        "step": "1",
        "description": "Процедура синтеза не найдена.",
        "reason": "Нет данных в ORD",
    }]


def _split_procedure_raw(text: str) -> list[dict[str, str]]:
    """Fallback: split procedure into sentences without translation."""
    sentences = re.split(r'(?<=[.!])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    if not sentences:
        return [{"step": "1", "description": text[:500], "reason": "ORD (оригинал)"}]

    steps: list[dict[str, str]] = []
    for i, sent in enumerate(sentences, 1):
        steps.append({
            "step": str(i),
            "description": sent,
            "reason": "ORD (оригинал, без перевода)",
        })
    return steps
