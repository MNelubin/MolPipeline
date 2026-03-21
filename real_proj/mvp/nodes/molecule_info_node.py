"""Molecule info node: gather data + LLM synthesis via OpenRouter.

Outputs everything in Russian. Includes physical description,
2D/3D images, experimental properties, LD50, CAS, GHS pictograms.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from ..config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL, LLM_TEMPERATURE
from ..tools import (
    pubchem_lookup,
    rdkit_properties,
    get_physical_description,
    get_molecule_images,
    get_experimental_properties,
    get_ld50,
    get_cas_number,
    enrich_ghs_pictograms,
)
from ..tools.pubchem import pubchem_lookup_by_cid

logger = logging.getLogger(__name__)

_MOLECULE_CARD_PROMPT = PromptTemplate.from_template("""
Ты — эксперт-химик. Заполни карточку молекулы на основе предоставленных данных.
Все текстовые поля заполняй НА РУССКОМ ЯЗЫКЕ.

Входные данные:
1. Запрос пользователя: {query}
2. Данные PubChem: {pubchem_data}
3. Данные RDKit: {rdkit_data}
4. Данные безопасности: {safety_data}
5. Физическое описание (PubChem): {physical_description}
6. Экспериментальные свойства (PubChem): {experimental_props}
7. LD50 данные: {ld50_data}

Верни JSON-объект со следующими полями. Если данных нет — используй свои знания.

Поля JSON:
- "name": название вещества (русское общеупотребительное + IUPAC в скобках)
- "synonyms": список синонимов на русском (список строк, 3-5 штук)
- "smiles": SMILES-строка
- "molecular_formula": брутто-формула
- "molecular_weight": молярная масса (число)
- "physical_description": описание физических свойств НА РУССКОМ — внешний вид, цвет, запах, агрегатное состояние. Опирайся на PubChem Physical Description.
- "properties": словарь с ключами:
    - "melting_point": температура плавления (°C, число или строка с единицами)
    - "boiling_point": температура кипения (°C, число или строка с единицами)
    - "solubility": растворимость (описание на русском)
    - "density": плотность (г/мл, число)
    - "logP": коэффициент распределения (число)
    - "physical_state": агрегатное состояние на русском (твёрдое/жидкое/газ)
    - "flash_point": температура вспышки (°C, число или null)
    - "vapor_pressure": давление паров (строка или null)
- "ghs_classification": список классов опасности GHS на русском (список строк)
- "spectral_notes": краткая заметка о спектральных данных (ИК, ЯМР) на русском
- "description": подробное описание вещества на русском (применение, значение, история, 2-3 предложения)
- "pubchem_cid": CID число (0 если неизвестно)

ВАЖНО:
- Данные RDKit (масса, logP, TPSA) приоритетнее вычисленных PubChem.
- Экспериментальные свойства PubChem (Т. пл., Т. кип., плотность) приоритетнее LLM-знаний.
- Верни ТОЛЬКО валидный JSON, без markdown-блоков кода.
""")


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def molecule_info_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: gather molecule info and produce a structured card."""
    query = state.get("query", "")
    smiles = state.get("smiles", "")
    guard_result = state.get("guard_result", {})
    cid: int | None = state.get("pubchem_cid") or None

    logger.info("[molecule_info] query=%r smiles=%r cid=%s", query, smiles, cid)

    # 1. RDKit + PubChem basic
    pubchem_result = {}
    rdkit_result = {}

    if smiles:
        rdkit_result = rdkit_properties(smiles)

    # Prefer CID-based lookup (unambiguous) over name/SMILES lookup
    if cid:
        pubchem_data = pubchem_lookup_by_cid(cid)
        if "error" not in pubchem_data:
            pubchem_result = pubchem_data
    elif smiles:
        pubchem_data = pubchem_lookup(smiles)
        if "error" not in pubchem_data:
            pubchem_result = pubchem_data
        else:
            pubchem_data = pubchem_lookup(query)
            if "error" not in pubchem_data:
                pubchem_result = pubchem_data

    if not cid:
        cid = pubchem_result.get("cid")

    # 2. Physical description
    phys_desc_list = get_physical_description(smiles, cid=cid) if smiles else []
    phys_desc_str = " | ".join(phys_desc_list[:5]) if phys_desc_list else ""

    # 3. Images
    images = get_molecule_images(smiles, cid=cid)

    # 4. Experimental properties from PUG View
    exp_props: dict[str, Any] = {}
    if cid:
        exp_props = get_experimental_properties(cid)
        logger.info("[molecule_info] experimental: mp=%s bp=%s density=%s",
                     exp_props.get("melting_point"), exp_props.get("boiling_point"),
                     exp_props.get("density"))

    # 5. LD50
    ld50_data: dict[str, Any] = {}
    if cid:
        ld50_data = get_ld50(cid)
        logger.info("[molecule_info] ld50: %s", {k: bool(v) for k, v in ld50_data.items()})

    # 6. CAS number
    cas_number: str | None = None
    if cid:
        cas_number = get_cas_number(cid)
        logger.info("[molecule_info] cas=%s", cas_number)

    # 7. GHS pictograms enriched
    safety_data = guard_result.get("safety_data", {})
    ghs_codes = safety_data.get("ghs_pictograms", [])
    ghs_enriched = enrich_ghs_pictograms(ghs_codes)

    # 8. LLM synthesis
    llm = _get_llm()
    prompt_value = _MOLECULE_CARD_PROMPT.format(
        query=query,
        pubchem_data=json.dumps(pubchem_result, ensure_ascii=False),
        rdkit_data=json.dumps(rdkit_result, ensure_ascii=False),
        safety_data=json.dumps(safety_data, ensure_ascii=False),
        physical_description=phys_desc_str,
        experimental_props=json.dumps(exp_props, ensure_ascii=False),
        ld50_data=json.dumps(ld50_data, ensure_ascii=False),
    )

    try:
        llm_response = llm.invoke([HumanMessage(content=prompt_value)])
        text = llm_response.content
    except Exception as e:
        logger.error("[molecule_info] LLM error: %s", e)
        text = "{}"

    # Parse JSON
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        parsed = json.loads(text)
    except Exception as e:
        logger.warning("[molecule_info] JSON parse error: %s", e)
        parsed = {}

    # 9. Build molecule_info — merge LLM + raw data
    props = parsed.get("properties", {})

    rdkit_weight = rdkit_result.get("molecular_weight")
    parsed_weight = parsed.get("molecular_weight")
    final_weight = _safe_float(rdkit_weight if rdkit_weight else parsed_weight)

    parsed_cid = parsed.get("pubchem_cid")
    # Prefer the known CID (from state or pubchem lookup) — never let LLM override it
    final_cid = _safe_int(cid or parsed_cid or 0)

    # Experimental values take priority over LLM guesses
    melting_point = exp_props.get("melting_point") or props.get("melting_point", "Н/Д")
    boiling_point = exp_props.get("boiling_point") or props.get("boiling_point", "Н/Д")
    density = exp_props.get("density") or props.get("density", "Н/Д")
    solubility = exp_props.get("solubility") or props.get("solubility", "Н/Д")
    flash_point = exp_props.get("flash_point") or props.get("flash_point")
    vapor_pressure = exp_props.get("vapor_pressure") or props.get("vapor_pressure")

    # Name: prefer known PubChem IUPAC name over LLM output (LLM can hallucinate)
    known_iupac = pubchem_result.get("iupac") or ""
    llm_name = parsed.get("name", "")
    final_name = known_iupac if known_iupac else (llm_name or "Неизвестно")

    molecule_info = {
        "name": final_name,
        "synonyms": parsed.get("synonyms", pubchem_result.get("synonyms", [])),
        "smiles": parsed.get("smiles") or smiles or "",
        "molecular_formula": parsed.get("molecular_formula", pubchem_result.get("formula", "")),
        "molecular_weight": final_weight,
        "cas_number": cas_number,
        "physical_description": parsed.get("physical_description", phys_desc_str or "Нет данных"),
        "properties": {
            "melting_point": melting_point,
            "boiling_point": boiling_point,
            "solubility": solubility,
            "density": density,
            "logP": props.get("logP", rdkit_result.get("logp")),
            "physical_state": props.get("physical_state", "Н/Д"),
            "tpsa": rdkit_result.get("tpsa") or props.get("tpsa"),
            "h_bond_donors": rdkit_result.get("h_bond_donors") or props.get("h_bond_donors"),
            "h_bond_acceptors": rdkit_result.get("h_bond_acceptors") or props.get("h_bond_acceptors"),
            "rotatable_bonds": rdkit_result.get("rotatable_bonds") or props.get("rotatable_bonds"),
            "ring_count": rdkit_result.get("ring_count") or props.get("ring_count"),
            "flash_point": flash_point,
            "vapor_pressure": vapor_pressure,
            "spectral_notes": parsed.get("spectral_notes", "Н/Д"),
        },
        "toxicity": {
            "ld50_oral": ld50_data.get("ld50_oral"),
            "ld50_dermal": ld50_data.get("ld50_dermal"),
            "ld50_inhalation": ld50_data.get("ld50_inhalation"),
        },
        "description": parsed.get("description", ""),
        "ghs_classification": parsed.get("ghs_classification", []),
        "ghs_pictograms": ghs_enriched,
        "pubchem_cid": final_cid,
        "image_2d": images["image_2d"],
        "image_3d": images["image_3d"],
        "pubchem_url": images["pubchem_url"],
    }

    # 10. Build final text (Russian)
    guard_status = guard_result.get("overall_status", "НЕИЗВЕСТНО")
    status_ru = {
        "SAFE": "БЕЗОПАСНО",
        "WARNING": "ПРЕДУПРЕЖДЕНИЕ",
        "CRITICAL_STOP": "КРИТИЧЕСКАЯ ОСТАНОВКА",
    }.get(guard_status, guard_status)

    ppe_list = guard_result.get("ppe_recommendations", [])
    h_phrases = safety_data.get("h_phrases", [])
    p = molecule_info["properties"]

    # GHS pictogram display
    ghs_display = ""
    for pic in ghs_enriched:
        ghs_display += f"    {pic['code']} — {pic['name_ru']}: {pic['description']}\n"
        ghs_display += f"      Картинка: {pic['image_svg']}\n"

    # LD50 display
    ld50_display = ""
    for route, label in [("ld50_oral", "Перорально"), ("ld50_dermal", "Дермально"), ("ld50_inhalation", "Ингаляционно")]:
        val = ld50_data.get(route)
        if val:
            ld50_display += f"    {label}: {val}\n"

    final_text = (
        f"{'='*60}\n"
        f"  КАРТОЧКА МОЛЕКУЛЫ: {molecule_info['name']}\n"
        f"{'='*60}\n"
        f"  SMILES:         {molecule_info['smiles']}\n"
        f"  Формула:        {molecule_info['molecular_formula']}\n"
        f"  Мол. масса:     {molecule_info['molecular_weight']:.2f} г/моль\n"
        f"  CAS:            {molecule_info['cas_number'] or 'Н/Д'}\n"
        f"  PubChem CID:    {molecule_info['pubchem_cid']}\n"
        f"\n"
        f"  Физ. описание:  {molecule_info['physical_description']}\n"
        f"\n"
        f"  Свойства:\n"
        f"    Т. плавления:   {p['melting_point']} °C\n"
        f"    Т. кипения:     {p['boiling_point']} °C\n"
        f"    Растворимость:  {p['solubility']}\n"
        f"    Плотность:      {p['density']} г/мл\n"
        f"    LogP:           {p['logP']}\n"
        f"    Состояние:      {p['physical_state']}\n"
        f"    TPSA:           {p['tpsa']}\n"
        f"    Т. вспышки:     {p['flash_point'] or 'Н/Д'} °C\n"
        f"    H-доноры:       {p['h_bond_donors']}  H-акцепторы: {p['h_bond_acceptors']}\n"
        f"    Враш. связи:    {p['rotatable_bonds']}  Кольца: {p['ring_count']}\n"
        f"\n"
        f"  Описание: {molecule_info['description']}\n"
        f"\n"
        f"  Изображения:\n"
        f"    2D: {molecule_info['image_2d']}\n"
        f"    3D: {molecule_info['image_3d']}\n"
        f"    PubChem: {molecule_info['pubchem_url']}\n"
        f"\n"
        f"{'='*60}\n"
        f"  ОТЧЁТ О БЕЗОПАСНОСТИ\n"
        f"{'='*60}\n"
        f"  Статус:         {status_ru}\n"
        f"\n"
        f"  GHS пиктограммы:\n"
        f"{ghs_display if ghs_display else '    Нет\n'}"
        f"\n"
        f"  H-фразы:        {'; '.join(h_phrases[:5]) if h_phrases else 'Нет'}\n"
        f"  СИЗ:            {', '.join(ppe_list) if ppe_list else 'Стандартное лаб. оборудование'}\n"
    )

    if ld50_display:
        final_text += f"\n  Токсичность (LD50):\n{ld50_display}"

    final_text += f"{'='*60}\n"

    return {
        "molecule_info": molecule_info,
        "final_answer": final_text,
    }
