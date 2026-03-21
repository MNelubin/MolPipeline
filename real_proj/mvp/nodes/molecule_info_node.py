"""Molecule info node: gather data + LLM synthesis via OpenRouter.

Outputs everything in Russian. Includes physical description,
2D/3D images, and full safety report.
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
)

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

Верни JSON-объект со следующими полями. Если данных нет — используй свои знания.

Поля JSON:
- "name": название вещества (русское общеупотребительное + IUPAC в скобках)
- "synonyms": список синонимов на русском (список строк)
- "smiles": SMILES-строка
- "molecular_formula": брутто-формула
- "molecular_weight": молярная масса (число)
- "physical_description": описание физических свойств НА РУССКОМ — внешний вид, цвет, запах, агрегатное состояние, кристаллическая структура. Опирайся на данные из PubChem Physical Description.
- "properties": словарь с ключами:
    - "melting_point": температура плавления (°C)
    - "boiling_point": температура кипения (°C)
    - "solubility": растворимость (описание на русском)
    - "density": плотность (г/мл)
    - "logP": коэффициент распределения
    - "physical_state": агрегатное состояние на русском (твёрдое/жидкое/газ)
    - "tpsa": площадь полярной поверхности
    - "h_bond_donors": доноры водородных связей (число)
    - "h_bond_acceptors": акцепторы водородных связей (число)
    - "rotatable_bonds": вращаемые связи (число)
    - "ring_count": количество колец (число)
- "ghs_classification": список классов опасности GHS на русском (список строк)
- "spectral_notes": краткая заметка о спектральных данных (ИК, ЯМР) на русском
- "description": подробное описание вещества на русском (применение, значение, история)
- "pubchem_cid": CID число (0 если неизвестно)

ВАЖНО: Данные RDKit (масса, logP, TPSA и т.д.) приоритетнее вычисленных PubChem.
Верни ТОЛЬКО валидный JSON, без markdown-блоков кода.
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
    """LangGraph node: gather molecule info and produce a structured card.

    Reads: state["query"], state["smiles"], state["guard_result"]
    Writes: state["molecule_info"], state["final_answer"]
    """
    query = state.get("query", "")
    smiles = state.get("smiles", "")
    guard_result = state.get("guard_result", {})
    cid: int | None = state.get("pubchem_cid") or None

    logger.info("[molecule_info] query=%r smiles=%r cid=%s", query, smiles, cid)

    # 1. Get data from PubChem and RDKit
    pubchem_result = {}
    rdkit_result = {}

    if smiles:
        rdkit_result = rdkit_properties(smiles)
        pubchem_data = pubchem_lookup(smiles)
        if "error" not in pubchem_data:
            pubchem_result = pubchem_data
        else:
            pubchem_data = pubchem_lookup(query)
            if "error" not in pubchem_data:
                pubchem_result = pubchem_data

    # Use CID from state (resolved in validate), fallback to pubchem_result
    if not cid:
        cid = pubchem_result.get("cid")

    # 2. Physical description from PubChem (pass CID directly)
    phys_desc_list = get_physical_description(smiles, cid=cid) if smiles else []
    phys_desc_str = " | ".join(phys_desc_list[:5]) if phys_desc_list else ""
    logger.info("[molecule_info] physical_description entries: %d", len(phys_desc_list))

    # 3. 2D/3D image URLs (pass CID directly)
    images = get_molecule_images(smiles, cid=cid)
    logger.info("[molecule_info] images: 2d=%s 3d=%s", bool(images["image_2d"]), bool(images["image_3d"]))

    # 4. Safety data
    safety_data = guard_result.get("safety_data", {})

    # 5. LLM synthesis (Russian)
    llm = _get_llm()
    prompt_value = _MOLECULE_CARD_PROMPT.format(
        query=query,
        pubchem_data=json.dumps(pubchem_result, ensure_ascii=False),
        rdkit_data=json.dumps(rdkit_result, ensure_ascii=False),
        safety_data=json.dumps(safety_data, ensure_ascii=False),
        physical_description=phys_desc_str,
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

    # 6. Build molecule_info
    props = parsed.get("properties", {})

    rdkit_weight = rdkit_result.get("molecular_weight")
    parsed_weight = parsed.get("molecular_weight")
    final_weight = _safe_float(rdkit_weight if rdkit_weight else parsed_weight)

    parsed_cid = parsed.get("pubchem_cid")
    final_cid = _safe_int(parsed_cid if parsed_cid not in ("", None) else cid)

    molecule_info = {
        "name": parsed.get("name", pubchem_result.get("iupac", "Неизвестно")),
        "synonyms": parsed.get("synonyms", pubchem_result.get("synonyms", [])),
        "smiles": parsed.get("smiles") or smiles or "",
        "molecular_formula": parsed.get("molecular_formula", pubchem_result.get("formula", "")),
        "molecular_weight": final_weight,
        "physical_description": parsed.get("physical_description", phys_desc_str or "Нет данных"),
        "properties": {
            "melting_point": props.get("melting_point", "Н/Д"),
            "boiling_point": props.get("boiling_point", "Н/Д"),
            "solubility": props.get("solubility", "Н/Д"),
            "density": props.get("density", "Н/Д"),
            "logP": props.get("logP", rdkit_result.get("logp")),
            "physical_state": props.get("physical_state", "Н/Д"),
            "tpsa": props.get("tpsa", rdkit_result.get("tpsa")),
            "h_bond_donors": props.get("h_bond_donors", rdkit_result.get("h_bond_donors")),
            "h_bond_acceptors": props.get("h_bond_acceptors", rdkit_result.get("h_bond_acceptors")),
            "rotatable_bonds": props.get("rotatable_bonds", rdkit_result.get("rotatable_bonds")),
            "ring_count": props.get("ring_count", rdkit_result.get("ring_count")),
            "spectral_notes": parsed.get("spectral_notes", "Н/Д"),
        },
        "description": parsed.get("description", ""),
        "ghs_classification": parsed.get("ghs_classification", []),
        "pubchem_cid": final_cid,
        "image_2d": images["image_2d"],
        "image_3d": images["image_3d"],
        "pubchem_url": images["pubchem_url"],
    }

    # 7. Build final text answer (Russian)
    guard_status = guard_result.get("overall_status", "НЕИЗВЕСТНО")
    status_ru = {
        "SAFE": "БЕЗОПАСНО",
        "WARNING": "ПРЕДУПРЕЖДЕНИЕ",
        "CRITICAL_STOP": "КРИТИЧЕСКАЯ ОСТАНОВКА",
    }.get(guard_status, guard_status)

    ppe_list = guard_result.get("ppe_recommendations", [])
    h_phrases = safety_data.get("h_phrases", [])
    ghs_pics = safety_data.get("ghs_pictograms", [])
    p = molecule_info["properties"]

    final_text = (
        f"{'='*60}\n"
        f"  КАРТОЧКА МОЛЕКУЛЫ: {molecule_info['name']}\n"
        f"{'='*60}\n"
        f"  SMILES:         {molecule_info['smiles']}\n"
        f"  Формула:        {molecule_info['molecular_formula']}\n"
        f"  Мол. масса:     {molecule_info['molecular_weight']:.2f} г/моль\n"
        f"  PubChem CID:    {molecule_info['pubchem_cid']}\n"
        f"\n"
        f"  Физ. описание:  {molecule_info['physical_description']}\n"
        f"\n"
        f"  Свойства:\n"
        f"    Т. плавления:   {p['melting_point']}\n"
        f"    Т. кипения:     {p['boiling_point']}\n"
        f"    Растворимость:  {p['solubility']}\n"
        f"    Плотность:      {p['density']}\n"
        f"    LogP:           {p['logP']}\n"
        f"    Состояние:      {p['physical_state']}\n"
        f"    TPSA:           {p['tpsa']}\n"
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
        f"  GHS пиктограммы: {', '.join(ghs_pics) if ghs_pics else 'Нет'}\n"
        f"  H-фразы:        {'; '.join(h_phrases[:5]) if h_phrases else 'Нет'}\n"
        f"  СИЗ:            {', '.join(ppe_list) if ppe_list else 'Стандартное лаб. оборудование'}\n"
        f"{'='*60}\n"
    )

    return {
        "molecule_info": molecule_info,
        "final_answer": final_text,
    }
