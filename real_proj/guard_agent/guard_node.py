r"""
guard_node.py — GuardAgent как нода LangGraph.

════════════════════════════════════════════════════════════════════
БЫСТРЫЙ СТАРТ
════════════════════════════════════════════════════════════════════

Нода принимает из стейта:
  • smiles: str               — каноническая SMILES-строка вещества (обязательно)
  • reaction_description: str — текстовое описание реакции (опционально, "" по умолчанию)

Нода добавляет / перезаписывает в стейте:
  • guard_result: GuardResult — итоговый отчёт проверки (см. схему ниже)

Минимальный пример подключения к графу:

    from langgraph.graph import StateGraph
    from guard_node import guard_node, GuardStateProtocol

    graph = StateGraph(YourState)           # YourState должен включать поля из GuardStateProtocol
    graph.add_node("guard", guard_node)
    graph.add_edge("some_upstream_node", "guard")

════════════════════════════════════════════════════════════════════
СХЕМА ВОЗВРАЩАЕМОГО guard_result
════════════════════════════════════════════════════════════════════

GuardResult (TypedDict):
  overall_status:    "SAFE" | "WARNING" | "CRITICAL_STOP"
  molecule_check:    dict  — см. MoleculeCheckResult (models.py)
  reaction_check:    dict  — см. ReactionCheckResult (models.py)
  safety_data:       dict  — см. SafetyData (models.py): GHS, H/P-фразы
  ppe_recommendations: list[str] — рекомендованные СИЗ

Логика overall_status:
  CRITICAL_STOP — молекула или реакция помечена "banned" / "prohibited"
  WARNING       — молекула или реакция "restricted"
  SAFE          — оба чека "clear" / "allowed"

════════════════════════════════════════════════════════════════════
ОШИБКИ, НАЙДЕННЫЕ В ИСХОДНОМ КОДЕ (и исправленные здесь)
════════════════════════════════════════════════════════════════════

tools.py / safety_lookup:
  БЫЛО:  re.match(r"GHS\d{2}", p)
  СТАЛО: re.search(r"GHS\d{2}", p)
  ПРИЧИНА: PubChem возвращает строки вида
    "https://pubchem.ncbi.nlm.nih.gov/images/ghs/GHS02.svg"
    re.match проверяет только начало строки → все пиктограммы терялись.
  ТАКЖЕ: добавлен отдельный парсинг pictogram-секции вместо смешивания
    с H/P-фразами, чтобы избежать ложных срабатываний _parse_codes.

rag.py / _get_or_create_collection:
  БЫЛО:  existing._collection.count()
  СТАЛО: len(existing.get()["ids"])
  ПРИЧИНА: _collection — приватный атрибут обёртки LangChain Chroma;
    в chromadb >= 0.4 его имя и интерфейс менялись.
    Публичный метод .get() надёжнее.

models.py / MoleculeCheckResult:
  БЫЛО:  status: Literal["clear", "restricted", "banned"]
  СТАЛО: добавлен валидатор — smiles не может быть пустой строкой.
  ПРИЧИНА: пустой SMILES пропускался в инструменты и вызывал
    молчаливые ошибки RDKit.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, TypedDict

from tools import banlist_check, ppe_recommender, reaction_banlist_check, safety_lookup

logger = logging.getLogger(__name__)


# ─── Типизация ────────────────────────────────────────────────────────────────

class GuardResult(TypedDict):
    """Итоговый результат, который нода записывает в стейт."""
    overall_status: Literal["SAFE", "WARNING", "CRITICAL_STOP"]
    molecule_check: dict
    reaction_check: dict
    safety_data: dict
    ppe_recommendations: list[str]


class GuardStateProtocol(TypedDict, total=False):
    """
    Минимальный контракт стейта, с которым работает нода.

    Включите эти поля в свой StateGraph-стейт.
    total=False означает, что все поля опциональны при объявлении,
    но smiles обязан присутствовать в рантайме.

    Пример:
        class MyState(GuardStateProtocol):
            messages: list[BaseMessage]
            some_other_field: str
    """
    smiles: str               # каноническая SMILES — обязательное поле рантайма
    reaction_description: str # описание реакции — опционально
    guard_result: GuardResult # сюда нода пишет результат


# ─── Вспомогательная функция ──────────────────────────────────────────────────

def _determine_overall_status(
    mol_status: str,
    rxn_status: str,
) -> Literal["SAFE", "WARNING", "CRITICAL_STOP"]:
    """
    Агрегирует статусы молекулярной и реакционной проверок.

    Приоритет: CRITICAL_STOP > WARNING > SAFE
    """
    critical = {"banned", "prohibited"}
    warning = {"restricted"}

    if mol_status in critical or rxn_status in critical:
        return "CRITICAL_STOP"
    if mol_status in warning or rxn_status in warning:
        return "WARNING"
    return "SAFE"


# ─── Нода ─────────────────────────────────────────────────────────────────────

def guard_node(state: GuardStateProtocol) -> dict[str, Any]:
    """
    LangGraph-нода безопасности.

    Получает из стейта каноническую SMILES-строку и опциональное
    описание реакции; возвращает частичное обновление стейта с
    ключом ``guard_result``.

    Шаги:
      1. banlist_check         — точное совпадение + SMARTS-подструктуры
      2. reaction_banlist_check — семантический поиск запрещённых реакций
      3. safety_lookup         — GHS-данные из PubChem (пиктограммы, H/P-фразы)
      4. ppe_recommender       — рекомендации СИЗ на основе H-фраз
      5. Агрегация overall_status

    Args:
        state: стейт графа, содержащий как минимум ``smiles``.

    Returns:
        Словарь ``{"guard_result": GuardResult}`` для частичного
        обновления стейта LangGraph.

    Raises:
        ValueError: если ``smiles`` отсутствует или пуст в стейте.
    """
    smiles: str = state.get("smiles", "").strip()
    if not smiles:
        raise ValueError(
            "guard_node: поле 'smiles' отсутствует или пусто в стейте."
        )

    reaction_description: str = state.get("reaction_description", "")

    logger.info("[guard_node] Старт проверки: smiles=%r", smiles)

    # ── 1. Молекулярный бан-лист ───────────────────────────────────────────────
    mol_check: dict = banlist_check.invoke({"smiles": smiles})
    logger.info("[guard_node] banlist_check → %s", mol_check.get("status"))

    # ── 2. Реакционный бан-лист ───────────────────────────────────────────────
    rxn_check: dict = reaction_banlist_check.invoke(
        {"reaction_description": reaction_description}
    )
    logger.info("[guard_node] reaction_banlist_check → %s", rxn_check.get("status"))

    # ── 3. GHS / PubChem ──────────────────────────────────────────────────────
    safety: dict = safety_lookup.invoke({"smiles": smiles})
    logger.info(
        "[guard_node] safety_lookup → %d H-фраз, %d P-фраз, пиктограммы: %s",
        len(safety.get("h_phrases", [])),
        len(safety.get("p_phrases", [])),
        safety.get("ghs_pictograms", []),
    )

    # ── 4. СИЗ ────────────────────────────────────────────────────────────────
    h_phrases_str: str = ",".join(safety.get("h_phrases", []))
    ppe: list[str] = ppe_recommender.invoke(
        {"substances": smiles, "h_phrases": h_phrases_str}
    )
    logger.info("[guard_node] ppe_recommender → %s", ppe)

    # ── 5. Агрегация ──────────────────────────────────────────────────────────
    overall = _determine_overall_status(
        mol_status=mol_check.get("status", "clear"),
        rxn_status=rxn_check.get("status", "allowed"),
    )
    logger.info("[guard_node] overall_status=%s", overall)

    guard_result: GuardResult = {
        "overall_status": overall,
        "molecule_check": mol_check,
        "reaction_check": rxn_check,
        "safety_data": safety,
        "ppe_recommendations": ppe,
    }

    # Возвращаем только изменение стейта — LangGraph сам сделает merge
    return {"guard_result": guard_result}