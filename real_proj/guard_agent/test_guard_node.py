"""
test_guard_node.py — Микротест guard_node без реальных зависимостей.

Стратегия: инжектируем фиктивные модули в sys.modules ДО импорта guard_node,
чтобы обойти цепочку langchain → chromadb → rdkit → HuggingFace.
Тестируем саму логику ноды: маршрутизацию стейта и агрегацию статусов.

Запуск:
    python -m pytest test_guard_node.py -v
    # или напрямую:
    python test_guard_node.py
"""

import sys
import types
import unittest
from unittest.mock import MagicMock


# ─── 1. Заглушки тяжёлых зависимостей ────────────────────────────────────────
# Вставляем фиктивный модуль tools ЦЕЛИКОМ до любого импорта guard_node.
# Это единственный надёжный способ: patch() уже требует, чтобы модуль
# был импортирован, а guard_node при импорте тянет tools → rag → rdkit и т.д.

def _make_fake_tool(return_value):
    """Объект, имитирующий @tool LangChain: вызов через .invoke()."""
    fake = MagicMock()
    fake.invoke = MagicMock(return_value=return_value)
    return fake


# Создаём фиктивный модуль tools с нейтральными заглушками по умолчанию
_fake_tools_module = types.ModuleType("tools")
_fake_tools_module.banlist_check          = _make_fake_tool({})
_fake_tools_module.reaction_banlist_check = _make_fake_tool({})
_fake_tools_module.safety_lookup          = _make_fake_tool({})
_fake_tools_module.ppe_recommender        = _make_fake_tool([])

sys.modules["tools"] = _fake_tools_module

# Убеждаемся, что guard_node будет импортирован заново (чистый кэш)
sys.modules.pop("guard_node", None)

# ─── 2. Фикстуры ответов инструментов ────────────────────────────────────────

CLEAR_MOL = {"smiles": "CCO", "name": None, "status": "clear",
             "category": None, "danger_level": None,
             "reason": "Not found in banlists."}

BANNED_MOL = {"smiles": "C1=CC=CC=C1", "name": "Benzene", "status": "banned",
              "category": "carcinogen", "danger_level": "high",
              "reason": "Exact match in banlist: Benzene."}

RESTRICTED_MOL = {"smiles": "CCO", "name": "Ethanol", "status": "restricted",
                  "category": "flammable", "danger_level": "medium",
                  "reason": "Restricted substance."}

ALLOWED_RXN = {"status": "allowed", "reason": "No prohibited patterns.", "matched_pattern": None}
PROHIBITED_RXN = {"status": "prohibited", "reason": "Semantic match: nitroglycerin synthesis.",
                  "matched_pattern": "[N+](=O)[O-]"}
RESTRICTED_RXN = {"status": "restricted", "reason": "Restricted process.", "matched_pattern": None}

SAFETY_FULL = {
    "ghs_pictograms": ["GHS02", "GHS07"],
    "h_phrases": ["H225 Highly flammable liquid", "H319 Causes serious eye irritation"],
    "p_phrases": ["P210 Keep away from heat"],
    "ld50": None,
    "flash_point": None,
}
SAFETY_EMPTY = {"ghs_pictograms": [], "h_phrases": [], "p_phrases": [],
                "ld50": None, "flash_point": None}

PPE_BASIC = ["Lab coat", "Nitrile gloves", "Safety goggles"]
PPE_FULL  = ["Explosion-proof equipment nearby", "Flame-resistant lab coat",
             "Lab coat", "Nitrile gloves", "Safety goggles"]


# ─── 3. Тесты ─────────────────────────────────────────────────────────────────

class TestGuardNode(unittest.TestCase):

    def _run(self, mol, rxn, safety, ppe, smiles="CCO", reaction_desc="test reaction"):
        """Вспомогательный метод: подменяет инструменты прямо в модуле guard_node и вызывает ноду."""
        import guard_node as gn
        # Подменяем атрибуты модуля напрямую — guard_node уже импортирован
        _orig = (gn.banlist_check, gn.reaction_banlist_check,
                 gn.safety_lookup, gn.ppe_recommender)
        gn.banlist_check          = _make_fake_tool(mol)
        gn.reaction_banlist_check = _make_fake_tool(rxn)
        gn.safety_lookup          = _make_fake_tool(safety)
        gn.ppe_recommender        = _make_fake_tool(ppe)
        try:
            state = {"smiles": smiles, "reaction_description": reaction_desc}
            return gn.guard_node(state)
        finally:
            # Восстанавливаем оригинальные заглушки после каждого теста
            (gn.banlist_check, gn.reaction_banlist_check,
             gn.safety_lookup, gn.ppe_recommender) = _orig

    # ── Базовая структура ответа ───────────────────────────────────────────────

    def test_returns_guard_result_key(self):
        """Нода возвращает словарь с ключом guard_result."""
        result = self._run(CLEAR_MOL, ALLOWED_RXN, SAFETY_EMPTY, PPE_BASIC)
        self.assertIn("guard_result", result)

    def test_guard_result_has_all_fields(self):
        """guard_result содержит все обязательные поля."""
        gr = self._run(CLEAR_MOL, ALLOWED_RXN, SAFETY_EMPTY, PPE_BASIC)["guard_result"]
        for field in ("overall_status", "molecule_check", "reaction_check",
                      "safety_data", "ppe_recommendations"):
            with self.subTest(field=field):
                self.assertIn(field, gr)

    # ── overall_status агрегация ───────────────────────────────────────────────

    def test_safe_when_all_clear(self):
        """SAFE: молекула clear, реакция allowed."""
        gr = self._run(CLEAR_MOL, ALLOWED_RXN, SAFETY_EMPTY, PPE_BASIC)["guard_result"]
        self.assertEqual(gr["overall_status"], "SAFE")

    def test_critical_stop_on_banned_molecule(self):
        """CRITICAL_STOP: молекула banned, реакция allowed."""
        gr = self._run(BANNED_MOL, ALLOWED_RXN, SAFETY_EMPTY, PPE_BASIC)["guard_result"]
        self.assertEqual(gr["overall_status"], "CRITICAL_STOP")

    def test_critical_stop_on_prohibited_reaction(self):
        """CRITICAL_STOP: молекула clear, реакция prohibited."""
        gr = self._run(CLEAR_MOL, PROHIBITED_RXN, SAFETY_EMPTY, PPE_BASIC)["guard_result"]
        self.assertEqual(gr["overall_status"], "CRITICAL_STOP")

    def test_critical_stop_when_both_critical(self):
        """CRITICAL_STOP: и молекула banned, и реакция prohibited."""
        gr = self._run(BANNED_MOL, PROHIBITED_RXN, SAFETY_EMPTY, PPE_BASIC)["guard_result"]
        self.assertEqual(gr["overall_status"], "CRITICAL_STOP")

    def test_warning_on_restricted_molecule(self):
        """WARNING: молекула restricted, реакция allowed."""
        gr = self._run(RESTRICTED_MOL, ALLOWED_RXN, SAFETY_EMPTY, PPE_BASIC)["guard_result"]
        self.assertEqual(gr["overall_status"], "WARNING")

    def test_warning_on_restricted_reaction(self):
        """WARNING: молекула clear, реакция restricted."""
        gr = self._run(CLEAR_MOL, RESTRICTED_RXN, SAFETY_EMPTY, PPE_BASIC)["guard_result"]
        self.assertEqual(gr["overall_status"], "WARNING")

    def test_critical_beats_warning(self):
        """CRITICAL_STOP превалирует над WARNING (banned + restricted)."""
        gr = self._run(BANNED_MOL, RESTRICTED_RXN, SAFETY_EMPTY, PPE_BASIC)["guard_result"]
        self.assertEqual(gr["overall_status"], "CRITICAL_STOP")

    # ── Проброс данных инструментов ───────────────────────────────────────────

    def test_safety_data_passed_through(self):
        """safety_data из safety_lookup попадает в guard_result без изменений."""
        gr = self._run(CLEAR_MOL, ALLOWED_RXN, SAFETY_FULL, PPE_FULL)["guard_result"]
        self.assertEqual(gr["safety_data"]["ghs_pictograms"], ["GHS02", "GHS07"])
        self.assertEqual(len(gr["safety_data"]["h_phrases"]), 2)

    def test_ppe_list_passed_through(self):
        """Список СИЗ из ppe_recommender попадает в guard_result."""
        gr = self._run(CLEAR_MOL, ALLOWED_RXN, SAFETY_FULL, PPE_FULL)["guard_result"]
        self.assertIn("Lab coat", gr["ppe_recommendations"])

    def test_molecule_check_passed_through(self):
        """molecule_check содержит данные из banlist_check."""
        gr = self._run(BANNED_MOL, ALLOWED_RXN, SAFETY_EMPTY, PPE_BASIC)["guard_result"]
        self.assertEqual(gr["molecule_check"]["name"], "Benzene")
        self.assertEqual(gr["molecule_check"]["status"], "banned")

    # ── Граничные случаи ──────────────────────────────────────────────────────

    def test_raises_on_empty_smiles(self):
        """ValueError при пустом smiles в стейте."""
        import guard_node as gn
        with self.assertRaises(ValueError):
            gn.guard_node({"smiles": ""})

    def test_raises_on_missing_smiles(self):
        """ValueError при отсутствии smiles в стейте."""
        import guard_node as gn
        with self.assertRaises(ValueError):
            gn.guard_node({})

    def test_reaction_description_defaults_to_empty(self):
        """Нода работает без reaction_description в стейте."""
        result = self._run(CLEAR_MOL, ALLOWED_RXN, SAFETY_EMPTY, PPE_BASIC,
                           reaction_desc="")
        self.assertIn("guard_result", result)

    def test_return_is_partial_state_update(self):
        """Нода возвращает ровно один ключ — только изменение стейта."""
        result = self._run(CLEAR_MOL, ALLOWED_RXN, SAFETY_EMPTY, PPE_BASIC)
        self.assertEqual(list(result.keys()), ["guard_result"])


# ─── 4. Запуск ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(unittest.TestLoader().loadTestsFromTestCase(TestGuardNode))
    sys.exit(0 if result.wasSuccessful() else 1)
