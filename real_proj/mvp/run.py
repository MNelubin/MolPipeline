#!/usr/bin/env python3
"""Точка входа для MVP пайплайна.

Использование:
    python -m real_proj.mvp.run "аспирин"
    python -m real_proj.mvp.run "CC(=O)Oc1ccccc1C(O)=O"
    python -m real_proj.mvp.run  # интерактивный режим
"""

from __future__ import annotations

import sys
import logging

# Загружаем конфиг первым (устанавливает env vars для LangSmith)
from . import config as _cfg  # noqa: F401
from .graph import build_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("mvp")


def run(query: str) -> dict:
    """Запуск MVP графа для одного запроса. Возвращает финальный стейт."""
    logger.info("Построение графа...")
    app = build_graph()

    logger.info("Запрос: %r", query)
    result = app.invoke({"query": query})

    # Вывод результата
    if result.get("error"):
        print(f"\n{'!'*60}")
        print(f"  ОШИБКА: {result['error']}")
        print(f"{'!'*60}")

        guard = result.get("guard_result", {})
        if guard:
            mol_check = guard.get("molecule_check", {})
            rxn_check = guard.get("reaction_check", {})
            if mol_check.get("status") in ("banned", "restricted"):
                print(f"\n  Вещество:   {mol_check.get('name', 'Неизвестно')}")
                print(f"  Статус:     {mol_check.get('status')}")
                print(f"  Категория:  {mol_check.get('category')}")
                print(f"  Причина:    {mol_check.get('reason')}")
            if rxn_check.get("status") in ("prohibited", "restricted"):
                print(f"\n  Реакция:    {rxn_check.get('reason')}")

        validation = result.get("validation", {})
        if validation and not validation.get("is_valid"):
            print(f"\n  Ошибка валидации: {validation.get('error')}")

    elif result.get("final_answer"):
        print(f"\n{result['final_answer']}")

    else:
        print("\n  Результат не получен.")

    return result


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("MVP Пайплайн: валидация → проверка безопасности → информация о молекуле")
        print("=" * 60)
        query = input("Введите молекулу (название или SMILES): ").strip()
        if not query:
            print("Пустой ввод. Выход.")
            sys.exit(0)

    run(query)


if __name__ == "__main__":
    main()
