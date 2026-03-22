# MolPipeline — ИИ-агент для ретросинтеза и планирования эксперимента

> **Хакатон-проект** — мультиагентный пайплайн на LangGraph, который принимает целевую молекулу и выдаёт полный пошаговый протокол синтеза, подкреплённый реальными химическими базами данных.

**Демо:** `https://hack.humaneconomy.ru`

---

## Что делает система

Химик вводит молекулу (название, SMILES, CAS или текст на русском). Система:

1. **Идентифицирует** молекулу через PubChem — канонический SMILES, формула, свойства, безопасность
2. **Проверяет** по банлисту контролируемых веществ — немедленно блокирует запрещённые
3. **Запускает ретросинтез** — сначала Open Reaction Database (ORD), затем веб-поиск (PubMed + DuckDuckGo), затем нейромодель ASKCOS
4. **Строит дерево синтеза** рекурсивно, пока каждый лист не окажется коммерчески доступным или заблокированным
5. **Оценивает и ранжирует** до 5 маршрутов по реализуемости, выходу и доступности реагентов
6. **Рассчитывает стехиометрию** — точные массы, объёмы, эквиваленты под заданную целевую массу
7. **Генерирует экспериментальный протокол** — пошаговая процедура для каждой стадии с обоснованиями, на русском языке

---

## Преимущества перед обычным LLM

| Критерий | Обычный LLM-промпт | MolPipeline |
|---|---|---|
| Источник реакций | Галлюцинированные / общие | ORD (реальные реакции) + веб-поиск + нейромодель |
| Количества реагентов | Отсутствуют или приблизительные | Точные г, мл, моль, экв под целевую массу |
| Глубина процедуры | 2–3 размытых предложения | 8 детальных шагов на стадию с обоснованием |
| Дерево синтеза | Один шаг | Рекурсивное дерево до покупаемых листьев |
| Проверка безопасности | Нет | Банлист + GHS-паспорт на каждом узле |
| Ранжирование путей | Нет | Score = выход × доступность × качество процедуры |
| Многостадийность | Не связаны | Стадии связаны — продукт стадии N = реагент стадии N+1 |

---

## Архитектура

```
Ввод пользователя
    │
    ▼
classify_node              ← эвристика: SMILES / название / исследовательский запрос
    │
    ▼
validate_and_guard_node    ← PubChem resolve + проверка банлиста
    │ (not_found)
    ▼
research_node              ← веб-поиск + LLM-синтез информации о молекуле
    │ (found)
    ▼
molecule_info_node         ← LLM-обогащение данных PubChem → карточка молекулы
    │
  [ПРЕРЫВАНИЕ #1 — пользователь подтверждает продолжение]
    │
    ▼
retrosynthesis_node        ← ORD → Web → ASKCOS → score_route → дерево
    │
  ┌─┴─────────────────┐
  ▼                   ▼
guard_safety_node   reagent_node     ← параллельный fan-out
  └─────────┬─────────┘
            ▼
      aggregate_node       ← объединение, ранжирование, выбор лучшего пути
            │
  [ПРЕРЫВАНИЕ #2 — пользователь выбирает маршрут + целевую массу]
            │
            ▼
    stoichiometry_node     ← точный расчёт масс/объёмов
            │
            ▼
  experiment_planner_node  ← генерация процедуры по стадиям через LLM
            │
           END → протокол, готовый к PDF
```

### Логика маршрутизации

| Условие | Следующий узел |
|---|---|
| SMILES/название найдено в PubChem | `molecule_info` |
| Не найдено, без предыдущего research | `research_node` (веб + LLM) |
| Не найдено после research | `END` с ошибкой |
| Попадание в банлист | `END` с предупреждением |
| ORD содержит реакцию | Использовать маршрут из ORD |
| ORD пуст | Веб-поиск → ASKCOS нейромодель |
| Лист покупаемый | `status=buyable`, остановка ветки |
| Глубина ≥ 6 или таймаут | `status=depth_limit/timeout` |

---

## Стек технологий

| Слой | Технология |
|---|---|
| Граф агентов | LangGraph `StateGraph` с `MemorySaver` checkpointer |
| LLM | OpenRouter API (GPT-4o и другие модели) |
| Ретросинтез (БД) | Open Reaction Database (ORD) — SQLite, 1M+ реакций |
| Ретросинтез (веб) | PubMed + DuckDuckGo + LLM-экстракция маршрутов |
| Ретросинтез (ML) | ASKCOS Molecular Transformer |
| Валидация молекул | RDKit + PubChem REST API |
| Безопасность | Банлист (ФСКН, КХО, двойного назначения) + PubChem GHS |
| Backend API | FastAPI + SSE-стриминг |
| Frontend | React 18 + Vite, ReactFlow визуализация графов |
| Стехиометрия | Калькулятор масс/объёмов с PubChem-данными о плотности |
| Деплой | Systemd + Caddy reverse proxy |

---

## Структура проекта

```
real_proj/
├── mvp/                              # Основной бэкенд
│   ├── api.py                        # FastAPI приложение (SSE, прерывания)
│   ├── graph.py                      # LangGraph StateGraph — определение графа
│   ├── state.py                      # MVPState TypedDict
│   ├── config.py                     # Фабрика LLM, прокси, переменные окружения
│   ├── journal.py                    # Журнал решений агента
│   ├── procedure_inference.py        # Генерация процедур через LLM
│   ├── tree_expansion.py             # Рекурсивное раскрытие дерева синтеза
│   ├── retro_predictor.py            # Standalone ретросинтез-модель
│   ├── retro_tools.py                # ORD-поиск, веб-поиск, скоринг
│   ├── tools.py                      # PubChem-тулы, банлист, покупаемость
│   ├── nodes/                        # Узлы графа (11 шт.)
│   │   ├── classify_node.py          #   Классификатор ввода
│   │   ├── validate_and_guard_node.py #  PubChem-резолв + банлист
│   │   ├── research_node.py          #   Веб-исследование (fallback)
│   │   ├── molecule_info_node.py     #   LLM-карточка молекулы
│   │   ├── retrosynthesis_node.py    #   ORD + Web + ASKCOS + дерево
│   │   ├── guard_safety_node.py      #   GHS-безопасность по маршрутам
│   │   ├── reagent_node.py           #   Проверка покупаемости
│   │   ├── aggregate_node.py         #   Объединение и ранжирование
│   │   ├── stoichiometry_node.py     #   Расчёт стехиометрии
│   │   └── experiment_planner_node.py #  Генерация протокола
│   ├── tools/                        # Инструменты (8 модулей)
│   │   ├── calculations.py           #   Стехиометрический калькулятор
│   │   ├── rdkit_tools.py            #   RDKit: свойства, SMILES, парсинг реакций
│   │   ├── pubchem.py                #   PubChem API: плотность, имена, CID
│   │   ├── retro_tools.py            #   ORD + веб-поиск + скоринг
│   │   ├── safety.py                 #   Проверка безопасности
│   │   ├── research.py               #   Поиск информации
│   │   └── rag_search.py             #   RAG-поиск
│   ├── models/                       # Pydantic-модели
│   │   ├── calculations.py           #   StoichiometryRequest, CalculationResult
│   │   ├── validation.py             #   ValidationResult
│   │   └── research.py               #   ResearchResult
│   ├── services/                     # Внешние сервисы
│   │   ├── research_llm.py           #   LLM-исследование
│   │   ├── web_search.py             #   DuckDuckGo + PubMed
│   │   └── web_scraper.py            #   Парсинг веб-страниц
│   ├── rag/                          # RAG-система
│   │   ├── bm25.py                   #   BM25-ранжирование
│   │   ├── embeddings.py             #   Эмбеддинги
│   │   ├── retriever.py              #   Ретривер
│   │   └── models.py                 #   Модели RAG
│   └── tests/                        # Тесты (284 юнит-теста)
│       ├── test_classify_node.py
│       ├── test_validate_and_guard_node.py
│       ├── test_molecule_info_node.py
│       ├── test_retrosynthesis_node.py
│       ├── test_stoichiometry_node.py
│       ├── test_tree_expansion.py
│       ├── test_graph.py
│       └── ...
├── backend/                          # Вспомогательный бэкенд
│   ├── main.py                       # FastAPI: 2D/3D молекул, калькулятор
│   └── calculator_combined.py        # Стехиометрический калькулятор (standalone)
└── frontend/                         # React-фронтенд
    ├── index.html
    ├── package.json
    ├── vite.config.js
    ├── public/
    │   └── favicon.svg
    └── src/
        ├── App.jsx                   # Главный компонент + localStorage-персистенция
        ├── main.jsx
        ├── components/
        │   ├── MoleculeCard.jsx      #   Карточка молекулы (вкладки)
        │   ├── RetroCard.jsx         #   Маршруты синтеза
        │   ├── SynthesisGraph.jsx    #   ReactFlow-визуализация дерева
        │   ├── SynthesisTree.jsx     #   Раскрывающееся дерево
        │   ├── ExperimentProtocol.jsx #  Протокол эксперимента
        │   ├── ProtocolGraph.jsx     #   Визуализация протокола
        │   ├── CalculatorCard.jsx    #   Стехиометрический калькулятор
        │   ├── PathwaySelector.jsx   #   Выбор маршрута синтеза
        │   ├── ModelSelector.jsx     #   Выбор LLM-модели
        │   ├── PipelineProgress.jsx  #   Индикатор прогресса
        │   ├── TestPage.jsx          #   Страница запуска тестов
        │   └── Viewer3D.jsx          #   3D-визуализация молекулы
        ├── hooks/
        │   ├── useInteractivePipeline.js  # Хук управления пайплайном
        │   └── useSSEPipeline.js     #   SSE-подключение
        └── styles/
            └── global.css            # Глобальные стили
```

---

## Данные

Проект использует следующие данные (не включены в репозиторий):

| Файл / директория | Описание | Как получить |
|---|---|---|
| `mvp/data/banned_chemicals.json` | Банлист контролируемых веществ (150 записей). Источники: CWC Schedules, ФСКН РФ, списки двойного назначения. Уровни: critical / high / medium | Скрипт `scripts/download_banned_data.py` |
| `mvp/data/banned_reactions.json` | Запрещённые типы реакций (19 записей). Источники: маршруты синтеза по КХО, реакции двойного назначения | Скрипт `scripts/download_banned_data.py` |
| ORD SQLite БД | Open Reaction Database — 1M+ реакций, индексированных по продукту. Используется для поиска реальных синтезов | Скрипт `scripts/build_ord_index.py` |

### Формат банлиста химикатов

```json
{
  "_meta": {
    "total": 150,
    "stats": { "critical": 36, "high": 78, "medium": 36 },
    "sources": ["CWC Schedules", "ФСКН РФ", ...]
  },
  "chemicals": [
    {
      "name": "...",
      "smiles": "...",
      "cas": "...",
      "severity": "critical",
      "category": "CWC Schedule 1",
      "reason": "..."
    }
  ]
}
```

### Формат банлиста реакций

```json
{
  "_meta": { "total": 19, "stats": { "critical": 9, "high": 8, "medium": 2 } },
  "reactions": [
    {
      "name": "...",
      "pattern": "...",
      "severity": "critical",
      "reason": "..."
    }
  ]
}
```

---

## API-эндпоинты

| Метод | Путь | Описание |
|---|---|---|
| `POST` | `/analyze` | Запуск пайплайна (SSE-стрим) |
| `GET` | `/stream/{thread_id}` | Возобновление интерактивной сессии |
| `POST` | `/resume/{thread_id}` | Продолжение после прерывания (выбор маршрута) |
| `POST` | `/tree/expand` | Рекурсивное дерево синтеза для маршрута |
| `POST` | `/api/calculate` | Стехиометрический калькулятор |
| `POST` | `/tests/run` | Запуск pytest, возвращает JSON-результаты |
| `GET` | `/health` | Health check |

### SSE-события

```
pipeline_start  →  { query, model }
node_start      →  { node, label }
node_complete   →  { node, label, output }
interrupt       →  { phase: "card_ready" | "select_pathway", payload }
pipeline_done   →  {}
error           →  { message }
```

---

## Переменные окружения

```env
# Обязательные
OPENROUTER_API_KEY=sk-or-...       # Ключ OpenRouter для LLM
LLM_MODEL=openai/gpt-4o            # Модель по умолчанию

# Опциональные
SOCKS_PROXY=socks5://user:pass@host:port   # Прокси для гео-блокировок
LANGSMITH_API_KEY=lsv2_pt_...              # LangSmith трейсинг
LANGSMITH_PROJECT=hackaton
ASKCOS_BASE_URL=http://localhost:9100      # ASKCOS (self-hosted)
```

---

## Запуск локально

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Настроить переменные окружения
cp .env.example .env
# Заполнить OPENROUTER_API_KEY

# 3. Подготовить данные
python scripts/download_banned_data.py
python scripts/build_ord_index.py

# 4. Запустить API агента
uvicorn real_proj.mvp.api:app --port 8765 --reload

# 5. Запустить бэкенд калькулятора / молекул
uvicorn real_proj.backend.main:app --port 8002 --reload

# 6. Запустить фронтенд
cd real_proj/frontend
npm install && npm run dev
# → http://localhost:5173
```

---

## Запуск тестов

```bash
# Быстрые юнит-тесты (без сети, без LLM) — ~2.3 мин, 284 теста
pytest real_proj/mvp/tests/ -m "not integration and not slow and not llm"

# Все тесты включая интеграционные (PubChem + ORD) — ~5 мин
pytest real_proj/mvp/tests/
```

---

## Лицензия

Хакатон-проект. Все права защищены.
