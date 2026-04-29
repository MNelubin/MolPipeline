# MolPipeline — ИИ-агент для ретросинтеза и планирования эксперимента

> Мультиагентный пайплайн на LangGraph, который принимает целевую молекулу и выдаёт полный пошаговый протокол синтеза, подкреплённый реальными химическими базами данных.

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
| Ретросинтез (БД) | Open Reaction Database (ORD) — SQLite, 2.3M реакций |
| Ретросинтез (веб) | PubMed + DuckDuckGo + LLM-экстракция маршрутов |
| Ретросинтез (ML) | Template-relevance нейромодель (163K шаблонов, ~192 MB) |
| RAG | SPECTER2 эмбеддинги + ChromaDB + BM25 гибридный поиск |
| Валидация молекул | RDKit + PubChem REST API |
| Безопасность | Банлист (ФСКН, КХО, двойного назначения) + PubChem GHS |
| Покупаемость | БД buyables (690K+ молекул от eMolecules, Mcule, ChemBridge, ChemSpace) |
| Backend API | FastAPI + SSE-стриминг |
| Frontend | React 18 + Vite, ReactFlow визуализация графов |
| Стехиометрия | Калькулятор масс/объёмов с PubChem-данными о плотности |
| Деплой | Systemd + Caddy reverse proxy |

---

## Структура проекта

```
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
│   │   ├── retro_tools.py            #   ORD + веб-поиск + скоринг + покупаемость
│   │   ├── safety.py                 #   Проверка безопасности
│   │   ├── research.py               #   Поиск информации
│   │   └── rag_search.py             #   RAG-поиск по научной литературе
│   ├── models/                       # Pydantic-модели
│   │   ├── calculations.py           #   StoichiometryRequest, CalculationResult
│   │   ├── validation.py             #   ValidationResult
│   │   └── research.py               #   ResearchResult
│   ├── services/                     # Внешние сервисы
│   │   ├── research_llm.py           #   LLM-исследование
│   │   ├── web_search.py             #   DuckDuckGo + PubMed
│   │   └── web_scraper.py            #   Парсинг веб-страниц
│   ├── rag/                          # RAG-система (Retrieval-Augmented Generation)
│   │   ├── retriever.py              #   Гибридный ретривер (SPECTER2 + BM25)
│   │   ├── embeddings.py             #   Эмбеддинги научных текстов (SPECTER2)
│   │   ├── bm25.py                   #   BM25Okapi ранжирование
│   │   ├── tracking.py               #   Трекинг проиндексированных документов
│   │   └── models.py                 #   Модели: DocumentSource, LiteratureDocument
│   └── tests/                        # Юнит-тесты (284 шт.)
├── backend/                          # Вспомогательный бэкенд
│   ├── main.py                       # FastAPI: 2D/3D молекул, калькулятор
│   └── calculator_combined.py        # Стехиометрический калькулятор (standalone)
└── frontend/                         # React-фронтенд
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── App.jsx                   # Главный компонент + localStorage-персистенция
        ├── components/
        │   ├── MoleculeCard.jsx      #   Карточка молекулы (вкладки)
        │   ├── RetroCard.jsx         #   Маршруты синтеза
        │   ├── SynthesisGraph.jsx    #   ReactFlow-визуализация дерева
        │   ├── SynthesisTree.jsx     #   Раскрывающееся дерево
        │   ├── ExperimentProtocol.jsx #  Протокол эксперимента
        │   ├── CalculatorCard.jsx    #   Стехиометрический калькулятор
        │   ├── PathwaySelector.jsx   #   Выбор маршрута синтеза
        │   ├── ModelSelector.jsx     #   Выбор LLM-модели
        │   ├── PipelineProgress.jsx  #   Индикатор прогресса
        │   ├── TestPage.jsx          #   Страница запуска тестов
        │   ├── ChatMessage.jsx       #   Сообщения чата
        │   ├── ProtocolGraph.jsx     #   Визуализация протокола
        │   └── Viewer3D.jsx          #   3D-визуализация молекулы
        ├── hooks/
        │   ├── useInteractivePipeline.js  # Хук управления пайплайном
        │   └── useSSEPipeline.js     #   SSE-подключение
        └── styles/
            └── global.css            # Глобальные стили
```

---

## Данные

Все данные хранятся на сервере в директории `data/` и **не включены в репозиторий**.

### Базы данных

| Файл | Размер | Описание |
|---|---|---|
| `ord_reactions.db` | 1.8 GB | Open Reaction Database — 2.3M реакций, индексированных по продукту (SQLite). Основной источник для ретросинтеза |
| `buyables.db` | 22 MB | База коммерчески доступных реагентов — 690K+ молекул (SQLite). Источники: eMolecules, Mcule, ChemBridge, ChemSpace |

### Сырые данные покупаемых реагентов

| Файл | Размер | Описание |
|---|---|---|
| `buyables/buyables.json.gz` | 16 MB | Объединённый каталог покупаемых молекул |
| `buyables/chemspace_buyables_dedup_id_pub.json.gz` | 11 MB | ChemSpace каталог |
| `buyables/mcule_buyables_fd2.json.gz` | 14 MB | Mcule каталог |
| `buyables/chembridge_buyables.json.gz` | 821 KB | ChemBridge каталог |

### Модель ретросинтеза

| Файл | Размер | Описание |
|---|---|---|
| `retro_model/model_latest.pt` | 192 MB | Веса template-relevance нейромодели (извлечена из ASKCOS v2) |
| `retro_model/templates.jsonl` | 74 MB | 163K шаблонов реакций (SMARTS) |
| `retro_model/*.py` | ~30 KB | Код инференса модели (handler, parser, utils) |

### Банлисты безопасности

| Файл | Размер | Описание |
|---|---|---|
| `banned_chemicals.json` | 17 KB | 150 контролируемых веществ (CWC, ФСКН РФ, двойное назначение). Поля: name, SMILES, CAS, severity (critical/high/medium), SMARTS-паттерны |
| `banned_reactions.json` | 6 KB | 19 запрещённых типов реакций (синтез ОВ, взрывчатки, наркотиков). Поля: name, SMARTS pattern, severity |

### RAG-система

Гибридный поиск по научной литературе (SPECTER2 + BM25):

| Компонент | Описание |
|---|---|
| ChromaDB vectorstore | Векторные эмбеддинги научных статей (SPECTER2, 384-dim) |
| `literature_tracking.db` | SQLite: метаданные проиндексированных документов, parent/child чанки |
| Источники документов | PMC, BigQuery Patents, USPTO, S2ORC, ручная индексация |

### Рантайм-данные (создаются автоматически)

| Файл | Описание |
|---|---|
| `mvp/data/checkpoints.db` | LangGraph checkpoints — состояние сессий для возобновления |
| `mvp/logs/*.jsonl` | Журнал решений агента (JSONL) |

---

## API-эндпоинты

| Метод | Путь | Описание |
|---|---|---|
| `POST` | `/analyze` | Запуск пайплайна (SSE-стрим) |
| `GET` | `/stream/{thread_id}` | Возобновление интерактивной сессии |
| `POST` | `/resume/{thread_id}` | Продолжение после прерывания (выбор маршрута) |
| `POST` | `/tree/expand` | Рекурсивное дерево синтеза для маршрута |
| `POST` | `/api/calculate` | Стехиометрический калькулятор |
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

# 3. Подготовить данные (см. раздел "Данные")
# Разместить data/ рядом с проектом

# 4. Запустить API агента
uvicorn mvp.api:app --port 8765 --reload

# 5. Запустить бэкенд калькулятора / молекул
uvicorn backend.main:app --port 8002 --reload

# 6. Запустить фронтенд
cd frontend
npm install && npm run dev
# → http://localhost:5173
```

---

## Запуск тестов

```bash
# Быстрые юнит-тесты (без сети, без LLM) — 284 теста
pytest mvp/tests/ -m "not integration and not slow and not llm"

# Все тесты включая интеграционные (PubChem + ORD)
pytest mvp/tests/
```

---

## Вдохновение и заимствования

При проектировании MolPipeline мы также смотрели на проект [arqoofficial/itmo-chemcrow2](https://github.com/arqoofficial/itmo-chemcrow2) — AI-ассистент для химиков с FastAPI/React-стеком, LangGraph-агентом и набором хемоинформатических сервисов вокруг ретросинтеза, поиска литературы и safety-check'ов.

Некоторые архитектурные идеи и продуктовые решения могут быть адаптированы из этого проекта в MolPipeline. Репозиторий `itmo-chemcrow2` распространяется по лицензии MIT, поэтому при использовании существенных фрагментов кода или прямых адаптаций необходимо сохранять исходное уведомление об авторских правах и текст лицензии.

Источник:
- Репозиторий: `https://github.com/arqoofficial/itmo-chemcrow2`
- Лицензия: `https://github.com/arqoofficial/itmo-chemcrow2/blob/main/LICENSE`

---

## Лицензия

Все права защищены.
