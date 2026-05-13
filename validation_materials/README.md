# Chemistry Benchmark

Локальный mini-benchmark для сравнения голой модели `deepseek/deepseek-v4-flash` и полной системы MolPipeline ChemChat.

Runner не требует локального RDKit: bare model вызывается напрямую через OpenRouter из `.env`, а MolPipeline вызывается удаленно через публичный `/chat/message`.

## Dataset

Основная таблица:

```text
validation_materials/chemistry_benchmark_dataset.csv
```

Обязательные колонки:

- `level` - уровень задания: `school`, `university`, `research`.
- `prompt` - вопрос, который отправляется в модель или систему.
- `expected_contains` - строка, которая должна встретиться в ответе после нормализации текста.

Дополнительные колонки можно добавлять свободно. Сейчас таблица также содержит `id`, `source_pdf`, `task_location`, `answer_location`.

## Источники

Все проверочные материалы сохранены локально в `validation_materials/sources/`.

- `school_ncert_class12_unit13_amines.pdf` - школьный уровень, NCERT Class XII Amines.
- `school_ncert_class12_unit12_aldehydes_ketones_carboxylic_acids.pdf` - школьный уровень, NCERT Class XII carbonyl/carboxylic acids.
- `university_openstax_organic_chemistry_10e_study_guide.pdf` - университетский уровень, OpenStax Organic Chemistry study guide.
- `research_segler_2018_nature_supplementary_information.pdf` - research-level, supplementary information к статье Segler, Preuss, Waller, Nature 2018 про neural retrosynthesis planning.

## Run

```bash
python validation_materials/run_chem_benchmark.py
```

По умолчанию MolPipeline берется с:

```text
https://hack.humaneconomy.ru
```

Можно указать другой URL:

```bash
python validation_materials/run_chem_benchmark.py --molpipeline-url http://127.0.0.1:8765
```

Smoke без вызова bare model:

```bash
python validation_materials/run_chem_benchmark.py --skip-bare --limit 2
```

Проверить только CSV без вызова модели и сайта:

```bash
python validation_materials/run_chem_benchmark.py --validate-only
```

Своя таблица:

```bash
python validation_materials/run_chem_benchmark.py --dataset path/to/questions.csv --outdir validation_materials/runs/my_run
```

## Outputs

Runner создает:

- `results.jsonl` - полный ответ по каждой системе и строке.
- `results.csv` - табличный результат.
- `summary.json` - агрегированные метрики.
- `graphs/overall_accuracy.png` и `.svg` - общий граф DeepSeek vs MolPipeline.
- `graphs/by_level_accuracy.png` и `.svg` - сравнение по уровням.
- `graphs/school_accuracy.png`, `graphs/university_accuracy.png`, `graphs/research_accuracy.png` и `.svg` - отдельные графики по уровням.

Графики строятся через обычный `matplotlib` в стандартной темной теме.

## Optional RAG Indexing

Чтобы MolPipeline отвечал на research-level вопросы по локально сохраненным материалам не через hardcoded rules, а через RAG, проиндексируй `.txt` источники:

```bash
python scripts/index_literature_texts.py --source-dir validation_materials/sources --force
```

По умолчанию строится keyword/BM25-ready индекс без загрузки embedding-модели. Для vector search можно отдельно запустить с `--with-embeddings`, если окружение готово.

## Grading Rule

Проверка намеренно строгая и простая: ответ считается правильным, если `expected_contains` встречается в ответе после нормализации. Это не semantic grading и не LLM-as-judge.
