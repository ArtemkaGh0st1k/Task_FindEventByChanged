# CLAUDE.md — память проекта Task_FindEventByChanged

Этот файл Claude Code читает автоматически в каждой сессии и передаёт
субагентам из `.claude/agents/`. Здесь — то, что нельзя быстро вывести из кода.

## Задача

Детекция геолого-технических мероприятий (ГТМ) на нефтяных скважинах по
суточным замерам. Основной фокус — ГТМ **«увеличение частоты F»** (подъём
частоты двигателя ЭЦН): найти мероприятие и даты **начала** и **окончания**
эффекта так, как их ставят технологи.

Рабочая ветка ML: `feature/create-llm-model`. Ветка `feature/increase-f-gtm` —
детектор на правилах; это **бейзлайн, который ML пока не обгоняет** (найдено 89 %,
начало точно 78 %). Любое улучшение сравнивается и с ним.

## Структура

| Путь | Что это | Статус |
|------|---------|--------|
| `gtm_detector/ml_freq/` | ML-детектор: Transformer (PyTorch) + scikit-learn, CLI, тесты | **активная разработка** |
| `neural-network/` | первая версия нейросети | legacy, не трогать без запроса |
| `prompt-engineering/` | запросы к DeepSeek/Ollama | legacy, не трогать без запроса |
| `constants/`, `helpers/`, `main.py` | общие утилиты старых подходов | legacy |
| `resources/` | исходные Excel (замеры, разметка) | **только чтение** |
| `results/ml_freq/` | отчёты `compare`, `predict`, `audit` | генерируется |

Подробное описание архитектуры, признаков и результатов —
`gtm_detector/ml_freq/README.md`. Читай его перед любой работой в `ml_freq`.

Ключевые абстракции `ml_freq`:

- `data.WellSeries` — скважина: `frame` (дни × `SERIES_COLUMNS`), `labels: list[Interval]`,
  `target()` и `training_mask(context_days)`.
- `features.FeatureBuilder` — `sequence()` (15 каналов для Transformer),
  `tabular()` (106 признаков для sklearn). Порядок `SEQUENCE_CHANNELS` зашит в веса `.pt`.
- `models.EventSegmenter` — общий интерфейс `fit / predict_proba / save / load`.
  Новая модель = наследник + строка в `model_registry()` (`models/__init__.py`).
- `evaluation.SegmentDecoder` — вероятности → интервалы, одинаково для всех моделей.
- `evaluation.CrossValidator` — k-fold **по скважинам**, `EventMatcher` (±7 дней), `Score`.
- `config.py` — все гиперпараметры в frozen-dataclass'ах. Магических чисел в коде нет.

## Команды

Окружение: Windows, Python 3.12, `.venv` в корне. Запускать из корня проекта.

```bash
.venv/Scripts/python -m pytest gtm_detector/ml_freq/tests -q          # тесты (секунды)
.venv/Scripts/ruff check gtm_detector                                # линтер PEP8
.venv/Scripts/ruff format --check gtm_detector                       # формат
.venv/Scripts/python -m gtm_detector.ml_freq compare --models gboost extra_trees   # ~2 мин
.venv/Scripts/python -m gtm_detector.ml_freq compare                 # все модели, ~15 мин (CPU)
.venv/Scripts/python -m gtm_detector.ml_freq compare --models transformer --epochs 20  # быстрый смоук
.venv/Scripts/python -m gtm_detector.ml_freq audit                   # аудит данных 2021–2026
```

Долгие команды (`compare` всех моделей, `train --model transformer`) запускай в фоне.
Первое чтение Excel ~30 с, потом кэш в `%TEMP%/gtm_ml_freq_cache`. Если менялись
`data/readers.py`, `data/statistics.py`, `data/repair.py` или `data/dataset.py` —
кэш может быть устаревшим: проверь ключ кэша или удали папку кэша.

## Стиль кода

- **PEP 8**, проверяется `ruff` (настройки в `pyproject.toml`, длина строки 99).
  Старый код местами длиннее — исправляй только в строках, которые и так меняешь;
  **не переформатируй файлы целиком** без просьбы (раздувает diff).
- `from __future__ import annotations`, аннотации типов у всех публичных функций,
  `X | None` вместо `Optional[X]`, `collections.abc` для `Sequence/Mapping/Callable`.
- Импорты внутри пакета — относительные (`from ..data import WellSeries`).
- Docstring'и и комментарии — **на русском**, коротко, объясняют *почему*.
  Имена — на английском, `snake_case` / `PascalCase`.
- Конфигурация — frozen `@dataclass` в `config.py`; новые гиперпараметры добавлять туда
  со значением по умолчанию и комментарием-обоснованием.
- `logging`, а не `print`, внутри библиотечного кода; `print` — только в `cli.py`.
- Пути — `pathlib.Path`; имена файлов с кириллицей и латинской «c»
  (`Итоговый_файл_c_2021_2026.xlsx`) — копировать из `config.py`, не набирать руками.

## Правила ML (обязательные)

1. **Никакой утечки между скважинами.** Разбиение только по скважинам
   (`CrossValidator.folds`). Нормировки и статистики — внутри скважины или по train-фолду.
2. **Признаки только из `FeatureBuilder`**; добавил канал → обнови `SEQUENCE_CHANNELS`,
   тесты и README; старые `.pt` станут несовместимы (это проверяется в `load`).
3. **Обучение только на `training_mask`** (±`CONTEXT_DAYS` вокруг разметки). Дни после
   `censored`-мероприятий («ПРОДОЛЖАЕТСЯ ЭФФЕКТ») в потерях не участвуют.
4. **Воспроизводимость:** `seed` из конфига → `torch.manual_seed`, `np.random.default_rng`,
   `random_state` sklearn. Без глобального `np.random.*`.
5. **Сравнение честное:** одинаковые фолды, один `SegmentDecoder`, метрики из `Score`.
   Утверждение «стало лучше» — только с таблицей до/после из `compare` и, для Transformer,
   по нескольким seed (разброс между seed сопоставим с эффектом).
6. Precision занижена by design (разметка неполная) — главные метрики: recall
   (найдено), точность даты начала ±1/±3, окончания ±3/±7, ROC-AUC по дням.

## Данные — подводные камни

- Объединённая статистика 2021–2026 склеена макросом, часть выгрузок **сдвинута на 180 дней**.
  Исправляет `data/repair.py` (`BlockShiftRepair`); 17 % ячеек исключается. Не «чини»
  данные в обход этого модуля.
- Из 786 строк разметки в обучение идёт ~135 мероприятий на ~122 скважинах. Данных мало:
  крупные модели переобучаются (d_model 64 хуже, чем 32).
- `resources/*.xlsx` и `*.rar` — не изменять, не удалять, не коммитить новые версии.
  `Статистика_2021-2026.rar` в обучении не используется.

## Git

- Не коммить и не пушь без явной просьбы. `results/`, кэши и новые веса — не коммитить,
  если не попросили.
- Сообщения коммитов — на русском, в стиле истории (`Добавил: ...`, `Поправил ...`).
