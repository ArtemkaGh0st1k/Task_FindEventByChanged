## @module  [07dbb2d5]
Командная строка пакета: `python -m gtm_detector.ml_freq <команда>`. Модуль только связывает готовые части — загрузку данных (`data`), модели (`models`), проверку и отчёт (`evaluation`), аудит (`audit`) — и не содержит собственных вычислений.

Четыре команды:

| Команда | Что делает | Что создаёт |
|---------|-----------|-------------|
| `compare` | перекрёстная проверка по скважинам, сравнение моделей | `comparison.xlsx`, `comparison.png` в `results/ml_freq/<дата_время>/` |
| `train` | обучение одной модели на всей разметке | веса в `gtm_detector/ml_freq/weights/` (`.pt` или `.joblib`) |
| `predict` | поиск мероприятий обученной моделью (разметка не нужна) | `events_<модель>.xlsx` |
| `audit` | отчёт о качестве данных | `data_audit.xlsx`, `training_data.xlsx` и др. |

Примеры запуска из корня проекта:

```bash
.venv/Scripts/python -m gtm_detector.ml_freq compare --models gboost extra_trees      # около 2 минут
.venv/Scripts/python -m gtm_detector.ml_freq compare                                  # все модели, около 15 минут
.venv/Scripts/python -m gtm_detector.ml_freq compare --models transformer --epochs 20 # быстрая проверка
.venv/Scripts/python -m gtm_detector.ml_freq train --model gboost
.venv/Scripts/python -m gtm_detector.ml_freq predict --model gboost
.venv/Scripts/python -m gtm_detector.ml_freq audit
```

Те же команды заведены в `.vscode/launch.json` (вкладка Run and Debug, F5): «compare — gboost (быстро)», «compare — transformer, 2 эпохи», «train — выбранная модель (перезапишет веса)», «predict — выбранная модель». Отладочные запуски пишут в `results/ml_freq/debug/`.

`print` разрешён только здесь: это пользовательский вывод, а не лог библиотеки.

## _run_dir  [2ff7d12d]
Возвращает папку для результатов запуска: `base / "дд-мм-гггг_чч-мм-сс"` по текущему времени. Принимает базовую папку (обычно `results/ml_freq`), саму папку не создаёт. Уникальное имя не даёт новому запуску затереть старый отчёт.

```python
_run_dir(Path("results/ml_freq"))   # results/ml_freq/29-10-2026_14-05-31
```

## _registry  [6ef05c30]
Формирует словарь `{имя: фабрика_модели}` для запуска. Берёт `model_registry(seed=args.seed)` и, если указан `--epochs`, заменяет фабрику `transformer` так, чтобы модель училась заданное число эпох (по умолчанию 80, `TrainingConfig().epochs`). Остальные модели не затрагиваются. Зерно `--seed` передаётся и в модели, и в `TrainingConfig` — запуск воспроизводим.

```python
args.epochs = 2
_registry(args)["transformer"]()   # TransformerSegmenter, 2 эпохи
```

## load_wells  [a5f4390d]
Загружает скважины по выбранному набору данных. Принимает `args` и `labeled`:

- `--dataset merged` (по умолчанию): `MergedWellRepository(statistics, labels, reference).load()` — статистика 2020–2026: папка полугодовых `.xls` (по умолчанию) или один склеенный файл (тогда исправляются сдвиги на 180 дней); `--no-reference` отключает сверку склеенного файла с эталонной выгрузкой;
- `--dataset test`: `WellRepository(measurements, results)` — первые тестовые файлы (`resources/test_datas`, в проекте этой папки больше нет); разметка читается, только если `labeled=True`.

Если `labeled=True` (команды `compare`, `train`), остаются только скважины с разметкой — 369 на merged-наборе. При `labeled=False` (команда `predict`) возвращаются все 611 скважин с данными, в том числе 242 без разметки: именно на них ищутся новые мероприятия.

## run_compare  [1b0aea51]
Команда `compare`. Порядок:

1. Загружает размеченные скважины и реестр моделей.
2. Проверяет `--models`; неизвестное имя — сообщение в лог и код 1.
3. Печатает «Размеченных скважин: 369, мероприятий: 424; модели: …; фолдов: 5».
4. `CrossValidator(ExperimentConfig(folds, seed)).run(...)` — перекрёстная проверка по скважинам.
5. `ComparisonReport(results).save(out_dir)` и вывод `to_text()`.

Параметры: `--models` (по умолчанию все: `transformer`, `logreg`, `random_forest`, `extra_trees`, `gboost`), `--folds` (5), `--seed`, `--epochs`, `--out-dir`. Возвращает 0.

Результаты на 424 мероприятиях (`results/ml_freq/comparison/сравнение_качества_моделей_(20-26 год, полугодия).xlsx`, зерно 0): Градиентный бустинг — найдено 74 %, начало точно 36 %, ±3 дня 89 %, F1 0,136; Случайный лес — 68 %, F1 0,141; Extra Trees — 63 %, F1 0,131; Transformer — 46 %, F1 0,081; Логистическая регрессия — 61 %, F1 0,069. Precision и F1 с прежним прогоном (135 мероприятий, F1 0,46–0,48) сравнивать нельзя: теперь проверяется вся история скважин, а реальные подъёмы вне разметки считаются «лишними». Детектор на правилах (89 %) считали на других данных.

```bash
.venv/Scripts/python -m gtm_detector.ml_freq compare --models gboost --out-dir results/ml_freq/debug/compare
```

## run_train  [6c74f4e3]
Команда `train`: обучает одну модель на **всех** размеченных скважинах (без отложенной выборки — для боевого использования) и сохраняет веса в `weights_dir / "<имя><суффикс>"`: `transformer.pt` у PyTorch-модели, `gboost.joblib` и т. п. у моделей scikit-learn. Печатает «Градиентный бустинг (HistGB): обучена на 369 скважинах -> …\weights\gboost.joblib». Параметр `--model` по умолчанию `transformer`; обучение Transformer на CPU долгое, поэтому запускайте в фоне. Размер весов: `extra_trees.joblib` 178 МБ, `random_forest.joblib` 85 МБ, остальные — килобайты и около 1 МБ.

Команда перезаписывает существующие веса.

```bash
.venv/Scripts/python -m gtm_detector.ml_freq train --model gboost
```

## run_predict  [11c583ca]
Команда `predict`: загружает веса (`--weights` или найденные `_find_weights`), прогоняет модель по всем скважинам, декодирует вероятности тем же `SegmentDecoder`, что и при проверке, и пишет Excel `events_<модель>.xlsx` со столбцами «скважина», «куст», «начало эффекта», «окончание эффекта», «статус». Для интервала без окончания «статус» = «эффект продолжается».

Результат `results/ml_freq/predict_gboost_merged/events_gboost.xlsx` (получен старыми весами, обученными на 122 скважинах склеенного файла): 770 мероприятий на 468 скважинах, из них 112 «эффект продолжается». Нашлось больше мероприятий, чем было размечено (135), потому что предсказания идут по всем скважинам, а не только по размеченным; технолог должен проверить найденное. С новыми весами результат будет другим.

```bash
.venv/Scripts/python -m gtm_detector.ml_freq predict --model gboost --out-dir results/ml_freq/debug/predict
```

## _find_weights  [d01f5e82]
Ищет в папке файл весов модели с именем `name`: сначала `<имя>.pt`, затем `<имя>.joblib`. Если ничего нет — `FileNotFoundError` с подсказкой «сначала выполните train». Папка `gtm_detector/ml_freq/weights` сейчас содержит `transformer.pt`, `gboost.joblib`, `logreg.joblib`, `random_forest.joblib`, `extra_trees.joblib`.

```python
_find_weights(data.weights_dir, "gboost")   # …/weights/gboost.joblib
```

## build_parser  [7325d69f]
Строит парсер с подкомандами `compare`, `train`, `predict`, `audit`. Общий блок аргументов (`common`) для `compare`, `train`, `predict`: `--dataset {merged,test}`, `--statistics` (папка полугодовых файлов или склеенный файл), `--labels` (разметка технологов), `--reference` (эталон для склеенного файла, по умолчанию `None`), `--no-reference`, `--measurements`, `--results`, `--seed` (0), `--epochs`. Пути по умолчанию берутся из `DataConfig`: `resources/statistics` и `resources/labeled_data.xlsx`. Флаг `-v/--verbose` включает подробный лог. Для `audit` аргументы добавляет `audit.add_arguments`. Каждой подкоманде назначен `handler` (`run_compare`, `run_train`, `run_predict`, `run_audit`).

Доступные модели берутся из `model_registry()`; значит, новая модель, добавленная в реестр, сразу появляется в `--models` и `--model`.

```bash
.venv/Scripts/python -m gtm_detector.ml_freq compare --help
```

## main  [c783a82a]
Точка входа. Разбирает аргументы, настраивает `logging` (по умолчанию уровень WARNING, с `-v` — INFO; логгер `evaluation` всегда на INFO, чтобы при `compare` было видно ход фолдов: «gboost: фолд 3/5 готов») и вызывает `args.handler(args)`. Возвращает код выхода команды, который передаётся в `sys.exit` в блоке `__main__`.

```python
main(["audit", "--out-dir", "results/ml_freq/debug/audit"])   # 0
```
