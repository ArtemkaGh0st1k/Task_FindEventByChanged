## @module  [b1638777]
Командная строка пакета: `python -m gtm_detector.ml_freq <команда>`. Модуль только связывает готовые части — загрузку данных (`data`), модели (`models`), проверку и отчёт (`evaluation`), аудит (`audit`) — и не содержит собственных вычислений.

Шесть команд:

| Команда | Что делает | Что создаёт |
|---------|-----------|-------------|
| `compare` | перекрёстная проверка по скважинам, сравнение моделей | `comparison.xlsx`, `comparison.png` в `results/ml_freq/<дата_время>/` |
| `compare-windows` | обучение на разметке с узким и широким окном и началом по дате технолога или по дню скачка, проверка на одних мероприятиях | `windows.xlsx` (лист «по зёрнам», «среднее, проверка N», «разница, проверка N») |
| `train` | обучение одной модели на всей разметке | веса в `gtm_detector/ml_freq/weights/` (`.pt` или `.joblib`) и обновлённый `weights/weights.zip` |
| `predict` | поиск мероприятий обученной моделью (разметка не нужна) | `events_<модель>.xlsx` |
| `pack-weights` | запись всех весов из `weights/` в архив | `weights/weights.zip` |
| `audit` | отчёт о качестве данных | `data_audit.xlsx`, `training_data.xlsx` и др. |

Примеры запуска из корня проекта:

```bash
.venv/Scripts/python -m gtm_detector.ml_freq compare --models gboost extra_trees      # около 2 минут
.venv/Scripts/python -m gtm_detector.ml_freq compare                                  # все модели, около 15 минут
.venv/Scripts/python -m gtm_detector.ml_freq compare --models transformer --epochs 20 # быстрая проверка
.venv/Scripts/python -m gtm_detector.ml_freq compare-windows --models gboost --seeds 0 1 2 --label-start both
.venv/Scripts/python -m gtm_detector.ml_freq train --model gboost
.venv/Scripts/python -m gtm_detector.ml_freq predict --model gboost
.venv/Scripts/python -m gtm_detector.ml_freq pack-weights
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

## _all_wells  [bc58ee5f]
Загружает все пригодные скважины набора с окном отбора разметки `lookback`/`lookahead` дней (до и после даты технолога). Для `merged` собирает `BlockShiftRepair(check=FrequencyCheck(lookback, lookahead))` и передаёт его в `MergedWellRepository`; для `test` окно не применяется. Скважины без разметки не отбрасывает — это делают `load_wells` и `run_compare_windows`.

```python
_all_wells(args, 14, 3)   # разметка отобрана с окном «14 дней до, 3 после»
```

## load_wells  [95560680]
Загружает скважины по выбранному набору данных (через `_all_wells` с окном `--label-lookback`/`--label-lookahead`, по умолчанию 3/3). Принимает `args` и `labeled`:

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

## _event_count  [4682b731]
Число размеченных мероприятий во всех скважинах списка: `sum(len(well.labels))`. Используется в сводке `compare-windows`.

```python
_event_count(narrow)   # 424 при окне 3/3
```

## _window_tables  [111be127]
Таблицы для одной проверки `compare-windows`. Берёт строки `frame` с нужным значением столбца «проверка» и возвращает пару: среднее метрик по зёрнам для каждой пары (модель, вариант) и разницу каждого варианта с `baseline` в виде строки «среднее ± std по зёрнам» (например, `+0.012 ± 0.008`). Разница считается по зёрнам, а не по средним, чтобы показать разброс: у Transformer он сравним с эффектом.

### Пример
```python
mean, delta = _window_tables(frame, JUMP, "узкое окно / начало по дате технолога")
delta.loc[("широкое окно / начало по дню скачка", "Градиентный бустинг (HistGB)")]
```

## run_compare_windows  [81359a30]
Команда `compare-windows`: проверяет, помогает ли обучению более широкое окно отбора разметки (подъём до 14 дней раньше даты технолога). Порядок:

1. Загружает скважины дважды: с узким окном (`DataConfig.label_lookback/lookahead`, 3/3) и с широким (`--label-lookback/--label-lookahead`, у этой команды по умолчанию 14/3). Остаются скважины, размеченные при широком окне; списки должны совпасть по `well_id`, иначе код 1.
2. Печатает сводку: число скважин и мероприятий в каждом варианте.
3. Строит варианты обучения: узкое и широкое окно × начало по дате технолога (`TECH`) или по дню скачка (`JUMP`, `WellSeries.with_jump_starts`). Набор начал задаёт `--label-start {tech, jump, both}` (по умолчанию `both`). База — «узкое окно / начало по дате технолога», она добавляется всегда: без неё нечего вычитать.
4. Для каждого зерна из `--seeds` (или `--seed`) и каждого варианта запускает `CrossValidator.run_multi`: одно обучение, проверка на двух наборах мероприятий узкого окна — с началом по дате технолога и по дню скачка. Фолды общие, поэтому разница метрик — эффект обучающей разметки.
5. Пишет `windows.xlsx`: «по зёрнам» (столбцы «вариант» и «проверка»), а для каждой проверки — «среднее, проверка N» и «разница, проверка N» (разница с базой, среднее ± std по зёрнам; считает `_window_tables`). Метрики: найдено, точность, начало ±1/±3, конец ±3/±7, ROC-AUC по дням.

Работает только с `--dataset merged`. Разброс между зёрнами сравним с эффектом, поэтому для вывода нужно несколько зёрен.

```bash
.venv/Scripts/python -m gtm_detector.ml_freq compare-windows --models gboost --seeds 0 1 2 --label-start both --out-dir results/ml_freq/debug/windows
```

## run_train  [525d1dee]
Команда `train`: обучает одну модель на **всех** размеченных скважинах (без отложенной выборки — для боевого использования) и сохраняет веса в `weights_dir / "<имя><суффикс>"`: `transformer.pt` у PyTorch-модели, `gboost.joblib` и т. п. у моделей scikit-learn. Печатает «Градиентный бустинг (HistGB): обучена на 369 скважинах -> …\weights\gboost.joblib». Параметр `--model` по умолчанию `transformer`; обучение Transformer на CPU долгое, поэтому запускайте в фоне. Размер весов: `extra_trees.joblib` 178 МБ, `random_forest.joblib` 85 МБ, остальные — килобайты и около 1 МБ.

После сохранения вызывает `WeightsArchive.add([path])`: файл попадает в `weights/weights.zip`, остальные модели в архиве сохраняются. Так сделано, потому что GitHub не принимает файлы больше 100 МБ, а в git лежит только архив. Пересборка архива занимает около 75 секунд — `train` любой модели стал дольше на столько же. Команда перезаписывает существующие веса и их запись в архиве.

```bash
.venv/Scripts/python -m gtm_detector.ml_freq train --model gboost
```

## run_predict  [de0a1ec0]
Команда `predict`: загружает веса (`--weights` или `WeightsArchive.locate(args.model)`: при необходимости модель распаковывается из `weights.zip`), прогоняет модель по всем скважинам, декодирует вероятности тем же `SegmentDecoder`, что и при проверке, и пишет Excel `events_<модель>.xlsx` со столбцами «скважина», «куст», «начало эффекта», «окончание эффекта», «статус». Для интервала без окончания «статус» = «эффект продолжается».

Результат `python -m gtm_detector.ml_freq predict --model gboost --out-dir results/ml_freq/predict_gboost_halfyear` (около 47 секунд) — `events_gboost.xlsx`: 6548 мероприятий на 609 скважинах из 611, из них 81 «эффект продолжается». По годам начала: 2020 — 801, 2021 — 886, 2022 — 967, 2023 — 1076, 2024 — 892, 2025 — 1119, 2026 — 807; медианная длительность 15 дней. Например, у скважины 10055 (куст 2301) найдены 15.12.2021–24.12.2021, 03.04.2022–17.04.2022, 02.06.2022–23.06.2022.

Это список кандидатов, а не готовый результат: в разметке всего 786 строк, а precision бустинга на перекрёстной проверке около 0,07 (разметка неполная). Предсказания идут по всем скважинам, а не только по размеченным, поэтому технолог должен проверить найденное выборочно. Прежний результат на старых весах (`predict_gboost_merged`, 770 мероприятий на 468 скважинах) устарел.

```bash
.venv/Scripts/python -m gtm_detector.ml_freq predict --model gboost --out-dir results/ml_freq/debug/predict
```

## _weights_archive  [ee6dc8af]
Собирает `WeightsArchive(data.weights_archive, data.weights_dir)` из настроек `DataConfig`: архив `weights/weights.zip` и папка рабочих копий `weights/`. Общий помощник для `run_train`, `run_pack_weights` и `run_predict`, чтобы пути не дублировались.

```python
_weights_archive(DataConfig()).path   # …/gtm_detector/ml_freq/weights/weights.zip
```

## run_pack_weights  [383a04ff]
Команда `pack-weights`: записывает в архив все `.pt` и `.joblib` из `weights/` (`WeightsArchive.pack`) и печатает имена и размер архива. Нужна, когда веса скопированы вручную или обучены до появления архива. Если в папке нет весов, выбрасывается `FileNotFoundError`. Возвращает 0.

```bash
.venv/Scripts/python -m gtm_detector.ml_freq pack-weights
# В архив …\weights\weights.zip (79.4 МБ) записаны: extra_trees.joblib, gboost.joblib, logreg.joblib, random_forest.joblib, transformer.pt
```

## build_parser  [b91e058e]
Строит парсер с подкомандами `compare`, `compare-windows`, `train`, `predict`, `pack-weights`, `audit`. Общий блок аргументов (`common`) для `compare`, `compare-windows`, `train`, `predict`: `--dataset {merged,test}`, `--statistics` (папка полугодовых файлов или склеенный файл), `--labels` (разметка технологов), `--reference` (эталон для склеенного файла, по умолчанию `None`), `--no-reference`, `--measurements`, `--results`, `--label-lookback` и `--label-lookahead` (окно отбора разметки, по умолчанию 3 и 3; у `compare-windows` по умолчанию 14 и 3), `--seed` (0), `--epochs`. У `compare-windows` есть ещё `--models`, `--label-start {tech,jump,both}` (по умолчанию `both`), `--folds` (5), `--seeds`, `--out-dir`. Пути по умолчанию берутся из `DataConfig`: `resources/statistics` и `resources/labeled_data.xlsx`. Флаг `-v/--verbose` включает подробный лог. Для `audit` аргументы добавляет `audit.add_arguments`. Каждой подкоманде назначен `handler` (`run_compare`, `run_compare_windows`, `run_train`, `run_predict`, `run_pack_weights`, `run_audit`). У `predict --weights` значения по умолчанию нет: веса берутся из архива `weights.zip`; у `pack-weights` аргументов нет.

Доступные модели берутся из `model_registry()`; значит, новая модель, добавленная в реестр, сразу появляется в `--models` и `--model`.

```bash
.venv/Scripts/python -m gtm_detector.ml_freq compare --help
```

## main  [c783a82a]
Точка входа. Разбирает аргументы, настраивает `logging` (по умолчанию уровень WARNING, с `-v` — INFO; логгер `evaluation` всегда на INFO, чтобы при `compare` было видно ход фолдов: «gboost: фолд 3/5 готов») и вызывает `args.handler(args)`. Возвращает код выхода команды, который передаётся в `sys.exit` в блоке `__main__`.

```python
main(["audit", "--out-dir", "results/ml_freq/debug/audit"])   # 0
```
