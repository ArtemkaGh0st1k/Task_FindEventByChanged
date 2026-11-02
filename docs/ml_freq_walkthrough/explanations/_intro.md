### Задача

На нефтяных скважинах с насосом ЭЦН технологи периодически **поднимают частоту** двигателя (например, с 48 до 50 Гц), чтобы получить больше жидкости. Это геолого-техническое мероприятие (ГТМ) «увеличение частоты F». Технолог записывает в итоговом файле дату начала мониторинга и дату окончания мониторинга: период, пока эффект от подъёма заметен в суточных замерах (дебит жидкости и нефти, обводнённость, давления).

Пакет `gtm_detector/ml_freq` решает обратную задачу: **по суточным замерам скважины найти мероприятие и поставить даты начала и окончания эффекта так, как их ставил бы технолог.** Модель для каждого дня отвечает «этот день внутри эффекта или нет», а затем из вероятностей по дням собираются интервалы.

Особенности, из-за которых код устроен именно так:

- **Данных немного.** В обучение попадает 424 мероприятия на 369 скважинах (раньше, на склеенном файле, было 135 на 122), поэтому модели небольшие, а проверка всегда идёт по скважинам, а не по дням.
- **Разметка неполная.** Технолог размечает не все реальные подъёмы частоты, поэтому «вне разметки» не значит «эффекта нет». Модели учатся только на днях вокруг размеченных мероприятий, а точность (precision) занижена намеренно.
- **Источник данных — исходные полугодовые выгрузки.** В прежнем склеенном файле часть выгрузок была сдвинута на 180 дней (это исправляет `BlockShiftRepair`, но только для склеенного файла); в 14 исходных `.xls` 2020–2026 сдвига нет.
- **Есть опорный уровень.** Детектор на правилах (ветка `feature/increase-f-gtm`) находит 89 % мероприятий и точно ставит дату начала в 78 %. Модели сравниваются с ним.

### Путь данных

```
Excel
  resources/statistics/<год>/*.xls            - замеры по дням, 14 полугодовых файлов 2020-2026, листы Qж, Обв, Qн, Рлин, Fэцн, Рзаб
  resources/labeled_data.xlsx                 - разметка технологов («Лист1»)
        |
        v   HalfYearStatisticsReader / MonitoringLabelsReader (data/statistics.py)
  таблицы «скважина x дата» + записи разметки
        |
        v   BlockShiftRepair                                  (data/repair.py)
  только для склеенного .xlsx: блоки, сдвинутые на 180 дней, возвращены на место или исключены
        |
        v   MergedWellRepository                              (data/dataset.py)
  список WellSeries: таблица frame (дни x ряды) + labels (интервалы)
        |
        v   FeatureBuilder                                    (features/builder.py)
  15 каналов (Transformer) или 106 признаков (scikit-learn) на каждый день
        |
        v   модели: transformer, logreg, random_forest, extra_trees, gboost   (models/)
  вероятность «день в эффекте» для каждого дня
        |
        v   SegmentDecoder                                    (evaluation/decoding.py)
  интервалы: начало, окончание (или «эффект продолжается»)
        |
        v   EventMatcher, Score, CrossValidator               (evaluation/)
  сопоставление с разметкой (допуск +-7 дней), метрики, перекрёстная проверка по скважинам
        |
        v   ComparisonReport, audit                           (evaluation/report.py, audit.py)
  отчёты в results/ml_freq/
```

Пути к данным берутся из `config.py`, вручную их не набирают (в прежних именах файлов встречалась латинская «c» вместо кириллической).

### Модули пакета

| Модуль | Что делает |
|--------|-----------|
| `config.py` | Все настройки в неизменяемых dataclass: пути, декодер, Transformer, обучение, эксперимент |
| `data/models.py` | `Interval` (мероприятие) и `WellSeries` (скважина, `target()`, `training_mask()`) |
| `data/readers.py` | Чтение Excel, разбор номеров и дат с опечатками (`normalize_id`, `parse_date`), тестовый набор замеров |
| `data/statistics.py` | Чтение полугодовых файлов `.xls` (`HalfYearStatisticsReader`), склеенной статистики и файла разметки |
| `data/repair.py` | `FrequencyCheck` и `BlockShiftRepair`: проверка подъёма частоты; поиск и исправление сдвига на 180 дней (для склеенного файла) |
| `data/dataset.py` | `MergedWellRepository`: сборка скважин, проверка разметки по частоте, кэш, `DataAudit` |
| `features/builder.py` | `FeatureBuilder`: `sequence()` и `tabular()` |
| `models/base.py` | Интерфейс `EventSegmenter`: `fit`, `predict_proba`, `save`, `load` |
| `models/transformer.py` | Transformer-энкодер для временных рядов и его обёртка `TransformerSegmenter` |
| `models/classic.py` | Модели scikit-learn: `logreg`, `random_forest`, `extra_trees`, `gboost` |
| `models/__init__.py` | Реестр моделей `model_registry()` и `load_model()` |
| `evaluation/decoding.py` | `SegmentDecoder`: вероятности по дням в интервалы |
| `evaluation/metrics.py` | `EventMatcher` и `Score`: сопоставление и метрики |
| `evaluation/experiment.py` | `CrossValidator`: перекрёстная проверка по скважинам |
| `evaluation/report.py` | `ComparisonReport`: таблицы и графики сравнения моделей |
| `audit.py` | Аудит данных 2020–2026: что отброшено и почему, выгрузка обучающих данных |
| `cli.py`, `__main__.py` | Командная строка `python -m gtm_detector.ml_freq` |
| `tests/` | Тесты на синтетических скважинах |

### Главные команды

Запускать из корня проекта, в виртуальном окружении `.venv`:

```bash
# сравнить модели перекрёстной проверкой по скважинам (gboost и extra_trees - около 2 минут)
.venv/Scripts/python -m gtm_detector.ml_freq compare --models gboost extra_trees
# все модели, около 15 минут на процессоре (Transformer на 424 мероприятиях — около часа)
.venv/Scripts/python -m gtm_detector.ml_freq compare

# обучить модель на всей разметке и сохранить веса
.venv/Scripts/python -m gtm_detector.ml_freq train --model transformer

# найти мероприятия обученной моделью
.venv/Scripts/python -m gtm_detector.ml_freq predict --model gboost

# аудит данных 2020-2026
.venv/Scripts/python -m gtm_detector.ml_freq audit

# тесты
.venv/Scripts/python -m pytest gtm_detector/ml_freq/tests -q
```

Отчёты команд попадают в `results/ml_freq/`.

### Как читать этот документ

Документ — «построчный разбор»: для каждого модуля пакета сначала краткое пояснение, зачем он нужен, затем каждый класс, функция и метод с номерами строк, как в редакторе. У объекта сначала идёт пояснение (что делает, что принимает и возвращает, почему так) и пример с реальными данными проекта — номерами скважин, датами, значениями из `config.py` и `tests/conftest.py`; ниже сам код.

Лучше читать в порядке потока данных: `config.py` → `data/` → `features/` → `models/` → `evaluation/` → `cli.py`; тесты в конце показывают на маленьких синтетических скважинах, что каждая часть должна делать.

### Как обновляется документ

- Код в PDF попадает из файлов проекта автоматически, пояснения хранятся в `docs/ml_freq_walkthrough/explanations/<путь модуля>.md`, этот раздел — в `explanations/_intro.md`.
- Каждое пояснение помечено отпечатком кода объекта. Если код изменился, а пояснение нет, `build.py --check` сообщит, что оно устарело.
- PDF собирает `build.py`: `.venv/Scripts/python -X utf8 docs/ml_freq_walkthrough/build.py` (около минуты, печать через Microsoft Edge без окна). Проверка пояснений: ключ `--check`, пометка проверенного — `--stamp`.
- Обновляет пояснения и PDF агент `docs-maintainer` (`.claude/agents/docs-maintainer.md`). Stop-хук `.claude/hooks/docs_sync.py` после каждого ответа Claude Code, в котором менялись файлы проекта, просит вызвать этого агента.
