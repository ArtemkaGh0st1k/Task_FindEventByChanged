## @module  [d34717c3]
Все настройки проекта в одном месте: неизменяемые (`frozen=True`) dataclass-классы со значениями по умолчанию. Чисел, вписанных прямо в код, в пакете нет — любой порог ищите здесь или в константах классов `data/repair.py` и `data/dataset.py`.

Модуль задаёт три корневых пути: `PROJECT_ROOT` (папка проекта, два уровня выше `config.py`), `PACKAGE_DIR` (`gtm_detector/ml_freq`) и `RESOURCES` (`resources/`, исходные Excel — только чтение). Константа `CONTEXT_DAYS = 45` — сколько дней до начала и после окончания размеченного мероприятия используется для обучения (см. `WellSeries.training_mask`).

### Пример
```python
from gtm_detector.ml_freq.config import ExperimentConfig, CONTEXT_DAYS
cfg = ExperimentConfig()
cfg.folds, cfg.decoder.threshold, CONTEXT_DAYS   # (5, 0.5, 45)
cfg.folds = 3   # FrozenInstanceError: настройки менять нельзя, только создавать новые
```
Чтобы изменить значение, создают новый объект: `ExperimentConfig(folds=3)`.

## DataConfig  [552bf13b]
Где лежат входные файлы и куда писать результаты. Поля — пути `pathlib.Path`:

| Поле | Значение по умолчанию |
|------|-----------------------|
| `dataset` | `"merged"` (файлы 2021–2026) или `"test"` (тестовые файлы 2026 г.) |
| `statistics` | `resources/Статистика_с_2021_2026_в_одном_файле.xlsx` |
| `labels` | `resources/Итоговый_файл_c_2021_2026.xlsx` (буква «c» в имени латинская — имя копируют только отсюда) |
| `reference` | `resources/test_datas/input_dataset.xlsx` — исходная выгрузка 2026 г., нужна только для проверки сдвигов блоков, в обучение не идёт |
| `measurements`, `results` | `input_dataset.xlsx` и `output_dataset.xlsm` для режима `"test"` |
| `output_dir` | `results/ml_freq` |
| `weights_dir` | `gtm_detector/ml_freq/weights` (файлы `.pt`) |

### Пример
```python
cfg = DataConfig()
cfg.statistics.exists()          # True, если лежит рядом с проектом
DataConfig(dataset="test")       # режим тестовых файлов (без объединённой статистики)
```

## DecoderConfig  [ab34b346]
Параметры `SegmentDecoder`, превращающего вероятности по дням в интервалы; одни и те же для всех моделей, иначе сравнение нечестное.

- `threshold = 0.5` — день «в эффекте», если сглаженная вероятность выше;
- `smooth_days = 5` — окно скользящего среднего вероятностей (дни);
- `max_gap = 3` — разрывы короче 3 дней склеиваются в один период;
- `min_length = 2` — периоды короче 2 дней отбрасываются.

### Пример
Если эффект «мигнул» на 2 дня (вероятность упала ниже 0,5 между двумя высокими участками), при `max_gap=3` это один период, а не два. Порог 0,5 и окно 5 дней применяются ко всем моделям одинаково.

## TransformerConfig  [605c2e29]
Архитектура Transformer-энкодера: `d_model=32`, `n_heads=4`, `n_layers=2`, `dim_feedforward=64`, `dropout=0.2`, свёртка `conv_kernel=5` перед энкодером (ловит локальные ступеньки частоты), `max_length=4096` дней в позиционном кодировании. Модель маленькая (~20 тыс. параметров) намеренно: размеченных мероприятий всего 135, и большая модель переобучается.

### Пример
Вариант `d_model=64`, 3 слоя на перекрёстной проверке дал точное начало 16 % против 24 % у `d_model=32` (комментарий в `config.py`). Поэтому значения по умолчанию такие.

## TrainingConfig  [8ba947c5]
Обучение Transformer: `epochs=80`, `batch_size=16`, `learning_rate=1e-3`, `weight_decay=1e-4`. Обучение идёт на случайных окнах ряда `crop_length=128` дней, по `crops_per_well=4` окна на скважину за эпоху — это аугментация при малом числе данных. `seed=0` передаётся в `torch.manual_seed` и генераторы NumPy (воспроизводимость).

### Пример
Быстрая проверка из CLAUDE.md: `compare --models transformer --epochs 20` подменяет только `epochs=80` на 20, остальное остаётся. Для сравнения зёрен создают `TrainingConfig(seed=1)`, `seed=2` и т. д.

## ExperimentConfig  [541f8a88]
Настройки перекрёстной проверки по скважинам: `folds=5`, `seed=0` (перемешивание скважин по фолдам), `start_tolerance=7` (допуск ±7 дней при сопоставлении найденного мероприятия с разметкой в `EventMatcher`) и вложенный `decoder` (`DecoderConfig`, создаётся через `default_factory`, чтобы объект не был общим изменяемым значением).

### Пример
```python
ExperimentConfig().decoder.smooth_days   # 5
ExperimentConfig(folds=3, seed=1)        # другое разбиение скважин на 3 фолда
```
