## @module  [28fbb68a]
Пакет `data` — первый шаг цепочки «Excel → скважины → признаки → модель → интервалы». Он ничего не вычисляет сам, а собирает публичный интерфейс из шести модулей и перечисляет его в `__all__`, поэтому остальной код пишет `from ..data import WellSeries`.

| Что экспортируется | Из модуля | Назначение |
|---|---|---|
| `SERIES_COLUMNS`, `Interval`, `WellSeries` | `models.py` | модель скважины и мероприятия |
| `MeasurementsReader`, `LabelsReader`, `WellRepository`, `normalize_id`, `parse_date` | `readers.py` | чтение тестовых файлов 2026 г. |
| `HalfYearStatisticsReader`, `MergedStatisticsReader`, `MonitoringLabelsReader`, `LabelRecord` | `statistics.py` | чтение полугодовых файлов `.xls`, прежнего склеенного файла и разметки 2020–2026 |
| `BlockShiftRepair`, `FrequencyCheck`, `Segment` | `repair.py` | проверка подъёма частоты; исправление блоков, сдвинутых на 180 дней (только для склеенного файла) |
| `MergedWellRepository`, `DataAudit` | `dataset.py` | итоговая загрузка и отчёт об отброшенном |

### Пример
```python
from gtm_detector.ml_freq.data import MergedWellRepository
from gtm_detector.ml_freq.config import DataConfig
c = DataConfig()
wells, audit = MergedWellRepository(c.statistics, c.labels, c.reference).build()   # reference = None
len(wells)   # 611 скважин, из них 369 с разметкой (424 мероприятия)
```
