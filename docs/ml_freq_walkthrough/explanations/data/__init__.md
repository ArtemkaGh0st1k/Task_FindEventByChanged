## @module  [7a023ba9]
Пакет `data` — первый шаг цепочки «Excel → скважины → признаки → модель → интервалы». Он ничего не вычисляет сам, а собирает публичный интерфейс из шести модулей и перечисляет его в `__all__`, поэтому остальной код пишет `from ..data import WellSeries`.

| Что экспортируется | Из модуля | Назначение |
|---|---|---|
| `SERIES_COLUMNS`, `Interval`, `WellSeries` | `models.py` | модель скважины и мероприятия |
| `MeasurementsReader`, `LabelsReader`, `WellRepository`, `normalize_id`, `parse_date` | `readers.py` | чтение тестовых файлов 2026 г. |
| `MergedStatisticsReader`, `MonitoringLabelsReader`, `LabelRecord` | `statistics.py` | чтение объединённых файлов 2021–2026 |
| `BlockShiftRepair`, `FrequencyCheck`, `Segment` | `repair.py` | исправление сдвинутых на 180 дней блоков |
| `MergedWellRepository`, `DataAudit` | `dataset.py` | итоговая загрузка и отчёт об отброшенном |

### Пример
```python
from gtm_detector.ml_freq.data import MergedWellRepository
from gtm_detector.ml_freq.config import DataConfig
c = DataConfig()
wells, audit = MergedWellRepository(c.statistics, c.labels, c.reference).build()
len(wells)   # 698 скважин, из них 122 с разметкой (135 мероприятий)
```
