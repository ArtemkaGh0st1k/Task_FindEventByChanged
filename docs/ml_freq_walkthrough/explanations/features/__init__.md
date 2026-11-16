## @module  [d1d88f6a]
Пакет `features` — маленькая «витрина»: он только реэкспортирует класс `FeatureBuilder` из `features/builder.py`, чтобы в остальном коде писать `from ..features import FeatureBuilder`.

В цепочке «Excel → скважины → признаки → модель → интервалы» пакет стоит между данными и моделями. На входе — `WellSeries` (таблица `frame`: дни × `SERIES_COLUMNS`), на выходе — числовые матрицы «день × признак». Все модели (`TransformerSegmenter`, `SklearnSegmenter`) получают признаки **только** отсюда; это правило проекта: добавил канал — обнови `SEQUENCE_CHANNELS`, тесты и README.

```python
from gtm_detector.ml_freq.features import FeatureBuilder

fb = FeatureBuilder()
fb.sequence_size        # 15 каналов для Transformer
len(fb.tabular_names)   # 106 признаков для scikit-learn
```
