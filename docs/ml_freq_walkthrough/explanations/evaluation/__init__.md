## @module  [0a3888e5]
Пакет `evaluation` — всё, что происходит **после** того, как модель выдала вероятность «в этот день эффект есть» для каждого дня скважины. Он собирает четыре шага в одну цепочку:

1. `SegmentDecoder` (`decoding.py`) превращает ряд вероятностей в список интервалов «начало — окончание».
2. `EventMatcher` и `Score` (`metrics.py`) сопоставляют найденные интервалы с разметкой технологов и считают метрики.
3. `CrossValidator` и `ModelResult` (`experiment.py`) повторяют это для каждой модели на одних и тех же фолдах, разбитых по скважинам.
4. `ComparisonReport` (`report.py`) пишет итог в `comparison.xlsx` и `comparison.png`.

Файл `__init__.py` ничего не вычисляет: он только собирает шесть публичных имён (`ComparisonReport`, `CrossValidator`, `EventMatcher`, `ModelResult`, `Score`, `SegmentDecoder`), чтобы в других местах писать `from .evaluation import CrossValidator`. Так делает `cli.py` в командах `compare` и `predict`.

Цепочка в проекте: «Excel → скважины (`data`) → признаки (`features`) → модель (`models`) → **интервалы и метрики (`evaluation`)**».

### Пример
```python
from gtm_detector.ml_freq.evaluation import ComparisonReport, CrossValidator, SegmentDecoder
```

Для `compare --models gboost extra_trees` `cli.run_compare` создаёт `CrossValidator`, получает `list[ModelResult]` и отдаёт его в `ComparisonReport`. Для `predict` используется только `SegmentDecoder`: разметки нет, метрики считать не с чем.
