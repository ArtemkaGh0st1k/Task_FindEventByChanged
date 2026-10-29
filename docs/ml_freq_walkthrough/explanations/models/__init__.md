## @module  [355b6b72]
Пакет `models` собирает все модели в одном месте и скрывает различия между ними: с точки зрения остального кода (`CrossValidator`, `cli`) любая модель — это `EventSegmenter` с методами `fit`, `predict_proba`, `save`, `load`.

В цепочке «Excel → скважины → признаки → модель → интервалы» пакет — четвёртое звено: берёт признаки из `FeatureBuilder`, учится на днях из `training_mask`, возвращает вероятность «день внутри эффекта». Даты начала и окончания получает уже `SegmentDecoder`, одинаково для всех моделей.

Пакет содержит: `base.py` (общий интерфейс), `classic.py` (4 модели scikit-learn), `transformer.py` (своя сеть на PyTorch). Новая модель — это наследник `EventSegmenter` и одна строка в `model_registry`.

## model_registry  [ac67e9a3]
Возвращает словарь «имя модели → фабрика необученной модели». Имена — те, что принимает `--models` в командной строке.

Параметр `seed` (по умолчанию 0) попадает в `random_state` моделей scikit-learn. У Transformer зерно хранится в `TrainingConfig.seed`, поэтому фабрика — сам класс `TransformerSegmenter`.

Фабрика, а не готовый объект, нужна потому, что `CrossValidator` на каждом из 5 фолдов обучает **новую** модель с нуля: переиспользование обученного объекта дало бы утечку между фолдами.

### Пример
```python
reg = model_registry()
list(reg)            # ['transformer', 'logreg', 'random_forest', 'extra_trees', 'gboost']
model = reg["gboost"]()          # необученный SklearnSegmenter
model.title                      # 'Градиентный бустинг (HistGB)'
```
В командной строке: `python -m gtm_detector.ml_freq compare --models gboost extra_trees`.

## load_model  [31efa844]
Загружает сохранённую модель, определяя тип по расширению файла: `.pt` — `TransformerSegmenter.load`, всё остальное (`.joblib`) — `SklearnSegmenter.load`. Нужна команде `predict`, которой всё равно, какая модель лежит в `weights/`.

Параметр `path` — путь к файлу (`str` или `Path`). Результат — обученная модель, готовая к `predict_proba`.

### Пример
```python
model = load_model("gtm_detector/ml_freq/weights/gboost.joblib")   # SklearnSegmenter
model = load_model("gtm_detector/ml_freq/weights/transformer.pt")  # TransformerSegmenter
```
Если для Transformer набор каналов `SEQUENCE_CHANNELS` изменился с момента обучения, `load` выбросит `ValueError("Набор признаков изменился — переобучите модель")`.
