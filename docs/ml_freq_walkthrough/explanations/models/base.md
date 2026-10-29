## @module  [2c1b319e]
Общий интерфейс всех моделей. Он фиксирует единый контракт: «получи размеченные скважины — научись; получи скважину — верни по вероятности на каждый день». Поэтому `CrossValidator`, `SegmentDecoder` и командная строка не знают, что внутри — Transformer или случайный лес, а сравнение моделей честное: одни фолды, один декодер, одни метрики.

Классы-наследники: `SklearnSegmenter` (`classic.py`) и `TransformerSegmenter` (`transformer.py`).

## EventSegmenter  [f432e13f]
Абстрактный класс (`ABC`) модели разметки ряда по дням. Четыре абстрактных метода — `fit`, `predict_proba`, `save`, `load` — и общие атрибуты:

- `name`, `title` — машинное имя (`gboost`) и русское название для таблиц и графиков;
- `features` — `FeatureBuilder`, общий для всех моделей;
- `context_days` — радиус маски обучения (по умолчанию `CONTEXT_DAYS` = 45).

Контракт `predict_proba`: массив `float32/float64` длины `len(well.frame)`, значения в [0; 1] — вероятность, что именно в этот день идёт эффект мероприятия.

### Пример
Создать новую модель — значит написать наследника и добавить его в `model_registry`:

```python
class MyModel(EventSegmenter):
    name, title = "my", "Моя модель"
    def fit(self, wells): ...
    def predict_proba(self, well): ...
    def save(self, path): ...
    @classmethod
    def load(cls, path): ...
```
Если забыть один из методов, `MyModel()` выбросит `TypeError: Can't instantiate abstract class`.

## EventSegmenter.__init__  [7df2a653]
Создаёт модель с набором признаков и радиусом маски обучения.

Параметры: `features` — готовый `FeatureBuilder` (по умолчанию создаётся новый, он без состояния); `context_days` — сколько дней до начала и после окончания размеченного мероприятия используется в обучении (по умолчанию 45; `None` — все дни скважины).

**Почему 45:** в разметке есть не все реальные подъёмы частоты, поэтому «день вне разметки» не значит «эффекта нет». Надёжны только окрестности размеченных мероприятий; их и берёт `WellSeries.training_mask`. После мероприятий «ПРОДОЛЖАЕТСЯ ЭФФЕКТ» (`censored`) дни после даты окончания не берутся совсем.

### Пример
```python
model = SklearnSegmenter("gboost", "Градиентный бустинг", factory)   # context_days = 45
```
Для скважины 8822 (начало 30.07.2022, окончание 05.09.2022) из 1587 дней ряда обучению достанется 128 (от 15.06.2022 до 20.10.2022), а целевыми будут 38 дней эффекта.

## EventSegmenter.fit  [29a571d2]
Абстрактный метод: обучить модель на размеченных скважинах. Принимает последовательность `WellSeries` (только обучающие скважины фолда — разбиение делает `CrossValidator.folds`), возвращает `self`, чтобы можно было писать `model.fit(train).predict_proba(well)`.

Обязанности реализации: брать признаки только из `self.features`, учиться только на днях `well.training_mask(self.context_days)`, цель — `well.target()`.

### Пример
```python
model = reg["extra_trees"]().fit(train_wells)    # train_wells: ~110 скважин фолда
```

## EventSegmenter.predict_proba  [a5d669cc]
Абстрактный метод: вероятность «день внутри эффекта» для каждого дня одной скважины. Принимает `WellSeries` (разметка не нужна), возвращает одномерный массив длины `len(well.frame)`. Вероятности потом сглаживает и режет по порогу `SegmentDecoder` (порог 0,5, окно сглаживания 5 дней).

### Пример
Для скважины 8822 `predict_proba` возвращает массив из 1587 чисел; окно дней около 30.07.2022 — ближе к 1, остальные — близко к 0.

## EventSegmenter.save  [c8370f0d]
Абстрактный метод: записать обученную модель в файл `path` (родительские папки создаются). Формат выбирает наследник: `.joblib` у scikit-learn, `.pt` у Transformer (см. `file_suffix`).

### Пример
```python
model.save(Path("gtm_detector/ml_freq/weights") / f"{model.name}{model.file_suffix}")
```

## EventSegmenter.load  [af60bb63]
Абстрактный классовый метод: восстановить обученную модель из файла. Возвращает готовый к `predict_proba` экземпляр без повторного обучения.

### Пример
```python
model = TransformerSegmenter.load(Path("gtm_detector/ml_freq/weights/transformer.pt"))
```

## EventSegmenter.file_suffix  [abcf7606]
Расширение файла для сохранения: по умолчанию `.joblib`. Transformer переопределяет его на `.pt`. Благодаря свойству код сохранения пишет `name + file_suffix`, не зная типа модели.

### Пример
```python
SklearnSegmenter(...).file_suffix    # '.joblib'
TransformerSegmenter().file_suffix   # '.pt'
```
