## @module  [ffa54e0a]
Четыре готовые модели scikit-learn, обёрнутые под общий интерфейс `EventSegmenter`: логистическая регрессия, случайный лес, Extra Trees и градиентный бустинг (`HistGradientBoostingClassifier`). Каждая решает задачу «строка = день скважины, 106 признаков из `FeatureBuilder.tabular`, ответ 0/1 — день внутри эффекта». Контекст соседних дней уже зашит в признаки (сдвиги ±1…14 дней, средние за неделю), поэтому моделям не нужны ни последовательности, ни внимание.

Классические модели на ~120 скважинах оказались лучшими в проекте (результаты `compare` на данных 2021–2026, 5 фолдов по скважинам):

| Модель | полнота | точность | начало ±3 дн | ROC-AUC | обучение, с |
|--------|---------|----------|--------------|---------|-------------|
| Случайный лес | 0,615 | 0,388 | 0,783 | 0,919 | 41,6 |
| Extra Trees | 0,615 | 0,388 | 0,771 | 0,920 | 34,9 |
| Градиентный бустинг (HistGB) | 0,622 | 0,370 | 0,810 | 0,917 | 112,2 |
| Transformer | 0,481 | 0,267 | 0,831 | 0,831 | 1522 |
| Логистическая регрессия | 0,481 | 0,208 | 0,769 | 0,858 | 25,1 |

Точность занижена намеренно: разметка неполная, многие настоящие подъёмы частоты в ней отсутствуют.

## SklearnSegmenter  [e0715931]
Обёртка над любым классификатором scikit-learn с методами `predict_proba` (пайплайн, лес, бустинг). Хранит не готовую модель, а **фабрику** `estimator_factory`: на каждом `fit` создаётся новый необученный классификатор. Это важно для перекрёстной проверки — иначе веса одного фолда протекли бы в следующий.

Поля: `name` / `title` (как в `model_registry`), `estimator` — обученный классификатор или `None`.

### Пример
```python
reg = classic_models()
model = reg["random_forest"]()      # SklearnSegmenter, estimator is None
model.title                          # 'Случайный лес'
model.fit(train_wells)               # estimator -> RandomForestClassifier(...)
```

## SklearnSegmenter.__init__  [4a684995]
Принимает: `name` — короткое имя (`gboost`), `title` — русское название для таблиц, `estimator_factory` — функцию без аргументов, возвращающую необученный классификатор, `features` — `FeatureBuilder` (по умолчанию новый). `context_days` берётся по умолчанию (45) из `EventSegmenter`.

### Пример
```python
SklearnSegmenter("logreg", "Логистическая регрессия", lambda: LogisticRegression())
```

## SklearnSegmenter.fit  [7390ad5e]
Обучает классификатор на днях из маски обучения.

1. Для каждой скважины берётся `training_mask(45)`: ±45 дней вокруг размеченного мероприятия (у `censored`-мероприятия «ПРОДОЛЖАЕТСЯ ЭФФЕКТ» дни после даты окончания в маску не входят: что там происходило, неизвестно).
2. Матрица признаков `tabular(well)[mask]` — форма `(дней_в_маске, 106)`; цель `target()[mask]` — 0/1.
3. Строки всех скважин складываются (`vstack`), фабрика создаёт классификатор, вызывается `fit(x, y)`.

Баланс классов: положительных дней заметно меньше (по всем размеченным скважинам 5 213 из 14 247 строк маски — 37 %), поэтому в моделях стоит `class_weight` (`balanced`).

Возвращает `self`. Зерно случайности не здесь, а в `random_state` фабрики (`classic_models(seed)`).

### Пример
```python
model = classic_models()["gboost"]().fit(train_wells)
```
Если взять все скважины сразу, маска даёт 14 247 строк × 106 признаков (из 201 918 дней рядов 698 скважин; остальное — неразмеченные дни); обучение `gboost` на пяти фолдах заняло 112 секунд, `random_forest` — 42.

## SklearnSegmenter.predict_proba  [9ce6ee93]
Возвращает вероятность класса «эффект» для каждого дня скважины: `estimator.predict_proba(tabular(well))[:, 1]`, массив длины `len(well.frame)`. Предсказание идёт по **всем** дням (маска нужна только на обучении), чтобы декодер видел весь ряд. Если модель не обучена, выбрасывается `RuntimeError("Модель gboost не обучена")`.

### Пример
```python
p = model.predict_proba(well)      # well — 1587 дней
p.shape                            # (1587,)
p[well.target() == 1].mean()       # заметно выше, чем вне эффекта
```

## SklearnSegmenter.save  [f83598c7]
Сохраняет через `joblib.dump` словарь `{name, title, estimator}` — модель вместе с названием, чтобы `load` не нуждался в реестре. Расширение `.joblib`, папки создаются.

### Пример
```python
model.save(Path("gtm_detector/ml_freq/weights/gboost.joblib"))
```

## SklearnSegmenter.load  [e932bd8f]
Классовый метод: читает файл `.joblib` и собирает `SklearnSegmenter` с сохранённым классификатором. Фабрика в загруженной модели возвращает тот же **уже обученный** объект (`lambda: state["estimator"]`), поэтому повторный `fit` на загруженной модели переобучит именно его; для новой проверки создавайте модель через `model_registry`.

### Пример
```python
model = SklearnSegmenter.load(Path("gtm_detector/ml_freq/weights/gboost.joblib"))
model.name    # 'gboost'
```

## classic_models  [fa5a93dc]
Возвращает словарь «имя → фабрика» четырёх моделей. Гиперпараметры подобраны консервативно под малые данные (в `config.py` они не вынесены, остались в этой функции).

| Ключ | Модель | Параметры |
|------|--------|-----------|
| `logreg` | `StandardScaler` + `LogisticRegression` | `C=0.5`, `class_weight="balanced"`, `max_iter=2000` |
| `random_forest` | `RandomForestClassifier` | 300 деревьев, `min_samples_leaf=5`, `balanced_subsample` |
| `extra_trees` | `ExtraTreesClassifier` | то же, но случайные пороги |
| `gboost` | `HistGradientBoostingClassifier` | 300 итераций, `learning_rate=0.05`, 31 лист, `l2_regularization=1.0`, `balanced` |

`min_samples_leaf=5` и L2-регуляризация сдерживают переобучение на 120 скважинах. Логистическая регрессия единственная масштабирует признаки (линейной модели важны масштабы; деревьям — нет). Параметр `seed` (по умолчанию 0) идёт в `random_state` всех моделей, поэтому результаты воспроизводимы.

Тонкость: лямбда `lambda name=name, title=title, factory=factory: ...` связывает переменные цикла сразу; без значений по умолчанию все четыре фабрики создавали бы последнюю модель (`gboost`).

### Пример
```python
classic_models(seed=0).keys()   # dict_keys(['logreg', 'random_forest', 'extra_trees', 'gboost'])
```
Запуск сравнения: `python -m gtm_detector.ml_freq compare --models gboost extra_trees` (около 2 минут).
