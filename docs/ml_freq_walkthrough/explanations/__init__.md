## @module  [da39a3ee]
Корень пакета `gtm_detector.ml_freq`. Файл содержит только строку документации: ничего не импортирует и не запускает, поэтому `import gtm_detector.ml_freq` мгновенный (не тянет PyTorch и scikit-learn).

Пакет решает одну задачу: для каждого дня скважины оценить вероятность, что идёт эффект мероприятия «увеличение частоты F», а затем общим декодером получить даты начала и окончания. Цепочка такая: Excel → `data` (скважины) → `features` (признаки) → `models` (Transformer, scikit-learn) → `evaluation` (интервалы и метрики) → `cli`.

### Пример
```python
import gtm_detector.ml_freq as pkg
print(pkg.__doc__.splitlines()[0])   # ML-детекция мероприятий «увеличение частоты F».
```
Запуск самого детектора идёт не отсюда, а через `python -m gtm_detector.ml_freq compare` (см. `__main__.py`).
