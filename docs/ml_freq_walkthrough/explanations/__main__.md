## @module  [1c2ef65f]
Точка входа `python -m gtm_detector.ml_freq <команда>`. Модуль делает единственное: вызывает `cli.main()` и передаёт её код возврата в `sys.exit`, чтобы оболочка видела успех (0) или ошибку.

Вся логика команд (`compare`, `train`, `predict`, `audit`) лежит в `cli.py`; здесь её нет, чтобы модуль можно было импортировать в тестах без побочных эффектов.

### Пример
```bash
.venv/Scripts/python -m gtm_detector.ml_freq audit                              # аудит данных 2021–2026
.venv/Scripts/python -m gtm_detector.ml_freq compare --models gboost extra_trees # около 2 минут
```
Обе команды запускают `main()` из `cli.py`; результаты попадают в `results/ml_freq/`.
