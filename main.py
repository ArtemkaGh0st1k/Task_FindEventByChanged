"""Точка входа проекта: детектор ГТМ «увеличение частоты F».

Запуск из любой папки (пути к данным берутся из constants/path.py)::

    python main.py                       # поиск ГТМ + сравнение с разметкой + графики
    python main.py detect                # только поиск ГТМ (без сравнения и графиков)
    python main.py detect --plots        # поиск + графики (без разметки — все найденные)
    python main.py calibrate --cv 5      # подбор параметров по разметке
    python main.py --help                # все команды и параметры

Сама логика — в пакете gtm_detector/increase_freq.
Старые подходы (нейросеть, промпты) — в neural-network/ и prompt-engineering/.
"""

import sys

from constants.path import Constant
from gtm_detector.increase_freq.cli import main

# Без аргументов: полный прогон — найти ГТМ, сравнить с расставленными датами,
# построить сводку качества и графики размеченных скважин.
DEFAULT_ARGS = ["detect", "--results", str(Constant.OUTPUT_DATASET), "--plots"]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or DEFAULT_ARGS, prog="python main.py"))
