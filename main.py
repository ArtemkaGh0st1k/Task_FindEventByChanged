"""Точка входа проекта: детектор ГТМ «увеличение частоты F».

Запуск из любой папки (пути к данным берутся из constants/path.py)::

    python main.py                       # поиск ГТМ + сравнение с расставленными датами
    python main.py detect                # поиск ГТМ без сравнения
    python main.py calibrate --cv 5      # подбор параметров по разметке
    python main.py --help                # все команды и параметры

Сама логика — в пакете gtm_detector/increase_freq.
Старые подходы (нейросеть, промпты) — в neural-network/ и prompt-engineering/.
"""

import sys

from constants.path import Constant
from gtm_detector.increase_freq.cli import main

# Без аргументов: найти ГТМ во входных данных и сравнить с расставленными датами.
DEFAULT_ARGS = ["detect", "--results", str(Constant.OUTPUT_DATASET)]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or DEFAULT_ARGS, prog="python main.py"))
