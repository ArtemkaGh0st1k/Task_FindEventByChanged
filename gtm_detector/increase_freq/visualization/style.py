"""Общий стиль графиков.

Цвет несёт только смысл «найдено / разметка» (первые два слота проверенной
категориальной палитры — различимы и при дальтонизме). Ряды замеров — нейтральный
серый, у каждого параметра своя панель (никаких двух шкал на одной оси).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # без окна: только сохранение в файлы

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

DETECTED = "#2a78d6"  # найдено детектором
LABELED = "#eb6834"  # разметка технологов
SERIES = "#52514e"  # ряды замеров
MUTED = "#a9a8a0"  # второстепенное: простои, «нет в разметке»
TEXT = "#2b2a28"
TEXT_SECONDARY = "#6b6a65"
GRID = "#e4e3de"

DATE_FORMAT = mdates.DateFormatter("%d.%m")
MONTHS = mdates.MonthLocator()  # деления по 1-м числам: быстрее авторасстановки
MONTH_NAMES = ("янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек")


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",  # есть кириллица
            "font.size": 9,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 110,
        }
    )
