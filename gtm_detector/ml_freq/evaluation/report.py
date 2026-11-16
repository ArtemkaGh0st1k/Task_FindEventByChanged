"""Отчёт сравнения моделей: Excel-таблица и график."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .experiment import ModelResult

logger = logging.getLogger(__name__)

BLUE = "#2a78d6"
TEXT_SECONDARY = "#6b6a65"
GRID = "#e4e3de"


class ComparisonReport:
    """Пишет ``comparison.xlsx`` (метрики + предсказания по моделям) и ``comparison.png``."""

    KEY_METRICS = (
        "найдено размеченных (recall)",
        "начало: точно",
        "начало: ±3 дн",
        "конец: точно",
        "конец: ±7 дн",
        "ROC-AUC по дням",
    )

    def __init__(self, results: Sequence[ModelResult]) -> None:
        self.results = list(results)

    def table(self) -> pd.DataFrame:
        table = pd.DataFrame([result.as_row() for result in self.results]).set_index("модель")
        return table.sort_values("F1", ascending=False)

    def to_text(self) -> str:
        table = self.table()[list(self.KEY_METRICS) + ["F1"]]
        return table.to_string(float_format=lambda value: f"{value:.2f}")

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(out_dir / "comparison.xlsx") as writer:
            self.table().to_excel(writer, sheet_name="Сравнение")
            for result in self.results:
                pd.DataFrame(result.predictions).to_excel(writer, sheet_name=result.name[:31], index=False)
        self._plot(out_dir / "comparison.png")
        logger.info("Отчёт: %s", out_dir)

    def _plot(self, path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        table = self.table()
        figure, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharey=True)
        for axis, metric in zip(axes.flat, self.KEY_METRICS, strict=True):
            values = table[metric].fillna(0)
            bars = axis.barh(table.index, values, color=BLUE, height=0.6)
            for bar, value in zip(bars, values, strict=True):
                axis.text(
                    value + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.0%}",
                    va="center",
                    fontsize=9,
                    color=TEXT_SECONDARY,
                )
            axis.set_xlim(0, 1.12)
            axis.set_title(metric, loc="left", fontsize=10)
            axis.invert_yaxis()
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
            axis.grid(axis="x", color=GRID)
            axis.set_axisbelow(True)
        figure.suptitle("Сравнение моделей (перекрёстная проверка по скважинам)", x=0.01, ha="left")
        figure.tight_layout()
        figure.savefig(path, dpi=110)
        plt.close(figure)
