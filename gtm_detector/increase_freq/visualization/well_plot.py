"""Вариант 1: график по скважине — замеры, найденные и размеченные периоды."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..evaluation import MatchResult
from ..models import Event, Label, Well
from .style import DATE_FORMAT, DETECTED, LABELED, MONTHS, MUTED, SERIES, TEXT_SECONDARY, apply_style


def _span_end(day: date) -> date:
    """Правая граница заливки: период включает последний день целиком."""
    return day + timedelta(days=1)


def _format_period(start: date, end: date | None) -> str:
    return f"{start:%d.%m}–{end:%d.%m}" if end else f"{start:%d.%m}–…"


class WellPlotter:
    """Рисует скважину: панели F, Qж, Qн, обводнённость и полоса простоев с общей осью дат.

    * синяя заливка — период эффекта, найденный детектором; синяя линия — скачок частоты;
    * оранжевая штриховка — период мониторинга из разметки; оранжевый пунктир — «Дата выполнения»;
    * синий пунктир на Qж и Qн — базовый уровень «до», по нему детектор закрывает эффект.
    """

    PANELS = (
        ("frequency", "F, Гц"),
        ("debit_liquid", "Qж, м³/сут"),
        ("debit_oil", "Qн, т/сут"),
        ("watercut", "Обв, %"),
    )
    BASELINES = {"debit_liquid": "base_liquid", "debit_oil": "base_oil"}

    def __init__(self) -> None:
        apply_style()

    def plot(self, well: Well, events: Sequence[Event], match: MatchResult | None, path: Path) -> None:
        panels = [(column, title) for column, title in self.PANELS if column in well.data]
        figure, axes = plt.subplots(
            len(panels) + 1,
            1,
            sharex=True,
            figsize=(13, 2.0 * len(panels) + 2.2),
            gridspec_kw={"height_ratios": [3] * len(panels) + [0.5]},
        )
        data = well.data
        last_day = data.index[-1].date()

        for axis, (column, title) in zip(axes, panels, strict=False):
            axis.plot(data.index, data[column], color=SERIES, linewidth=1.2)
            axis.set_ylabel(title)
            self._draw_periods(axis, events, well.labels, last_day)
            if column in self.BASELINES:
                self._draw_baselines(axis, events, self.BASELINES[column], last_day)

        self._draw_state(axes[-1], well)
        axes[-1].xaxis.set_major_locator(MONTHS)
        axes[-1].xaxis.set_major_formatter(DATE_FORMAT)
        axes[-1].set_xlim(data.index[0], data.index[-1])

        figure.suptitle(
            f"Скважина {well.number}, куст {well.cluster}", x=0.01, y=0.99, ha="left", fontsize=13
        )
        figure.text(
            0.01,
            0.955,
            self._caption(events, well.labels, match),
            ha="left",
            va="top",
            fontsize=9,
            color=TEXT_SECONDARY,
        )
        figure.legend(handles=self._legend_handles(), loc="upper right", ncol=3, bbox_to_anchor=(0.99, 0.995))
        # Поля задаём явно: tight_layout перерисовывает рисунок несколько раз и сильно тормозит.
        figure.subplots_adjust(left=0.07, right=0.99, bottom=0.04, top=0.88, hspace=0.12)

        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path)
        plt.close(figure)

    @staticmethod
    def _draw_periods(axis, events: Sequence[Event], labels: Sequence[Label], last_day: date) -> None:
        for event in events:
            axis.axvspan(
                event.start, _span_end(event.end or last_day), color=DETECTED, alpha=0.14, linewidth=0
            )
            axis.axvline(event.date_step, color=DETECTED, linewidth=1.4)
        for label in labels:
            axis.axvspan(
                label.start,
                _span_end(label.end or last_day),
                facecolor="none",
                edgecolor=LABELED,
                hatch="//",
                linewidth=0,
                alpha=0.55,
            )
            if label.date_done:
                axis.axvline(label.date_done, color=LABELED, linewidth=1.4, linestyle=(0, (4, 3)))

    @staticmethod
    def _draw_baselines(axis, events: Sequence[Event], attribute: str, last_day: date) -> None:
        for event in events:
            level = getattr(event, attribute)
            if level is not None:
                axis.hlines(
                    level,
                    event.date_step,
                    _span_end(event.end or last_day),
                    color=DETECTED,
                    linewidth=1.2,
                    linestyle=(0, (1, 2)),
                )

    @staticmethod
    def _draw_state(axis, well: Well) -> None:
        stopped = ~well.data["state"].to_numpy()
        axis.fill_between(well.data.index, 0, 1, where=stopped, color=MUTED, step="mid", linewidth=0)
        axis.set_ylim(0, 1)
        axis.set_yticks([])
        axis.set_ylabel("простой", rotation=0, ha="right", va="center")
        axis.grid(False)

    @staticmethod
    def _caption(events: Sequence[Event], labels: Sequence[Label], match: MatchResult | None) -> str:
        lines = []
        if match is not None:
            for label, event in match.pairs:
                labeled = f"разметка {_format_period(label.start, label.end)}"
                if event is None:
                    lines.append(f"{labeled}  →  не найдено")
                    continue
                start_delta = (event.start - label.start).days
                end_delta = (event.end - label.end).days if event.end and label.end else None
                end_text = f"{end_delta:+d}" if end_delta is not None else "—"
                lines.append(
                    f"{labeled}  →  найдено {_format_period(event.start, event.end)} "
                    f"(Δ начала {start_delta:+d}, Δ конца {end_text} дн; {event.end_reason_title})"
                )
            extra = len(match.unmatched_events)
        else:
            extra = len(events)
        if extra:
            lines.append(f"найдено вне разметки: {extra}")
        return "\n".join(lines[:4] + (["…"] if len(lines) > 4 else []))

    @staticmethod
    def _legend_handles() -> list:
        return [
            Patch(facecolor=DETECTED, alpha=0.14, label="найдено: эффект"),
            Line2D([], [], color=DETECTED, linewidth=1.4, label="найдено: скачок F"),
            Line2D([], [], color=DETECTED, linewidth=1.2, linestyle=(0, (1, 2)), label="уровень «до»"),
            Patch(facecolor="none", edgecolor=LABELED, hatch="///", label="разметка: мониторинг"),
            Line2D([], [], color=LABELED, linewidth=1.4, linestyle=(0, (4, 3)), label="разметка: выполнено"),
            Patch(facecolor=MUTED, label="простой"),
        ]


def well_file_name(well: Well) -> str:
    """Имя файла без символов, недопустимых в Windows."""
    raw = f"{well.number}_куст_{well.cluster}" if well.cluster else well.number
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw) + ".png"
