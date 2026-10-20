"""Вариант 2: одна страница со сводкой качества относительно разметки."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..evaluation import EventMatcher, Score
from ..models import Event, Label, Well
from .style import (
    DATE_FORMAT,
    DETECTED,
    GRID,
    LABELED,
    MONTH_NAMES,
    MONTHS,
    MUTED,
    TEXT,
    TEXT_SECONDARY,
    apply_style,
)

CLIP_DAYS = 15  # ошибки больше — в крайние столбцы гистограммы


def reason_group(reason: str | None) -> str:
    """Свести причины окончания из разметки к нескольким группам."""
    text = (reason or "").strip().lower()
    if text.startswith("по обвод"):
        return "по обводнённости"
    if text.startswith("по жидк"):
        return "по жидкости"
    if "останов" in text or text == "идн":
        return "остановка / ИДН"
    if "отказ" in text or "эцн" in text:
        return "отказ / смена ЭЦН"
    return "прочее"


@dataclass
class _Collected:
    """Пары «разметка — найдено» и несопоставленные события по всем скважинам."""

    pairs: list[tuple[Label, Event | None]] = field(default_factory=list)
    unmatched: list[Event] = field(default_factory=list)


class SummaryPlotter:
    """Сводка на одной странице (2 × 3):

    1. ошибка даты начала (найдено − разметка), гистограмма;
    2. ошибка даты окончания, гистограмма;
    3. окончание: разметка против найденного (точки на диагонали — точное попадание);
    4. доля окончаний с ошибкой ≤ 3 дней по причине окончания в разметке;
    5. найденные мероприятия по месяцам: совпавшие с разметкой / вне разметки / пропущенные;
    6. ключевые метрики.
    """

    def __init__(self, matcher: EventMatcher | None = None) -> None:
        self.matcher = matcher or EventMatcher()
        apply_style()

    def plot(
        self,
        wells: Sequence[Well],
        events_by_well: Mapping[str, Sequence[Event]],
        score: Score,
        path: Path,
    ) -> None:
        collected = self._collect(wells, events_by_well)
        figure, axes = plt.subplots(2, 3, figsize=(17, 10))
        (start_ax, end_ax, scatter_ax), (reason_ax, month_ax, metrics_ax) = axes

        self._error_histogram(start_ax, score.start_errors, "Ошибка начала эффекта, дн")
        self._error_histogram(end_ax, score.end_errors, "Ошибка окончания эффекта, дн")
        self._end_scatter(scatter_ax, collected)
        self._end_accuracy_by_reason(reason_ax, collected)
        self._by_month(month_ax, collected)
        self._metrics(metrics_ax, score, collected)

        figure.suptitle(
            "Детектор «увеличение частоты F»: сравнение с разметкой", x=0.01, ha="left", fontsize=14
        )
        figure.subplots_adjust(left=0.1, right=0.98, bottom=0.08, top=0.88, wspace=0.28, hspace=0.42)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path)
        plt.close(figure)

    def _collect(self, wells: Sequence[Well], events_by_well: Mapping[str, Sequence[Event]]) -> _Collected:
        collected = _Collected()
        for well in wells:
            if not well.labels:
                continue
            result = self.matcher.match(events_by_well.get(well.well_id, []), well.labels)
            collected.pairs.extend(result.pairs)
            collected.unmatched.extend(result.unmatched_events)
        return collected

    @staticmethod
    def _error_histogram(axis, errors: Sequence[int], title: str) -> None:
        clipped = np.clip(np.asarray(errors, dtype=int), -CLIP_DAYS, CLIP_DAYS)
        bins = np.arange(-CLIP_DAYS - 0.5, CLIP_DAYS + 1.5)
        axis.hist(clipped, bins=bins, color=DETECTED, edgecolor="white", linewidth=1)
        axis.axvline(0, color=TEXT, linewidth=0.8)
        exact = np.mean(clipped == 0) if len(clipped) else 0
        within3 = np.mean(np.abs(clipped) <= 3) if len(clipped) else 0
        axis.set_title(f"{title}\nточно: {exact:.0%}, ±3 дн: {within3:.0%}, n = {len(clipped)}", loc="left")
        axis.set_xlabel(f"найдено − разметка, дн (крайние столбцы: ≤ −{CLIP_DAYS} и ≥ +{CLIP_DAYS})")
        axis.set_ylabel("мероприятий")

    @staticmethod
    def _end_scatter(axis, collected: _Collected) -> None:
        points = [
            (label.end, event.end)
            for label, event in collected.pairs
            if event is not None and event.end is not None and label.end is not None
        ]
        if points:
            true_end, found_end = zip(*points, strict=True)
            axis.scatter(
                true_end, found_end, s=22, color=DETECTED, edgecolor="white", linewidth=0.8, zorder=3
            )
            low, high = min(true_end + found_end), max(true_end + found_end)
            axis.plot([low, high], [low, high], color=MUTED, linewidth=1, zorder=2)
        for date_axis in (axis.xaxis, axis.yaxis):
            date_axis.set_major_locator(MONTHS)
            date_axis.set_major_formatter(DATE_FORMAT)
        axis.grid(True, axis="both")
        axis.set_xlabel("окончание по разметке")
        axis.set_ylabel("окончание найденное")
        axis.set_title("Окончание: разметка и найденное\nна диагонали — точное попадание", loc="left")

    @staticmethod
    def _end_accuracy_by_reason(axis, collected: _Collected) -> None:
        totals: Counter[str] = Counter()
        hits: Counter[str] = Counter()
        for label, event in collected.pairs:
            if event is None or label.end is None:
                continue
            group = reason_group(label.end_reason)
            totals[group] += 1
            if event.end is not None and abs((event.end - label.end).days) <= 3:
                hits[group] += 1
        groups = sorted(totals, key=totals.get)
        shares = [hits[group] / totals[group] for group in groups]
        bars = axis.barh(groups, shares, color=DETECTED, height=0.6)
        for bar, group in zip(bars, groups, strict=True):
            axis.text(
                bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{hits[group]} из {totals[group]}",
                va="center",
                color=TEXT_SECONDARY,
            )
        axis.set_xlim(0, 1.15)
        axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        axis.grid(True, axis="x")
        axis.grid(False, axis="y")
        axis.set_title("Окончание с ошибкой ≤ 3 дн\nпо причине окончания в разметке", loc="left")

    @staticmethod
    def _by_month(axis, collected: _Collected) -> None:
        def months(days: Sequence[date]) -> pd.Series:
            return pd.Series(1, index=pd.to_datetime(list(days))).resample("MS").sum()

        matched = months([event.date_step for _, event in collected.pairs if event is not None])
        extra = months([event.date_step for event in collected.unmatched])
        missed = months([label.start for label, event in collected.pairs if event is None])
        table = pd.concat(
            {"совпало с разметкой": matched, "вне разметки": extra, "пропущено": missed}, axis=1
        ).fillna(0)

        labels = [MONTH_NAMES[month.month - 1] for month in table.index]
        bottom = np.zeros(len(table))
        for column, color in (
            ("совпало с разметкой", DETECTED),
            ("вне разметки", MUTED),
            ("пропущено", LABELED),
        ):
            axis.bar(
                labels,
                table[column],
                bottom=bottom,
                color=color,
                label=column,
                edgecolor="white",
                linewidth=1.5,
                width=0.7,
            )
            bottom += table[column].to_numpy()
        axis.legend(loc="upper right")
        axis.set_ylabel("мероприятий")
        axis.set_title("Мероприятия по месяцам (скважины с разметкой)\nпо дате скачка частоты", loc="left")

    @staticmethod
    def _metrics(axis, score: Score, collected: _Collected) -> None:
        axis.axis("off")
        start, end = np.asarray(score.start_errors), np.asarray(score.end_errors)
        rows = [
            (
                "Найдено размеченных",
                f"{score.recall:.0%}",
                f"{score.true_positives} из {score.true_positives + score.false_negatives}",
            ),
            ("Начало: точно", f"{np.mean(start == 0):.0%}", f"±1 дн: {np.mean(np.abs(start) <= 1):.0%}"),
            ("Окончание: точно", f"{np.mean(end == 0):.0%}", f"±3 дн: {np.mean(np.abs(end) <= 3):.0%}"),
            (
                "Вне разметки",
                f"{len(collected.unmatched)}",
                f"из них рядом с известными ГТМ: {score.neutral}",
            ),
        ]
        axis.set_title("Ключевые метрики", loc="left")
        for i, (name, value, note) in enumerate(rows):
            y = 0.86 - i * 0.24
            axis.text(0.0, y, name, fontsize=10, color=TEXT_SECONDARY, transform=axis.transAxes)
            axis.text(0.0, y - 0.11, value, fontsize=24, color=TEXT, transform=axis.transAxes)
            axis.text(0.42, y - 0.09, note, fontsize=9, color=TEXT_SECONDARY, transform=axis.transAxes)
        axis.axhline(0.0, color=GRID)
        axis.text(
            0.0,
            -0.12,
            "Точность (precision) по разметке не оценить:\nв ней только рекомендации из одного расчёта.",
            fontsize=8,
            color=TEXT_SECONDARY,
            transform=axis.transAxes,
        )
