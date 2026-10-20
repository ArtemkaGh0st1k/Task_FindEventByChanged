"""Сборка всех графиков прогона: сводка + графики скважин по папкам."""

from __future__ import annotations

import logging
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from pathlib import Path

from ..evaluation import EventMatcher, MatchResult, Score
from ..models import Event, Well
from .summary import SummaryPlotter
from .well_plot import WellPlotter, well_file_name

logger = logging.getLogger(__name__)


class WellStatus(str, Enum):
    """Папка, в которую попадает график скважины."""

    MATCHED = "совпало"  # все метки найдены, начало ±1 дн, конец ±3 дн
    DATE_ERROR = "ошибка_дат"  # все метки найдены, но даты расходятся сильнее
    MISSED = "пропущено"  # хотя бы одна метка не найдена
    UNLABELED = "нет_в_разметке"  # разметки нет, детектор что-то нашёл
    FOUND = "найдено"  # режим без разметки


class PlotReport:
    """Строит ``summary.png`` и ``wells/<статус>/<скважина>.png``.

    Если разметка есть — графики размеченных скважин раскладываются по статусам
    (``совпало`` / ``ошибка_дат`` / ``пропущено``), а скважины без разметки рисуются
    только при ``include_unlabeled``. Без разметки — все скважины с найденными
    мероприятиями в папку ``найдено``.
    """

    START_TOLERANCE = 1
    END_TOLERANCE = 3

    def __init__(
        self,
        matcher: EventMatcher | None = None,
        include_unlabeled: bool = False,
        workers: int | None = None,
    ) -> None:
        self.matcher = matcher or EventMatcher()
        self.include_unlabeled = include_unlabeled
        # Графики независимы — рисуем параллельно (~1 с на скважину в одном процессе).
        # Не больше 4 процессов: каждый держит pandas + matplotlib (~300 МБ), на 8 ГБ ОЗУ больше не влезает.
        self.workers = workers or max(1, min(4, (os.cpu_count() or 2) - 1))
        self.summary_plotter = SummaryPlotter(self.matcher)

    def build(
        self,
        wells: Sequence[Well],
        events_by_well: Mapping[str, Sequence[Event]],
        out_dir: Path,
        score: Score | None = None,
    ) -> Counter[WellStatus]:
        has_labels = any(well.labels for well in wells)
        if has_labels and score is not None:
            labeled = [well for well in wells if well.labels]
            self.summary_plotter.plot(labeled, events_by_well, score, out_dir / "summary.png")

        tasks = []
        counts: Counter[WellStatus] = Counter()
        for well in wells:
            events = list(events_by_well.get(well.well_id, []))
            match = self.matcher.match(events, well.labels) if well.labels else None
            status = self._status(well, events, match, has_labels)
            if status is None:
                continue
            tasks.append((well, events, match, out_dir / "wells" / status.value / well_file_name(well)))
            counts[status] += 1

        if self.workers > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=self.workers) as pool:
                list(pool.map(_plot_well, tasks, chunksize=4))
        else:
            for task in tasks:
                _plot_well(task)
        logger.info("Графики скважин: %s", dict(counts))
        return counts

    def _status(
        self, well: Well, events: Sequence[Event], match: MatchResult | None, has_labels: bool
    ) -> WellStatus | None:
        if match is None:
            if not events:
                return None
            if not has_labels:
                return WellStatus.FOUND
            return WellStatus.UNLABELED if self.include_unlabeled else None

        if any(event is None for _, event in match.pairs):
            return WellStatus.MISSED
        for label, event in match.pairs:
            start_ok = abs((event.start - label.start).days) <= self.START_TOLERANCE
            end_ok = label.end is None or (
                event.end is not None and abs((event.end - label.end).days) <= self.END_TOLERANCE
            )
            if not (start_ok and end_ok):
                return WellStatus.DATE_ERROR
        return WellStatus.MATCHED


_plotter: WellPlotter | None = None


def _plot_well(task: tuple) -> None:
    """Нарисовать одну скважину (функция верхнего уровня — нужна для ProcessPoolExecutor)."""
    global _plotter
    if _plotter is None:
        _plotter = WellPlotter()
    well, events, match, path = task
    _plotter.plot(well, events, match, path)
