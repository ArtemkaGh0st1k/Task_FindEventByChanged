"""Модели данных: скважина с суточными рядами и интервалы мероприятий."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

SERIES_COLUMNS = (
    "frequency",
    "debit_liquid",
    "debit_oil",
    "watercut",
    "pressure_zab",
    "pressure_line",
)


@dataclass(frozen=True)
class Interval:
    """Мероприятие: период эффекта. ``end=None`` — эффект продолжается до конца данных."""

    start: date
    end: date | None
    date_done: date | None = None  # «Дата выполнения» (только в разметке)
    # Мониторинг остановлен, а эффект продолжался («ПРОДОЛЖАЕТСЯ ЭФФЕКТ»):
    # ``end`` — не конец эффекта, а последний день наблюдения.
    censored: bool = False

    def covers(self, day: date, last_day: date) -> bool:
        return self.start <= day <= (self.end or last_day)


@dataclass
class WellSeries:
    """Скважина: суточные ряды на непрерывном календаре + разметка."""

    number: str
    cluster: str
    frame: pd.DataFrame  # индекс — даты, колонки — SERIES_COLUMNS и state
    labels: list[Interval] = field(default_factory=list)

    @property
    def well_id(self) -> str:
        return f"{self.number}/{self.cluster}"

    @property
    def dates(self) -> list[date]:
        return [day.date() for day in self.frame.index]

    def __len__(self) -> int:
        return len(self.frame)

    def target(self) -> np.ndarray:
        """1 — день внутри размеченного периода эффекта, 0 — вне."""
        days = self.frame.index
        target = np.zeros(len(days), dtype=np.float32)
        for label in self.labels:
            end = pd.Timestamp(label.end) if label.end else days[-1]
            target[(days >= pd.Timestamp(label.start)) & (days <= end)] = 1.0
        return target

    def training_mask(self, context_days: int | None) -> np.ndarray:
        """Дни, на которых можно учиться.

        В разметке есть не все реальные подъёмы частоты, поэтому «вне разметки» не
        значит «эффекта нет». Надёжны только окрестности размеченных мероприятий:
        ``context_days`` дней до начала и после окончания. ``None`` — все дни.
        После окончания ``censored``-мероприятия эффект неизвестен — эти дни не берутся.
        """
        days = self.frame.index
        if context_days is None:
            return np.ones(len(days), dtype=bool)
        mask = np.zeros(len(days), dtype=bool)
        margin = pd.Timedelta(days=context_days)
        for label in self.labels:
            end = pd.Timestamp(label.end) if label.end else days[-1]
            after = end if label.censored else end + margin
            mask[(days >= pd.Timestamp(label.start) - margin) & (days <= after)] = True
        return mask
