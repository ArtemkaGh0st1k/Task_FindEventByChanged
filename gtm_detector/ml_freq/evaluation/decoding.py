"""Вероятности по дням -> мероприятия с датами начала и окончания."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd

from ..config import DecoderConfig
from ..data import Interval


class SegmentDecoder:
    """Общий для всех моделей шаг: одинаковые правила — честное сравнение.

    1. Вероятности сглаживаются скользящим средним (``smooth_days``).
    2. Дни выше ``threshold`` — «в эффекте».
    3. Разрывы не длиннее ``max_gap`` дней склеиваются.
    4. Периоды короче ``min_length`` дней отбрасываются.
    5. Период, дошедший до конца данных, — эффект продолжается (``end=None``).
    """

    def __init__(self, config: DecoderConfig | None = None) -> None:
        self.config = config or DecoderConfig()

    def decode(self, proba: np.ndarray, dates: Sequence[date]) -> list[Interval]:
        cfg = self.config
        smoothed = pd.Series(proba).rolling(cfg.smooth_days, center=True, min_periods=1).mean().to_numpy()
        active = smoothed > cfg.threshold
        runs = self._runs(active)
        runs = self._merge(runs, cfg.max_gap)
        last = len(proba) - 1
        return [
            Interval(start=dates[begin], end=None if end == last else dates[end])
            for begin, end in runs
            if end - begin + 1 >= cfg.min_length
        ]

    @staticmethod
    def _runs(active: np.ndarray) -> list[tuple[int, int]]:
        edges = np.diff(np.r_[0, active.astype(int), 0])
        starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
        return list(zip(starts.tolist(), stops.tolist(), strict=True))

    @staticmethod
    def _merge(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []
        for begin, end in runs:
            if merged and begin - merged[-1][1] - 1 <= max_gap:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((begin, end))
        return merged
