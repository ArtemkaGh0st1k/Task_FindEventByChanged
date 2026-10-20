"""Шаг 1: поиск скачков частоты вверх."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DetectorConfig
from ..models import FrequencyStep


class StepDetector:
    """Находит дни, когда частоту насоса подняли.

    1. Берутся только рабочие дни (``state=True`` и частота > ``min_working_freq``),
       поэтому пуск после простоя (0 -> 50 Гц) скачком не считается.
    2. Частота сглаживается скользящей медианой — одиночные выбросы пропадают.
    3. День с приростом >= ``min_step`` — ступень. Ступени с разрывом не более
       ``max_gap`` рабочих дней и без снижения между ними объединяются в один скачок.
    4. Скачок принимается, если суммарный прирост >= ``min_total`` и новый уровень
       держится ``hold_days`` рабочих дней.
    """

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config

    def find(self, data: pd.DataFrame) -> list[FrequencyStep]:
        cfg = self.config
        working = data["state"].to_numpy() & (data["frequency"].to_numpy() > cfg.min_working_freq)
        positions = np.flatnonzero(working)  # позиции рабочих дней в data
        if len(positions) < 2:
            return []

        frequency = data["frequency"].iloc[positions].reset_index(drop=True)
        smoothed = frequency.rolling(max(1, cfg.smooth_window), center=True, min_periods=1).median()
        diff = smoothed.diff().to_numpy()
        is_up = diff >= cfg.min_step
        is_down = diff <= -cfg.min_step
        after_stop = np.r_[False, np.diff(positions) > 1]

        steps = []
        for group in self._group_steps(is_up, is_down):
            step = self._validate(group, smoothed, positions, after_stop)
            if step is not None:
                steps.append(step)
        return steps

    def _group_steps(self, is_up: np.ndarray, is_down: np.ndarray) -> list[list[int]]:
        """Объединить соседние ступени роста в группы (ступенчатый вывод на режим)."""
        groups: list[list[int]] = []
        for i in np.flatnonzero(is_up):
            if groups:
                previous = groups[-1][-1]
                close_enough = i - previous - 1 <= self.config.max_gap
                if close_enough and not is_down[previous + 1 : i].any():
                    groups[-1].append(int(i))
                    continue
            groups.append([int(i)])
        return groups

    def _validate(
        self,
        group: list[int],
        smoothed: pd.Series,
        positions: np.ndarray,
        after_stop: np.ndarray,
    ) -> FrequencyStep | None:
        cfg = self.config
        first, last = group[0], group[-1]

        before = smoothed.iloc[max(0, first - cfg.pre_window) : first].dropna()
        hold = smoothed.iloc[last : last + cfg.hold_days]
        if before.empty or len(hold) < cfg.hold_days:
            return None

        f_before = float(before.median())
        f_after = float(hold.median())
        delta = f_after - f_before
        if delta < cfg.min_total:
            return None
        if hold.min() < f_after - cfg.hold_tolerance * delta:
            return None  # кратковременный всплеск — частоту сразу вернули

        after_restart = bool(after_stop[first : last + 1].any())
        if cfg.skip_after_restart and after_restart:
            return None

        return FrequencyStep(
            index=int(positions[first]),
            previous_index=int(positions[first - 1]),
            f_before=f_before,
            f_after=f_after,
            n_steps=len(group),
            after_restart=after_restart,
        )
