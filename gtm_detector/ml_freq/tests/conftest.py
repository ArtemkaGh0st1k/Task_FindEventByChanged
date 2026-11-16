"""Синтетические скважины: подъём частоты и отклик добычи с известными датами."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from gtm_detector.ml_freq.data import Interval, WellSeries

START = date(2026, 1, 1)


def day(offset: int) -> date:
    return START + timedelta(days=offset)


def make_well(n_days: int = 120, step: int | None = 40, effect: int = 30, seed: int = 0) -> WellSeries:
    """Частота 48 -> 50 Гц в день ``step``, Qж/Qн выше базы ``effect`` дней, затем возврат."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(START, periods=n_days, freq="D")
    frequency = np.full(n_days, 48.0)
    liquid = np.full(n_days, 100.0)
    labels = []
    if step is not None:
        frequency[step:] = 50.0
        liquid[step + 2 : step + 2 + effect] = 110.0
        labels.append(Interval(day(step + 2), day(step + 1 + effect), day(step)))
    frame = pd.DataFrame(
        {
            "frequency": frequency,
            "debit_liquid": liquid + rng.normal(0, 0.3, n_days),
            "debit_oil": liquid * 0.1,
            "watercut": np.full(n_days, 90.0),
            "pressure_zab": np.full(n_days, 50.0),
            "pressure_line": np.full(n_days, 12.0),
            "state": np.ones(n_days),
        },
        index=index,
    )
    return WellSeries(number=f"W{seed}", cluster="1", frame=frame, labels=labels)


@pytest.fixture
def wells() -> list[WellSeries]:
    return [make_well(step=30 + 3 * i, effect=25 + 2 * i, seed=i) for i in range(8)]
