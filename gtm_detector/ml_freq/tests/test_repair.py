"""Исправление сдвинутых блоков и проверка разметки по частоте."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gtm_detector.ml_freq.data import BlockShiftRepair, FrequencyCheck, Interval
from gtm_detector.ml_freq.data.repair import LabelPoint
from gtm_detector.ml_freq.evaluation import EventMatcher

from .conftest import day, make_well

SHIFT = 30  # в тестах сдвиг короче настоящих 180 дней


def test_frequency_check():
    check = FrequencyCheck()
    series = np.r_[np.full(20, 48.0), np.full(20, 50.0)]
    assert check.confirms(series, 20, 48.0, 50.0)
    assert check.confirms(series, 18, 48.0, 50.0)  # дата технолога на 2 дня раньше скачка
    assert check.confirms(series + 3, 20, 48.0, 50.0)  # телеметрия смещена, прирост тот же
    assert not check.confirms(series, 20, 48.0, 56.0)  # прирост 2 из запланированных 8 Гц
    assert not check.confirms(np.full(40, 48.0), 20, 48.0, 50.0)
    assert check.confirms(np.full(40, np.nan), 20, 48.0, 50.0) is None


def _shifted_block() -> tuple[np.ndarray, np.ndarray, list[LabelPoint], pd.DatetimeIndex]:
    """Скважины 0–5: подъёмы на дни 10+k, но блок 0–59 записан на SHIFT дней позже."""
    n_days = 120
    frequency = np.full((6, n_days), np.nan)
    labels = []
    for k in range(6):
        frequency[k, SHIFT : SHIFT + 60] = 48.0
        frequency[k, SHIFT + 10 + k : SHIFT + 60] = 50.0
        labels.append(LabelPoint(row=k, day=10 + k, current=48.0, new=50.0))
    return frequency, ~np.isnan(frequency), labels, pd.date_range("2026-01-01", periods=n_days)


def test_repair_finds_and_undoes_shift():
    frequency, presence, labels, dates = _shifted_block()
    repair = BlockShiftRepair(shifts=(0, SHIFT))
    segments = repair.segments(presence, dates)
    assert [(s.first, s.last) for s in segments] == [(SHIFT, SHIFT + 59)]

    repair.assess(segments, presence, frequency, labels)
    assert segments[0].shift == SHIFT
    assert segments[0].labels[SHIFT].confirmed == 6 and segments[0].labels[0].confirmed == 0

    table = pd.DataFrame(frequency, columns=dates)
    repaired = repair.apply({"frequency": table, "debit_liquid": table}, segments)["frequency"].to_numpy()
    assert np.isnan(repaired[:, 60:]).all()
    assert repaired[0, 9] == 48.0 and repaired[0, 10] == 50.0


def test_repair_drops_segment_contradicting_labels():
    frequency, presence, labels, dates = _shifted_block()
    wrong = [LabelPoint(p.row, p.day + 7, p.current, p.new) for p in labels]  # ни 0, ни SHIFT
    repair = BlockShiftRepair(shifts=(0, SHIFT))
    segments = repair.segments(presence, dates)
    repair.assess(segments, presence, frequency, wrong)
    assert segments[0].shift is None


def test_seam_takes_neighbour_decision():
    frequency, presence, labels, dates = _shifted_block()
    # Стык: за день до блока данные есть у тех же скважин и ещё у пяти других.
    frequency = np.vstack([frequency, np.full((5, frequency.shape[1]), np.nan)])
    presence = np.vstack([presence, np.zeros((5, presence.shape[1]), dtype=bool)])
    presence[:, SHIFT - 1] = True
    repair = BlockShiftRepair(shifts=(0, SHIFT))
    segments = repair.segments(presence, dates)
    repair.assess(segments, presence, frequency, labels)
    assert segments[0].days == 1 and segments[0].shift == segments[1].shift == SHIFT


def test_censored_label():
    well = make_well(n_days=200, step=40, effect=30)
    end = well.labels[0].end
    well.labels = [Interval(well.labels[0].start, end, censored=True)]
    mask = well.training_mask(45)
    after_end = np.asarray(well.frame.index > pd.Timestamp(end))
    assert not mask[after_end].any() and mask[~after_end].any()

    found = [Interval(well.labels[0].start, day(150))]
    score = EventMatcher().score(found, well.labels)
    assert score.true_positives == 1 and score.end_errors == [] and score.open_ends == 0
