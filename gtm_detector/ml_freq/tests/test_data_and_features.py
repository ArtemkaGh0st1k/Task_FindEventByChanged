from datetime import date

import numpy as np
import pytest

from gtm_detector.ml_freq.data import normalize_id, parse_date
from gtm_detector.ml_freq.features import FeatureBuilder
from gtm_detector.ml_freq.tests.conftest import make_well


def test_target_and_training_mask():
    well = make_well(step=40, effect=30)
    target = well.target()
    assert target.sum() == 30 and target[42] == 1 and target[41] == 0
    mask = well.training_mask(context_days=10)
    assert mask[32] and not mask[31]  # 10 дней до начала (42)
    assert mask[81] and not mask[82]  # 10 дней после окончания (71)
    assert well.training_mask(None).all()


def _channel(builder: FeatureBuilder, sequence: np.ndarray, name: str) -> np.ndarray:
    return sequence[:, builder.SEQUENCE_CHANNELS.index(name)]


def test_oil_end_features_follow_return_to_base():
    builder = FeatureBuilder()
    well = make_well(step=40, effect=30)  # нефть 10, в днях 42..71 — 11, затем снова 10
    sequence = builder.sequence(well)
    below = _channel(builder, sequence, "oil_below_base")
    run = _channel(builder, sequence, "oil_days_below")
    ratio = _channel(builder, sequence, "oil_vs_base")
    drawdown = _channel(builder, sequence, "oil_drawdown")
    assert below[:40].sum() == 0  # до первого скачка частоты признаков нет
    assert below[50] == 0 and ratio[50] == pytest.approx(0.1, abs=1e-3)  # эффект: нефть выше базы
    assert below[72] == 1 and run[72] == pytest.approx(1 / 14)  # вернулась к базе
    assert drawdown[72] == pytest.approx(0.1, abs=1e-3)  # просадка от пика эффекта
    assert run[90] == 1.0  # серия дней ниже базы насытилась


def test_oil_end_features_with_several_steps():
    """Несколько скачков частоты: у каждого своя база, массивы разных отрезков не смешиваются."""
    builder = FeatureBuilder()
    well = make_well(step=30, effect=20)
    well.frame.iloc[70:, well.frame.columns.get_loc("frequency")] = 52.0  # второй скачок
    well.frame.iloc[100:, well.frame.columns.get_loc("frequency")] = 54.0  # третий скачок
    for sequence in (builder.sequence(well), builder.tabular(well)):
        assert sequence.shape[0] == len(well) and not np.isnan(sequence).any()
    sequence = builder.sequence(well)
    assert _channel(builder, sequence, "days_since_step")[70] == 0  # отсчёт от нового скачка
    assert _channel(builder, sequence, "oil_below_base")[75] == 1  # нефть на базе у второго скачка


def test_oil_missing_is_not_read_as_return_to_base():
    builder = FeatureBuilder()
    well = make_well(step=40, effect=30)
    well.frame.iloc[60:63, well.frame.columns.get_loc("debit_oil")] = np.nan
    sequence = builder.sequence(well)
    missing = _channel(builder, sequence, "oil_missing")
    assert missing[60:63].all() and missing.sum() == 3
    assert not _channel(builder, sequence, "oil_below_base")[60:63].any()
    assert not _channel(builder, sequence, "oil_vs_base")[60:63].any()

    well.frame["debit_oil"] = np.nan  # нефти нет совсем: признаки нулевые, без ошибок
    empty = builder.sequence(well)
    assert not np.isnan(empty).any()
    assert not _channel(builder, empty, "oil_below_base").any()
    assert _channel(builder, empty, "oil_missing").all()


def test_features_shapes_and_values():
    builder = FeatureBuilder()
    well = make_well()
    sequence, tabular = builder.sequence(well), builder.tabular(well)
    assert sequence.shape == (len(well), builder.sequence_size)
    assert tabular.shape == (len(well), len(builder.tabular_names))
    assert not np.isnan(tabular).any() and np.abs(tabular).max() <= 3.0
    channel = builder.SEQUENCE_CHANNELS.index
    assert sequence[40, channel("f_diff")] > 0  # скачок частоты виден
    assert sequence[45, channel("liq_vs_step")] > 0.05  # Qж выше уровня до скачка


def test_parsing_helpers():
    assert normalize_id(" 561E ") == "561Е" and normalize_id(7782.0) == "7782"
    assert parse_date("21.04.2026г") == date(2026, 4, 21)
    assert parse_date("08.02.206") == date(2026, 2, 8)
    assert parse_date("нет") is None
