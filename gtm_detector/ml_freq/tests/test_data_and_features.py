from datetime import date

import numpy as np

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
