from datetime import date

import numpy as np

from gtm_detector.ml_freq.data import Interval, normalize_id, parse_date
from gtm_detector.ml_freq.features import FeatureBuilder
from gtm_detector.ml_freq.tests.conftest import day, make_well


def test_target_and_training_mask():
    well = make_well(step=40, effect=30)
    target = well.target()
    assert target.sum() == 30 and target[42] == 1 and target[41] == 0
    mask = well.training_mask(context_days=10)
    assert mask[32] and not mask[31]  # 10 дней до начала (42)
    assert mask[81] and not mask[82]  # 10 дней после окончания (71)
    assert well.training_mask(None).all()


def test_with_jump_starts_moves_only_start():
    well = make_well(step=40, effect=30)
    end = well.labels[0].end
    well.labels = [
        Interval(day(42), end, jump=day(35)),  # скачок раньше даты технолога
        Interval(day(80), day(90), jump=day(95)),  # скачок позже окончания: начало не позже конца
        Interval(day(100), day(110)),  # скачок не определён: без изменений
    ]
    moved = well.with_jump_starts()
    assert [(label.start, label.end) for label in moved.labels] == [
        (day(35), end),
        (day(90), day(90)),
        (day(100), day(110)),
    ]
    assert well.labels[0].start == day(42)  # исходная скважина не меняется
    assert moved.target()[35] == 1 and well.target()[35] == 0  # метка действительно сдвинута


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
