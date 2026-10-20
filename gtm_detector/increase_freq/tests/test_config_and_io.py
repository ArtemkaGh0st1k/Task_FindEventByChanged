from datetime import date

import pytest

from gtm_detector.increase_freq import DetectorConfig, EndRule, StartMode, Well
from gtm_detector.increase_freq.excel import ResultRecord, normalize_id, parse_date


def test_config_json_roundtrip(tmp_path):
    config = DetectorConfig(start_mode=StartMode.RISE, end_rules=(EndRule.OIL,), min_step=0.1)
    path = tmp_path / "params.json"
    config.save(path)
    assert DetectorConfig.load(path) == config


def test_config_accepts_strings_and_old_names():
    config = DetectorConfig.from_dict({"start_mode": "rise", "end_rules": ["liq"], "liq_tol": 0.02})
    assert config.start_mode is StartMode.RISE
    assert config.end_rules == (EndRule.LIQUID,)
    assert config.liquid_tolerance == 0.02
    with pytest.raises(ValueError):
        DetectorConfig.from_dict({"no_such_param": 1})


def test_well_from_dict_fills_missing_days_and_aliases():
    well = Well.from_dict(
        {
            "well_id": "W1",
            "date": [date(2026, 1, 1), date(2026, 1, 3)],
            "frequency": [50, 51],
            "pressue_zab": [40, 41],  # опечатка из исходного формата
            "state": [True, True],
        }
    )
    assert len(well.data) == 3
    assert not well.data["state"].iloc[1]  # пропущенный день — не работала
    assert "pressure_zab" in well.data


def test_parsing_helpers():
    assert normalize_id(" 561E ") == "561Е"
    assert normalize_id(7782.0) == "7782"
    assert parse_date("21.04.2026г").date() == date(2026, 4, 21)
    assert parse_date("08.02.206").date() == date(2026, 2, 8)
    assert parse_date("нет даты") is None


def test_result_record_label_filter():
    good = ResultRecord("1", "2", "Увеличение F", "да", None, date(2026, 1, 2), date(2026, 1, 9), None)
    assert good.is_label
    assert not ResultRecord("1", "2", "Увеличение F", "нет", None, good.start, good.end, None).is_label
    assert not ResultRecord("1", "2", "Увеличение F", "да", None, good.start, None, None).is_label
    assert not ResultRecord(
        "1", "2", "Изменение режима работы ЦЭС", "да", None, good.start, good.end, None
    ).is_label
