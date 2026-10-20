"""Общие фикстуры: построение скважины из коротких рядов."""

from datetime import date, timedelta

import pytest

from gtm_detector.increase_freq import DetectorConfig, EndRule, StartMode, Well

START_DATE = date(2026, 1, 1)


def day(offset: int) -> date:
    return START_DATE + timedelta(days=offset)


def make_well(frequency, liquid=None, oil=None, state=None, watercut=None, **extra) -> Well:
    n = len(frequency)
    record = {
        "well_id": "T",
        "date": [day(i) for i in range(n)],
        "frequency": frequency,
        "debit_liquid": liquid or [100.0] * n,
        "debit_oil": oil or [10.0] * n,
        "watercut": watercut or [90.0] * n,
        "state": state or [True] * n,
        **extra,
    }
    return Well.from_dict(record)


@pytest.fixture
def config() -> DetectorConfig:
    """Параметры, близкие к подобранным по разметке."""
    return DetectorConfig(
        smooth_window=3,
        min_step=0.1,
        min_total=0.3,
        hold_days=5,
        start_mode=StartMode.RISE,
        end_rules=(EndRule.OIL,),
        end_confirm_days=3,
        stop_days=2,
    )
