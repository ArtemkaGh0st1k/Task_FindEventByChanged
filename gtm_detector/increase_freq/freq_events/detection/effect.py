"""Шаги 2–4: базовый режим, начало и окончание эффекта."""

from __future__ import annotations

import math

import pandas as pd

from freq_events.config import DetectorConfig, EndRule, StartMode
from freq_events.models import Baseline, EndReason, FrequencyStep


def value_at(data: pd.DataFrame, column: str, position: int) -> float | None:
    """Значение колонки в позиции или None, если колонки/значения нет."""
    if column not in data:
        return None
    value = data[column].iat[position]
    return None if pd.isna(value) else float(value)


def baseline_before(data: pd.DataFrame, step: FrequencyStep) -> Baseline:
    """Шаг 2: режим в последний рабочий день перед скачком частоты."""
    position = step.previous_index
    return Baseline(
        liquid=value_at(data, "debit_liquid", position),
        oil=value_at(data, "debit_oil", position),
        watercut=value_at(data, "watercut", position),
    )


class EffectStartFinder:
    """Шаг 3: день начала эффекта (≈ «дата начала мониторинга»).

    * ``StartMode.RISE`` — первый день после скачка, когда Qж или Qн выше базовых;
    * ``StartMode.CHANGE`` — первый день, когда пришёл новый замер Qж / обводнённости.

    Если за ``start_max_days`` признака нет — день скачка + ``start_default``.
    """

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config

    def find(self, data: pd.DataFrame, step: FrequencyStep, baseline: Baseline) -> int:
        cfg = self.config
        first = max(step.index if cfg.start_same_day else step.index + 1, 1)
        last = min(step.index + cfg.start_max_days, len(data) - 1)

        if cfg.start_mode is StartMode.RISE:
            found = self._first_rise(data, first, last, baseline)
        else:
            found = self._first_change(data, first, last)
        if found is not None:
            return found
        return min(step.index + cfg.start_default, len(data) - 1)

    @staticmethod
    def _first_rise(data: pd.DataFrame, first: int, last: int, baseline: Baseline) -> int | None:
        checks = [("debit_liquid", baseline.liquid), ("debit_oil", baseline.oil)]
        for position in range(first, last + 1):
            for column, base in checks:
                value = value_at(data, column, position)
                if base is not None and value is not None and value > base:
                    return position
        return None

    @staticmethod
    def _first_change(data: pd.DataFrame, first: int, last: int) -> int | None:
        columns = [column for column in ("debit_liquid", "watercut") if column in data]
        for position in range(first, last + 1):
            for column in columns:
                previous, current = data[column].iat[position - 1], data[column].iat[position]
                if pd.notna(previous) and pd.notna(current) and previous != current:
                    return position
        return None

    def has_response(self, data: pd.DataFrame, start: int, baseline: Baseline) -> bool:
        """Был ли рост Qж или Qн выше базовых в первые ``response_days`` дней эффекта."""
        window = data.iloc[start : start + self.config.response_days]
        for column, base in (("debit_liquid", baseline.liquid), ("debit_oil", baseline.oil)):
            if column in window and base is not None and (window[column] > base).any():
                return True
        return False


class EffectEndFinder:
    """Шаг 4: окончание эффекта (≈ «дата окончания мониторинга»).

    Окончание — день перед первым из событий:

    * остановка скважины на ``stop_days`` дней подряд (``EndReason.STOP``);
    * потеря эффекта по правилам ``end_rules`` (сравнение с базовым режимом),
      если она держится ``end_confirm_days`` рабочих дней подряд.

    Первые ``min_effect_days`` дней после начала потеря эффекта не проверяется.
    Если ничего не произошло до конца данных — эффект продолжается (``None``).
    """

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config

    def find(self, data: pd.DataFrame, start: int, baseline: Baseline) -> tuple[int | None, EndReason | None]:
        cfg = self.config
        state = data["state"].to_numpy()
        series = {
            column: data[column].to_numpy() if column in data else None
            for column in ("debit_liquid", "watercut", "debit_oil")
        }

        stopped_days = 0
        lost_days, lost_since, lost_reason = 0, None, None
        for position in range(start + 1, len(data)):
            if not state[position]:
                stopped_days += 1
                if stopped_days >= cfg.stop_days:
                    return position - stopped_days, EndReason.STOP
                continue
            stopped_days = 0
            if position - start < cfg.min_effect_days:
                continue

            reason = self._lost_effect(series, position, baseline)
            if reason is None:
                lost_days, lost_since = 0, None
                continue
            if lost_days == 0:
                lost_since, lost_reason = position, reason
            lost_days += 1
            if lost_days >= cfg.end_confirm_days:
                return lost_since - 1, lost_reason
        return None, None

    def _lost_effect(self, series: dict, position: int, baseline: Baseline) -> EndReason | None:
        """Какое правило потери эффекта сработало в этот день (порядок: Qж, обв., Qн)."""
        cfg = self.config
        liquid = self._value(series["debit_liquid"], position)
        watercut = self._value(series["watercut"], position)
        oil = self._value(series["debit_oil"], position)

        if EndRule.LIQUID in cfg.end_rules and None not in (liquid, baseline.liquid):
            if liquid <= baseline.liquid * (1 - cfg.liquid_tolerance):
                return EndReason.LIQUID
        if EndRule.WATERCUT in cfg.end_rules and None not in (watercut, baseline.watercut):
            if watercut > baseline.watercut + cfg.watercut_tolerance:
                return EndReason.WATERCUT
        if EndRule.OIL in cfg.end_rules and None not in (oil, baseline.oil):
            if oil <= baseline.oil * (1 - cfg.oil_tolerance):
                return EndReason.OIL
        return None

    @staticmethod
    def _value(values, position: int) -> float | None:
        if values is None or math.isnan(values[position]):
            return None
        return float(values[position])
