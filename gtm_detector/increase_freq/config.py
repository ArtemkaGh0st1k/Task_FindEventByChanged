"""Конфигурация детектора мероприятий «увеличение частоты F»."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from enum import Enum
from pathlib import Path
from typing import Any


class StartMode(str, Enum):
    """Способ определения дня начала эффекта."""

    CHANGE = "change"  # первый новый замер Qж / обводнённости после скачка
    RISE = "rise"  # первый день, когда Qж или Qн стали выше базовых


class EndRule(str, Enum):
    """Условия потери эффекта (окончания мероприятия)."""

    LIQUID = "liq"  # Qж опустился до базового уровня
    WATERCUT = "wc"  # обводнённость выросла выше базовой
    OIL = "oil"  # Qн опустился до базового уровня


@dataclass(frozen=True)
class DetectorConfig:
    """Параметры всех шагов алгоритма.

    Значения по умолчанию — разумная отправная точка; рабочие значения
    подбираются по разметке классом :class:`~gtm_detector.increase_freq.evaluation.Calibrator`
    и хранятся в JSON (см. :meth:`save` / :meth:`load`).
    """

    # --- шаг 1: скачок частоты
    min_working_freq: float = 10.0  # Гц: ниже — насос не работает
    smooth_window: int = 3  # окно скользящей медианы по рабочим дням (1 — без сглаживания)
    min_step: float = 0.3  # Гц: минимальный суточный прирост, считающийся подъёмом
    min_total: float = 0.3  # Гц: минимальный суммарный прирост за мероприятие
    max_gap: int = 3  # рабочих дней между ступенями одного мероприятия
    pre_window: int = 7  # рабочих дней для оценки частоты «до»
    hold_days: int = 3  # рабочих дней, которые новый уровень должен держаться
    hold_tolerance: float = 0.5  # доля прироста, которую можно «потерять» в окне удержания
    skip_after_restart: bool = False  # отбрасывать скачки, совпавшие с пуском после простоя

    # --- шаг 3: начало эффекта
    start_mode: StartMode = StartMode.CHANGE
    start_same_day: bool = False  # учитывать замер в сам день скачка
    start_max_days: int = 14  # сколько дней ждать признак начала
    start_default: int = 2  # сдвиг начала от скачка, если признака не дождались
    require_response: bool = True  # отбрасывать скачки без роста Qж/Qн
    response_days: int = 7  # окно (дней от начала эффекта) для проверки отклика

    # --- шаг 4: окончание эффекта
    end_rules: tuple[EndRule, ...] = (EndRule.LIQUID, EndRule.WATERCUT)
    liquid_tolerance: float = 0.0  # доля от базового Qж
    watercut_tolerance: float = 0.0  # процентные пункты обводнённости
    oil_tolerance: float = 0.0  # доля от базового Qн
    stop_days: int = 2  # дней простоя подряд, означающих окончание
    min_effect_days: int = 0  # первые N дней после начала окончание не проверяется
    end_confirm_days: int = 1  # условие потери эффекта должно держаться N рабочих дней подряд

    def __post_init__(self) -> None:
        # Допускаем строки на входе (из JSON или при ручном создании).
        object.__setattr__(self, "start_mode", StartMode(self.start_mode))
        object.__setattr__(self, "end_rules", tuple(EndRule(rule) for rule in self.end_rules))

    def replace(self, **changes: Any) -> DetectorConfig:
        """Копия конфигурации с изменёнными полями."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["start_mode"] = self.start_mode.value
        data["end_rules"] = [rule.value for rule in self.end_rules]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DetectorConfig:
        # Старые имена полей (первая версия params.json).
        renamed = {"liq_tol": "liquid_tolerance", "wc_tol": "watercut_tolerance", "oil_tol": "oil_tolerance"}
        data = {renamed.get(key, key): value for key, value in data.items()}
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Неизвестные параметры: {sorted(unknown)}")
        return cls(**data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> DetectorConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
