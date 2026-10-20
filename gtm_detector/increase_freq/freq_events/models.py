"""Модели данных: скважина, разметка, скачок частоты, мероприятие."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any

import pandas as pd

# Канонические имена колонок -> допустимые ключи во входном словаре
# (в исходной постановке есть опечатки: "pressue_zab", "pressure_lineqr").
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "debit_oil": ("debit_oil",),
    "frequency": ("frequency",),
    "debit_liquid": ("debit_liquid",),
    "watercut": ("watercut",),
    "pressure_zab": ("pressure_zab", "pressue_zab"),
    "pressure_line": ("pressure_line", "pressure_lineqr", "pressure_linear"),
    "state": ("state",),
}
REQUIRED_COLUMNS = ("frequency", "state")


def to_date(value: Any) -> date | None:
    """Привести datetime / Timestamp / строку к ``date``; пустые значения -> None."""
    if value is None or (not isinstance(value, (str, date)) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


class EndReason(str, Enum):
    """Почему закончился эффект мероприятия."""

    OIL = "oil"
    LIQUID = "liq"
    WATERCUT = "wc"
    STOP = "stop"

    @property
    def title(self) -> str:
        return _END_REASON_TITLES[self]


_END_REASON_TITLES = {
    EndReason.OIL: "по нефти",
    EndReason.LIQUID: "по жидкости",
    EndReason.WATERCUT: "по обводнённости",
    EndReason.STOP: "остановка",
}
ONGOING_TITLE = "эффект продолжается"


@dataclass(frozen=True)
class Label:
    """Размеченное мероприятие (из файла результатов)."""

    start: date
    end: date | None
    date_done: date | None = None
    end_reason: str | None = None


@dataclass
class Well:
    """Скважина: суточные замеры + (необязательно) разметка.

    ``data`` — DataFrame с непрерывным суточным индексом и колонками
    ``frequency``, ``state`` и, если есть, ``debit_liquid``, ``debit_oil``,
    ``watercut``, ``pressure_zab``, ``pressure_line``.
    """

    well_id: str
    data: pd.DataFrame
    number: str = ""
    cluster: str = ""
    labels: list[Label] = field(default_factory=list)
    # Даты прочих известных мероприятий с частотой (не попавших в разметку):
    # срабатывание детектора рядом с ними не считается ложным.
    known_event_dates: list[date] = field(default_factory=list)

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> Well:
        """Создать из словаря входного формата::

        {"well_id": str, "date": [...], "frequency": [...], "state": [...],
         "debit_liquid": [...], "debit_oil": [...], ...,
         "events": [{"start": ..., "end": ..., "date_done": ...}]}  # необязательно
        """
        well_id = str(record.get("well_id"))
        dates = pd.to_datetime(record["date"]).normalize()
        columns: dict[str, list] = {}
        for column, aliases in COLUMN_ALIASES.items():
            key = next((alias for alias in aliases if alias in record), None)
            if key is None:
                if column in REQUIRED_COLUMNS:
                    raise KeyError(f"Скважина {well_id!r}: нет обязательного поля {column!r}")
                continue
            values = list(record[key])
            if len(values) != len(dates):
                raise ValueError(
                    f"Скважина {well_id!r}: длина {key!r} ({len(values)}) != длина date ({len(dates)})"
                )
            columns[column] = values

        labels = [
            Label(
                start=to_date(event["start"]),
                end=to_date(event.get("end")),
                date_done=to_date(event.get("date_done")),
                end_reason=event.get("end_reason"),
            )
            for event in record.get("events") or []
        ]
        return cls(
            well_id=well_id,
            data=cls._build_frame(columns, dates),
            number=str(record.get("well", "")),
            cluster=str(record.get("kust", "")),
            labels=labels,
            known_event_dates=[to_date(d) for d in record.get("neutral_dates", [])],
        )

    @staticmethod
    def _build_frame(columns: dict[str, list], dates: pd.DatetimeIndex) -> pd.DataFrame:
        frame = pd.DataFrame(columns, index=dates)
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        for column in frame.columns:
            if column != "state":
                frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
        frame["state"] = frame["state"].fillna(False).astype(bool)

        # Пропущенные календарные дни — строки с NaN и state=False.
        full_range = pd.date_range(frame.index.min(), frame.index.max(), freq="D")
        frame = frame.reindex(full_range)
        frame["state"] = frame["state"].fillna(False).astype(bool)
        return frame


@dataclass(frozen=True)
class FrequencyStep:
    """Скачок частоты вверх (одна или несколько ступеней подряд)."""

    index: int  # позиция дня скачка в Well.data
    previous_index: int  # позиция последнего рабочего дня перед скачком
    f_before: float
    f_after: float
    n_steps: int
    after_restart: bool  # между ступенями или перед скачком был простой

    @property
    def delta(self) -> float:
        return self.f_after - self.f_before


@dataclass(frozen=True)
class Baseline:
    """Режим скважины «до» мероприятия."""

    liquid: float | None
    oil: float | None
    watercut: float | None


@dataclass
class Event:
    """Найденное мероприятие «увеличение частоты F».

    ``date_step`` — день изменения частоты (≈ «Дата выполнения мероприятия»);
    ``start`` / ``end`` — начало и окончание эффекта (≈ даты мониторинга);
    ``end=None`` — эффект продолжается до конца данных.
    """

    well_id: str
    date_step: date
    start: date
    end: date | None
    end_reason: EndReason | None = None
    f_before: float | None = None
    f_after: float | None = None
    delta_f: float | None = None
    n_steps: int | None = None
    base_liquid: float | None = None
    base_oil: float | None = None
    base_watercut: float | None = None
    liquid_at_start: float | None = None
    oil_at_start: float | None = None
    responded: bool | None = None
    after_restart: bool = False
    confidence: float | None = None

    @property
    def end_reason_title(self) -> str:
        return self.end_reason.title if self.end_reason else ONGOING_TITLE

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("date_step", "start", "end"):
            data[key] = data[key].isoformat() if data[key] else None
        data["end_reason"] = self.end_reason.value if self.end_reason else None
        return data
