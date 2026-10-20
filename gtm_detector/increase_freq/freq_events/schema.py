"""Входной формат данных по скважине и приведение его к DataFrame."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime

import pandas as pd

# Канонические имена колонок -> допустимые варианты ключей во входном словаре
# (во входном формате есть опечатки: "pressue_zab", "pressure_lineqr").
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "debit_oil": ("debit_oil",),
    "frequency": ("frequency",),
    "debit_liquid": ("debit_liquid",),
    "watercut": ("watercut",),
    "pressure_zab": ("pressure_zab", "pressue_zab"),
    "pressure_line": ("pressure_line", "pressure_lineqr", "pressure_linear"),
    "state": ("state",),
}
REQUIRED = ("frequency", "state")


@dataclass
class Event:
    """Мероприятие «увеличение частоты F» на скважине.

    date_step — день изменения частоты (≈ «Дата выполнения мероприятия»),
    start/end — начало и окончание эффекта (≈ даты начала/окончания мониторинга);
    end=None — эффект продолжается до конца данных.
    """

    well_id: str
    date_step: date
    start: date
    end: date | None
    end_reason: str | None = None
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

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("date_step", "start", "end"):
            d[k] = None if d[k] is None else d[k].isoformat()
        return d


def _to_date(x) -> date | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return pd.Timestamp(x).date()


def well_to_frame(well: dict) -> pd.DataFrame:
    """Словарь одной скважины -> DataFrame с ежедневным индексом.

    Пропущенные календарные дни добавляются строками с NaN и state=False.
    """
    dates = pd.to_datetime(well["date"]).normalize()
    data = {}
    for col, aliases in COLUMN_ALIASES.items():
        key = next((a for a in aliases if a in well), None)
        if key is None:
            if col in REQUIRED:
                raise KeyError(f"well {well.get('well_id')!r}: нет обязательного поля {col!r}")
            continue
        values = list(well[key])
        if len(values) != len(dates):
            raise ValueError(
                f"well {well.get('well_id')!r}: длина {key!r}={len(values)} != длина date={len(dates)}"
            )
        data[col] = values

    df = pd.DataFrame(data, index=dates)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for col in df.columns:
        if col != "state":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df["state"] = df["state"].fillna(False).astype(bool)

    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full)
    df["state"] = df["state"].fillna(False).astype(bool)
    return df


def labels_of(well: dict) -> list[dict]:
    """Размеченные мероприятия из well["events"] = [{"start", "end", ["date_done"]}, ...]."""
    out = []
    for e in well.get("events", []) or []:
        out.append({"start": _to_date(e["start"]), "end": _to_date(e.get("end")),
                    "date_done": _to_date(e.get("date_done"))})
    return out
