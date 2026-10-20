"""Детектор мероприятия «увеличение частоты F» по суточным данным скважины.

Шаг 1. Скачок частоты (≈ «Дата выполнения»):
  берём рабочие дни (state=True, частота > min_working_freq), сглаживаем частоту
  скользящей медианой и ищем дни роста >= min_step Гц. Подъёмы подряд (разрыв
  <= max_gap рабочих дней, без снижения между ними) — одно мероприятие.
  Суммарный прирост >= min_total, новый уровень держится hold_days дней.

Шаг 2. Базовый режим «до»: Qж, Qн, обводнённость в последний рабочий день перед скачком.

Шаг 3. Начало эффекта (≈ «дата начала мониторинга»):
  первый день после скачка, когда пришёл новый замер Qж/обводнённости
  (start_mode="change") или когда Qж/Qн стал выше базового (start_mode="rise").
  Если за start_max_days такого дня нет — скачок + start_default.
  Скачки без роста Qж/Qн в первые response_days дней отбрасываются (require_response).

Шаг 4. Окончание эффекта (≈ «дата окончания мониторинга») — день перед первым из:
  * "liq": Qж <= базового Qж * (1 - liq_tol)          («по жидкости»);
  * "wc":  обводнённость > базовой + wc_tol             («по обводнённости»);
  * "oil": Qн <= базового Qн * (1 - oil_tol);
  * остановка скважины на >= stop_days дней подряд      («остановлена», ИДН, отказ).
  Условие должно держаться end_confirm_days рабочих дней подряд (разовые колебания
  замеров не считаются), первые min_effect_days дней после начала не проверяются.
  Если ни одно условие не сработало до конца данных — end=None (эффект продолжается).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .schema import Event, well_to_frame


@dataclass(frozen=True)
class DetectorParams:
    # --- скачок частоты
    min_working_freq: float = 10.0  # Гц: ниже — насос не работает
    smooth_window: int = 3          # окно медианы по рабочим дням (1 = без сглаживания)
    min_step: float = 0.3           # Гц: минимальный суточный прирост, считающийся подъёмом
    min_total: float = 0.3          # Гц: минимальный суммарный прирост за мероприятие
    max_gap: int = 3                # рабочих дней между ступенями одного мероприятия
    pre_window: int = 7             # рабочих дней для оценки частоты «до»
    hold_days: int = 3              # рабочих дней, которые новый уровень должен держаться
    hold_tolerance: float = 0.5     # доля прироста, которую можно «потерять» в окне удержания
    skip_after_restart: bool = False  # отбрасывать скачки, совпавшие с пуском после простоя
    # --- начало эффекта
    start_mode: str = "change"      # "change" — первый новый замер; "rise" — первый рост Qж/Qн выше базы
    start_same_day: bool = False    # засчитывать замер в сам день скачка
    start_max_days: int = 14        # сколько дней ждать новый замер
    start_default: int = 2          # сдвиг начала, если замера не дождались
    require_response: bool = True   # отбрасывать скачки без роста Qж/Qн
    response_days: int = 7          # окно (дней от начала эффекта) для проверки отклика
    # --- окончание эффекта
    end_rules: tuple[str, ...] = ("liq", "wc")
    liq_tol: float = 0.0
    wc_tol: float = 0.0             # п.п. обводнённости
    oil_tol: float = 0.0
    stop_days: int = 2
    min_effect_days: int = 0        # первые N дней после начала окончание не проверяем
    end_confirm_days: int = 1       # условие потери эффекта должно держаться N рабочих дней подряд

    def with_(self, **kw) -> "DetectorParams":
        return replace(self, **kw)


def _median(x: pd.Series) -> float | None:
    x = x.dropna()
    return float(x.median()) if len(x) else None


def _num(x) -> float | None:
    return None if x is None or pd.isna(x) else float(x)


def find_steps(df: pd.DataFrame, p: DetectorParams) -> list[dict]:
    """Скачки частоты вверх: [{"i": позиция дня скачка в df, ...}]."""
    working = df["state"].to_numpy() & (df["frequency"].to_numpy() > p.min_working_freq)
    pos = np.flatnonzero(working)
    if len(pos) < 2:
        return []
    f = df["frequency"].iloc[pos].reset_index(drop=True)
    fs = f.rolling(max(1, p.smooth_window), center=True, min_periods=1).median()
    d = fs.diff().to_numpy()
    up, down = d >= p.min_step, d <= -p.min_step
    after_stop = np.r_[False, np.diff(pos) > 1]

    groups: list[list[int]] = []
    for i in np.flatnonzero(up):
        if groups:
            prev = groups[-1][-1]
            if i - prev - 1 <= p.max_gap and not down[prev + 1 : i].any():
                groups[-1].append(i)
                continue
        groups.append([i])

    steps = []
    for g in groups:
        s, e = g[0], g[-1]
        f_before = _median(fs.iloc[max(0, s - p.pre_window) : s])
        hold = fs.iloc[e : e + p.hold_days]
        if f_before is None or len(hold) < p.hold_days:
            continue
        f_after = _median(hold)
        delta = f_after - f_before
        if delta < p.min_total or hold.min() < f_after - p.hold_tolerance * delta:
            continue
        restart = bool(after_stop[s : e + 1].any())
        if p.skip_after_restart and restart:
            continue
        steps.append({"i": int(pos[s]), "i_prev": int(pos[s - 1]), "f_before": f_before,
                      "f_after": f_after, "delta": delta, "n_steps": len(g), "after_restart": restart})
    return steps


def _effect_start(df: pd.DataFrame, i: int, base: dict, p: DetectorParams) -> int:
    first = max(i if p.start_same_day else i + 1, 1)
    last = min(i + p.start_max_days, len(df) - 1)
    if p.start_mode == "rise":  # первый день, когда Qж или Qн выше базового
        for t in range(first, last + 1):
            for c, b in (("debit_liquid", base["liq"]), ("debit_oil", base["oil"])):
                if c in df and b is not None and pd.notna(df[c].iat[t]) and df[c].iat[t] > b:
                    return t
    else:  # "change": первый новый замер Qж / обводнённости
        cols = [c for c in ("debit_liquid", "watercut") if c in df]
        for t in range(first, last + 1):
            for c in cols:
                a, b = df[c].iat[t - 1], df[c].iat[t]
                if pd.notna(a) and pd.notna(b) and a != b:
                    return t
    return min(i + p.start_default, len(df) - 1)


def _effect_end(df: pd.DataFrame, start: int, base: dict, p: DetectorParams) -> tuple[int | None, str | None]:
    state = df["state"].to_numpy()
    liq = df["debit_liquid"].to_numpy() if "debit_liquid" in df else None
    wc = df["watercut"].to_numpy() if "watercut" in df else None
    oil = df["debit_oil"].to_numpy() if "debit_oil" in df else None
    stopped = 0
    lost, lost_since, lost_reason = 0, None, None  # подряд идущие рабочие дни без эффекта
    for t in range(start + 1, len(df)):
        if not state[t]:
            stopped += 1
            if stopped >= p.stop_days:
                return t - stopped, "stop"
            continue
        stopped = 0
        if t - start < p.min_effect_days:
            continue
        reason = None
        if "liq" in p.end_rules and liq is not None and base["liq"] is not None and not np.isnan(liq[t]):
            if liq[t] <= base["liq"] * (1 - p.liq_tol):
                reason = "liq"
        if reason is None and "wc" in p.end_rules and wc is not None and base["wc"] is not None \
                and not np.isnan(wc[t]) and wc[t] > base["wc"] + p.wc_tol:
            reason = "wc"
        if reason is None and "oil" in p.end_rules and oil is not None and base["oil"] is not None \
                and not np.isnan(oil[t]) and oil[t] <= base["oil"] * (1 - p.oil_tol):
            reason = "oil"
        if reason is None:
            lost, lost_since = 0, None
            continue
        if lost == 0:
            lost_since, lost_reason = t, reason
        lost += 1
        if lost >= p.end_confirm_days:
            return lost_since - 1, lost_reason
    return None, None


def detect_events(well: dict, params: DetectorParams | None = None,
                  frame: pd.DataFrame | None = None) -> list[Event]:
    p = params or DetectorParams()
    df = frame if frame is not None else well_to_frame(well)
    well_id = str(well.get("well_id"))
    idx = df.index

    def col(c, t):
        return _num(df[c].iat[t]) if c in df else None

    events = []
    for st in find_steps(df, p):
        i, ip = st["i"], st["i_prev"]
        base = {"liq": col("debit_liquid", ip), "wc": col("watercut", ip), "oil": col("debit_oil", ip)}
        s = _effect_start(df, i, base, p)
        liq_s, oil_s = col("debit_liquid", s), col("debit_oil", s)
        # Отклик: в первые response_days дней эффекта Qж или Qн выше базовых.
        win = df.iloc[s : s + p.response_days]
        responded = bool(
            ("debit_liquid" in df and base["liq"] is not None and (win["debit_liquid"] > base["liq"]).any())
            or ("debit_oil" in df and base["oil"] is not None and (win["debit_oil"] > base["oil"]).any())
        )
        if p.require_response and not responded:
            continue
        e, reason = _effect_end(df, s, base, p)
        if e is not None and e < s:
            e = s

        conf = 0.5 * min(1.0, st["delta"] / max(p.min_total, 1e-9) / 3) + (0.5 if responded else 0.0)
        if st["after_restart"]:
            conf *= 0.7
        events.append(Event(
            well_id=well_id,
            date_step=idx[i].date(),
            start=idx[s].date(),
            end=None if e is None else idx[e].date(),
            end_reason=reason,
            f_before=round(st["f_before"], 3),
            f_after=round(st["f_after"], 3),
            delta_f=round(st["delta"], 3),
            n_steps=st["n_steps"],
            base_liquid=base["liq"],
            base_oil=base["oil"],
            base_watercut=base["wc"],
            liquid_at_start=liq_s,
            oil_at_start=oil_s,
            responded=responded,
            after_restart=st["after_restart"],
            confidence=round(conf, 3),
        ))
    return events


def detect_many(wells: list[dict], params: DetectorParams | None = None) -> dict[str, list[Event]]:
    return {str(w["well_id"]): detect_events(w, params) for w in wells}
