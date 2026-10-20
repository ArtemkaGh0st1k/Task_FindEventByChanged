"""Сравнение найденных мероприятий с разметкой."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from .schema import Event


def _mae(x):
    return float(np.mean(np.abs(x))) if len(x) else float("nan")


def _share(x, k):
    return float(np.mean(np.abs(x) <= k)) if len(x) else float("nan")


@dataclass
class Score:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    neutral: int = 0          # найдено там, где есть известное мероприятие вне выборки
    start_err: list[int] = field(default_factory=list)  # pred - true, дней
    end_err: list[int] = field(default_factory=list)
    end_open: int = 0         # у разметки конец есть, а детектор считает, что эффект продолжается
    step_err: list[int] = field(default_factory=list)   # date_step - «Дата выполнения»

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def mae_start(self) -> float:
        return _mae(self.start_err)

    @property
    def mae_end(self) -> float:
        return _mae(self.end_err)

    def __add__(self, o: "Score") -> "Score":
        return Score(self.tp + o.tp, self.fp + o.fp, self.fn + o.fn, self.neutral + o.neutral,
                     self.start_err + o.start_err, self.end_err + o.end_err,
                     self.end_open + o.end_open, self.step_err + o.step_err)

    def summary(self) -> str:
        return (
            f"TP={self.tp} FN={self.fn} FP={self.fp} (+{self.neutral} нейтр.) | "
            f"recall={self.recall:.3f} precision={self.precision:.3f} F1={self.f1:.3f}\n"
            f"    начало: MAE={self.mae_start:.2f} дн, точно={_share(self.start_err, 0):.0%}, "
            f"±1={_share(self.start_err, 1):.0%}, ±3={_share(self.start_err, 3):.0%}\n"
            f"    конец:  MAE={self.mae_end:.2f} дн, точно={_share(self.end_err, 0):.0%}, "
            f"±3={_share(self.end_err, 3):.0%}, ±7={_share(self.end_err, 7):.0%}; "
            f"не закрыто детектором: {self.end_open}\n"
            f"    скачок частоты vs «Дата выполнения»: MAE={_mae(self.step_err):.2f} дн, "
            f"точно={_share(self.step_err, 0):.0%}, ±2={_share(self.step_err, 2):.0%}"
        )


def match(pred: list[Event], true: list[dict], tol: int = 7, step_tol: int = 3,
          neutral_dates: list[date] | None = None) -> Score:
    """Жадное 1-к-1 сопоставление: пара засчитывается, если даты начала отличаются
    не более чем на tol дней ИЛИ скачок частоты не дальше step_tol дней от
    «Даты выполнения» (начало мониторинга иногда ставят через недели).

    Несопоставленное предсказание, у которого скачок частоты в пределах tol
    от даты известного мероприятия вне выборки (neutral_dates), не считается FP.
    """
    cand = []
    for i, e in enumerate(pred):
        for j, t in enumerate(true):
            d = abs((e.start - t["start"]).days)
            ds = abs((e.date_step - t["date_done"]).days) if t.get("date_done") else 10**6
            if d <= tol or ds <= step_tol:
                cand.append((min(d, ds), i, j))
    cand.sort()
    used_p, used_t = set(), set()
    s = Score()
    for _, i, j in cand:
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        e, t = pred[i], true[j]
        s.tp += 1
        s.start_err.append((e.start - t["start"]).days)
        if t["end"] is not None:
            if e.end is None:
                s.end_open += 1
            else:
                s.end_err.append((e.end - t["end"]).days)
        if t.get("date_done") is not None:
            s.step_err.append((e.date_step - t["date_done"]).days)
    for i, e in enumerate(pred):
        if i in used_p:
            continue
        if neutral_dates and any(abs((e.date_step - d).days) <= tol for d in neutral_dates):
            s.neutral += 1
        else:
            s.fp += 1
    s.fn = len(true) - len(used_t)
    return s
