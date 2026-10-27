"""Сопоставление найденных мероприятий с разметкой и метрики."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..data import Interval


def _share(errors: Sequence[int], days: int) -> float:
    return float(np.mean(np.abs(errors) <= days)) if len(errors) else float("nan")


def _mae(errors: Sequence[int]) -> float:
    return float(np.mean(np.abs(errors))) if len(errors) else float("nan")


@dataclass
class Score:
    """Метрики уровня мероприятий; складываются оператором ``+``."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    start_errors: list[int] = field(default_factory=list)  # найдено − разметка, дней
    end_errors: list[int] = field(default_factory=list)
    open_ends: int = 0  # в разметке конец есть, модель считает эффект продолжающимся

    @property
    def recall(self) -> float:
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total else 0.0

    @property
    def precision(self) -> float:
        found = self.true_positives + self.false_positives
        return self.true_positives / found if found else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def __add__(self, other: Score) -> Score:
        return Score(
            self.true_positives + other.true_positives,
            self.false_positives + other.false_positives,
            self.false_negatives + other.false_negatives,
            self.start_errors + other.start_errors,
            self.end_errors + other.end_errors,
            self.open_ends + other.open_ends,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "найдено размеченных (recall)": self.recall,
            "точность (precision)": self.precision,
            "F1": self.f1,
            "начало: точно": _share(self.start_errors, 0),
            "начало: ±1 дн": _share(self.start_errors, 1),
            "начало: ±3 дн": _share(self.start_errors, 3),
            "начало: MAE, дн": _mae(self.start_errors),
            "конец: точно": _share(self.end_errors, 0),
            "конец: ±3 дн": _share(self.end_errors, 3),
            "конец: ±7 дн": _share(self.end_errors, 7),
            "конец: MAE, дн": _mae(self.end_errors),
            "TP": self.true_positives,
            "FN": self.false_negatives,
            "FP": self.false_positives,
            "не закрыт конец": self.open_ends,
        }


class EventMatcher:
    """Один-к-одному по дате начала (|Δ| ≤ ``start_tolerance``), сначала ближайшие пары."""

    def __init__(self, start_tolerance: int = 7) -> None:
        self.start_tolerance = start_tolerance

    def match(
        self, predicted: Sequence[Interval], labels: Sequence[Interval]
    ) -> list[tuple[Interval, Interval | None]]:
        candidates = sorted(
            (abs((p.start - t.start).days), i, j)
            for i, p in enumerate(predicted)
            for j, t in enumerate(labels)
            if abs((p.start - t.start).days) <= self.start_tolerance
        )
        used_predicted: set[int] = set()
        found: dict[int, int] = {}
        for _, i, j in candidates:
            if i not in used_predicted and j not in found:
                used_predicted.add(i)
                found[j] = i
        return [(label, predicted[found[j]] if j in found else None) for j, label in enumerate(labels)]

    def score(self, predicted: Sequence[Interval], labels: Sequence[Interval]) -> Score:
        pairs = self.match(predicted, labels)
        score = Score()
        for label, hit in pairs:
            if hit is None:
                score.false_negatives += 1
                continue
            score.true_positives += 1
            score.start_errors.append((hit.start - label.start).days)
            if label.end is not None and not label.censored:  # у censored конец эффекта неизвестен
                if hit.end is None:
                    score.open_ends += 1
                else:
                    score.end_errors.append((hit.end - label.end).days)
        score.false_positives = len(predicted) - score.true_positives
        return score
