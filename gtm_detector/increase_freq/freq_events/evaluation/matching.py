"""Сопоставление найденных мероприятий с разметкой и метрики качества."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from freq_events.models import Event, Label


def _mean_abs(values: Sequence[int]) -> float:
    return float(np.mean(np.abs(values))) if len(values) else float("nan")


def _share_within(values: Sequence[int], days: int) -> float:
    return float(np.mean(np.abs(values) <= days)) if len(values) else float("nan")


@dataclass
class Score:
    """Метрики качества на наборе скважин. Складываются оператором ``+``."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    neutral: int = 0  # срабатывания рядом с известными мероприятиями вне разметки
    start_errors: list[int] = field(default_factory=list)  # найдено - разметка, дней
    end_errors: list[int] = field(default_factory=list)
    open_ends: int = 0  # в разметке конец есть, а детектор считает, что эффект продолжается
    step_errors: list[int] = field(default_factory=list)  # скачок F - «Дата выполнения»

    @property
    def precision(self) -> float:
        found = self.true_positives + self.false_positives
        return self.true_positives / found if found else 0.0

    @property
    def recall(self) -> float:
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total else 0.0

    def fbeta(self, beta: float = 1.0) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return (1 + beta**2) * p * r / (beta**2 * p + r)

    @property
    def f1(self) -> float:
        return self.fbeta(1.0)

    @property
    def start_mae(self) -> float:
        return _mean_abs(self.start_errors)

    @property
    def end_mae(self) -> float:
        return _mean_abs(self.end_errors)

    def __add__(self, other: Score) -> Score:
        return Score(
            true_positives=self.true_positives + other.true_positives,
            false_positives=self.false_positives + other.false_positives,
            false_negatives=self.false_negatives + other.false_negatives,
            neutral=self.neutral + other.neutral,
            start_errors=self.start_errors + other.start_errors,
            end_errors=self.end_errors + other.end_errors,
            open_ends=self.open_ends + other.open_ends,
            step_errors=self.step_errors + other.step_errors,
        )

    def summary(self) -> str:
        start, end, step = self.start_errors, self.end_errors, self.step_errors
        return "\n".join(
            [
                f"TP={self.true_positives} FN={self.false_negatives} FP={self.false_positives} "
                f"(+{self.neutral} нейтр.) | recall={self.recall:.3f} "
                f"precision={self.precision:.3f} F1={self.f1:.3f}",
                f"    начало: MAE={self.start_mae:.2f} дн, точно={_share_within(start, 0):.0%}, "
                f"±1={_share_within(start, 1):.0%}, ±3={_share_within(start, 3):.0%}",
                f"    конец:  MAE={self.end_mae:.2f} дн, точно={_share_within(end, 0):.0%}, "
                f"±3={_share_within(end, 3):.0%}, ±7={_share_within(end, 7):.0%}; "
                f"не закрыто детектором: {self.open_ends}",
                f"    скачок частоты vs «Дата выполнения»: MAE={_mean_abs(step):.2f} дн, "
                f"точно={_share_within(step, 0):.0%}, ±2={_share_within(step, 2):.0%}",
            ]
        )


@dataclass
class MatchResult:
    """Результат сопоставления по одной скважине."""

    pairs: list[tuple[Label, Event | None]]  # каждая метка и найденное ей событие (или None)
    unmatched_events: list[Event]


class EventMatcher:
    """Сопоставляет найденные мероприятия с разметкой один-к-одному.

    Пара засчитывается, если даты начала отличаются не более чем на
    ``start_tolerance`` дней ИЛИ скачок частоты не дальше ``step_tolerance``
    дней от «Даты выполнения» (начало мониторинга иногда ставят через недели).
    Сначала сопоставляются самые близкие пары.
    """

    def __init__(self, start_tolerance: int = 7, step_tolerance: int = 3) -> None:
        self.start_tolerance = start_tolerance
        self.step_tolerance = step_tolerance

    def match(self, events: Sequence[Event], labels: Sequence[Label]) -> MatchResult:
        candidates = []
        for i, event in enumerate(events):
            for j, label in enumerate(labels):
                distance = self._distance(event, label)
                if distance is not None:
                    candidates.append((distance, i, j))
        candidates.sort()

        event_for_label: dict[int, int] = {}
        used_events: set[int] = set()
        for _, i, j in candidates:
            if i in used_events or j in event_for_label:
                continue
            used_events.add(i)
            event_for_label[j] = i

        pairs = [
            (label, events[event_for_label[j]] if j in event_for_label else None)
            for j, label in enumerate(labels)
        ]
        unmatched = [event for i, event in enumerate(events) if i not in used_events]
        return MatchResult(pairs=pairs, unmatched_events=unmatched)

    def _distance(self, event: Event, label: Label) -> int | None:
        start_gap = abs((event.start - label.start).days)
        step_gap = abs((event.date_step - label.date_done).days) if label.date_done else None
        if start_gap <= self.start_tolerance or (step_gap is not None and step_gap <= self.step_tolerance):
            return min(start_gap, step_gap) if step_gap is not None else start_gap
        return None

    def score(
        self,
        events: Sequence[Event],
        labels: Sequence[Label],
        known_event_dates: Sequence[date] = (),
    ) -> Score:
        """Метрики по одной скважине.

        Несопоставленное событие, чей скачок частоты не дальше ``start_tolerance``
        дней от известного мероприятия вне разметки, считается нейтральным, а не FP.
        """
        result = self.match(events, labels)
        score = Score()
        for label, event in result.pairs:
            if event is None:
                score.false_negatives += 1
                continue
            score.true_positives += 1
            score.start_errors.append((event.start - label.start).days)
            if label.end is not None:
                if event.end is None:
                    score.open_ends += 1
                else:
                    score.end_errors.append((event.end - label.end).days)
            if label.date_done is not None:
                score.step_errors.append((event.date_step - label.date_done).days)

        for event in result.unmatched_events:
            near_known = any(
                abs((event.date_step - known).days) <= self.start_tolerance for known in known_event_dates
            )
            if near_known:
                score.neutral += 1
            else:
                score.false_positives += 1
        return score
