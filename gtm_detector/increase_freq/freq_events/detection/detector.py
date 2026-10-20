"""Детектор мероприятий «увеличение частоты F» — объединяет все шаги."""

from __future__ import annotations

from collections.abc import Iterable

from freq_events.config import DetectorConfig
from freq_events.detection.effect import (
    EffectEndFinder,
    EffectStartFinder,
    baseline_before,
    value_at,
)
from freq_events.detection.steps import StepDetector
from freq_events.models import Event, FrequencyStep, Well


class FrequencyIncreaseDetector:
    """Ищет мероприятия «увеличение частоты F» по суточным замерам скважины.

    Пример::

        detector = FrequencyIncreaseDetector(DetectorConfig.load("params.json"))
        events = detector.detect(well)
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self.step_detector = StepDetector(self.config)
        self.start_finder = EffectStartFinder(self.config)
        self.end_finder = EffectEndFinder(self.config)

    def detect(self, well: Well) -> list[Event]:
        events = []
        for step in self.step_detector.find(well.data):
            event = self._build_event(well, step)
            if event is not None:
                events.append(event)
        return events

    def detect_all(self, wells: Iterable[Well]) -> dict[str, list[Event]]:
        return {well.well_id: self.detect(well) for well in wells}

    def _build_event(self, well: Well, step: FrequencyStep) -> Event | None:
        data = well.data
        baseline = baseline_before(data, step)

        start = self.start_finder.find(data, step, baseline)
        responded = self.start_finder.has_response(data, start, baseline)
        if self.config.require_response and not responded:
            return None

        end, end_reason = self.end_finder.find(data, start, baseline)
        if end is not None:
            end = max(end, start)

        dates = data.index
        return Event(
            well_id=well.well_id,
            date_step=dates[step.index].date(),
            start=dates[start].date(),
            end=dates[end].date() if end is not None else None,
            end_reason=end_reason,
            f_before=round(step.f_before, 3),
            f_after=round(step.f_after, 3),
            delta_f=round(step.delta, 3),
            n_steps=step.n_steps,
            base_liquid=baseline.liquid,
            base_oil=baseline.oil,
            base_watercut=baseline.watercut,
            liquid_at_start=value_at(data, "debit_liquid", start),
            oil_at_start=value_at(data, "debit_oil", start),
            responded=responded,
            after_restart=step.after_restart,
            confidence=self._confidence(step, responded),
        )

    def _confidence(self, step: FrequencyStep, responded: bool) -> float:
        """Эвристическая уверенность 0..1: величина скачка + наличие отклика добычи."""
        size_score = min(1.0, step.delta / max(self.config.min_total, 1e-9) / 3)
        confidence = 0.5 * size_score + (0.5 if responded else 0.0)
        if step.after_restart:
            confidence *= 0.7
        return round(confidence, 3)
