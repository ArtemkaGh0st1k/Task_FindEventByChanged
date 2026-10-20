"""Подбор параметров детектора по размеченным скважинам."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import DetectorConfig, EndRule, StartMode
from ..detection import FrequencyIncreaseDetector
from ..evaluation.matching import EventMatcher, Score
from ..models import Well

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_SPACE: dict[str, list[Any]] = {
    "smooth_window": [1, 3],
    "min_step": [0.1, 0.3, 0.5, 1.0],
    "min_total": [0.3, 0.5, 1.0],
    "max_gap": [0, 3, 7, 14],
    "hold_days": [1, 3, 5],
    "require_response": [False, True],
    "skip_after_restart": [False, True],
    "start_mode": [StartMode.CHANGE, StartMode.RISE],
    "start_same_day": [False, True],
    "response_days": [1, 3, 7, 14],
    "start_max_days": [3, 7, 14, 30],
    "start_default": [0, 1, 2, 3],
    "end_rules": [
        (EndRule.LIQUID,),
        (EndRule.WATERCUT,),
        (EndRule.OIL,),
        (EndRule.LIQUID, EndRule.WATERCUT),
        (EndRule.LIQUID, EndRule.WATERCUT, EndRule.OIL),
    ],
    "liquid_tolerance": [0.0, 0.01, 0.02, 0.05],
    "watercut_tolerance": [0.0, 0.5, 1.0, 2.0],
    "oil_tolerance": [0.0, 0.02, 0.05],
    "stop_days": [1, 2, 3, 5],
    "min_effect_days": [0, 3, 7, 14],
    "end_confirm_days": [1, 2, 3, 5, 7],
}


class Evaluator:
    """Оценивает конфигурацию детектора на размеченных скважинах."""

    def __init__(self, matcher: EventMatcher | None = None) -> None:
        self.matcher = matcher or EventMatcher()

    def evaluate(self, config: DetectorConfig, wells: Sequence[Well]) -> Score:
        detector = FrequencyIncreaseDetector(config)
        total = Score()
        for well in wells:
            events = detector.detect(well)
            total += self.matcher.score(events, well.labels, well.known_event_dates)
        return total


@dataclass
class CalibrationResult:
    config: DetectorConfig
    score: Score


class Calibrator:
    """Покоординатный подбор параметров.

    По очереди перебирает значения каждого параметра из ``search_space``,
    оставляя остальные фиксированными, и принимает изменение, если растёт
    :meth:`objective`. Повторяет проходы, пока есть улучшения (не более ``rounds``).

    Целевая функция — F-beta (``beta=2``: полнота важнее точности, т.к. в разметке
    есть не все реальные подъёмы частоты) минус штраф 0.01 за день ошибки дат,
    ошибка обрезается на ``clip_days``, чтобы единичные грубые промахи не
    перевешивали долю точных попаданий.
    """

    def __init__(
        self,
        search_space: dict[str, list[Any]] | None = None,
        evaluator: Evaluator | None = None,
        rounds: int = 3,
        beta: float = 2.0,
        clip_days: int = 14,
    ) -> None:
        self.search_space = search_space or DEFAULT_SEARCH_SPACE
        self.evaluator = evaluator or Evaluator()
        self.rounds = rounds
        self.beta = beta
        self.clip_days = clip_days

    def objective(self, score: Score) -> float:
        if not score.true_positives:
            return -1.0
        clip = self.clip_days
        start_penalty = np.mean(np.minimum(np.abs(score.start_errors), clip))
        end_count = len(score.end_errors) + score.open_ends
        end_penalty = (np.sum(np.minimum(np.abs(score.end_errors), clip)) + clip * score.open_ends) / max(
            1, end_count
        )
        return score.fbeta(self.beta) - 0.01 * (start_penalty + 0.5 * end_penalty)

    def fit(self, wells: Sequence[Well], initial: DetectorConfig | None = None) -> CalibrationResult:
        best_config = initial or DetectorConfig()
        best_score = self.evaluator.evaluate(best_config, wells)
        best_value = self.objective(best_score)

        for round_number in range(self.rounds):
            improved = False
            for name, values in self.search_space.items():
                for value in values:
                    if getattr(best_config, name) == value:
                        continue
                    candidate = best_config.replace(**{name: value})
                    score = self.evaluator.evaluate(candidate, wells)
                    candidate_value = self.objective(score)
                    if candidate_value > best_value + 1e-9:
                        best_config, best_score, best_value = candidate, score, candidate_value
                        improved = True
                        logger.info(
                            "[проход %d] %s=%s: цель=%.4f F1=%.3f MAE начала=%.2f конца=%.2f",
                            round_number,
                            name,
                            _format(value),
                            candidate_value,
                            score.f1,
                            score.start_mae,
                            score.end_mae,
                        )
            if not improved:
                break
        return CalibrationResult(config=best_config, score=best_score)

    def cross_validate(self, wells: Sequence[Well], folds: int = 5, seed: int = 0) -> Score:
        """k-fold по скважинам: подбор на k-1 частях, оценка на отложенной."""
        order = np.random.default_rng(seed).permutation(len(wells))
        parts = np.array_split(order, folds)
        total = Score()
        for i, test_part in enumerate(parts):
            train = [wells[j] for k, part in enumerate(parts) if k != i for j in part]
            test = [wells[j] for j in test_part]
            logger.info("Фолд %d/%d: обучение %d скважин, проверка %d", i + 1, folds, len(train), len(test))
            result = self.fit(train)
            total += self.evaluator.evaluate(result.config, test)
        return total


def _format(value: Any) -> str:
    if isinstance(value, tuple):
        return "(" + ", ".join(_format(item) for item in value) + ")"
    return str(value.value) if hasattr(value, "value") else repr(value)
