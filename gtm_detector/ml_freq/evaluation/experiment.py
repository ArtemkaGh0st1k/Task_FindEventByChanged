"""Сравнение моделей: k-fold по скважинам, одинаковые разбиения для всех моделей."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score

from ..config import ExperimentConfig
from ..data import WellSeries
from ..models import EventSegmenter
from .decoding import SegmentDecoder
from .metrics import EventMatcher, Score

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Итог перекрёстной проверки одной модели."""

    name: str
    title: str
    score: Score = field(default_factory=Score)
    day_auc: float = float("nan")  # ROC-AUC по дням: качество вероятностей до декодирования
    train_seconds: float = 0.0
    predictions: list[dict] = field(default_factory=list)  # строки «метка — найдено»

    def as_row(self) -> dict:
        return {
            "модель": self.title,
            **self.score.as_dict(),
            "ROC-AUC по дням": self.day_auc,
            "обучение, с (все фолды)": round(self.train_seconds, 1),
        }


class CrossValidator:
    """k-fold по скважинам: модель не видит скважины, на которых её проверяют."""

    def __init__(self, config: ExperimentConfig | None = None) -> None:
        self.config = config or ExperimentConfig()
        self.decoder = SegmentDecoder(self.config.decoder)
        self.matcher = EventMatcher(self.config.start_tolerance)

    def folds(self, wells: Sequence[WellSeries]) -> list[list[int]]:
        order = np.random.default_rng(self.config.seed).permutation(len(wells))
        return [part.tolist() for part in np.array_split(order, self.config.folds)]

    def run(
        self, wells: Sequence[WellSeries], models: Mapping[str, Callable[[], EventSegmenter]]
    ) -> list[ModelResult]:
        folds = self.folds(wells)
        return [self._evaluate(name, factory, wells, folds) for name, factory in models.items()]

    def _evaluate(
        self,
        name: str,
        factory: Callable[[], EventSegmenter],
        wells: Sequence[WellSeries],
        folds: list[list[int]],
    ) -> ModelResult:
        result: ModelResult | None = None
        targets, probas = [], []
        for k, test_idx in enumerate(folds):
            test_set = set(test_idx)
            train = [well for i, well in enumerate(wells) if i not in test_set]
            model = factory()
            if result is None:
                result = ModelResult(name=name, title=model.title)
            started = time.perf_counter()
            model.fit(train)
            result.train_seconds += time.perf_counter() - started

            for i in test_idx:
                well = wells[i]
                proba = model.predict_proba(well)
                targets.append(well.target())
                probas.append(proba)
                predicted = self.decoder.decode(proba, well.dates)
                result.score += self.matcher.score(predicted, well.labels)
                result.predictions.extend(self._rows(well, predicted))
            logger.info("%s: фолд %d/%d готов", name, k + 1, len(folds))

        y, p = np.concatenate(targets), np.concatenate(probas)
        result.day_auc = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")
        logger.info("%s: recall=%.3f F1=%.3f", name, result.score.recall, result.score.f1)
        return result

    def _rows(self, well: WellSeries, predicted: Sequence) -> list[dict]:
        rows = []
        for label, hit in self.matcher.match(predicted, well.labels):
            rows.append(
                {
                    "скважина": well.number,
                    "куст": well.cluster,
                    "начало (разметка)": label.start,
                    "окончание (разметка)": label.end,
                    "начало (модель)": hit.start if hit else None,
                    "окончание (модель)": hit.end if hit else None,
                    "Δ начала, дн": (hit.start - label.start).days if hit else None,
                    "Δ окончания, дн": (
                        (hit.end - label.end).days
                        if hit and hit.end and label.end and not label.censored
                        else None
                    ),
                    "эффект продолжался": label.censored,
                }
            )
        return rows
