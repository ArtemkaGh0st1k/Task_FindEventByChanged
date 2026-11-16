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
        self,
        wells: Sequence[WellSeries],
        models: Mapping[str, Callable[[], EventSegmenter]],
        test_wells: Sequence[WellSeries] | None = None,
    ) -> list[ModelResult]:
        """``test_wells`` — те же скважины в том же порядке, но с другой разметкой для проверки.

        Нужен, чтобы сравнить разные способы отбора обучающей разметки на одних и тех же
        фолдах и тех же проверочных мероприятиях. Скважины без проверочной разметки
        в оценку не входят.
        """
        tests = {"": wells if test_wells is None else test_wells}
        return self.run_multi(wells, models, tests)[""]

    def run_multi(
        self,
        wells: Sequence[WellSeries],
        models: Mapping[str, Callable[[], EventSegmenter]],
        tests: Mapping[str, Sequence[WellSeries]],
    ) -> dict[str, list[ModelResult]]:
        """Одно обучение на ``wells``, проверка сразу на нескольких наборах разметки.

        ``tests`` — имя набора -> те же скважины в том же порядке с другой разметкой
        (например, с началом по дате технолога и по дню скачка частоты). Вероятности
        считаются один раз, поэтому наборы сравнимы между собой день в день.
        """
        for label, test_wells in tests.items():
            if len(test_wells) != len(wells):
                raise ValueError(f"Набор «{label}» должен соответствовать wells один к одному")
        folds = self.folds(wells)
        evaluated = [
            self._evaluate(name, factory, wells, folds, tests) for name, factory in models.items()
        ]
        return {label: [one[label] for one in evaluated] for label in tests}

    def _evaluate(
        self,
        name: str,
        factory: Callable[[], EventSegmenter],
        wells: Sequence[WellSeries],
        folds: list[list[int]],
        tests: Mapping[str, Sequence[WellSeries]],
    ) -> dict[str, ModelResult]:
        results: dict[str, ModelResult] = {}
        targets = {label: [] for label in tests}
        probas = {label: [] for label in tests}
        for k, test_idx in enumerate(folds):
            test_set = set(test_idx)
            train = [well for i, well in enumerate(wells) if i not in test_set]
            model = factory()
            if not results:
                results = {label: ModelResult(name=name, title=model.title) for label in tests}
            started = time.perf_counter()
            model.fit(train)
            elapsed = time.perf_counter() - started
            for result in results.values():
                result.train_seconds += elapsed

            for i in test_idx:
                active = {label: test[i] for label, test in tests.items() if test[i].labels}
                if not active:
                    continue
                proba = model.predict_proba(wells[i])
                predicted = self.decoder.decode(proba, wells[i].dates)
                for label, well in active.items():
                    targets[label].append(well.target())
                    probas[label].append(proba)
                    results[label].score += self.matcher.score(predicted, well.labels)
                    results[label].predictions.extend(self._rows(well, predicted))
            logger.info("%s: фолд %d/%d готов", name, k + 1, len(folds))

        for label, result in results.items():
            y, p = np.concatenate(targets[label]), np.concatenate(probas[label])
            result.day_auc = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")
            logger.info("%s: recall=%.3f F1=%.3f", name, result.score.recall, result.score.f1)
        return results

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
