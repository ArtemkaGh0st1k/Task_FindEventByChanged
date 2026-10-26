"""Готовые модели scikit-learn: классификация каждого дня по признакам окна."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..data import WellSeries
from ..features import FeatureBuilder
from .base import EventSegmenter


class SklearnSegmenter(EventSegmenter):
    """Обёртка над классификатором scikit-learn: строка = день скважины."""

    def __init__(
        self,
        name: str,
        title: str,
        estimator_factory: Callable[[], ClassifierMixin],
        features: FeatureBuilder | None = None,
    ) -> None:
        super().__init__(features)
        self.name = name
        self.title = title
        self.estimator_factory = estimator_factory
        self.estimator: ClassifierMixin | None = None

    def fit(self, wells: Sequence[WellSeries]) -> SklearnSegmenter:
        masks = [well.training_mask(self.context_days) for well in wells]
        x = np.vstack([self.features.tabular(well)[mask] for well, mask in zip(wells, masks, strict=True)])
        y = np.concatenate([well.target()[mask] for well, mask in zip(wells, masks, strict=True)]).astype(int)
        self.estimator = self.estimator_factory()
        self.estimator.fit(x, y)
        return self

    def predict_proba(self, well: WellSeries) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError(f"Модель {self.name} не обучена")
        return self.estimator.predict_proba(self.features.tabular(well))[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"name": self.name, "title": self.title, "estimator": self.estimator}, path)

    @classmethod
    def load(cls, path: Path) -> SklearnSegmenter:
        state = joblib.load(path)
        model = cls(state["name"], state["title"], estimator_factory=lambda: state["estimator"])
        model.estimator = state["estimator"]
        return model


def classic_models(seed: int = 0) -> dict[str, Callable[[], SklearnSegmenter]]:
    """Набор готовых моделей для сравнения: имя -> фабрика необученной модели."""

    def logistic() -> ClassifierMixin:
        return make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
        )

    def random_forest() -> ClassifierMixin:
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )

    def extra_trees() -> ClassifierMixin:
        return ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )

    def gradient_boosting() -> ClassifierMixin:
        return HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=seed,
        )

    specs = {
        "logreg": ("Логистическая регрессия", logistic),
        "random_forest": ("Случайный лес", random_forest),
        "extra_trees": ("Extra Trees", extra_trees),
        "gboost": ("Градиентный бустинг (HistGB)", gradient_boosting),
    }
    return {
        name: (lambda name=name, title=title, factory=factory: SklearnSegmenter(name, title, factory))
        for name, (title, factory) in specs.items()
    }
