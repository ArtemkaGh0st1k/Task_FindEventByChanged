"""Общий интерфейс моделей: вероятность «день внутри эффекта мероприятия»."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..config import CONTEXT_DAYS
from ..data import WellSeries
from ..features import FeatureBuilder


class EventSegmenter(ABC):
    """Модель разметки ряда по дням.

    ``fit`` учится на размеченных скважинах, ``predict_proba`` возвращает для
    каждого дня скважины вероятность того, что идёт эффект мероприятия.
    Даты начала и окончания из вероятностей получает
    :class:`~gtm_detector.ml_freq.evaluation.SegmentDecoder` — одинаково для всех моделей.
    """

    name: str = "base"
    title: str = "Базовая модель"

    def __init__(
        self, features: FeatureBuilder | None = None, context_days: int | None = CONTEXT_DAYS
    ) -> None:
        self.features = features or FeatureBuilder()
        self.context_days = context_days  # см. WellSeries.training_mask

    @abstractmethod
    def fit(self, wells: Sequence[WellSeries]) -> EventSegmenter: ...

    @abstractmethod
    def predict_proba(self, well: WellSeries) -> np.ndarray: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> EventSegmenter: ...

    @property
    def file_suffix(self) -> str:
        return ".joblib"
