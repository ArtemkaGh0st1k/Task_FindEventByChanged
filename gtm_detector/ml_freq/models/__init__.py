"""Модели: собственный Transformer (PyTorch) и готовые модели scikit-learn."""

from collections.abc import Callable

from .base import EventSegmenter
from .classic import SklearnSegmenter, classic_models
from .transformer import TimeSeriesTransformer, TransformerSegmenter


def model_registry(seed: int = 0) -> dict[str, Callable[[], EventSegmenter]]:
    """Все доступные модели: имя -> фабрика необученной модели."""
    registry: dict[str, Callable[[], EventSegmenter]] = {"transformer": TransformerSegmenter}
    registry.update(classic_models(seed))
    return registry


def load_model(path) -> EventSegmenter:
    """Загрузить сохранённую модель по расширению файла."""
    return TransformerSegmenter.load(path) if str(path).endswith(".pt") else SklearnSegmenter.load(path)


__all__ = [
    "EventSegmenter",
    "SklearnSegmenter",
    "TimeSeriesTransformer",
    "TransformerSegmenter",
    "classic_models",
    "load_model",
    "model_registry",
]
