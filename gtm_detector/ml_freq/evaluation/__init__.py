"""Декодирование вероятностей, метрики, перекрёстная проверка и отчёт."""

from .decoding import SegmentDecoder
from .experiment import CrossValidator, ModelResult
from .metrics import EventMatcher, Score
from .report import ComparisonReport

__all__ = ["ComparisonReport", "CrossValidator", "EventMatcher", "ModelResult", "Score", "SegmentDecoder"]
