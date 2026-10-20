"""Поиск мероприятий «увеличение частоты F» по суточным замерам скважин.

Основные классы::

    from gtm_detector.increase_freq import DetectorConfig, FrequencyIncreaseDetector, Well

    well = Well.from_dict({...})
    detector = FrequencyIncreaseDetector(DetectorConfig.load(Constant.INCREASE_FREQ_PARAMS))
    events = detector.detect(well)
"""

from .config import DetectorConfig, EndRule, StartMode
from .detection import FrequencyIncreaseDetector
from .evaluation import Calibrator, Evaluator, EventMatcher, Score
from .models import Baseline, EndReason, Event, FrequencyStep, Label, Well

__all__ = [
    "Baseline",
    "Calibrator",
    "DetectorConfig",
    "EndReason",
    "EndRule",
    "Event",
    "EventMatcher",
    "Evaluator",
    "FrequencyIncreaseDetector",
    "FrequencyStep",
    "Label",
    "Score",
    "StartMode",
    "Well",
]
