"""Поиск мероприятий «увеличение частоты F» по суточным замерам скважин.

Основные классы::

    from freq_events import DetectorConfig, FrequencyIncreaseDetector, Well

    well = Well.from_dict({...})
    detector = FrequencyIncreaseDetector(DetectorConfig.load("params.json"))
    events = detector.detect(well)
"""

from freq_events.config import DetectorConfig, EndRule, StartMode
from freq_events.detection import FrequencyIncreaseDetector
from freq_events.evaluation import Calibrator, Evaluator, EventMatcher, Score
from freq_events.models import Baseline, EndReason, Event, FrequencyStep, Label, Well

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
