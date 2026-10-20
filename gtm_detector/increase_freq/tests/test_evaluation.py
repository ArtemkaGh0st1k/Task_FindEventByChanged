from gtm_detector.increase_freq import EventMatcher, FrequencyIncreaseDetector, Label
from gtm_detector.increase_freq.tests.conftest import day, make_well


def _detect_one(config):
    well = make_well(
        frequency=[48.0] * 10 + [50.0] * 50,
        liquid=[100.0] * 12 + [110.0] * 48,
        oil=[10.0] * 12 + [11.0] * 48,
    )
    return FrequencyIncreaseDetector(config).detect(well)


def test_match_by_step_date_when_monitoring_starts_late(config):
    events = _detect_one(config)
    # Технолог поставил начало мониторинга через 3 недели — сопоставляем по «Дате выполнения».
    label = Label(start=day(33), end=day(50), date_done=day(10))
    score = EventMatcher().score(events, [label])
    assert (score.true_positives, score.false_positives) == (1, 0)
    assert score.start_errors == [-21]
    assert score.open_ends == 1  # детектор считает, что эффект продолжается


def test_unlabeled_event_near_known_date_is_neutral(config):
    events = _detect_one(config)
    score = EventMatcher().score(events, [], known_event_dates=[day(11)])
    assert (score.neutral, score.false_positives) == (1, 0)


def test_scores_add_up(config):
    events = _detect_one(config)
    matcher = EventMatcher()
    hit = matcher.score(events, [Label(start=day(12), end=None)])
    miss = matcher.score([], [Label(start=day(100), end=None)])
    total = hit + miss
    assert (total.true_positives, total.false_negatives, total.recall) == (1, 1, 0.5)
