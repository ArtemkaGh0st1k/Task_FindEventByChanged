import pytest

from gtm_detector.increase_freq import EventMatcher, FrequencyIncreaseDetector, Label, Score
from gtm_detector.increase_freq.tests.conftest import day, make_well

pytest.importorskip("matplotlib")

from gtm_detector.increase_freq.visualization import PlotReport, WellStatus  # noqa: E402
from gtm_detector.increase_freq.visualization.summary import reason_group  # noqa: E402


def _well_with_event(well_id="T", labels=()):
    well = make_well(
        frequency=[48.0] * 10 + [50.0] * 50,
        liquid=[100.0] * 12 + [110.0] * 48,
        oil=[10.0] * 12 + [11.0] * 28 + [10.0] * 20,
    )
    well.well_id, well.number, well.cluster = well_id, well_id, "1"
    well.labels = list(labels)
    return well


def _build(tmp_path, config, wells, include_unlabeled=False):
    detector = FrequencyIncreaseDetector(config)
    events = {well.well_id: detector.detect(well) for well in wells}
    score = Score()
    for well in wells:
        if well.labels:
            score += EventMatcher().score(events[well.well_id], well.labels)
    report = PlotReport(include_unlabeled=include_unlabeled, workers=1)
    return report.build(wells, events, tmp_path, score)


def test_wells_sorted_by_status_and_files_written(tmp_path, config):
    exact = _well_with_event("A", [Label(start=day(12), end=day(39), date_done=day(10))])
    late_end = _well_with_event("B", [Label(start=day(12), end=day(55), date_done=day(10))])
    missed = _well_with_event("C", [Label(start=day(45), end=day(50), date_done=day(44))])
    unlabeled = _well_with_event("D")

    counts = _build(tmp_path, config, [exact, late_end, missed, unlabeled])

    assert counts == {WellStatus.MATCHED: 1, WellStatus.DATE_ERROR: 1, WellStatus.MISSED: 1}
    assert (tmp_path / "summary.png").stat().st_size > 0
    assert (tmp_path / "wells" / "совпало" / "A_куст_1.png").exists()
    assert (tmp_path / "wells" / "ошибка_дат" / "B_куст_1.png").exists()
    assert (tmp_path / "wells" / "пропущено" / "C_куст_1.png").exists()


def test_unlabeled_wells_plotted_on_request(tmp_path, config):
    labeled = _well_with_event("A", [Label(start=day(12), end=day(39), date_done=day(10))])
    counts = _build(tmp_path, config, [labeled, _well_with_event("D")], include_unlabeled=True)
    assert counts[WellStatus.UNLABELED] == 1


def test_without_labels_everything_goes_to_found(tmp_path, config):
    counts = _build(tmp_path, config, [_well_with_event("A"), _well_with_event("B")])
    assert counts == {WellStatus.FOUND: 2}
    assert not (tmp_path / "summary.png").exists()


def test_reason_groups():
    assert reason_group("по обводнености") == "по обводнённости"
    assert reason_group("по жидкости, перевод в ПР") == "по жидкости"
    assert reason_group("ИДН") == "остановка / ИДН"
    assert reason_group("Отказ ЭЦН") == "отказ / смена ЭЦН"
    assert reason_group(None) == "прочее"
