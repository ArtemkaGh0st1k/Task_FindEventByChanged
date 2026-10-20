from gtm_detector.increase_freq import EndReason, FrequencyIncreaseDetector
from gtm_detector.increase_freq.detection import StepDetector
from gtm_detector.increase_freq.tests.conftest import day, make_well


def test_step_start_and_end_by_oil(config):
    # F 48 -> 50 на 10-й день, Qж/Qн растут на 12-й, Qн возвращается к базе на 40-й.
    well = make_well(
        frequency=[48.0] * 10 + [50.0] * 50,
        liquid=[100.0] * 12 + [110.0] * 48,
        oil=[10.0] * 12 + [11.0] * 28 + [10.0] * 20,
    )
    [event] = FrequencyIncreaseDetector(config).detect(well)
    assert (event.date_step, event.start, event.end) == (day(10), day(12), day(39))
    assert event.end_reason is EndReason.OIL
    assert event.delta_f == 2.0
    assert event.responded


def test_effect_continues_to_data_end(config):
    well = make_well(
        frequency=[48.0] * 10 + [50.0] * 30,
        liquid=[100.0] * 11 + [110.0] * 29,
        oil=[10.0] * 11 + [11.0] * 29,
    )
    [event] = FrequencyIncreaseDetector(config).detect(well)
    assert event.start == day(11)
    assert event.end is None
    assert event.end_reason_title == "эффект продолжается"


def test_end_by_stop(config):
    well = make_well(
        frequency=[48.0] * 10 + [50.0] * 20 + [0.0] * 10,
        state=[True] * 30 + [False] * 10,
        liquid=[100.0] * 11 + [110.0] * 29,
        oil=[10.0] * 11 + [11.0] * 29,
    )
    [event] = FrequencyIncreaseDetector(config).detect(well)
    assert (event.end, event.end_reason) == (day(29), EndReason.STOP)


def test_single_day_dip_is_not_end(config):
    well = make_well(
        frequency=[48.0] * 10 + [50.0] * 40,
        liquid=[100.0] * 11 + [110.0] * 39,
        oil=[10.0] * 11 + [11.0] * 9 + [9.0] + [11.0] * 29,  # разовый провал на 20-й день
    )
    [event] = FrequencyIncreaseDetector(config).detect(well)
    assert event.end is None


def test_no_production_response_means_no_event(config):
    well = make_well(frequency=[48.0] * 10 + [50.0] * 30)
    assert FrequencyIncreaseDetector(config).detect(well) == []


def test_staircase_is_one_step(config):
    well = make_well(frequency=[45.0] * 10 + [46.0, 47.0, 47.0, 48.0] + [48.0] * 10)
    [step] = StepDetector(config).find(well.data)
    assert (step.index, step.n_steps, step.delta) == (10, 3, 3.0)


def test_restart_spike_and_decrease_are_not_steps(config):
    detector = StepDetector(config)
    restart = make_well(
        frequency=[48.0] * 15 + [0.0] * 5 + [48.0] * 20,
        state=[True] * 15 + [False] * 5 + [True] * 20,
    )
    spike = make_well(frequency=[48.0] * 20 + [55.0] + [48.0] * 19)
    decrease = make_well(frequency=[50.0] * 20 + [48.0] * 20)
    for well in (restart, spike, decrease):
        assert detector.find(well.data) == []
