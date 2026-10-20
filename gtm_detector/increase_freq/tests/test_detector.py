from datetime import date, timedelta

from freq_events import DetectorParams, detect_events
from freq_events.excel_io import _parse_date, norm_id
from freq_events.metrics import match

D0 = date(2026, 1, 1)
P = DetectorParams(smooth_window=3, min_step=0.1, min_total=0.3, hold_days=5, start_mode="rise",
                   end_rules=("oil",), end_confirm_days=3, stop_days=2)


def day(i):
    return D0 + timedelta(days=i)


def _well(freq, liq, oil, state=None, wc=None):
    n = len(freq)
    return {"well_id": "T", "date": [day(i) for i in range(n)], "frequency": freq,
            "debit_liquid": liq, "debit_oil": oil, "watercut": wc or [90.0] * n,
            "state": state or [True] * n}


def test_step_start_and_end_by_oil():
    # F 48 -> 50 на 10-й день, Qж/Qн растут на 12-й, Qн падает до базы на 40-й.
    freq = [48.0] * 10 + [50.0] * 50
    liq = [100.0] * 12 + [110.0] * 48
    oil = [10.0] * 12 + [11.0] * 28 + [10.0] * 20
    [e] = detect_events(_well(freq, liq, oil), P)
    assert (e.date_step, e.start, e.end, e.end_reason) == (day(10), day(12), day(39), "oil")
    assert e.delta_f == 2.0 and e.responded


def test_effect_continues_to_data_end():
    freq = [48.0] * 10 + [50.0] * 30
    liq = [100.0] * 11 + [110.0] * 29
    oil = [10.0] * 11 + [11.0] * 29
    [e] = detect_events(_well(freq, liq, oil), P)
    assert e.start == day(11) and e.end is None


def test_end_by_stop():
    n = 40
    freq = [48.0] * 10 + [50.0] * 20 + [0.0] * 10
    state = [True] * 30 + [False] * 10
    liq = [100.0] * 11 + [110.0] * 29
    oil = [10.0] * 11 + [11.0] * 29
    [e] = detect_events(_well(freq, liq, oil, state=state), P)
    assert (e.end, e.end_reason) == (day(29), "stop")


def test_single_day_dip_is_not_end():
    freq = [48.0] * 10 + [50.0] * 40
    liq = [100.0] * 11 + [110.0] * 39
    oil = [10.0] * 11 + [11.0] * 9 + [9.0] + [11.0] * 29   # разовый провал на 20-й день
    [e] = detect_events(_well(freq, liq, oil), P)
    assert e.end is None


def test_no_response_no_event():
    freq = [48.0] * 10 + [50.0] * 30
    assert detect_events(_well(freq, [100.0] * 40, [10.0] * 40), P) == []


def test_restart_spike_decrease_are_not_events():
    flat = [100.0] * 40
    # пуск после простоя
    freq = [48.0] * 15 + [0.0] * 5 + [48.0] * 20
    state = [True] * 15 + [False] * 5 + [True] * 20
    assert detect_events(_well(freq, flat, [10.0] * 40, state=state), P) == []
    # одиночный выброс и снижение частоты
    assert detect_events(_well([48.0] * 20 + [55.0] + [48.0] * 19, flat, [10.0] * 40), P) == []
    assert detect_events(_well([50.0] * 20 + [48.0] * 20, flat, [10.0] * 40), P) == []


def test_match_by_step_date_when_start_is_late():
    freq = [48.0] * 10 + [50.0] * 50
    liq = [100.0] * 12 + [110.0] * 48
    oil = [10.0] * 12 + [11.0] * 48
    pred = detect_events(_well(freq, liq, oil), P)
    # технолог поставил начало мониторинга через 3 недели — сопоставляем по дате выполнения
    s = match(pred, [{"start": day(33), "end": day(50), "date_done": day(10)}])
    assert s.tp == 1 and s.fp == 0 and s.start_err == [-21]


def test_excel_helpers():
    assert norm_id(" 561E ") == "561Е" and norm_id(7782.0) == "7782"
    assert _parse_date("21.04.2026г").date() == date(2026, 4, 21)
    assert _parse_date("08.02.206").date() == date(2026, 2, 8)
