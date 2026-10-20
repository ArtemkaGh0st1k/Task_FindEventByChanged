"""Поиск мероприятий «увеличение частоты F» в файле замеров технологов.

Подбор параметров по размеченному файлу результатов (пока разметка есть):
    python run.py calibrate --measurements Статистика.xlsx --results Расчёт_НОРДС.xlsm

Поиск мероприятий (разметка не нужна; --results — только для сравнения):
    python run.py detect --measurements Статистика.xlsx [--results Расчёт_НОРДС.xlsm] --out events.xlsx
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from freq_events.calibrate import Dataset, calibrate, cross_validate, evaluate, load_params, save_params
from freq_events.detector import DetectorParams, detect_events
from freq_events.excel_io import attach_labels, load_measurements, read_results

REASONS = {"oil": "по нефти", "liq": "по жидкости", "wc": "по обводнённости", "stop": "остановка",
           None: "эффект продолжается"}


def _load(args):
    t = time.time()
    wells = load_measurements(args.measurements)
    print(f"Замеры: {len(wells)} скважин ({time.time() - t:.0f} с)")
    labeled = []
    if args.results:
        res = read_results(args.results)
        labeled, unmatched = attach_labels(wells, res)
        print(f"Разметка: {int(res['is_label'].sum())} мероприятий «Увеличение F», успешно, с датами; "
              f"скважин {len(labeled)}; не найдено в замерах: {len(unmatched)}")
    return wells, labeled


def cmd_calibrate(args):
    _, labeled = _load(args)
    if not labeled:
        sys.exit("Нет размеченных скважин — калибровать не по чему.")
    data = Dataset(labeled)
    print("\nПараметры по умолчанию:\n  " + evaluate(data, DetectorParams()).summary())
    if args.cv:
        print(f"\nКросс-валидация ({args.cv} фолдов по скважинам)...")
        print("  " + cross_validate(data, k=args.cv).summary())
    print("\nПодбор параметров...")
    p, s = calibrate(data, verbose=True)
    print("\nПодобранные параметры (на всей разметке):\n  " + s.summary())
    save_params(p, args.params)
    print(f"\nСохранено в {args.params}")


def cmd_detect(args):
    wells, labeled = _load(args)
    p = load_params(args.params) if args.params else DetectorParams()
    rows, cmp_rows = [], []
    for w in wells:
        events = detect_events(w, p)
        for e in events:
            d = e.to_dict()
            d.update(well=w["well"], kust=w["kust"])
            rows.append(d)
        if w.get("events"):
            for lab in w["events"]:
                cand = [e for e in events
                        if abs((e.start - lab["start"]).days) <= 7
                        or (lab["date_done"] and abs((e.date_step - lab["date_done"]).days) <= 3)]
                e = min(cand, key=lambda e: abs((e.start - lab["start"]).days)) if cand else None
                cmp_rows.append({
                    "well": w["well"], "kust": w["kust"],
                    "Дата выполнения": lab["date_done"], "Начало (разметка)": lab["start"],
                    "Окончание (разметка)": lab["end"], "Причина (разметка)": lab.get("end_reason"),
                    "Скачок F (найдено)": e.date_step if e else None,
                    "Начало (найдено)": e.start if e else None,
                    "Окончание (найдено)": e.end if e else None,
                    "Причина (найдено)": REASONS[e.end_reason] if e else None,
                    "Δ начала, дн": (e.start - lab["start"]).days if e else None,
                    "Δ окончания, дн": (e.end - lab["end"]).days if e and e.end else None,
                })

    cols = ["well", "kust", "date_step", "start", "end", "end_reason", "f_before", "f_after", "delta_f",
            "n_steps", "base_liquid", "liquid_at_start", "base_oil", "oil_at_start", "base_watercut",
            "responded", "after_restart", "confidence"]
    ev = pd.DataFrame(rows, columns=cols).rename(columns={
        "well": "Скважина", "kust": "Куст", "date_step": "Скачок частоты", "start": "Начало эффекта",
        "end": "Окончание эффекта", "end_reason": "Причина окончания", "f_before": "F до, Гц",
        "f_after": "F после, Гц", "delta_f": "ΔF, Гц", "n_steps": "Ступеней",
        "base_liquid": "Qж до", "liquid_at_start": "Qж в начале", "base_oil": "Qн до",
        "oil_at_start": "Qн в начале", "base_watercut": "Обв до, %", "responded": "Есть отклик",
        "after_restart": "После простоя", "confidence": "Уверенность",
    })
    ev["Причина окончания"] = ev["Причина окончания"].map(lambda r: REASONS[r if pd.notna(r) else None])
    with pd.ExcelWriter(args.out) as xw:
        ev.to_excel(xw, sheet_name="Мероприятия", index=False)
        if cmp_rows:
            pd.DataFrame(cmp_rows).to_excel(xw, sheet_name="Сравнение с разметкой", index=False)
        pd.DataFrame([p.__dict__]).T.rename(columns={0: "значение"}).to_excel(xw, sheet_name="Параметры")
    print(f"Найдено мероприятий: {len(ev)} на {ev['Скважина'].nunique()} скважинах -> {args.out}")
    if labeled:
        print("Сравнение с разметкой:\n  " + evaluate(Dataset(labeled), p).summary())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate", help="подбор параметров по разметке")
    c.add_argument("--measurements", required=True)
    c.add_argument("--results", required=True)
    c.add_argument("--params", default="params.json")
    c.add_argument("--cv", type=int, default=0, help="k-fold кросс-валидация перед подбором")
    c.set_defaults(func=cmd_calibrate)
    d = sub.add_parser("detect", help="поиск мероприятий")
    d.add_argument("--measurements", required=True)
    d.add_argument("--results", help="файл с разметкой — для сравнения (необязательно)")
    d.add_argument("--params", default="params.json")
    d.add_argument("--out", default="events.xlsx")
    d.set_defaults(func=cmd_detect)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()