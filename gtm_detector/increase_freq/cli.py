"""Командная строка детектора.

Запуск из корня проекта::

    python -m gtm_detector.increase_freq calibrate   # подбор параметров по разметке
    python -m gtm_detector.increase_freq detect      # поиск мероприятий
    python -m gtm_detector.increase_freq detect --plots   # + графики и сводка

По умолчанию берутся resources/input_dataset.xlsx (замеры) и
resources/output_dataset.xlsm (разметка) — см. constants/path.py.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from constants.path import Constant

from .config import DetectorConfig
from .detection import FrequencyIncreaseDetector
from .evaluation import Calibrator, Evaluator, EventMatcher, Score
from .excel import EventsExcelWriter, MeasurementsReader, ResultsReader, attach_labels
from .models import Well

PACKAGE = __name__.rsplit(".", 1)[0]  # gtm_detector.increase_freq
logger = logging.getLogger(PACKAGE)


def load_wells(measurements: Path, results: Path | None) -> tuple[list[Well], list[Well]]:
    """Прочитать замеры и (если задан файл результатов) привязать разметку.

    Возвращает все скважины и только размеченные.
    """
    wells = MeasurementsReader(measurements).read()
    if not results:
        return wells, []
    unmatched = attach_labels(wells, ResultsReader(results).read())
    labeled = [well for well in wells if well.labels]
    logger.info(
        "Разметка: %d мероприятий на %d скважинах; не найдено в замерах: %d",
        sum(len(well.labels) for well in labeled),
        len(labeled),
        len(unmatched),
    )
    return wells, labeled


def run_calibrate(args: argparse.Namespace) -> int:
    _, labeled = load_wells(args.measurements, args.results)
    if not labeled:
        logger.error("Нет размеченных скважин — подбирать параметры не по чему.")
        return 1

    evaluator = Evaluator()
    calibrator = Calibrator(evaluator=evaluator)
    print("Параметры по умолчанию:\n  " + evaluator.evaluate(DetectorConfig(), labeled).summary())
    if args.cv:
        print(f"\nКросс-валидация ({args.cv} фолдов по скважинам)...")
        print("  " + calibrator.cross_validate(labeled, folds=args.cv).summary())

    print("\nПодбор параметров...")
    result = calibrator.fit(labeled)
    print("\nПодобранные параметры (на всей разметке):\n  " + result.score.summary())
    result.config.save(args.params)
    print(f"\nСохранено в {args.params}")
    return 0


def run_detect(args: argparse.Namespace) -> int:
    wells, labeled = load_wells(args.measurements, args.results)
    config = DetectorConfig.load(args.params)
    events_by_well = FrequencyIncreaseDetector(config).detect_all(wells)

    out_dir = args.out_dir or default_run_dir()
    excel_path = out_dir / "events.xlsx"
    EventsExcelWriter(excel_path).write(wells, events_by_well, config)

    total = sum(len(events) for events in events_by_well.values())
    with_events = sum(1 for events in events_by_well.values() if events)
    print(f"Найдено мероприятий: {total} на {with_events} скважинах -> {excel_path}")

    score = None
    if labeled:
        matcher = EventMatcher()
        score = Score()
        for well in labeled:
            score += matcher.score(events_by_well[well.well_id], well.labels, well.known_event_dates)
        print("Сравнение с разметкой:\n  " + score.summary())

    if args.plots:
        # matplotlib нужен только для графиков — импортируем по требованию.
        from .visualization import PlotReport

        print("Строю графики...")
        report = PlotReport(include_unlabeled=args.all_wells, workers=args.workers)
        counts = report.build(wells, events_by_well, out_dir, score)
        if score is not None:
            print(f"  сводка: {out_dir / 'summary.png'}")
        for status, count in counts.items():
            print(f"  {out_dir / 'wells' / status.value}: {count} шт.")
    return 0


def default_run_dir() -> Path:
    """results/increase_freq/ДД-ММ-ГГГГ_ЧЧ-ММ-СС/"""
    return Constant.INCREASE_FREQ_RESULTS_DIR / datetime.now().strftime("%d-%m-%Y_%H-%M-%S")


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog or f"python -m {PACKAGE}",
        description="Поиск мероприятий «увеличение частоты F» по суточным замерам скважин.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный лог")
    commands = parser.add_subparsers(dest="command", required=True)

    calibrate = commands.add_parser("calibrate", help="подбор параметров по разметке")
    calibrate.add_argument(
        "--measurements", type=Path, default=Constant.INPUT_DATASET, help="файл замеров (.xlsx)"
    )
    calibrate.add_argument(
        "--results", type=Path, default=Constant.OUTPUT_DATASET, help="файл с разметкой (.xlsm)"
    )
    calibrate.add_argument(
        "--params", type=Path, default=Constant.INCREASE_FREQ_PARAMS, help="куда сохранить параметры"
    )
    calibrate.add_argument("--cv", type=int, default=0, help="k-fold кросс-валидация перед подбором")
    calibrate.set_defaults(handler=run_calibrate)

    detect = commands.add_parser("detect", help="поиск мероприятий")
    detect.add_argument(
        "--measurements", type=Path, default=Constant.INPUT_DATASET, help="файл замеров (.xlsx)"
    )
    detect.add_argument(
        "--results", type=Path, help="файл с разметкой — только для сравнения (необязательно)"
    )
    detect.add_argument(
        "--params", type=Path, default=Constant.INCREASE_FREQ_PARAMS, help="файл параметров (JSON)"
    )
    detect.add_argument(
        "--out-dir",
        type=Path,
        help="папка прогона (по умолчанию results/increase_freq/<дата_время>/)",
    )
    detect.add_argument("--plots", action="store_true", help="построить графики скважин и сводку качества")
    detect.add_argument(
        "--all-wells",
        action="store_true",
        help="с --plots: рисовать и скважины без разметки, где что-то найдено",
    )
    detect.add_argument(
        "--workers", type=int, help="с --plots: сколько процессов рисуют графики (по умолчанию — по ядрам)"
    )
    detect.set_defaults(handler=run_detect)
    return parser


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    args = build_parser(prog).parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Прогресс подбора выводим всегда — он длится минуты.
    logging.getLogger(f"{PACKAGE}.evaluation").setLevel(logging.INFO)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
