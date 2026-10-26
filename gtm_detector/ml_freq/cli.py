"""Командная строка ML-детектора.

Из корня проекта::

    python -m gtm_detector.ml_freq compare                      # сравнить все модели
    python -m gtm_detector.ml_freq compare --models transformer gboost
    python -m gtm_detector.ml_freq train --model transformer    # обучить на всех данных, сохранить веса
    python -m gtm_detector.ml_freq predict --model transformer  # найти мероприятия (разметка не нужна)
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import DataConfig, ExperimentConfig, TrainingConfig
from .data import WellRepository
from .evaluation import ComparisonReport, CrossValidator, SegmentDecoder
from .models import TransformerSegmenter, load_model, model_registry

logger = logging.getLogger(__package__)


def _run_dir(base: Path) -> Path:
    return base / datetime.now().strftime("%d-%m-%Y_%H-%M-%S")


def _registry(args: argparse.Namespace) -> dict:
    registry = model_registry(seed=args.seed)
    if args.epochs:
        registry["transformer"] = lambda: TransformerSegmenter(
            training=TrainingConfig(epochs=args.epochs, seed=args.seed)
        )
    return registry


def run_compare(args: argparse.Namespace) -> int:
    data = DataConfig(measurements=args.measurements, results=args.results)
    wells = [well for well in WellRepository(data.measurements, data.results).load() if well.labels]
    registry = _registry(args)
    unknown = set(args.models) - set(registry)
    if unknown:
        logger.error("Неизвестные модели: %s. Доступны: %s", sorted(unknown), sorted(registry))
        return 1
    print(
        f"Размеченных скважин: {len(wells)}, мероприятий: {sum(len(w.labels) for w in wells)}; "
        f"модели: {', '.join(args.models)}; фолдов: {args.folds}"
    )

    validator = CrossValidator(ExperimentConfig(folds=args.folds, seed=args.seed))
    results = validator.run(wells, {name: registry[name] for name in args.models})
    report = ComparisonReport(results)
    out_dir = args.out_dir or _run_dir(data.output_dir)
    report.save(out_dir)
    print("\n" + report.to_text())
    print(f"\nОтчёт: {out_dir / 'comparison.xlsx'}, график: {out_dir / 'comparison.png'}")
    return 0


def run_train(args: argparse.Namespace) -> int:
    data = DataConfig(measurements=args.measurements, results=args.results)
    wells = [well for well in WellRepository(data.measurements, data.results).load() if well.labels]
    model = _registry(args)[args.model]()
    model.fit(wells)
    path = data.weights_dir / f"{model.name}{model.file_suffix}"
    model.save(path)
    print(f"{model.title}: обучена на {len(wells)} скважинах -> {path}")
    return 0


def run_predict(args: argparse.Namespace) -> int:
    data = DataConfig(measurements=args.measurements)
    path = args.weights or _find_weights(data.weights_dir, args.model)
    model = load_model(path)
    wells = WellRepository(data.measurements, None).load()
    decoder = SegmentDecoder()

    rows = []
    for well in wells:
        for interval in decoder.decode(model.predict_proba(well), well.dates):
            rows.append(
                {
                    "скважина": well.number,
                    "куст": well.cluster,
                    "начало эффекта": interval.start,
                    "окончание эффекта": interval.end,
                    "статус": "эффект продолжается" if interval.end is None else "",
                }
            )
    out_dir = args.out_dir or _run_dir(data.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"events_{model.name}.xlsx"
    pd.DataFrame(rows).to_excel(out, index=False)
    print(
        f"{model.title}: найдено {len(rows)} мероприятий на "
        f"{len({(r['скважина'], r['куст']) for r in rows})} скважинах -> {out}"
    )
    return 0


def _find_weights(directory: Path, name: str) -> Path:
    for suffix in (".pt", ".joblib"):
        candidate = directory / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Нет весов модели {name} в {directory}: сначала выполните train")


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    defaults = DataConfig()
    names = list(model_registry())
    parser = argparse.ArgumentParser(
        prog=prog or f"python -m {__package__}",
        description="ML-детекция мероприятий «увеличение частоты F»: Transformer и модели scikit-learn.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный лог")
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser, with_results: bool = True) -> None:
        sub.add_argument("--measurements", type=Path, default=defaults.measurements, help="файл замеров")
        if with_results:
            sub.add_argument("--results", type=Path, default=defaults.results, help="файл с разметкой")
        sub.add_argument("--seed", type=int, default=0, help="зерно случайности")
        sub.add_argument("--epochs", type=int, help="эпох обучения Transformer (по умолчанию 40)")

    compare = commands.add_parser("compare", help="перекрёстная проверка и сравнение моделей")
    common(compare)
    compare.add_argument("--models", nargs="+", default=names, help=f"какие модели: {' '.join(names)}")
    compare.add_argument("--folds", type=int, default=5, help="фолдов по скважинам")
    compare.add_argument(
        "--out-dir", type=Path, help="папка отчёта (по умолчанию results/ml_freq/<дата_время>)"
    )
    compare.set_defaults(handler=run_compare)

    train = commands.add_parser("train", help="обучить модель на всей разметке и сохранить веса")
    common(train)
    train.add_argument("--model", choices=names, default="transformer")
    train.set_defaults(handler=run_train)

    predict = commands.add_parser("predict", help="найти мероприятия обученной моделью")
    common(predict, with_results=False)
    predict.add_argument("--model", choices=names, default="transformer")
    predict.add_argument("--weights", type=Path, help="файл весов (по умолчанию из ml_freq/weights)")
    predict.add_argument("--out-dir", type=Path, help="папка результата")
    predict.set_defaults(handler=run_predict)
    return parser


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    args = build_parser(prog).parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger(f"{__package__}.evaluation").setLevel(logging.INFO)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
