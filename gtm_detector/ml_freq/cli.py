"""Командная строка ML-детектора.

Из корня проекта::

    python -m gtm_detector.ml_freq compare                      # сравнить все модели
    python -m gtm_detector.ml_freq compare --models transformer gboost
    python -m gtm_detector.ml_freq train --model transformer    # обучить на всех данных, сохранить веса
    python -m gtm_detector.ml_freq predict --model transformer  # найти мероприятия (разметка не нужна)
    python -m gtm_detector.ml_freq audit                        # отчёт о качестве данных 2021–2026

По умолчанию — объединённые файлы 2021–2026 (``--dataset merged``); первые
тестовые файлы — ``--dataset test``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd

from .audit import add_arguments as audit_arguments
from .audit import run as run_audit
from .config import DataConfig, ExperimentConfig, TrainingConfig
from .data import BlockShiftRepair, FrequencyCheck, MergedWellRepository, WellRepository
from .evaluation import ComparisonReport, CrossValidator, SegmentDecoder
from .models import TransformerSegmenter, WeightsArchive, load_model, model_registry

logger = logging.getLogger(__package__)


def _weights_archive(data: DataConfig) -> WeightsArchive:
    return WeightsArchive(data.weights_archive, data.weights_dir)


def _run_dir(base: Path) -> Path:
    return base / datetime.now().strftime("%d-%m-%Y_%H-%M-%S")


def _registry(args: argparse.Namespace) -> dict:
    registry = model_registry(seed=args.seed)
    if args.epochs:
        registry["transformer"] = lambda: TransformerSegmenter(
            training=TrainingConfig(epochs=args.epochs, seed=args.seed)
        )
    return registry


def _all_wells(
    args: argparse.Namespace, lookback: int, lookahead: int, labeled: bool = True
) -> list:
    """Все пригодные скважины; разметка отобрана с окном ``lookback``/``lookahead`` дней."""
    if args.dataset == "merged":
        reference = None if args.no_reference else args.reference
        repair = BlockShiftRepair(check=FrequencyCheck(lookback, lookahead))
        return MergedWellRepository(args.statistics, args.labels, reference, repair=repair).load()
    return WellRepository(args.measurements, args.results if labeled else None).load()


def load_wells(args: argparse.Namespace, labeled: bool) -> list:
    """Скважины выбранного набора данных; ``labeled`` — только с разметкой."""
    wells = _all_wells(args, args.label_lookback, args.label_lookahead, labeled)
    return [well for well in wells if well.labels] if labeled else wells


def run_compare(args: argparse.Namespace) -> int:
    data = DataConfig()
    wells = load_wells(args, labeled=True)
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


WINDOW_METRICS = (
    "найдено размеченных (recall)",
    "точность (precision)",
    "начало: ±1 дн",
    "начало: ±3 дн",
    "конец: ±3 дн",
    "конец: ±7 дн",
    "ROC-AUC по дням",
)
NARROW, WIDE = "узкое окно", "широкое окно"
TECH, JUMP = "начало по дате технолога", "начало по дню скачка"
START_CHOICES = {"tech": (TECH,), "jump": (JUMP,), "both": (TECH, JUMP)}


def _event_count(wells: Sequence) -> int:
    return sum(len(well.labels) for well in wells)


def _window_tables(
    frame: pd.DataFrame, test: str, baseline: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Среднее метрик по зёрнам и разница с базовым вариантом (среднее ± std по зёрнам)."""
    metrics = list(WINDOW_METRICS)
    sub = frame[frame["проверка"] == test]
    mean = sub.groupby(["модель", "вариант"])[metrics].mean()
    by_seed = sub.set_index(["модель", "зерно", "вариант"])[metrics].unstack("вариант")
    base = by_seed.xs(baseline, axis=1, level=1)
    variants = [v for v in sub["вариант"].unique() if v != baseline]
    deltas = {v: by_seed.xs(v, axis=1, level=1) - base for v in variants}
    grouped = pd.concat(deltas, names=["вариант"]).groupby(level=["вариант", "модель"])
    stats = grouped.agg(["mean", "std"])
    means = stats.xs("mean", axis=1, level=1).apply(lambda col: col.map("{:+.3f}".format))
    spread = stats.xs("std", axis=1, level=1).apply(lambda col: col.map("{:.3f}".format))
    return mean, means + " ± " + spread


def run_compare_windows(args: argparse.Namespace) -> int:
    """Обучение на разметке с узким и широким окном отбора, проверка — на одних мероприятиях.

    Фолды по скважинам общие, проверочные мероприятия — те, что прошли узкое окно
    (как в прежних результатах), поэтому разница метрик — эффект обучающей разметки.
    """
    data = DataConfig()
    if args.dataset != "merged":
        logger.error("Окно отбора разметки есть только у --dataset merged")
        return 1
    narrow_all = _all_wells(args, data.label_lookback, data.label_lookahead)
    wide_all = _all_wells(args, args.label_lookback, args.label_lookahead)
    keep = [i for i, well in enumerate(wide_all) if well.labels]
    narrow, wide = [narrow_all[i] for i in keep], [wide_all[i] for i in keep]
    same = len(narrow_all) == len(wide_all) and all(
        a.well_id == b.well_id for a, b in zip(narrow, wide, strict=True)
    )
    if not same:
        logger.error("Наборы скважин при разных окнах не совпали")
        return 1
    print(
        f"Скважин: {len(wide)} (с узким окном размечено {sum(1 for w in narrow if w.labels)}); "
        f"обучающих мероприятий: {_event_count(narrow)} (узкое) и {_event_count(wide)} "
        f"(широкое: {args.label_lookback} дн. до и {args.label_lookahead} после); "
        f"проверка — на {_event_count(narrow)}"
    )

    # Проверка: те же мероприятия узкого окна, начало по дате технолога и по дню скачка.
    tests = {TECH: narrow, JUMP: [well.with_jump_starts() for well in narrow]}
    starts = START_CHOICES[args.label_start]
    variants = [
        (f"{window} / {start}", wells if start == TECH else [w.with_jump_starts() for w in wells])
        for window, wells in ((NARROW, narrow), (WIDE, wide))
        for start in starts
    ]
    baseline = f"{NARROW} / {TECH}"
    if baseline not in dict(variants):
        variants.insert(0, (baseline, narrow))  # без базы разницу считать не с чем

    rows = []
    for seed in args.seeds or [args.seed]:
        args.seed = seed
        registry = _registry(args)
        unknown = set(args.models) - set(registry)
        if unknown:
            logger.error("Неизвестные модели: %s. Доступны: %s", sorted(unknown), sorted(registry))
            return 1
        models = {name: registry[name] for name in args.models}
        validator = CrossValidator(ExperimentConfig(folds=args.folds, seed=seed))
        for variant, train in variants:
            for test, results in validator.run_multi(train, models, tests).items():
                for result in results:
                    row = {"модель": result.title, "зерно": seed}
                    row |= {"вариант": variant, "проверка": test}
                    row |= result.score.as_dict() | {"ROC-AUC по дням": result.day_auc}
                    rows.append(row)
            print(f"зерно {seed}: {variant} — готово")

    frame = pd.DataFrame(rows)
    out_dir = args.out_dir or _run_dir(data.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {test: _window_tables(frame, test, baseline) for test in tests}
    with pd.ExcelWriter(out_dir / "windows.xlsx") as writer:
        frame.to_excel(writer, sheet_name="по зёрнам", index=False)
        for n, (mean, delta) in enumerate(tables.values(), start=1):
            mean.to_excel(writer, sheet_name=f"среднее, проверка {n}")
            delta.to_excel(writer, sheet_name=f"разница, проверка {n}")
    with pd.option_context("display.width", 250, "display.max_columns", None):
        for test, (mean, delta) in tables.items():
            print(f"\n=== Проверка: {test} ===\nСреднее по зёрнам:\n{mean.round(3).to_string()}")
            print(f"\nРазница с «{baseline}» (среднее ± std по зёрнам):\n{delta.to_string()}")
    print(f"\nОтчёт: {out_dir / 'windows.xlsx'}")
    return 0


def run_train(args: argparse.Namespace) -> int:
    data = DataConfig()
    wells = load_wells(args, labeled=True)
    model = _registry(args)[args.model]()
    model.fit(wells)
    path = data.weights_dir / f"{model.name}{model.file_suffix}"
    model.save(path)
    _weights_archive(data).add([path])
    print(f"{model.title}: обучена на {len(wells)} скважинах -> {path}, {data.weights_archive}")
    return 0


def run_pack_weights(args: argparse.Namespace) -> int:
    data = DataConfig()
    names = _weights_archive(data).pack()
    size = data.weights_archive.stat().st_size / 2**20
    print(f"В архив {data.weights_archive} ({size:.1f} МБ) записаны: {', '.join(names)}")
    return 0


def run_predict(args: argparse.Namespace) -> int:
    data = DataConfig()
    path = args.weights or _weights_archive(data).locate(args.model)
    model = load_model(path)
    wells = load_wells(args, labeled=False)
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


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    defaults = DataConfig()
    names = list(model_registry())
    parser = argparse.ArgumentParser(
        prog=prog or f"python -m {__package__}",
        description="ML-детекция мероприятий «увеличение частоты F»: Transformer и модели scikit-learn.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный лог")
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--dataset",
            choices=("merged", "test"),
            default=defaults.dataset,
            help="merged — статистика 2020–2026 и разметка, test — первые тестовые (resources/test_datas)",
        )
        sub.add_argument("--statistics", type=Path, default=defaults.statistics, help="merged: папка полугодий или склеенный файл")
        sub.add_argument("--labels", type=Path, default=defaults.labels, help="merged: разметка технологов")
        sub.add_argument(
            "--reference", type=Path, default=defaults.reference, help="merged: выгрузка для сверки склеенного файла"
        )
        sub.add_argument("--no-reference", action="store_true", help="merged: не сверять с выгрузкой")
        sub.add_argument("--measurements", type=Path, default=defaults.measurements, help="test: замеры")
        sub.add_argument("--results", type=Path, default=defaults.results, help="test: разметка")
        sub.add_argument(
            "--label-lookback",
            type=int,
            default=defaults.label_lookback,
            help="merged: дней ДО даты технолога, где ищется подъём частоты при отборе разметки",
        )
        sub.add_argument(
            "--label-lookahead",
            type=int,
            default=defaults.label_lookahead,
            help="merged: дней ПОСЛЕ даты технолога, где ищется подъём частоты",
        )
        sub.add_argument("--seed", type=int, default=0, help="зерно случайности")
        sub.add_argument(
            "--epochs", type=int, help=f"эпох обучения Transformer (по умолчанию {TrainingConfig().epochs})"
        )

    compare = commands.add_parser("compare", help="перекрёстная проверка и сравнение моделей")
    common(compare)
    compare.add_argument("--models", nargs="+", default=names, help=f"какие модели: {' '.join(names)}")
    compare.add_argument("--folds", type=int, default=5, help="фолдов по скважинам")
    compare.add_argument(
        "--out-dir", type=Path, help="папка отчёта (по умолчанию results/ml_freq/<дата_время>)"
    )
    compare.set_defaults(handler=run_compare)

    windows = commands.add_parser(
        "compare-windows",
        help="обучение на разметке с узким и широким окном отбора; проверка на одних мероприятиях",
    )
    common(windows)
    windows.set_defaults(
        label_lookback=defaults.wide_lookback, label_lookahead=defaults.wide_lookahead
    )
    windows.add_argument(
        "--models", nargs="+", default=names, help=f"какие модели: {' '.join(names)}"
    )
    windows.add_argument(
        "--label-start",
        choices=sorted(START_CHOICES),
        default="both",
        help="начало обучающей метки: tech — дата технолога, jump — день скачка, both — оба",
    )
    windows.add_argument("--folds", type=int, default=5, help="фолдов по скважинам")
    windows.add_argument(
        "--seeds", type=int, nargs="+", help="зёрна случайности (по умолчанию --seed)"
    )
    windows.add_argument("--out-dir", type=Path, help="папка отчёта")
    windows.set_defaults(handler=run_compare_windows)

    train = commands.add_parser("train", help="обучить модель на всей разметке и сохранить веса")
    common(train)
    train.add_argument("--model", choices=names, default="transformer")
    train.set_defaults(handler=run_train)

    predict = commands.add_parser("predict", help="найти мероприятия обученной моделью")
    common(predict)
    predict.add_argument("--model", choices=names, default="transformer")
    predict.add_argument("--weights", type=Path, help="файл весов (по умолчанию из архива weights.zip)")
    predict.add_argument("--out-dir", type=Path, help="папка результата")
    predict.set_defaults(handler=run_predict)

    pack = commands.add_parser("pack-weights", help="записать все веса из weights/ в zip-архив")
    pack.set_defaults(handler=run_pack_weights)

    audit = commands.add_parser("audit", help="отчёт о качестве объединённых данных 2021–2026")
    audit_arguments(audit, defaults)
    audit.set_defaults(handler=run_audit)
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
