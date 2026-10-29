"""Отчёт о качестве объединённых данных 2021–2026: что исправлено, что исключено и почему.

Запуск из корня проекта::

    python -m gtm_detector.ml_freq.audit            # отчёт в консоль + Excel и график
    python -m gtm_detector.ml_freq audit            # то же через общую командную строку

Что не так со статистикой
-------------------------
Файл «Статистика_с_2021_2026_в_одном_файле.xlsx» склеен макросом из отдельных
выгрузок. При склейке:

1. **Пустые столбцы в формате даты.** 01.01–29.06.2020 и 28–31.12.2026 замеров
   нет, но ячейки отформатированы как даты: pandas читает их как NaT, а наивное
   ``pd.to_numeric`` превращает NaT в −9.2·10¹⁸. Читатель берёт только настоящие
   числа, поэтому это просто пропуски; ячейки с текстом или датой вместо числа
   были бы посчитаны в разделе 1 отчёта.
2. **Сдвиг на 180 дней.** Часть выгрузок записана на 180 столбцов правее, то есть
   на 180 дней позже. Нагляднее всего это видно в 2026 г.: данные 343 скважин
   стоят на 30.06–27.12.2026, хотя ноябрь–декабрь ещё не наступили. Сверка с
   исходной выгрузкой (``resources/test_datas/input_dataset.xlsx``) показывает:

   * 30.06.2026 и 04.09–27.12.2026 — это 01.01.2026 и 08.03–30.06.2026 со сдвигом
     +180 дн (совпадают 343 скважины из 343);
   * 01.07–03.09.2026 — настоящие июль–август, на своём месте; при склейке они
     затёрли начало сдвинутого блока, поэтому 02.01–07.03.2026 этих скважин нет.

   Тот же сдвиг есть в 2022–2025 гг.: подъём частоты из разметки технологов
   находится не на дате начала, а на 180 дней позже.

Как это исправляется (:mod:`gtm_detector.ml_freq.data.repair`)
--------------------------------------------------------------
Календарь делится на **сегменты** — периоды, где набор скважин с данными не
меняется (след одной выгрузки). Для каждого сегмента проверяются сдвиги 0 и 180
дней:

* **исходная выгрузка** (где есть): скважина совпала, если равны ≥ 90 % ячеек;
  сдвиг принят, если совпала хотя бы половина скважин;
* иначе **разметка**: сдвиг принят, если частота поднимается у ≥ 3 мероприятий
  и это ≥ 25 % проверенных;
* нечем проверить (нет мероприятий) — сегмент остаётся на месте; стык выгрузок
  (1–2 дня) получает решение соседнего сегмента;
* доказательства есть, но не подтверждают ни 0, ни 180 дней — **сегмент
  исключается**: где на самом деле стоят эти данные, установить нельзя.

Почему не исключать скважины целиком
-------------------------------------
Сдвиг — свойство выгрузки, а не скважины: у одной скважины одни годы стоят
верно, другие сдвинуты. Исключение по сегментам сохраняет верные годы.
Целиком не используются только скважины без телеметрии частоты (частота > 10 Гц
меньше чем в 10 % дней с замерами): подъём частоты на них не увидеть.

Какие мероприятия разметки идут в обучение
------------------------------------------
Только «Лист1» итогового файла (сводный список), только «Изменение частоты, Гц»
и «Увеличение F», успешные, выполненные, с обеими датами и с «Новым значением»
больше «Текущего». Затем по замерам после исправления: около даты начала
(±14 дней) есть данные, и частота действительно поднялась (± 3 дня от даты:
медиана за неделю после выше медианы за неделю до и дошла до «Нового значения»
или дала половину запланированного прироста). Мероприятие, которое замеры не
подтверждают, вредит обучению: модель училась бы находить подъём там, где его нет.

Если причина окончания «ПРОДОЛЖАЕТСЯ ЭФФЕКТ», дата окончания — конец
наблюдения, а не конец эффекта: при обучении после неё дни не используются,
при оценке ошибка даты окончания не считается.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .config import CONTEXT_DAYS, DataConfig  # noqa: E402
from .data import DataAudit, MergedWellRepository, WellSeries  # noqa: E402
from .data.dataset import ACCEPTED  # noqa: E402
from .data.statistics import MergedStatisticsReader  # noqa: E402

# Главные причины отказа -> пояснение для технологов (файл rejected_labels.csv).
REJECTED = {
    "скважины нет в статистике": "скважины с таким номером/кустом нет в файле статистики",
    "нет замеров около даты начала": "у скважины нет замеров за ±14 дней от даты начала",
    "частота в замерах не поднялась около даты начала": (
        "в ±3 дня от даты начала частота не выросла: ни до «Нового значения», "
        "ни на половину запланированного прироста"
    ),
}

# Листы training_data.xlsx — с именами и в порядке исходного файла статистики.
STATISTICS_SHEETS = {
    "debit_liquid": "Qж (из OIS)",
    "watercut": "Обв (из OIS)",
    "debit_oil": "Qн (из OIS)",
    "pressure_line": "Рлин (из OIS)",
    "frequency": "Fэцн ТМ(сут)",
    "pressure_zab": "Рзаб",
}

# Столбцы разметки -> заголовки итогового файла технологов.
LABEL_HEADERS = {
    "куст": "Куст",
    "скважина": "Скважина",
    "тип ГТМ": "Тип ГТМ",
    "начало": "дата начала мониторинга",
    "окончание": "дата окончания мониторинга",
    "частота до, Гц": "Текущее значение",
    "частота после, Гц": "Новое значение",
    "причина окончания": "Причина прекращения мониторинга",
    "год": "Год",
}

# Цвета: линии — нейтральный и первый категориальный; заливка решений — светлые ступени.
COLORS = {"before": "#9a9993", "after": "#2a78d6", "shifted": "#cde2fb", "dropped": "#fbd9cb"}


class DataAuditReport:
    """Текст, Excel и график по :class:`DataAudit`."""

    def __init__(self, audit: DataAudit) -> None:
        self.audit = audit

    def segments_table(self) -> pd.DataFrame:
        rows = []
        for s in self.audit.segments:
            rows.append(
                {
                    "начало": s.start,
                    "конец": s.end,
                    "дней": s.days,
                    "скважин": s.wells,
                    "ячеек с данными": s.cells,
                    "разметка, сдвиг 0": _evidence(s.labels.get(0)),
                    "разметка, сдвиг 180": _evidence(s.labels.get(180)),
                    "выгрузка, сдвиг 0": _evidence(s.reference.get(0)),
                    "выгрузка, сдвиг 180": _evidence(s.reference.get(180)),
                    "решение": s.decision,
                    "обоснование": s.verdict,
                    "наложений (оставлены прежние)": s.collisions,
                }
            )
        return pd.DataFrame(rows)

    def non_numeric_by_month(self) -> pd.Series:
        series = self.audit.non_numeric
        monthly = series.groupby(series.index.to_period("M")).sum()
        return monthly[monthly > 0]

    def to_text(self) -> str:
        a = self.audit
        segments = self.segments_table()
        labels, wells = a.labels, a.wells
        accepted = labels[labels["решение"] == ACCEPTED]
        cells = {
            d: segments.loc[segments["решение"] == d, "ячеек с данными"].sum() for d in segments["решение"]
        }

        lines = [
            "=" * 100,
            "1. Заполненные ячейки не с числом (текст, дата) — считаются пропусками",
            "=" * 100,
        ]
        monthly = self.non_numeric_by_month()
        lines += [f"   {period}: {count:>8,}".replace(",", " ") for period, count in monthly.items()]
        if monthly.empty:
            lines.append(
                "   нет: все заполненные ячейки — числа (пустые столбцы 2020 и конца 2026 г. — пропуски)"
            )

        lines += ["", "=" * 100, "2. Сегменты (след отдельных выгрузок) и решения", "=" * 100]
        lines.append(
            "   разметка «a/b» — подтверждено a мероприятий из b проверенных; "
            "выгрузка «a/b» — совпало a скважин из b"
        )
        lines.append(_indent(segments.drop(columns=["ячеек с данными"]).to_string(index=False)))
        lines.append("")
        total = sum(cells.values())
        for decision, count in sorted(cells.items(), key=lambda item: -item[1]):
            lines.append(f"   {decision:<16} {count:>9,} ячеек ({count / total:.0%})".replace(",", " "))
        if a.reference is not None:
            lines.append(f"   исходная выгрузка для сверки: {a.reference}")

        lines += ["", "=" * 100, "3. Мероприятия итогового файла («Лист1»)", "=" * 100]
        funnel = labels["решение"].value_counts()
        for reason, count in funnel.items():
            lines.append(f"   {count:>5}  {reason}")
        lines.append(f"   {len(labels):>5}  всего строк")
        lines.append("")
        by_year = pd.crosstab(labels["год"], labels["решение"] == ACCEPTED).rename(
            columns={True: "принято", False: "не принято"}
        )
        lines.append(_indent(by_year.to_string()))
        censored = int(accepted["эффект продолжался"].sum())
        lines.append(f"\n   из принятых «эффект продолжался» (конец не известен): {censored}")

        lines += ["", "=" * 100, "4. Скважины статистики", "=" * 100]
        for status, count in wells["статус"].value_counts().items():
            lines.append(f"   {count:>5}  {status}")

        lines += ["", "=" * 100, "ИТОГ ДЛЯ ОБУЧЕНИЯ", "=" * 100]
        used = wells[wells["статус"] == "используется, с разметкой"]
        lines.append(f"   скважин с разметкой: {len(used)}, мероприятий: {len(accepted)}")
        lines.append(
            f"   скважин без разметки (только для поиска мероприятий): "
            f"{(wells['статус'] == 'используется, без разметки').sum()}"
        )
        return "\n".join(lines)

    def rejected_table(self) -> pd.DataFrame:
        """Мероприятия, отброшенные по трём главным причинам, — для уточнения у технологов."""
        table = self.audit.labels[self.audit.labels["решение"].isin(REJECTED)].copy()
        table["пояснение"] = table["решение"].map(REJECTED)
        table = table.rename(columns={"решение": "причина"})
        return table.sort_values(["причина", "год", "скважина"])

    def save(self, out_dir: Path) -> tuple[Path, Path, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv = out_dir / "rejected_labels.csv"
        # «;» и BOM — чтобы Excel с русской локалью открыл файл без настройки импорта.
        self.rejected_table().to_csv(csv, sep=";", index=False, encoding="utf-8-sig")
        xlsx = out_dir / "data_audit.xlsx"
        with pd.ExcelWriter(xlsx) as writer:
            self.segments_table().to_excel(writer, sheet_name="сегменты", index=False)
            self.audit.labels.to_excel(writer, sheet_name="мероприятия", index=False)
            self.audit.wells.to_excel(writer, sheet_name="скважины", index=False)
            self.non_numeric_by_month().rename("ячеек «не число»").to_frame().to_excel(
                writer, sheet_name="не числа"
            )
        png = out_dir / "data_audit.png"
        self.plot(png)
        return xlsx, png, csv

    def plot(self, path: Path) -> None:
        a = self.audit
        figure, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, sharey=True)
        panels = (
            (axes[0], a.wells_per_day_before, COLORS["before"], "До исправления (как в файле)"),
            (axes[1], a.wells_per_day_after, COLORS["after"], "После исправления"),
        )
        for axis, series, color, title in panels:
            for s in a.segments:
                if s.shift is None or s.shift:
                    fill = COLORS["dropped"] if s.shift is None else COLORS["shifted"]
                    if axis is axes[0]:
                        axis.axvspan(pd.Timestamp(s.start), pd.Timestamp(s.end), color=fill, lw=0)
            axis.plot(series.index, series.to_numpy(), color=color, lw=1.5)
            axis.set_title(title, loc="left", fontsize=11)
            axis.set_ylabel("скважин с замерами")
            axis.grid(axis="y", color="#e6e5e0", lw=0.8)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=COLORS["shifted"], label="сегмент сдвинут на −180 дней"),
            plt.Rectangle((0, 0), 1, 1, color=COLORS["dropped"], label="сегмент исключён"),
        ]
        axes[0].legend(handles=handles, loc="upper left", frameon=False)
        figure.suptitle("Статистика 2020–2026: скважин с замерами по дням", x=0.01, ha="left", fontsize=13)
        figure.tight_layout()
        figure.savefig(path, dpi=130)
        plt.close(figure)


class TrainingDataReport:
    """Данные, на которых учатся модели: принятая разметка и исправленные замеры.

    Замеры — после исправления сдвига на 180 дней и исключения сегментов, только по
    скважинам с принятой разметкой, в раскладке исходного файла статистики:
    лист на параметр, строка — скважина, столбец — дата.
    """

    def __init__(
        self, wells: Sequence[WellSeries], audit: DataAudit, context_days: int = CONTEXT_DAYS
    ) -> None:
        self.wells = [well for well in wells if well.labels]
        self.audit = audit
        self.context_days = context_days
        start = min(well.frame.index[0] for well in self.wells)
        end = max(well.frame.index[-1] for well in self.wells)
        self.dates = pd.date_range(start, end)

    def labels_table(self) -> pd.DataFrame:
        table = self.audit.labels[self.audit.labels["решение"] == ACCEPTED]
        columns = ["строка Excel", *LABEL_HEADERS, "эффект продолжался"]
        table = table[columns].rename(columns=LABEL_HEADERS)
        return table.sort_values(["Скважина", "дата начала мониторинга"])

    def wide(self, rows: Sequence[pd.Series]) -> pd.DataFrame:
        """Ряды скважин -> таблица «скважина × дата» с заголовками как в исходном файле."""
        values = [row.reindex(self.dates).to_numpy() for row in rows]
        table = pd.DataFrame(values, columns=self.dates)
        table.columns = [day.strftime("%d.%m.%Y") for day in self.dates]
        ids = pd.DataFrame(
            {
                MergedStatisticsReader.NUMBER: [well.number for well in self.wells],
                MergedStatisticsReader.CLUSTER: [well.cluster for well in self.wells],
            }
        )
        return pd.concat([ids, table], axis=1)

    def target_table(self) -> pd.DataFrame:
        """1 — день эффекта по разметке, 0 — эффекта нет; пусто — день не участвует в обучении."""
        rows = []
        for well in self.wells:
            target = pd.Series(well.target(), index=well.frame.index)
            rows.append(target.where(well.training_mask(self.context_days)))
        return self.wide(rows)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        freeze = (1, 2)  # заголовок и номера скважины/куста видны при прокрутке
        with pd.ExcelWriter(path) as writer:
            self.labels_table().to_excel(writer, sheet_name="мероприятия", index=False)
            for column, sheet in STATISTICS_SHEETS.items():
                table = self.wide([well.frame[column] for well in self.wells])
                table.to_excel(writer, sheet_name=sheet, index=False, freeze_panes=freeze)
            self.target_table().to_excel(
                writer, sheet_name="разметка по дням", index=False, freeze_panes=freeze
            )
        return path


def _evidence(e) -> str:
    return f"{e.confirmed}/{e.tested}" if e is not None and e.tested else "—"


def _indent(text: str) -> str:
    return "\n".join("   " + line for line in text.splitlines())


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    defaults = DataConfig()
    parser = argparse.ArgumentParser(
        prog=prog or "python -m gtm_detector.ml_freq.audit",
        description="Отчёт о качестве объединённых данных 2021–2026.",
    )
    add_arguments(parser, defaults)
    return parser


def add_arguments(parser: argparse.ArgumentParser, defaults: DataConfig) -> None:
    parser.add_argument("--statistics", type=Path, default=defaults.statistics, help="склеенная статистика")
    parser.add_argument("--labels", type=Path, default=defaults.labels, help="итоговый файл технологов")
    parser.add_argument(
        "--reference", type=Path, default=defaults.reference, help="исходная выгрузка для сверки сдвигов"
    )
    parser.add_argument("--no-reference", action="store_true", help="не сверять с исходной выгрузкой")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=defaults.output_dir / "data_audit",
        help="куда сохранить Excel и график",
    )


def run(args: argparse.Namespace) -> int:
    reference = None if args.no_reference else args.reference
    wells, audit = MergedWellRepository(args.statistics, args.labels, reference).build()
    report = DataAuditReport(audit)
    print(report.to_text())
    xlsx, png, csv = report.save(args.out_dir)
    training = TrainingDataReport(wells, audit).save(args.out_dir / "training_data.xlsx")
    print(f"\nТаблицы: {xlsx}\nГрафик:  {png}\nОтброшенные мероприятия: {csv}")
    print(f"Данные для обучения: {training}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
