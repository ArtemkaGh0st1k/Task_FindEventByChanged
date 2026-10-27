"""Скважины из объединённых файлов 2021–2026 и разбор качества этих данных.

Порядок:

1. статистика: только числовые ячейки (:class:`.MergedStatisticsReader`);
2. сдвинутые блоки возвращаются на место или исключаются (:class:`.BlockShiftRepair`);
3. скважины без телеметрии частоты не используются: подъём F на них не увидеть;
4. каждое мероприятие разметки проверяется по замерам; для обучения остаются
   те, где частота около даты начала действительно поднялась (:class:`.FrequencyCheck`).

Всё, что отброшено, и почему — в :class:`DataAudit` (отчёт: ``python -m gtm_detector.ml_freq audit``).
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .models import SERIES_COLUMNS, Interval, WellSeries
from .readers import MeasurementsReader
from .repair import BlockShiftRepair, FrequencyCheck, LabelPoint, Segment
from .statistics import LabelRecord, MergedStatisticsReader, MonitoringLabelsReader

logger = logging.getLogger(__name__)

ACCEPTED = "принято"


@dataclass
class DataAudit:
    """Что было в файлах и что из этого взято для обучения."""

    segments: list[Segment]
    labels: pd.DataFrame  # строка итогового файла + «решение»
    wells: pd.DataFrame  # скважина + «статус»
    non_numeric: pd.Series  # ячеек «не число» по датам (все листы)
    wells_per_day_before: pd.Series  # скважин с данными по дням, до исправления
    wells_per_day_after: pd.Series
    reference: Path | None


class MergedWellRepository:
    """Скважины с проверенной разметкой из объединённых файлов; результат кэшируется."""

    VERSION = 1  # менять при изменении логики — сбрасывает кэш
    MIN_TELEMETRY_SHARE = 0.1  # доля дней с замерами, где частота > 10 Гц
    DATA_WINDOW = 14  # дней вокруг даты начала, где должны быть замеры

    def __init__(
        self,
        statistics: Path,
        labels: Path | None,
        reference: Path | None = None,
        cache_dir: Path | None = None,
        repair: BlockShiftRepair | None = None,
    ) -> None:
        self.statistics = Path(statistics)
        self.labels = Path(labels) if labels else None
        self.reference = Path(reference) if reference and Path(reference).exists() else None
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "gtm_ml_freq_cache"
        self.repair = repair or BlockShiftRepair()
        self.check = self.repair.check

    def load(self) -> list[WellSeries]:
        return self.build()[0]

    def build(self) -> tuple[list[WellSeries], DataAudit]:
        cache = self.cache_dir / f"merged_{self._fingerprint()}.pkl"
        if cache.exists():
            logger.info("Данные из кэша: %s", cache)
            with cache.open("rb") as file:
                return pickle.load(file)
        result = self._build()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with cache.open("wb") as file:
            pickle.dump(result, file)
        return result

    def _build(self) -> tuple[list[WellSeries], DataAudit]:
        reader = MergedStatisticsReader(self.statistics)
        tables = reader.read()
        index = tables["frequency"].index
        for table in tables.values():
            index = index.union(table.index)
        dates = tables["frequency"].columns
        tables = {name: table.reindex(index=index, columns=dates) for name, table in tables.items()}
        presence = np.any([table.notna().to_numpy() for table in tables.values()], axis=0)

        records = MonitoringLabelsReader(self.labels).read() if self.labels else []
        decisions, rows = self._prefilter(records, index)
        points = [
            LabelPoint(rows[i], dates.get_loc(pd.Timestamp(r.start)), r.current, r.new)
            for i, r in enumerate(records)
            if decisions[i] is None and pd.Timestamp(r.start) in dates
        ]

        reference, observed = self._reference(index, dates), tables["debit_liquid"].to_numpy()
        segments = self.repair.segments(presence, dates)
        self.repair.assess(segments, presence, tables["frequency"].to_numpy(), points, reference, observed)
        repaired = self.repair.apply(tables, segments)
        for segment in segments:
            logger.info(
                "Сегмент %s — %s: %s (%s)", segment.start, segment.end, segment.decision, segment.verdict
            )

        frequency = repaired["frequency"].to_numpy()
        has_data = np.any([table.notna().to_numpy() for table in repaired.values()], axis=0)
        telemetry = self._telemetry_share(frequency, has_data)
        dropped_day = self._dropped_days(segments, presence)

        for i, record in enumerate(records):
            if decisions[i] is None:
                decisions[i] = self._verify(
                    record, rows[i], dates, frequency, has_data, telemetry, dropped_day
                )

        wells, well_rows = self._wells(repaired, index, has_data, telemetry, records, rows, decisions)
        audit = DataAudit(
            segments=segments,
            labels=self._labels_table(records, decisions),
            wells=pd.DataFrame(well_rows),
            non_numeric=reader.non_numeric,
            wells_per_day_before=pd.Series(presence.sum(axis=0), index=dates),
            wells_per_day_after=pd.Series(has_data.sum(axis=0), index=dates),
            reference=self.reference,
        )
        logger.info(
            "Скважин: %d, с разметкой: %d, мероприятий: %d",
            len(wells),
            sum(1 for w in wells if w.labels),
            sum(len(w.labels) for w in wells),
        )
        return wells, audit

    @staticmethod
    def _prefilter(
        records: Sequence[LabelRecord], index: pd.MultiIndex
    ) -> tuple[list[str | None], list[int | None]]:
        """Причины, видные без замеров, и строка скважины в таблицах для каждой записи."""
        position = {key: i for i, key in enumerate(index)}
        by_number: dict[str, list[int]] = {}
        for i, (number, _) in enumerate(index):
            by_number.setdefault(number, []).append(i)

        decisions: list[str | None] = []
        rows: list[int | None] = []
        seen = set()
        for record in records:
            row = position.get(record.key)
            if row is None and len(by_number.get(record.number, [])) == 1:
                row = by_number[record.number][0]  # куст записан иначе, но номер уникален
            rows.append(row)
            reason = MonitoringLabelsReader.rejection(record)
            if reason is None and (record.number, record.start) in seen:
                reason = "дубликат (та же скважина и дата начала)"
            if reason is None and row is None:
                reason = "скважины нет в статистике"
            if reason is None:
                seen.add((record.number, record.start))
            decisions.append(reason)
        return decisions, rows

    def _verify(
        self,
        record: LabelRecord,
        row: int,
        dates: pd.DatetimeIndex,
        frequency: np.ndarray,
        has_data: np.ndarray,
        telemetry: np.ndarray,
        dropped_day: np.ndarray,
    ) -> str:
        if pd.Timestamp(record.start) not in dates:
            return "дата начала вне календаря статистики"
        day = dates.get_loc(pd.Timestamp(record.start))
        window = slice(max(day - self.DATA_WINDOW, 0), day + self.DATA_WINDOW + 1)
        if not has_data[row, window].any():
            if dropped_day[row, window].any():
                return "замеры около даты начала — в исключённом сегменте"
            return "нет замеров около даты начала"
        if telemetry[row] < self.MIN_TELEMETRY_SHARE:
            return "нет телеметрии частоты у скважины"
        result = self.check.confirms(frequency[row], day, record.current, record.new)
        if result is None:
            return "нет замеров частоты около даты начала"
        return ACCEPTED if result else "частота в замерах не поднялась около даты начала"

    def _wells(
        self,
        repaired: dict[str, pd.DataFrame],
        index: pd.MultiIndex,
        has_data: np.ndarray,
        telemetry: np.ndarray,
        records: Sequence[LabelRecord],
        rows: Sequence[int | None],
        decisions: Sequence[str],
    ) -> tuple[list[WellSeries], list[dict]]:
        labels: dict[int, list[Interval]] = {}
        for record, row, decision in zip(records, rows, decisions, strict=True):
            if decision == ACCEPTED:
                labels.setdefault(row, []).append(
                    Interval(record.start, record.end, censored=record.censored)
                )

        dates = repaired["frequency"].columns
        wells, table = [], []
        for row, (number, cluster) in enumerate(index):
            days = np.flatnonzero(has_data[row])
            if not len(days):
                status = "нет данных после исправления"
            elif telemetry[row] < self.MIN_TELEMETRY_SHARE:
                status = "не используется: нет телеметрии частоты"
            else:
                status = "используется, с разметкой" if row in labels else "используется, без разметки"
            table.append(
                {
                    "скважина": number,
                    "куст": cluster,
                    "дней с замерами": len(days),
                    "доля дней с частотой > 10 Гц": round(float(telemetry[row]), 3),
                    "мероприятий принято": len(labels.get(row, [])),
                    "статус": status,
                }
            )
            if not status.startswith("используется"):
                continue
            span = slice(days[0], days[-1] + 1)
            frame = pd.DataFrame(
                {name: repaired[name].iloc[row, span].to_numpy() for name in SERIES_COLUMNS},
                index=dates[span],
            )
            frame["state"] = (frame["frequency"] > 0).astype(float)
            intervals = sorted(labels.get(row, []), key=lambda interval: interval.start)
            wells.append(WellSeries(number=number, cluster=cluster, frame=frame, labels=intervals))
        return wells, table

    def _reference(self, index: pd.MultiIndex, dates: pd.DatetimeIndex) -> np.ndarray | None:
        """Исходная выгрузка (Qж) в координатах склеенного файла — только для проверки сдвигов."""
        if self.reference is None:
            return None
        table = MeasurementsReader(self.reference).read_parameter("Qж")
        return table.reindex(index=index, columns=dates).to_numpy(dtype=float)

    @staticmethod
    def _telemetry_share(frequency: np.ndarray, has_data: np.ndarray) -> np.ndarray:
        working = np.nan_to_num(frequency) > FrequencyCheck.MIN_WORKING
        days = has_data.sum(axis=1)
        return np.divide((working & has_data).sum(axis=1), days, out=np.zeros(len(days)), where=days > 0)

    @staticmethod
    def _dropped_days(segments: Sequence[Segment], presence: np.ndarray) -> np.ndarray:
        dropped = np.zeros_like(presence)
        for segment in segments:
            if segment.shift is None:
                span = slice(segment.first, segment.last + 1)
                dropped[:, span] = presence[:, span]
        return dropped

    @staticmethod
    def _labels_table(records: Sequence[LabelRecord], decisions: Sequence[str]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "строка Excel": r.row,
                    "скважина": r.number,
                    "куст": r.cluster,
                    "год": r.year,
                    "тип ГТМ": r.kind,
                    "начало": r.start,
                    "окончание": r.end,
                    "частота до, Гц": r.current,
                    "частота после, Гц": r.new,
                    "причина окончания": r.reason,
                    "эффект продолжался": r.censored,
                    "решение": decision,
                }
                for r, decision in zip(records, decisions, strict=True)
            ]
        )

    def _fingerprint(self) -> str:
        parts = [f"v{self.VERSION}", repr(self.repair)]
        for path in (self.statistics, self.labels, self.reference):
            if path is not None:
                stat = path.stat()
                parts.append(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}")
        return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]
