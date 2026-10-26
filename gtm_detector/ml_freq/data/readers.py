"""Чтение Excel-файлов технологов.

* Замеры (``input_dataset.xlsx``): лист на параметр, строка — скважина,
  столбцы ``ДД.ММ.ГГГГ`` — суточные значения; состояние — лист «Прим».
* Разметка (``output_dataset.xlsm``): лист «Результат__*».
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import re
import tempfile
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .models import Interval, WellSeries

logger = logging.getLogger(__name__)

_LATIN_TO_CYRILLIC = str.maketrans("ABEKMHOPCTXabekmhopctx", "АВЕКМНОРСТХавекмнорстх")
_DATE_IN_TEXT = re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{2,4})")
_DATE_HEADER = re.compile(r"\d{2}\.\d{2}\.\d{4}")


def normalize_id(value: Any) -> str:
    """Номер скважины/куста к единому виду: ``'561E '`` -> ``'561Е'``, ``7782.0`` -> ``'7782'``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip().upper().translate(_LATIN_TO_CYRILLIC)


def parse_date(value: Any) -> date | None:
    """Дата из ячейки, в том числе с опечатками: ``'21.04.2026г'``, ``'08.02.206'``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (pd.Timestamp, date)):
        timestamp = pd.Timestamp(value)
    else:
        found = _DATE_IN_TEXT.search(str(value))
        if not found:
            return None
        day, month, year = map(int, found.groups())
        year = {206: 2026, 26: 2026}.get(year, year)
        try:
            timestamp = pd.Timestamp(year=year, month=month, day=day)
        except ValueError:
            return None
    if pd.isna(timestamp) or timestamp.year < 2000:
        return None
    return timestamp.date()


class MeasurementsReader:
    """Файл замеров -> список :class:`WellSeries` (без разметки)."""

    SHEETS = {
        "debit_liquid": "Qж",
        "watercut": "Обв",
        "debit_oil": "Qн",
        "pressure_line": "Pлин",
        "frequency": "Fэцн ТМ",
        "pressure_zab": "Рзаб",
    }
    STATE_SHEET = "Прим"
    WORKING_STATES = frozenset({"раб.", "нак"})

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def read(self) -> list[WellSeries]:
        tables = {name: self._read_parameter(sheet) for name, sheet in self.SHEETS.items()}
        states = self._read_states()
        frequency = tables["frequency"]
        calendar = pd.date_range(frequency.columns.min(), frequency.columns.max(), freq="D")

        wells = []
        for key in frequency.index:
            frame = pd.DataFrame(index=calendar)
            for name, table in tables.items():
                frame[name] = table.loc[key].reindex(calendar) if key in table.index else float("nan")
            if key in states.index:
                frame["state"] = states.loc[key].reindex(calendar).isin(self.WORKING_STATES).to_numpy()
            else:
                frame["state"] = (frame["frequency"] > 0).to_numpy()
            wells.append(WellSeries(number=key[0], cluster=key[1], frame=frame.astype(float)))
        logger.info("Замеры: %d скважин, %s — %s", len(wells), calendar[0].date(), calendar[-1].date())
        return wells

    def _read_parameter(self, sheet: str) -> pd.DataFrame:
        table = pd.read_excel(self.path, sheet_name=sheet, header=1).iloc[1:]
        dates = {}
        for column in table.columns:
            if isinstance(column, (pd.Timestamp, date)):
                dates[column] = pd.Timestamp(column)
            elif isinstance(column, str) and _DATE_HEADER.fullmatch(column.strip()):
                dates[column] = pd.to_datetime(column.strip(), format="%d.%m.%Y")
        values = table[list(dates)].apply(pd.to_numeric, errors="coerce")
        values.columns = pd.DatetimeIndex(list(dates.values()))
        values.index = pd.MultiIndex.from_arrays(
            [table["№ скв."].map(normalize_id), table["№ куста"].map(normalize_id)]
        )
        return values[~values.index.duplicated()]

    def _read_states(self) -> pd.DataFrame:
        sheet = pd.read_excel(self.path, sheet_name=self.STATE_SHEET, header=None)
        is_state = sheet[4] == "Состояние"
        info = sheet.shift(-1)[is_state][1].astype(str)  # строка «Прим»: «..., куст 2079, ...»
        clusters = info.str.extract(r"куст\s+([^,]+)", expand=False).map(normalize_id)
        values = sheet[is_state].iloc[:, 7:].apply(lambda column: column.astype(str).str.strip())
        values.columns = pd.to_datetime(sheet.iloc[1, 7:].values)
        values.index = pd.MultiIndex.from_arrays([sheet[is_state][3].map(normalize_id), clusters])
        return values[~values.index.duplicated()]


class LabelsReader:
    """Лист «Результат__*»: мероприятия «Увеличение F», успешные, с обеими датами мониторинга."""

    # Номер столбца (с нуля) -> имя. Шапка листа — 4 строки.
    COLUMNS = {3: "cluster", 4: "number", 26: "event", 35: "date_done", 39: "success", 41: "start", 42: "end"}

    def __init__(self, path: Path, sheet: str | None = None) -> None:
        self.path = Path(path)
        self.sheet = sheet

    def read(self) -> dict[tuple[str, str], list[Interval]]:
        sheet = self.sheet or next(
            name for name in pd.ExcelFile(self.path).sheet_names if name.startswith("Результат")
        )
        table = pd.read_excel(self.path, sheet_name=sheet, header=None, skiprows=4)
        table = table[list(self.COLUMNS)].rename(columns=self.COLUMNS)

        labels: dict[tuple[str, str], list[Interval]] = {}
        for row in table.itertuples(index=False):
            if str(row.event).strip().lower() != "увеличение f" or str(row.success).strip().lower() != "да":
                continue
            start, end = parse_date(row.start), parse_date(row.end)
            if start is None or end is None or end < start:
                continue
            key = (normalize_id(row.number), normalize_id(row.cluster))
            labels.setdefault(key, []).append(Interval(start, end, parse_date(row.date_done)))
        logger.info("Разметка: %d мероприятий", sum(map(len, labels.values())))
        return labels


class WellRepository:
    """Скважины с привязанной разметкой; результат чтения Excel кэшируется на диске."""

    def __init__(self, measurements: Path, results: Path | None, cache_dir: Path | None = None) -> None:
        self.measurements = Path(measurements)
        self.results = Path(results) if results else None
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "gtm_ml_freq_cache"

    def load(self) -> list[WellSeries]:
        cache = self.cache_dir / f"wells_{self._fingerprint()}.pkl"
        if cache.exists():
            logger.info("Данные из кэша: %s", cache)
            with cache.open("rb") as file:
                return pickle.load(file)

        wells = MeasurementsReader(self.measurements).read()
        if self.results:
            self._attach(wells, LabelsReader(self.results).read())
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with cache.open("wb") as file:
            pickle.dump(wells, file)
        return wells

    @staticmethod
    def _attach(wells: Iterable[WellSeries], labels: dict[tuple[str, str], list[Interval]]) -> None:
        by_key = {(well.number, well.cluster): well for well in wells}
        by_number: dict[str, list[WellSeries]] = {}
        for well in by_key.values():
            by_number.setdefault(well.number, []).append(well)
        lost = 0
        for (number, cluster), intervals in labels.items():
            well = by_key.get((number, cluster))
            if well is None and len(by_number.get(number, [])) == 1:
                well = by_number[number][0]  # куст записан иначе, но номер уникален
            if well is None:
                lost += len(intervals)
                continue
            well.labels.extend(intervals)
        if lost:
            logger.warning("Не найдено скважин для %d мероприятий разметки", lost)

    def _fingerprint(self) -> str:
        parts = []
        for path in (self.measurements, self.results):
            if path is not None:
                stat = path.stat()
                parts.append(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}")
        return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]
