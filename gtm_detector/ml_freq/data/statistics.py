"""Чтение статистики и разметки 2020–2026.

* Статистика по полугодиям (``resources/statistics/<год>/*.xls``, 14 файлов): лист на
  параметр, строка — скважина, столбцы — даты периода. Основной источник
  (:class:`HalfYearStatisticsReader`): файлы исходные, сдвигов в них нет.
* Склеенная статистика (один ``.xlsx``, те же листы, даты 2020–2026): прежний источник
  (:class:`MergedStatisticsReader`). Склеена макросом, часть блоков записана со сдвигом —
  их чинит :mod:`.repair`.
* Разметка (``labeled_data.xlsx``): лист «Лист1», строка — мероприятие.
"""

from __future__ import annotations

import logging
import re
import struct
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xlrd

from .readers import normalize_id, parse_date

logger = logging.getLogger(__name__)

_DATE_HEADER = re.compile(r"\d{2}\.\d{2}\.\d{4}")


class MergedStatisticsReader:
    """Файл статистики -> таблицы «скважина × дата» по параметрам.

    В ячейку годится только число. Пустые столбцы в формате даты (2020 г., конец
    2026 г.) pandas читает как NaT — это пропуски, а не числа. Заполненные ячейки
    с текстом или датой тоже пропуски; их число по датам — в :attr:`non_numeric`.
    """

    # Имя параметра -> начало имени листа (листы: «Qж (из OIS)», «Fэцн ТМ(сут)», …).
    SHEETS = {
        "debit_liquid": "Qж",
        "watercut": "Обв",
        "debit_oil": "Qн",
        "pressure_line": "Рлин",
        "frequency": "Fэцн",
        "pressure_zab": "Рзаб",
    }
    HEADER_ROW = 1  # строка с названиями столбцов; данные — со строки HEADER_ROW + 2
    NUMBER, CLUSTER = "№ скв.", "№ куста"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.non_numeric = pd.Series(dtype=int)

    def read(self) -> dict[str, pd.DataFrame]:
        sheets = pd.ExcelFile(self.path).sheet_names
        tables, garbage = {}, []
        for name, prefix in self.SHEETS.items():
            sheet = next(s for s in sheets if normalize_id(s).startswith(normalize_id(prefix)))
            tables[name], bad = self._read_sheet(sheet)
            garbage.append(bad)
        self.non_numeric = pd.concat(garbage, axis=1).sum(axis=1)
        logger.info("Статистика: %d скважин, %d дат", len(tables["frequency"]), tables["frequency"].shape[1])
        return tables

    def _read_sheet(self, sheet: str) -> tuple[pd.DataFrame, pd.Series]:
        raw = pd.read_excel(self.path, sheet_name=sheet, header=None)
        header = raw.iloc[self.HEADER_ROW]
        body = raw.iloc[self.HEADER_ROW + 2 :]
        columns = {i: _header_date(value) for i, value in header.items()}
        columns = {i: day for i, day in columns.items() if day is not None}

        values = pd.DataFrame({day: _numbers(body[i]) for i, day in columns.items()})
        values.index = pd.MultiIndex.from_arrays(
            [
                body[_position(header, self.NUMBER)].map(normalize_id).to_numpy(),
                body[_position(header, self.CLUSTER)].map(normalize_id).to_numpy(),
            ]
        )
        filled = pd.DataFrame({day: body[i].notna().to_numpy() for i, day in columns.items()})
        bad = (filled.to_numpy() & values.isna().to_numpy()).sum(axis=0)

        keep = (values.index.get_level_values(0) != "") & ~values.index.duplicated()
        return values[keep], pd.Series(bad, index=values.columns)


class HalfYearStatisticsReader:
    """Папка с полугодовыми файлами ``.xls`` -> таблицы «скважина × дата» по параметрам.

    Результат — как у :class:`MergedStatisticsReader`: столбцы — непрерывный календарь
    от первой до последней даты файлов, скважины — объединение по всем файлам.
    """

    FILES = "*/*.xls"  # <год>/<файл>.xls

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.non_numeric = pd.Series(dtype=int)

    def files(self) -> list[Path]:
        return sorted(self.directory.glob(self.FILES))

    def read(self) -> dict[str, pd.DataFrame]:
        files = self.files()
        if not files:
            raise FileNotFoundError(f"Нет файлов статистики {self.FILES} в {self.directory}")
        sheets = MergedStatisticsReader.SHEETS
        parts: dict[str, list[pd.DataFrame]] = {name: [] for name in sheets}
        garbage = []
        for path in files:
            book = xlrd.open_workbook(file_contents=_workbook_stream(path), on_demand=True)
            for name, prefix in sheets.items():
                title = next(
                    s for s in book.sheet_names() if normalize_id(s).startswith(normalize_id(prefix))
                )
                table, bad = _sheet_table(book.sheet_by_name(title))
                parts[name].append(table)
                garbage.append(bad)
            book.release_resources()
        tables = {name: _join_periods(frames) for name, frames in parts.items()}
        self.non_numeric = pd.concat(garbage).groupby(level=0).sum()
        logger.info(
            "Статистика: %d файлов, %d скважин, %d дат",
            len(files),
            len(tables["frequency"]),
            tables["frequency"].shape[1],
        )
        return tables


def _workbook_stream(path: Path) -> bytes:
    """Поток ``Workbook`` из файла OLE2 (старый формат Excel), без разбора контейнера в xlrd.

    Выгрузки сохранены не самим Excel: цепочка секторов потока длиннее, чем размер в
    каталоге файла, и xlrd отказывается их открывать («size exceeds expected»). Лишнее —
    только заполнение последних секторов; саму книгу xlrd читает из потока как обычно.
    """
    data = Path(path).read_bytes()
    sector = 1 << struct.unpack_from("<H", data, 30)[0]

    def ints(sid: int) -> tuple[int, ...]:
        return struct.unpack_from(f"<{sector // 4}i", data, (sid + 1) * sector)

    fat_count, directory = struct.unpack_from("<ii", data, 44)
    extra = struct.unpack_from("<i", data, 68)[0]
    fat_sectors = list(struct.unpack_from("<109i", data, 76))
    while extra >= 0:  # больше 109 секторов таблицы: продолжение списка в отдельных секторах
        values = ints(extra)
        fat_sectors.extend(values[:-1])
        extra = values[-1]
    fat = [link for sid in fat_sectors[:fat_count] for link in ints(sid)]

    for sid in _chain(fat, directory):
        for offset in range(0, sector, 128):
            entry = data[(sid + 1) * sector + offset :][:128]
            length = struct.unpack_from("<H", entry, 64)[0]
            if entry[: max(length - 2, 0)].decode("utf-16-le") == "Workbook":
                start = struct.unpack_from("<i", entry, 116)[0]
                return b"".join(data[(s + 1) * sector :][:sector] for s in _chain(fat, start))
    raise ValueError(f"В файле {path} нет потока Workbook")


def _chain(fat: list[int], start: int) -> list[int]:
    """Номера секторов потока: отрицательная ссылка — конец цепочки."""
    chain = []
    while start >= 0:
        chain.append(start)
        start = fat[start]
    return chain


def _sheet_table(sheet) -> tuple[pd.DataFrame, pd.Series]:
    """Лист xlrd -> таблица «скважина × дата» и число не-чисел по датам."""
    header = pd.Series(sheet.row_values(MergedStatisticsReader.HEADER_ROW))
    number = _position(header, MergedStatisticsReader.NUMBER)
    cluster = _position(header, MergedStatisticsReader.CLUSTER)
    columns = {i: _header_date(value) for i, value in header.items()}
    columns = {i: day for i, day in columns.items() if day is not None}
    positions = list(columns)

    index, rows = [], []
    bad = np.zeros(len(positions), dtype=int)
    for r in range(MergedStatisticsReader.HEADER_ROW + 2, sheet.nrows):
        values, types = sheet.row_values(r), sheet.row_types(r)
        well = normalize_id(values[number])
        if not well:
            continue
        kinds = np.array([types[i] for i in positions])
        numeric = kinds == xlrd.XL_CELL_NUMBER
        bad += ~numeric & ~np.isin(kinds, (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK))
        row = np.full(len(positions), np.nan)
        row[numeric] = [values[i] for i, ok in zip(positions, numeric, strict=True) if ok]
        index.append((well, normalize_id(values[cluster])))
        rows.append(row)
    days = list(columns.values())
    table = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(index), columns=days, dtype=float)
    return table[~table.index.duplicated()], pd.Series(bad, index=days)


def _join_periods(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Таблицы периодов -> одна на непрерывном календаре; при наложении дат верен поздний файл."""
    table = pd.concat(frames, axis=1, sort=True)
    table = table.loc[:, ~table.columns.duplicated(keep="last")]
    return table.reindex(columns=pd.date_range(table.columns.min(), table.columns.max()))


def _header_date(value) -> pd.Timestamp | None:
    if isinstance(value, (pd.Timestamp, date)) and not pd.isna(value):
        return pd.Timestamp(value)
    if isinstance(value, str) and _DATE_HEADER.fullmatch(value.strip()):
        return pd.to_datetime(value.strip(), format="%d.%m.%Y")
    return None


def _position(header: pd.Series, title: str) -> int:
    return next(i for i, value in header.items() if str(value).strip() == title)


def _numbers(column: pd.Series) -> np.ndarray:
    """Только настоящие числа: даты, NaT, текст и bool -> NaN."""
    if pd.api.types.is_bool_dtype(column) or pd.api.types.is_datetime64_any_dtype(column):
        return np.full(len(column), np.nan)
    if pd.api.types.is_numeric_dtype(column):
        return column.to_numpy(dtype=float)
    return column.map(_number).to_numpy(dtype=float)


def _number(value) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        return np.nan
    return float(value)


@dataclass(frozen=True)
class LabelRecord:
    """Строка итогового файла технологов."""

    row: int  # номер строки в Excel (с шапкой)
    number: str
    cluster: str
    kind: str
    status: str
    success: str
    reason: str
    start: date | None
    end: date | None
    current: float  # частота до мероприятия, Гц
    new: float  # частота после мероприятия, Гц
    year: int | None

    @property
    def key(self) -> tuple[str, str]:
        return self.number, self.cluster

    @property
    def censored(self) -> bool:
        """Мониторинг остановлен, а эффект продолжался — ``end`` не конец эффекта."""
        return "продолжается" in self.reason.lower()


class MonitoringLabelsReader:
    """Итоговый файл технологов: мероприятия по изменению частоты.

    Берётся только «Лист1» — сводный список. «Лист2» — частичная копия 2025 г.:
    все его строки, которых нет в «Лист1» (кроме одной пустой), — тип «Изменение
    ЦЭС/Частоты на увеличение», который технологи в сводный список не включили.
    """

    SHEET = "Лист1"
    EVENT_TYPES = frozenset({"изменение частоты, гц", "увеличение f"})
    COLUMNS = {
        "Куст": "cluster",
        "Скважина": "number",
        "Тип ГТМ": "kind",
        "Статус ГТМ": "status",
        "Успешно/Неуспешно": "success",
        "Причина прекращения мониторинга": "reason",
        "дата начала мониторинга": "start",
        "дата окончания мониторинга": "end",
        "Текущее значение": "current",
        "Новое значение": "new",
        "Год": "year",
    }

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def read(self) -> list[LabelRecord]:
        table = pd.read_excel(self.path, sheet_name=self.SHEET)
        table = table[list(self.COLUMNS)].rename(columns=self.COLUMNS)
        records = [
            LabelRecord(
                row=i + 2,
                number=normalize_id(row.number),
                cluster=normalize_id(row.cluster),
                kind=_text(row.kind),
                status=_text(row.status),
                success=_text(row.success),
                reason=_text(row.reason),
                start=parse_date(row.start),
                end=parse_date(row.end),
                current=pd.to_numeric(row.current, errors="coerce"),
                new=pd.to_numeric(row.new, errors="coerce"),
                year=int(row.year) if pd.notna(row.year) else None,
            )
            for i, row in enumerate(table.itertuples(index=False))
        ]
        logger.info("Итоговый файл: %d строк", len(records))
        return records

    @classmethod
    def rejection(cls, record: LabelRecord) -> str | None:
        """Причина не брать строку, видная без данных замеров; ``None`` — строка годится."""
        if record.kind.lower() not in cls.EVENT_TYPES:
            return "другой тип ГТМ"
        if record.success.lower() != "да":
            return "неуспешное"
        if not record.status.lower().startswith("выполнено"):
            return "не выполнено"
        if record.start is None or record.end is None:
            return "нет даты начала или окончания"
        if record.end < record.start:
            return "окончание раньше начала"
        if not (record.new > record.current):
            return "частота не увеличена (новое ≤ текущего)"
        return None


def _text(value) -> str:
    return "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value).strip()
