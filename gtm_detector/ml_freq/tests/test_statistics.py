"""Чтение полугодовых файлов статистики: поток книги, лист, объединение периодов."""

from __future__ import annotations

import struct

import numpy as np
import pandas as pd
import pytest
import xlrd

from gtm_detector.ml_freq.data.statistics import (
    HalfYearStatisticsReader,
    _join_periods,
    _sheet_table,
    _workbook_stream,
)

SECTOR = 512
END_OF_CHAIN, FREE, FAT_SECTOR = -2, -1, -3


def _ole_file(stream: bytes, declared: int) -> bytes:
    """Минимальный файл OLE2: сектор таблицы, сектор каталога, затем поток ``Workbook``."""
    sectors = max(1, -(-len(stream) // SECTOR))
    links = [FAT_SECTOR, END_OF_CHAIN, *range(3, sectors + 2), END_OF_CHAIN]
    fat = struct.pack(f"<{SECTOR // 4}i", *links, *[FREE] * (SECTOR // 4 - len(links)))

    name = "Workbook".encode("utf-16-le") + b"\0\0"
    entry = bytearray(128)
    entry[: len(name)] = name
    struct.pack_into("<H", entry, 64, len(name))
    struct.pack_into("<iI", entry, 116, 2, declared)
    directory = bytes(entry) + bytes(SECTOR - 128)

    header = bytearray(SECTOR)
    header[:8] = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    struct.pack_into("<H", header, 30, 9)  # размер сектора 2**9
    struct.pack_into("<ii", header, 44, 1, 1)  # один сектор таблицы; каталог в секторе 1
    struct.pack_into("<ii", header, 68, END_OF_CHAIN, 0)  # продолжения таблицы нет
    struct.pack_into("<109i", header, 76, 0, *[FREE] * 108)
    return bytes(header) + fat + directory + stream.ljust(sectors * SECTOR, b"\0")


def test_workbook_stream_follows_chain_longer_than_declared(tmp_path):
    # Как в настоящих выгрузках: в каталоге размер меньше, чем секторов в цепочке.
    stream = bytes(range(256)) * 6  # 3 сектора
    path = tmp_path / "book.xls"
    path.write_bytes(_ole_file(stream, declared=SECTOR + 1))
    assert _workbook_stream(path) == stream


class _Sheet:
    """Лист с интерфейсом xlrd: строки значений и типов."""

    def __init__(self, rows: list[list]) -> None:
        self.rows = rows
        self.nrows = len(rows)

    def row_values(self, r: int) -> list:
        return self.rows[r]

    def row_types(self, r: int) -> list[int]:
        kinds = {float: xlrd.XL_CELL_NUMBER, str: xlrd.XL_CELL_TEXT}
        return [
            xlrd.XL_CELL_EMPTY if value == "" else kinds[type(value)] for value in self.rows[r]
        ]


def _sheet(first: str, second: str, wells: list[list]) -> _Sheet:
    # Как в выгрузке: первая строка — версия, вторая — заголовок, третья пустая.
    return _Sheet(
        [
            ["Версия", "", "", ""],
            ["№ скв.", "№ куста", first, second],
            ["", "", "", ""],
            *wells,
        ]
    )


def test_sheet_table_keeps_only_numbers():
    sheet = _sheet(
        "01.01.2021",
        "02.01.2021",
        [
            ["101", "5А", 48.0, "ремонт"],
            ["", "", 1.0, 2.0],  # строка без номера скважины — итоги, не скважина
            ["101", "5А", 7.0, 7.0],  # повтор скважины: верна первая строка
            [202.0, "6", "", 50.0],
        ],
    )
    table, bad = _sheet_table(sheet)
    assert list(table.index) == [("101", "5А"), ("202", "6")]
    assert table.loc[("101", "5А")].iloc[0] == 48.0 and np.isnan(table.loc[("101", "5А")].iloc[1])
    assert np.isnan(table.loc[("202", "6")].iloc[0]) and table.loc[("202", "6")].iloc[1] == 50.0
    assert bad.tolist() == [0, 1]  # текст считается, пустая ячейка — нет
    assert table.columns[0] == pd.Timestamp("2021-01-01")


def test_join_periods_makes_continuous_calendar():
    first = pd.DataFrame(
        {pd.Timestamp("2021-06-29"): [1.0], pd.Timestamp("2021-06-30"): [2.0]},
        index=pd.MultiIndex.from_tuples([("101", "5")]),
    )
    # Новая скважина появилась во втором файле, а между файлами есть пропущенный день.
    second = pd.DataFrame(
        {pd.Timestamp("2021-07-02"): [3.0, 4.0]},
        index=pd.MultiIndex.from_tuples([("101", "5"), ("202", "6")]),
    )
    table = _join_periods([first, second])
    assert list(table.columns) == list(pd.date_range("2021-06-29", "2021-07-02"))
    assert table.loc[("101", "5")].tolist()[:2] == [1.0, 2.0]
    assert np.isnan(table.loc[("101", "5")].iloc[2])  # 01.07 нет ни в одном файле
    assert table.loc[("101", "5")].iloc[3] == 3.0
    assert np.isnan(table.loc[("202", "6")].iloc[0]) and table.loc[("202", "6")].iloc[3] == 4.0


def test_reader_requires_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        HalfYearStatisticsReader(tmp_path).read()
