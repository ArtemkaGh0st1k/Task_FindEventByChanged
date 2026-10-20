"""Разбор «грязных» значений из Excel-файлов технологов."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

import pandas as pd

_LATIN_TO_CYRILLIC = str.maketrans("ABEKMHOPCTXabekmhopctx", "АВЕКМНОРСТХавекмнорстх")
_DATE_PATTERN = re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{2,4})")
_DATE_HEADER_PATTERN = re.compile(r"\d{2}\.\d{2}\.\d{4}")
# Опечатки в годе: '08.02.206' -> 2026, '08.02.26' -> 2026.
_YEAR_FIXES = {206: 2026, 26: 2026}


def normalize_id(value: Any) -> str:
    """Номер скважины/куста к единому виду: ``'561E '``, ``'561Е'`` -> ``'561Е'``, ``7782.0`` -> ``'7782'``.

    Латинские буквы, похожие на кириллические, заменяются кириллическими.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip().upper().translate(_LATIN_TO_CYRILLIC)


def parse_date(value: Any) -> pd.Timestamp | None:
    """Дата из ячейки: datetime или строка с опечатками (``'21.04.2026г'``, ``'08.02.206'``)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (pd.Timestamp, date)):
        timestamp = pd.Timestamp(value)
    else:
        found = _DATE_PATTERN.search(str(value))
        if not found:
            return None
        day, month, year = map(int, found.groups())
        year = _YEAR_FIXES.get(year, year)
        try:
            timestamp = pd.Timestamp(year=year, month=month, day=day)
        except ValueError:
            return None
    if pd.isna(timestamp) or timestamp.year < 2000:  # '206-08-02' и подобное
        return None
    return timestamp.normalize()


def date_columns(columns: Any) -> dict[Any, pd.Timestamp]:
    """Колонки-даты листа замеров (``'01.01.2026'`` или datetime) -> Timestamp."""
    result = {}
    for column in columns:
        if isinstance(column, (pd.Timestamp, date)):
            result[column] = pd.Timestamp(column)
        elif isinstance(column, str) and _DATE_HEADER_PATTERN.fullmatch(column.strip()):
            result[column] = pd.to_datetime(column.strip(), format="%d.%m.%Y")
    return result
