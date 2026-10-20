"""Чтение Excel-файлов технологов.

* Замеры — «Статистика_заполнения_замеров_*.xlsx»: по листу на параметр,
  строка = скважина, столбцы «ДД.ММ.ГГГГ» = суточные значения;
  состояние скважины — лист «Прим», строки «Состояние».
* Разметка — лист «Результат__*» файла «Расчёт_НОРДС*.xlsm».
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from ..excel.parsing import date_columns, normalize_id, parse_date
from ..models import Label, Well

logger = logging.getLogger(__name__)


class MeasurementsReader:
    """Читает файл замеров и возвращает список :class:`Well`."""

    PARAMETER_SHEETS = {
        "debit_liquid": "Qж",
        "watercut": "Обв",
        "debit_oil": "Qн",
        "pressure_line": "Pлин",
        "frequency": "Fэцн ТМ",
        "pressure_zab": "Рзаб",
    }
    STATE_SHEET = "Прим"
    # Остальные состояния (ост., б/д тг, б/д пр л, пьез, конс, набл) — скважина не работает.
    WORKING_STATES = frozenset({"раб.", "нак"})

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def read(self) -> list[Well]:
        parameters = {name: self._read_parameter(sheet) for name, sheet in self.PARAMETER_SHEETS.items()}
        states = self._read_states()
        dates = parameters["frequency"].columns

        wells = []
        for key in parameters["frequency"].index:
            number, cluster = key
            record = {
                "well_id": f"{number}/{cluster}",
                "well": number,
                "kust": cluster,
                "date": [day.date() for day in dates],
            }
            for name, table in parameters.items():
                row = table.loc[key] if key in table.index else pd.Series(index=dates, dtype=float)
                record[name] = row.reindex(dates).tolist()
            if key in states.index:
                record["state"] = [value in self.WORKING_STATES for value in states.loc[key].reindex(dates)]
            else:  # состояния нет — рабочими считаем дни с ненулевой частотой
                record["state"] = [bool(value and value > 0) for value in record["frequency"]]
            wells.append(Well.from_dict(record))
        logger.info("Замеры: %d скважин", len(wells))
        return wells

    def _read_parameter(self, sheet: str) -> pd.DataFrame:
        """Лист параметра -> DataFrame: индекс (скважина, куст), колонки — даты."""
        table = pd.read_excel(self.path, sheet_name=sheet, header=1).iloc[1:]
        columns = date_columns(table.columns)
        values = table[list(columns)].apply(pd.to_numeric, errors="coerce")
        values.columns = list(columns.values())
        values.index = pd.MultiIndex.from_arrays(
            [table["№ скв."].map(normalize_id), table["№ куста"].map(normalize_id)],
            names=["well", "cluster"],
        )
        return values[~values.index.duplicated()]

    def _read_states(self) -> pd.DataFrame:
        """Лист «Прим»: пары строк «Состояние» / «Прим» на скважину.

        Куст берётся из строки «Прим» (текст вида «..., куст 2079, ЗУ ...»).
        """
        sheet = pd.read_excel(self.path, sheet_name=self.STATE_SHEET, header=None)
        dates = sheet.iloc[1, 7:]
        is_state_row = sheet[4] == "Состояние"
        state_rows = sheet[is_state_row]
        info = sheet.shift(-1)[is_state_row][1].astype(str)
        clusters = info.str.extract(r"куст\s+([^,]+)", expand=False).map(normalize_id)

        values = state_rows.iloc[:, 7:].apply(lambda column: column.astype(str).str.strip())
        values.columns = pd.to_datetime(dates.values)
        values.index = pd.MultiIndex.from_arrays(
            [state_rows[3].map(normalize_id), clusters], names=["well", "cluster"]
        )
        return values[~values.index.duplicated()]


@dataclass(frozen=True)
class ResultRecord:
    """Строка листа результатов (рекомендация и её выполнение)."""

    number: str
    cluster: str
    event_name: str
    success: str
    date_done: date | None
    start: date | None
    end: date | None
    end_reason: str | None

    @property
    def is_label(self) -> bool:
        """Попадает в разметку: «Увеличение F», успешно, есть корректные даты начала и окончания."""
        return (
            self.event_name.strip().lower() == "увеличение f"
            and self.success.strip().lower() == "да"
            and self.start is not None
            and self.end is not None
            and self.end >= self.start
        )

    @property
    def is_frequency_event(self) -> bool:
        """Мероприятие, связанное с частотой (любой статус)."""
        name = self.event_name.lower()
        return "f" in name or "частот" in name

    def to_label(self) -> Label:
        return Label(start=self.start, end=self.end, date_done=self.date_done, end_reason=self.end_reason)


class ResultsReader:
    """Читает лист «Результат__*» с рекомендациями и датами мониторинга."""

    # Номер столбца (с нуля) -> поле ResultRecord. Шапка листа занимает 4 строки.
    COLUMNS = {
        3: "cluster",
        4: "number",
        26: "event_name",
        35: "date_done",
        39: "success",
        40: "end_reason",
        41: "start",
        42: "end",
    }
    HEADER_ROWS = 4
    SHEET_PREFIX = "Результат"

    def __init__(self, path: str | Path, sheet: str | None = None) -> None:
        self.path = Path(path)
        self.sheet = sheet

    def read(self) -> list[ResultRecord]:
        sheet = self.sheet or self._find_sheet()
        table = pd.read_excel(self.path, sheet_name=sheet, header=None, skiprows=self.HEADER_ROWS)
        table = table[list(self.COLUMNS)].rename(columns=self.COLUMNS)
        table = table[table["number"].notna()]

        records = []
        for row in table.itertuples(index=False):
            records.append(
                ResultRecord(
                    number=normalize_id(row.number),
                    cluster=normalize_id(row.cluster),
                    event_name=_text(row.event_name),
                    success=_text(row.success),
                    date_done=_date(row.date_done),
                    start=_date(row.start),
                    end=_date(row.end),
                    end_reason=_text(row.end_reason) or None,
                )
            )
        logger.info(
            "Лист результатов: %d строк, из них в разметку: %d",
            len(records),
            sum(record.is_label for record in records),
        )
        return records

    def _find_sheet(self) -> str:
        for name in pd.ExcelFile(self.path).sheet_names:
            if name.startswith(self.SHEET_PREFIX):
                return name
        raise ValueError(f"В {self.path.name} нет листа «{self.SHEET_PREFIX}…»")


class WellIndex:
    """Поиск скважины по номеру и кусту.

    Номера скважин на месторождении повторяются, поэтому ключ — пара (номер, куст).
    Если куст записан по-разному, но номер уникален — находим по одному номеру.
    """

    def __init__(self, wells: Iterable[Well]) -> None:
        self._by_key: dict[tuple[str, str], Well] = {}
        self._by_number: dict[str, list[Well]] = {}
        for well in wells:
            self._by_key[(well.number, well.cluster)] = well
            self._by_number.setdefault(well.number, []).append(well)

    def find(self, number: str, cluster: str) -> Well | None:
        well = self._by_key.get((number, cluster))
        if well is None and len(self._by_number.get(number, [])) == 1:
            well = self._by_number[number][0]
        return well


def attach_labels(wells: list[Well], records: Iterable[ResultRecord]) -> list[ResultRecord]:
    """Записать разметку в ``well.labels`` и даты прочих мероприятий с частотой
    в ``well.known_event_dates``. Возвращает метки, для которых скважина не найдена."""
    index = WellIndex(wells)
    for well in wells:
        well.labels, well.known_event_dates = [], []

    unmatched = []
    for record in records:
        well = index.find(record.number, record.cluster)
        if record.is_label:
            if well is None:
                unmatched.append(record)
            else:
                well.labels.append(record.to_label())
        elif well is not None and record.is_frequency_event:
            known_date = record.date_done or record.start
            if known_date is not None:
                well.known_event_dates.append(known_date)
    return unmatched


def _text(value: object) -> str:
    return "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)


def _date(value: object) -> date | None:
    timestamp = parse_date(value)
    return timestamp.date() if timestamp is not None else None
