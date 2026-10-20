"""Чтение Excel-файлов технологов во входной формат детектора.

Входные данные — «Статистика_заполнения_замеров_*.xlsx»: по листу на параметр,
строка = скважина, столбцы с датами «ДД.ММ.ГГГГ» = суточные значения.
Состояние скважины — лист «Прим», строки с «Состояние» в 5-м столбце.

Разметка — лист «Результат__*» файла «Расчёт_НОРДС*.xlsm».
"""
from __future__ import annotations

import re
from datetime import date

import pandas as pd

PARAM_SHEETS = {
    "debit_liquid": "Qж",
    "watercut": "Обв",
    "debit_oil": "Qн",
    "pressure_line": "Pлин",
    "frequency": "Fэцн ТМ",
    "pressure_zab": "Рзаб",
}
STATE_SHEET = "Прим"
WORKING_STATES = {"раб.", "нак"}  # остальные: ост., б/д тг, б/д пр л, пьез, конс, набл

_LAT2CYR = str.maketrans("ABEKMHOPCTXabekmhopctx", "АВЕКМНОРСТХавекмнорстх")


def norm_id(x) -> str:
    """Номер скважины/куста к единому виду: '561Е ' / '561E' / 561.0 -> '561Е'."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    return str(x).strip().upper().translate(_LAT2CYR)


def _date_columns(columns) -> dict:
    out = {}
    for c in columns:
        if isinstance(c, (pd.Timestamp, date)):
            out[c] = pd.Timestamp(c)
        elif isinstance(c, str) and re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", c.strip()):
            out[c] = pd.to_datetime(c.strip(), format="%d.%m.%Y")
    return out


def _read_param(path, sheet: str) -> pd.DataFrame:
    """-> DataFrame: index (well, kust), columns = даты."""
    df = pd.read_excel(path, sheet_name=sheet, header=1).iloc[1:]
    dcols = _date_columns(df.columns)
    key = pd.MultiIndex.from_arrays(
        [df["№ скв."].map(norm_id), df["№ куста"].map(norm_id)], names=["well", "kust"]
    )
    vals = df[list(dcols)].apply(pd.to_numeric, errors="coerce")
    vals.columns = list(dcols.values())
    vals.index = key
    return vals[~vals.index.duplicated()]


def _read_state(path) -> pd.DataFrame:
    pr = pd.read_excel(path, sheet_name=STATE_SHEET, header=None)
    dates = pr.iloc[1, 7:]
    st = pr[pr[4] == "Состояние"]
    info = pr.shift(-1)[pr[4] == "Состояние"][1].astype(str)  # строка ниже: «..., куст 2079, ...»
    kust = info.str.extract(r"куст\s+([^,]+)", expand=False).map(norm_id)
    vals = st.iloc[:, 7:].apply(lambda s: s.astype(str).str.strip())
    vals.columns = pd.to_datetime(dates.values)
    vals.index = pd.MultiIndex.from_arrays([st[3].map(norm_id), kust], names=["well", "kust"])
    return vals[~vals.index.duplicated()]


def load_measurements(path) -> list[dict]:
    """Все скважины файла замеров во входном формате детектора."""
    params = {k: _read_param(path, sh) for k, sh in PARAM_SHEETS.items()}
    state = _read_state(path)
    dates = params["frequency"].columns
    wells = []
    for key in params["frequency"].index:
        well = {"well_id": f"{key[0]}/{key[1]}", "well": key[0], "kust": key[1],
                "date": [d.date() for d in dates]}
        for k, df in params.items():
            row = df.loc[key] if key in df.index else pd.Series(index=dates, dtype=float)
            well[k] = row.reindex(dates).tolist()
        if key in state.index:
            s = state.loc[key].reindex(dates)
            well["state"] = [v in WORKING_STATES for v in s]
        else:  # состояния нет — считаем рабочими дни с ненулевой частотой
            well["state"] = [bool(v and v > 0) for v in well["frequency"]]
        wells.append(well)
    return wells


# --- разметка -----------------------------------------------------------------

LABEL_COLS = {
    1: "cdng", 3: "kust", 4: "well", 13: "f_before_tr", 26: "event", 32: "status",
    33: "f_fact", 35: "date_done", 39: "success", 40: "end_reason",
    41: "start", 42: "end",
}


def _parse_date(x) -> pd.Timestamp | None:
    """Даты в разметке бывают строками с опечатками: '21.04.2026г', '08.02.206'."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, (pd.Timestamp, date)):
        ts = pd.Timestamp(x)
    else:
        m = re.search(r"(\d{1,2})\.(\d{1,2})\.(\d{2,4})", str(x))
        if not m:
            return None
        d, mo, y = map(int, m.groups())
        y = {206: 2026, 26: 2026}.get(y, y)
        try:
            ts = pd.Timestamp(year=y, month=mo, day=d)
        except ValueError:
            return None
    if ts.year < 2000:  # '206-08-02' и подобное
        return None
    return ts.normalize()


def read_results(path, sheet: str | None = None) -> pd.DataFrame:
    """Все строки листа результатов с разобранными датами."""
    if sheet is None:
        sheet = next(s for s in pd.ExcelFile(path).sheet_names if s.startswith("Результат"))
    r = pd.read_excel(path, sheet_name=sheet, header=None, skiprows=4)
    r = r[list(LABEL_COLS)].rename(columns=LABEL_COLS)
    r = r[r["well"].notna()].copy()
    for c in ("date_done", "start", "end"):
        r[c] = r[c].map(_parse_date)
    r["well"] = r["well"].map(norm_id)
    r["kust"] = r["kust"].map(norm_id)
    r["is_label"] = (
        r["event"].astype(str).str.strip().str.lower().eq("увеличение f")
        & r["success"].astype(str).str.strip().str.lower().eq("да")
        & r["start"].notna() & r["end"].notna()
    )
    r.loc[r["is_label"], "is_label"] = r.loc[r["is_label"], "end"] >= r.loc[r["is_label"], "start"]
    return r.reset_index(drop=True)


def load_labels(path, sheet: str | None = None) -> pd.DataFrame:
    """Мероприятия «Увеличение F», успешные, с датами начала и окончания мониторинга."""
    r = read_results(path, sheet)
    return r[r["is_label"]].reset_index(drop=True)


def _find_well(row, by_key, by_well):
    w = by_key.get((row["well"], row["kust"]))
    if w is None and len(by_well.get(row["well"], [])) == 1:
        w = by_well[row["well"]][0]  # куст записан по-разному, но номер уникален
    return w


def attach_labels(wells: list[dict], results: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    """Проставить well["events"] (разметка) и well["neutral_dates"] — даты прочих
    мероприятий с частотой из листа результатов (неуспешные, без дат и т.п.):
    срабатывание детектора на них не считается ложным.
    Возвращает скважины с разметкой и не сопоставленные метки."""
    by_key = {(w["well"], w["kust"]): w for w in wells}
    by_well: dict[str, list[dict]] = {}
    for w in wells:
        by_well.setdefault(w["well"], []).append(w)
        w["events"], w["neutral_dates"] = [], []
    if "is_label" not in results:
        results = results.assign(is_label=True)
    unmatched = []
    for _, row in results.iterrows():
        w = _find_well(row, by_key, by_well)
        if row["is_label"]:
            if w is None:
                unmatched.append(row)
                continue
            w["events"].append({"start": row["start"].date(), "end": row["end"].date(),
                                "date_done": None if pd.isna(row["date_done"]) else row["date_done"].date(),
                                "end_reason": row["end_reason"]})
        elif w is not None and ("f" in str(row["event"]).lower() or "частот" in str(row["event"]).lower()):
            d = row["date_done"] if pd.notna(row["date_done"]) else row["start"]
            if pd.notna(d):
                w["neutral_dates"].append(d.date())
    labeled = [w for w in wells if w["events"]]
    return labeled, pd.DataFrame(unmatched)
