from dataclasses import dataclass


@dataclass
class WellConfig:
    # ключи - наименование листов

    has_unnamed: dict[str, bool]                       # есть ли склеинные столбцы
    start_date : dict[str, tuple[int, str]]            # идекс строки и столбца где находится временные данные
    col_well_idx: dict[str, int]                       # индекс столбца начала id скважины
    start_data : dict[str, tuple[int, str]]            # для каждого листа начала строки и столбца где начинаются данные
    col_cluster_well_id : dict[str, str]               # индекс столбце куста
    count_empty_rows_before_header : dict[str, int]    # число пустых строк до заголовка