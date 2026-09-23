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



class OutterExcelImportConfig():

    def __init__(self, 
                 sheet_name : str,
                 start_row_idx : int,
                 well_id_idx : str,
                 well_cluster_id_idx : str,
                 type_gtm_idx : str,
                 success_idx : str,
                 reason_stop_idx : str,
                 start_date_idx : str,
                 end_date_idx : str):

        """
        Конфигурация для импорта выходных данных. \n

        `sheet_name`: Имя листа \n
        `start_row_idx`: Индекс строки, с которой начинается чтение данных \n
        `well_id_idx`: Индекс столбца id скважины \n
        `well_cluster_id_idx`: Индекс столбца id куста \n
        `type_gtm`: Индекс столбца тип ГТМ \n
        `success_idx`: Индекс столбца вып.успешно \n
        `reason_stop_idx` : Индекс столбцп прич.остановки мониторинга \т
        `start_date_idx`: Индекс столбца нач.даты \n
        `end_date_idx`: Индекс столбца конеч.даты
        """
        
        self.sheet_name = sheet_name
        self.start_row_idx = start_row_idx
        self.well_id_idx = well_id_idx
        self.well_cluster_id_idx = well_cluster_id_idx
        self.type_gtm_idx = type_gtm_idx
        self.success_idx = success_idx
        self.reason_stop_idx = reason_stop_idx
        self.start_date_idx = start_date_idx
        self.end_date_idx = end_date_idx