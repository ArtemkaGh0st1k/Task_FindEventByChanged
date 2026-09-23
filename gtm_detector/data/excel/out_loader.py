import pandas as pd
import numpy as np
from openpyxl.utils import column_index_from_string

from gtm_detector.data.excel.base_loader import BaseDataLoader
from gtm_detector.data.excel.config import *
from gtm_detector.data.dto import *


class OutterDataLoader(BaseDataLoader):
    """Класс для выгрузки для выход.данных"""

    def __init__(self, file_path):
        super().__init__(file_path)


    def load_wells_by_sheets(self, config):
        super().load_wells_by_sheets(config)


    def load_succes_and_contain_dates(self, configs : list[OutterExcelImportConfig]
        ) -> dict[str, list[ResultDto]]:
        """Собирает выход.данные с excel в формате словаре, где
        ключ - имя листа, значение - список ResultDto"""

        data = {}
        with pd.ExcelFile(self.file_path) as file:
            for config in configs:
                if config.sheet_name in file.sheet_names:

                    df : pd.DataFrame = pd.read_excel(file, sheet_name=config.sheet_name)
                    #print(df.head)

                    unnamed_idx = set(super().check_unnamed_columns(df))
                
                    col_well_id_idx = column_index_from_string(config.well_id_idx) - 1
                    col_well_cluster_idx = column_index_from_string(config.well_cluster_id_idx) - 1
                    col_type_gtm = column_index_from_string(config.type_gtm_idx) - 1
                    col_success = column_index_from_string(config.success_idx) - 1
                    col_reason_stop = column_index_from_string(config.reason_stop_idx) - 1
                    col_start_date = column_index_from_string(config.start_date_idx) - 1
                    col_end_date = column_index_from_string(config.end_date_idx) - 1

                    if col_well_id_idx in unnamed_idx: col_well_id_idx = f"Unnamed: {col_well_id_idx}"
                    if col_well_cluster_idx in unnamed_idx: col_well_cluster_idx = f"Unnamed: {col_well_cluster_idx}"
                    if col_type_gtm in unnamed_idx: col_type_gtm = f"Unnamed: {col_type_gtm}"
                    if col_success in unnamed_idx: col_success = f"Unnamed: {col_success}"
                    if col_reason_stop in unnamed_idx: col_reason_stop = f"Unnamed: {col_reason_stop}"  
                    if col_start_date in unnamed_idx: col_start_date = f"Unnamed: {col_start_date}"
                    if col_end_date in unnamed_idx: col_end_date = f"Unnamed: {col_end_date}"       

                    start_row = config.start_row_idx - 2
                    last_row = df.iloc[:, 0].last_valid_index() + 1

                    results : list[ResultDto] = []
                    for row in range(start_row, last_row):

                        row_data = df.iloc[row]

                        success = row_data.loc[col_success]
                        start_date = row_data.loc[col_start_date]
                        end_date = row_data.loc[col_end_date]

                        isnan_succes = success is np.nan
                        isnan_start_date = start_date is np.nan
                        isnan_end_date = end_date is np.nan

                        well_id = row_data.loc[col_well_id_idx]
                        if ((isnan_succes and success != "да") or 
                            isnan_start_date or 
                            isnan_end_date):
                            continue

                        well_cluster_id = row_data.loc[col_well_cluster_idx]
                        gtm_type = row_data.loc[col_type_gtm]
                        reason_stop = row_data.loc[col_reason_stop]

                        resDto = ResultDto(well_id, well_cluster_id, gtm_type,
                                           reason_stop, start_date, end_date)
                        results.append(resDto)

                    data[config.sheet_name] = results

        return data