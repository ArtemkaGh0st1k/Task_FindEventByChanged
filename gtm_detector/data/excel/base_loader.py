from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
from openpyxl.utils import column_index_from_string

from gtm_detector.data.excel.config import WellConfig
from gtm_detector.data.dto import *


class BaseDataLoader(ABC):
    def __init__(self, file_path : str):
        self.file_path = Path(file_path)


    
    def load_well_by_id(self,
                        well_id : str,
                        config : WellConfig):

        with pd.ExcelFile(self.file_path) as file:
            if config.sheet_name in file.sheet_names:
                df : pd.DataFrame = pd.read_excel(file, sheet_name=config.sheet_name)

                col_well_id_idx = column_index_from_string(config.col_idx) - 1
                col_cluster_well_id_idx = column_index_from_string(config.col_cluster_well_id) - 1
                col_start_data_idx = column_index_from_string(config.col_start_data) - 1
                if config.has_unnamed:
                    col_well_id_idx = f"Unnamed: {col_well_id_idx}"
                    col_cluster_well_id_idx = f"Unnamed: {col_cluster_well_id_idx}"
                    col_start_data_idx = f"Unnamed: {col_start_data_idx}"

                well_row : pd.Series = df[df[col_well_id_idx] == well_id]

                oil_data = df.iloc[col_start_data_idx:]

                oilDto = OilDto(
                    well_id,
                    well_row[col_cluster_well_id_idx],
                    well_row.iloc[col_start_data_idx:].tolist()
                )

                var = 3
                # TODO: проверка на пустое значение well_row 

                
                

