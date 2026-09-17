from typing import Literal, Any
import pandas as pd
from openpyxl.utils import column_index_from_string

from gtm_detector.data.excel.base_loader import BaseDataLoader
from gtm_detector.data.excel.config import WellConfig
from gtm_detector.data.dto import *

class InnerWellDataLoader(BaseDataLoader):
    """Класс для выгрузки для вход.данных"""

    def __init__(self,
                 file_path,
                 req_sheets : list[str] = ["Qн", "Fэцн ТМ", "Прим"]):
        super().__init__(file_path)
        self.req_sheets = req_sheets


    def load_wells_by_sheets(self, config : WellConfig):

        data = {}

        sheet_summator = 0      # вспомогательный прибавлятель для листа "Примчания"
        with pd.ExcelFile(self.file_path) as file:
            for sheet in self.req_sheets:
                if sheet in file.sheet_names:
                    
                    df : pd.DataFrame = pd.read_excel(file, sheet_name=sheet)
                    print(df.head)
                    col_well_id_idx = column_index_from_string(config.col_well_idx[sheet]) - 1

                    if config.col_cluster_well_id[sheet]:
                        col_cluster_well_id_idx = column_index_from_string(config.col_cluster_well_id[sheet]) - 1
                    else: col_cluster_well_id_idx = None

                    start_data_row = config.start_data[sheet][0] - config.count_empty_rows_before_header[sheet] - 1
                    start_data_col = column_index_from_string(config.start_data[sheet][1]) - 1

                    #start_date_row = config.start_date[sheet][0] - config.count_empty_rows_before_header[sheet]
                    start_date_col = column_index_from_string(config.start_date[sheet][1]) - 1

                    if config.has_unnamed[sheet]:
                        col_well_id_idx = f"Unnamed: {col_well_id_idx}"
                        if config.col_cluster_well_id[sheet]: col_cluster_well_id_idx = f"Unnamed: {col_cluster_well_id_idx}"
                        start_data_col = f"Unnamed: {start_data_col}"
                        start_date_col = f"Unnamed: {start_date_col}"

                    well_data_by_sheet : list[BaseObjectDto] = []

                    well_date_row = df.iloc[0]
                    well_date = (well_date_row.loc[start_date_col :].tolist()
                                if config.has_unnamed[sheet]
                                else well_date_row.iloc[start_date_col :].tolist() )
                    times = pd.to_datetime(well_date, format='%d.%m.%Y')

                    # FIXME: pandas пропускает пустые строки до заголовка и из-за этого сбивается индексы в config
                    # TODO: учесть эти пропуски, а пока считывается файл по умолчанию

                    if sheet == "Прим": 
                        start_data_row -= 1
                        sheet_summator = 2

                    for row_idx in range(start_data_row, df.index[-1]):
                        if row_idx > df.index[-1] - 2: break

                        well_row : pd.Series = df.iloc[row_idx 
                                                       if start_data_row == row_idx
                                                        else row_idx + sheet_summator]

                        well_id = ( well_row.loc[col_well_id_idx]
                                    if config.has_unnamed[sheet]
                                    else well_row.iloc[col_well_id_idx] )
                        well_data = ( well_row.loc[start_data_col :].tolist()
                                      if config.has_unnamed[sheet]
                                      else well_row.iloc[start_data_col :].tolist())
                        if col_cluster_well_id_idx:
                            well_cluster_id = ( well_row.loc[col_cluster_well_id_idx]
                                                if config.has_unnamed[sheet]
                                                else well_row.iloc[col_cluster_well_id_idx])
                        else: well_cluster_id = None

                        well_datas = dict(zip
                                          (times, well_data))

                        dto : BaseObjectDto = None
                        match sheet:
                            case "Qн":
                                dto = OilDto(well_id, well_cluster_id, well_datas)

                            case "Fэцн ТМ":
                                dto = FrequencyDto(well_id, well_cluster_id, well_datas)

                            case "Прим":
                                comments_row = df.iloc[row_idx + 1]
                                end_idx = df.index[-1]
                                if row_idx == df.index[-1]:
                                    comments = None
                                else:
                                    comments = ( comments_row.loc[start_data_col:].tolist()
                                                if ( config.has_unnamed[sheet])
                                                else comments_row.iloc[start_data_col:].tolist() )
                                dto = StateDto(well_id,
                                               well_datas,
                                               comments)
                            case _:
                                raise ValueError(f"Лист {sheet} не найден!")
                        
                        well_data_by_sheet.append(dto)

                    data[sheet] = well_data_by_sheet

        return data       
                    