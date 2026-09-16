from pathlib import Path
import pandas as pd
from openpyxl.utils import column_index_from_string


class ExcelDataLoader:
    """Отвечает за загрузку и парсинг Excel-файла с временными рядами."""
    def __init__(self,
                 input_file_path : str,
                 output_file_path : str):
        
        self.input_file_path = Path(input_file_path)
        self.output_file_path = Path(output_file_path)

        self.__check_path_exist()

        self.input_required_sheets = ["Qн", "Fэцн ТМ", "Прим"]
        self.output_required_sheets = ["Результат__16_54_11"]


    def load_data(self) -> dict[str, pd.DataFrame]:
        """Загружает все необходимые листы из Excel."""

        data = {}
        with pd.ExcelFile(self.input_file_path) as input_dataset:
            for sheet in self.input_required_sheets:
                if sheet in input_dataset.sheet_names:
                    data[sheet] = pd.read_excel(input_dataset, sheet_name=sheet)
                else:
                    raise ValueError(
                        f"Отсутствует обязательный лист '{sheet}' в файле!"
                    )

        return data


    def get_complete_wells(self):
        """Собирает из выходного файла только те скважины, 
        на которых были завершены мероприятия"""

        complete_wells : list = None
        with pd.ExcelFile(self.output_file_path) as output_dataset:
            for sheet in self.output_required_sheets:
                if sheet in output_dataset.sheet_names:
                    df = pd.read_excel(output_dataset, sheet_name=sheet)

                    #print(df.columns)
                    #print(df.head)
                    #print(df.head(10).to_string())

                    # проверить значение has_complete_well
                    complete_col_idx = column_index_from_string("AN") - 1       # int индекс где столбец с Успешно/Не успешно
                    complete_unnamed_col_idx = f"Unnamed: {complete_col_idx}"   # преобразованный индекс столбца тк есть склееные столбцы

                    has_complete_well = None
                    try:
                        has_complete_well = df.loc[5, complete_unnamed_col_idx]        # Unnamed - из-за формата excel, склеинные столбцы

                        if pd.notna(has_complete_well):
                            has_complete_well = True
                        else: has_complete_well = False
                    except KeyError:
                        has_complete_well = False

                    # получим номера строк скважин, где успешно выполнены мероприятия

                    well_id_col_idx = column_index_from_string("E") - 1
                    unnamed_well_id_col_idx = f"Unnamed: {well_id_col_idx}"

                    df_copy = df.iloc[4:]
                    complete_rows = df_copy[df_copy[unnamed_well_id_col_idx].notna()].index.tolist()

                    # получим скважины, на которых закончились мероприятия 
                    complete_wells = df_copy.loc[complete_rows, unnamed_well_id_col_idx].tolist()

        return complete_wells


    def get_wells_where_start_and_end_date_exists(self):
        """Собирает успешные скважины, где есть дата начала и конца мониторинга"""
        
        mask = None
        with pd.ExcelFile(self.output_file_path) as output_dataset:
            for sheet in self.output_required_sheets:
                if sheet in output_dataset.sheet_names:
                    df = pd.read_excel(output_dataset, sheet_name=sheet)

                    # пропустим первые 4 строки тк там оглавление и ненужная инфа

                    unnamed_well_id_col_idx = f"Unnamed: {column_index_from_string("E") - 1}"
                    unnamed_start_date_col_idx = f"Unnamed: {column_index_from_string("AP") - 1}"
                    unnamed_end_date_col_idx = f"Unnamed: {column_index_from_string("AQ") - 1}"

                    df_copy = df.iloc[4:]

                    mask =\
                    (
                        df_copy[unnamed_well_id_col_idx].notna() &
                        df_copy[unnamed_start_date_col_idx].notna() & 
                        df_copy[unnamed_end_date_col_idx].notna()
                    )

        return df_copy.index[mask].tolist()


    def __check_path_exist(self):
        if not self.input_file_path.exists():
            raise FileNotFoundError(f"Файл {self.input_file_path} не найден!")

        if not self.output_file_path.exists():
            raise FileNotFoundError(f"Файл {self.output_file_path} не найден!")

                