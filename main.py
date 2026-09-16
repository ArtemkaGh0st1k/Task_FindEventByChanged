from os.path import join
from os import getcwd

from gtm_detector.data.loader import ExcelDataLoader

if __name__ == "__main__":

    input_path = join(getcwd(), "resources", "input_dataset.xlsx")
    output_path = join(getcwd(), "resources", "output_dataset.xlsm")

    excelLoader = ExcelDataLoader(input_path, output_path)
    test1 = excelLoader.load_data()
    test2 = excelLoader.get_complete_wells()
    test3 = excelLoader.get_wells_where_start_and_end_date_exists()