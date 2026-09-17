from os.path import join
from os import getcwd

from gtm_detector.data.excel.in_loader import InnerWellDataLoader
from gtm_detector.data.excel.config import WellConfig

if __name__ == "__main__":

    input_path = join(getcwd(), "resources", "input_dataset.xlsx")
    output_path = join(getcwd(), "resources", "output_dataset.xlsm")

    inLoader = InnerWellDataLoader(input_path)

    config = WellConfig\
    (
        has_unnamed={'Qн' : False, 'Fэцн ТМ' : False, 'Прим' : True},
        start_date={'Qн' : [2, "T"], 'Fэцн ТМ' : [2, "T"], "Прим" : [2, "H"]},
        col_well_idx={'Qн' : "E", "Fэцн ТМ" : "E", "Прим" : "D"},
        start_data={"Qн" : [4, "T"], "Fэцн ТМ" : [4, "T"], "Прим" : [3, "H"]},
        col_cluster_well_id={"Qн" : "I", "Fэцн ТМ" : "I", "Прим" : None},
        count_empty_rows_before_header={"Qн" : 1, "Fэцн ТМ" : 1, "Прим" : 0}
    )
    inLoader.load_wells_by_sheets(config)