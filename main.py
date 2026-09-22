import os
from os.path import join
from os import getcwd

from constants.path import Constant
from gtm_detector.data.excel.in_loader import InnerWellDataLoader
from gtm_detector.data.excel.config import WellConfig
from gtm_detector.config.config import *
from gtm_detector.features.feature_engineering import FeatureEngineer
from gtm_detector.pipeline import GTMPipeline
from gtm_detector.parsers.note_parser import NoteParser
from gtm_detector.models.gtm_detector import *
from helpers.time import TimeHelper


if __name__ == "__main__":

    input_path = join(getcwd(), "resources", "input_dataset.xlsx")
    output_path = join(getcwd(), "resources", "output_dataset.xlsm")


    #1. Настройка конфигурации сбора данных с Excel
    well_config = WellConfig\
    (
        has_unnamed={'Qн' : False, 'Fэцн ТМ' : False, 'Прим' : True},
        start_date={'Qн' : [2, "T"], 'Fэцн ТМ' : [2, "T"], "Прим" : [2, "H"]},
        col_well_idx={'Qн' : "E", "Fэцн ТМ" : "E", "Прим" : "D"},
        start_data={"Qн" : [4, "T"], "Fэцн ТМ" : [4, "T"], "Прим" : [3, "H"]},
        col_cluster_well_id={"Qн" : "I", "Fэцн ТМ" : "I", "Прим" : None},
        count_empty_rows_before_header={"Qн" : 1, "Fэцн ТМ" : 1, "Прим" : 0}
    )

    #2. Выгрузка данных в DTO
    inLoader = InnerWellDataLoader(input_path)
    wells = inLoader.load_wells_by_sheets(well_config)  #FIXME: Считывает похоже не все строки (2 строки последнее не прочитывает)

    #3. Расчёт признаков
    pipeline_config = PipelineConfig()
    note_parser = NoteParser()
    engineer = FeatureEngineer(config=pipeline_config, note_parser=note_parser)

    transformed_features = engineer.transform(wells)

    #4. Детекция ГТМ нейросетью
    detector = GTMDetector(
        config=pipeline_config,
        note_parser=note_parser,
        model_path=Constant.WEIGHT_PATH
    )
    #detector.save_weights(Constant.WEIGHT_PATH)

    # Обход всех скважин из исходной выгрузки

    oil_dtos = wells.get("Qн", [])
    unique_well_ids = list(dict.fromkeys([dto.well_id for dto in oil_dtos]))

    results_list = []
    for well_id in unique_well_ids:

        # Сборка финального DataFrame для визуализации или отладки
        df_well = detector.build_well_df(well_id, wells, transformed_features)

        # Анализ скважины
        verdict = detector.analyze_well_dtos(
          well_id, wells, transformed_features
        )

        print(f"\n[Скважина {well_id}]")
        print(f"  Размерность ряда: {len(df_well)} дней")
        print(f"  Результат: {verdict.reasoning}")

        # Формирование строки для итоговой таблицы
        results_list.append\
        (
            {
                "Скважина": verdict.well_id,
                "Наличие ГТМ": "Да" if verdict.has_gtm else "Нет",
                "Тип ГТМ": verdict.gtm_type if verdict.gtm_type else "-",
                "Дата начала": \
                (
                    verdict.start_date.strftime("%Y-%m-%d")
                    if verdict.start_date
                    else "-"
                ),
                "Дата окончания": \
                (
                    verdict.end_date.strftime("%Y-%m-%d") if verdict.end_date else "-"
                ),
                "Обоснование": verdict.reasoning,
            }
        )

    # 4. Сохранение результатов в Excel
    df_results = pd.DataFrame(results_list)
    df_results.to_csv(TimeHelper.get_path_h_m_s(), index=False)