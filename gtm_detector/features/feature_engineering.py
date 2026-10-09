import pandas as pd
import numpy as np

from gtm_detector.config.config import PipelineConfig
from gtm_detector.data.dto import *
from gtm_detector.parsers.note_parser import NoteParser


class FeatureEngineer:
    """
    Преобразует 3 исходных сигнала в 5 каналов для нейросети:
    Q_n_smooth: Сглаженный медианой дебит Q_н.
    F_ecn: Частота работы насоса F_{эцн}.
    Specific_Yield: Удельный дебит {Q_н}\{F_{эцн} (признак смены типа насоса).
    dF: Суточная разница частоты delta F.
    Is_Working: Флаг технологического режима из примечаний и замеров.
    """

    def __init__(self,
                 config : PipelineConfig = None,
                 note_parser : NoteParser = None):
        self.config = config if config else PipelineConfig()
        self.note_parser = note_parser if note_parser else NoteParser()


    def __get_q_smooth(self, oil_dtos : list[OilDto]):
        """Сглаживание дебита"""

        transform_oil : list[OilDto] = []
        for oil_dto in oil_dtos.copy():
            series = pd.Series(oil_dto.data).sort_index()
            smoothed = (
                series
                .rolling(window=self.config.smoothing_window)
                .mean()
            )
            oil_dto.data = smoothed

            transform_oil.append(oil_dto)

        return {"Q_n_smooth" : transform_oil}


    def __get_specific_yield(self,
                            oil_smooth_dtos : list[OilDto],
                            freq_dtos : list[FrequencyDto]):
        """Удельный дебит (удельный приток на 1 Гц частоты)"""

        transform_dtos = []

        oil_smooth_dtos.sort(key=lambda x: (x.well_id, x.well_cluster_id))
        freq_dtos.sort(key=lambda x: (x.well_id, x.well_cluster_id))

        common_date = oil_smooth_dtos[0].data.keys()
        for oil_smooth_dto, freq_dto in zip(oil_smooth_dtos, freq_dtos):

            if (oil_smooth_dto.well_id != freq_dto.well_id and
                oil_smooth_dto.well_cluster_id != freq_dto.well_cluster_id):
                raise ValueError("Id должны быть упорядочены!")

            oil_smooth_values = oil_smooth_dto.data
            freq_values = np.array(list(freq_dto.data.values()))

            spec_yield = np.where(
                freq_values > 0, 
                oil_smooth_values / freq_values,
                0.0
            )

            data = {key : value for key, value in zip(common_date, spec_yield)}
            baseDto = BaseObjectDto(oil_smooth_dto.well_id,
                                    oil_smooth_dto.well_cluster_id,
                                    data)

            transform_dtos.append(baseDto)

        return {"Specific_Yield" : transform_dtos}      


    def __get_dF(self, freq_dtos : list[FrequencyDto]):
        """Производная частоты (скачок режима)"""

        transform_freq : list[FrequencyDto] = []
        for freq_dto in freq_dtos.copy():
            freq_series = pd.Series(freq_dto.data.values()).sort_index()
            diff_freq = freq_series.diff().fillna(0.0)

            freq_dto.data = {time : diff for time, diff in zip(freq_dto.data.keys(), diff_freq)}

            transform_freq.append(freq_dto)

        return {"dF" : transform_freq}


    def transform(self, data : dict[str, list[BaseObjectDto]]):
        """Подготавливает данные с листов для дальнейшего использования"""

        q_n_smooth = self.__get_q_smooth(data["Qн"])
        is_working = { "Is_Working" : self.note_parser.convert_to_bools(data["Прим"], catch_err=True, to_int=True) }
        specific_yield = self.__get_specific_yield(q_n_smooth["Q_n_smooth"], data["Fэцн ТМ"])
        dF = self.__get_dF(data["Fэцн ТМ"])

        return q_n_smooth | is_working | specific_yield | dF
