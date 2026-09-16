import pandas as pd
from abc import ABC


class BaseObjectDto(ABC):
    def __init__(self,
                 well_id : str | int,
                 well_cluster_id : str | int,
                 data : list[float]):
        
        self.well_id = well_id
        self.well_cluster_id = well_cluster_id
        self.data = data
        super().__init__()


class OilDto(BaseObjectDto):
    """Класс для хранения данных на листе Qн"""

    def __init__(self,
                 well_id : str | int,
                 well_cluster_id : str | int,
                 production_oil : list[float]):

        """
        `well_id`: Id скважины  \n
        `well_cluster`: Id: Id куста \n
        `production_oil`: добыча нефти
        """

        super().__init__(well_id, well_cluster_id, production_oil)
        


class DateDto:
    """Класс для хранения единного времени"""

    def __init__(self, period : pd.Timestamp):
        self.period : pd.Timestamp = period



class FrequencyDto(BaseObjectDto):
    """Класс для хранения данных на листе Fэцн ТМ"""

    def __init__(self, well_id, well_cluster_id, data):
        super().__init__(well_id, well_cluster_id, data)