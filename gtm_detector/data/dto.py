import pandas as pd
from abc import ABC


class BaseObjectDto():
    def __init__(self,
                 well_id : str | int,
                 well_cluster_id : str | int,
                 data : dict[pd.Timestamp, float | bool]):
        
        self.well_id = well_id
        self.well_cluster_id = well_cluster_id
        self.data = data


class OilDto(BaseObjectDto):
    """Класс для хранения данных на листе Qн"""

    def __init__(self,
                 well_id : str | int,
                 well_cluster_id : str | int,
                 production_oil : dict[pd.Timestamp, float]):

        """
        `well_id`: Id скважины  \n
        `well_cluster`: Id: Id куста \n
        `production_oil`: добыча нефти 
        """

        super().__init__(well_id, well_cluster_id, production_oil)


class FrequencyDto(BaseObjectDto):
    """Класс для хранения данных на листе Fэцн ТМ"""

    def __init__(self, well_id, well_cluster_id, data):
        super().__init__(well_id, well_cluster_id, data)


class StateDto(BaseObjectDto):
    """Класс для хранения состояния насосов"""

    def __init__(self,
                 well_id : str, 
                 state_data : dict[pd.Timestamp, str],
                 comments : dict[pd.Timestamp, str] = None):
        """
        `data`: Id скважины \n
        `state_data`: Ключ - дата, значение - активана\неавтивна \n
        'comments': Комментарии технологов в определенные даты
        """

        super().__init__(well_id, None, state_data)
        self.comments = comments