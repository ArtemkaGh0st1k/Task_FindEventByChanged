import pandas as pd
from abc import ABC
from datetime import datetime


#################################### Для вх.данных ####################################

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


class InnerResultDto():

    def __init__(self,
                 well_id : str,
                 well_cluster_id : str,
                 data : list):

        self.well_id = well_id
        self.well_cluster_id = well_cluster_id
        self.data = data


    def __eq__(self, other):
        return isinstance(other, InnerResultDto) and self.well_id == other.well_id

    
    def __hash__(self):
        return hash(self.well_id)

#################################### Для вх.данных ####################################


class ResultDto():
    """Класс для хранения результата вых.данных"""
    
    def __init__(self,
        well_id : str,
        well_cluster_id : str,
        gtm_type : str,
        reason_stop : str,
        start_date : datetime,
        end_date : datetime):
            
            self.well_id = well_id
            self.well_cluster_id = well_cluster_id
            self.gtm_type = gtm_type
            self.reason_stop = reason_stop
            self.start_date = start_date
            self.end_date = end_date