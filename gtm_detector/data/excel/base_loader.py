from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
from openpyxl.utils import column_index_from_string

from gtm_detector.data.excel.config import WellConfig
from gtm_detector.data.dto import *


class BaseDataLoader(ABC):
    """Класс-родитель для работы с Excel"""
    def __init__(self, file_path : str):
        self.file_path = Path(file_path)


    @abstractmethod
    def load_wells_by_sheets(self, config):
        pass