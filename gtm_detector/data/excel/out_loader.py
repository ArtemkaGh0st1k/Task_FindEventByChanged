

from gtm_detector.data.excel.base_loader import BaseDataLoader


class OutterDataLoader(BaseDataLoader):
    """Класс для выгрузки для выход.данных"""

    def __init__(self, file_path):
        super().__init__(file_path)