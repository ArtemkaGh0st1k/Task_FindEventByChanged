from datetime import datetime
from pathlib import Path
from os.path import join

class TimeHelper:

    @staticmethod
    def get_elapsed_time(start_time: float, end_time: float) -> str:
        """Возвращает строку с прошедшим временем в формате 'X ч Y м Z с'."""
        elapsed_seconds = int(end_time - start_time)
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours} ч {minutes} м {seconds} с"


    @staticmethod
    def create_mkdir_y_m_d():
        """Создаёт в папке results папку в формате
        {текущий день-текущий месяц-текущий год}"""

        today_y_m_d = datetime.now().strftime("%d-%m-%Y")
        path = join("results", today_y_m_d)

        folder = Path(path)
        folder.mkdir(exist_ok=True)
        
        return path


    @staticmethod
    def get_path_h_m_s():
        """Создаёт файл по пути 
        results/{текущая дата d-m-y}/{текущая дата h-m-s}"""

        cur_date = TimeHelper.create_mkdir_y_m_d()
        cur_time = datetime.now().strftime("%H-%M-%S")

        path = join(cur_date, f"gtm_result_{cur_time}.csv")
        return path




    