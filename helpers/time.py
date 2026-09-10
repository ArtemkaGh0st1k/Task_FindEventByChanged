

class TimeHelper:

    @staticmethod
    def get_elapsed_time(start_time: float, end_time: float) -> str:
        """Возвращает строку с прошедшим временем в формате 'X ч Y м Z с'."""
        elapsed_seconds = int(end_time - start_time)
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours} ч {minutes} м {seconds} с"