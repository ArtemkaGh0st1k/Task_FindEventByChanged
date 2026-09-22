from typing import Optional

from gtm_detector.data.dto import *


class NoteParser():

    def __init__(self):
        pass


    def convert_to_bools(self, notes : list[StateDto], catch_err = True, to_int = False):
        """
        Конвертирует массив строк, состоящих из ["раб", "нераб"]
        в массив [True/False] для списка StateDto

        `notes`: Объекты list[StateDto] \n
        'catch_err': Нужно ли выбрасывать исключение при неверном типе \n
        `to_int`: Нужно ли переводить True/False в 0 или 1
        """

        convert_notes = [self.convert_to_bool(note, catch_err, to_int) for note in notes]

        return convert_notes


    def convert_to_bool(self, note : StateDto, catch_err = True, to_int = False):
        """
        Конвертирует массив строк, состоящих из ["раб", "нераб"]
        в массив [True/False] для одного объекта StateDto

        `note`: Объект StateDto \n
        'catch_err': Нужно ли выбрасывать исключение при неверном типе \n
        `to_int`: Нужно ли переводить True/False в 0 или 1
        """

        if not isinstance(note, StateDto):
            if catch_err:
                raise TypeError(f"{note} имеет не тот тип!")
            else: return note

        if to_int:
            convert_data = {key : int(value == "раб." or value == "раб") for key, value in note.data.items()}
        else:
            convert_data = {key : (value == "раб." or value == "раб") for key, value in note.data.items()}

        note.data = convert_data

        return note


    def extract_gtm_events(
      self, note: Optional[StateDto], gtm_keywords: list[str]
  ) -> list[pd.Timestamp]:
        """Извлекает даты проведения сторонних ГТМ из комментариев в StateDto."""
        if not note or not note.comments:
            return []

        gtm_dates = []
        keywords_lower = [k.lower() for k in gtm_keywords]

        # Обработка ситуаций, когда comments передан словарем или списком
        if isinstance(note.comments, dict):
            items = note.comments.items()
        elif isinstance(note.comments, (list, tuple)):
            items = zip(note.data.keys(), note.comments)
        else:
            return []

        for date, comment in items:
            if comment is None or pd.isna(comment):
                continue
            comment_str = str(comment).lower()
            if any(kw in comment_str for kw in keywords_lower):
                gtm_dates.append(pd.Timestamp(date))

        return sorted(gtm_dates)
        