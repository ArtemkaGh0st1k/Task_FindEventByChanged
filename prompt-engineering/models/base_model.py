from parsers.base_parser import BaseParser


class BaseModel:
    def __init__(self, 
                 model_name : str = None,
                 base_prompt : str = None,
                 parser : BaseParser = None):
        self.model_name = model_name if model_name is not None else "deepseek-r1:8b"
        self.base_prompt = base_prompt if base_prompt is not None else self.__set_default_prompt()
        self.parser = parser

    def __set_default_prompt(self):
            return \
                "Ты специалист по добыче нейти." \
                "Задача слудующая - есть данные по добыче нефти, жидкости для определенной скважины." \
                "Необходимо по этим данным понять было ли произведено на скважине какое-либо мероприятие," \
                "связанное с дополнительной добычей нефти." \
                "Мы рассматриваем такие мероприятия как изменение частоты оборотов насоса, замена насоса." \
                "На выходе необходим ответ в формате JSON, а именно:" \
                "Ответ: { 'мероприятие' : 'да/нет', 'дата' : 'дд.мм.гггг'}"
            