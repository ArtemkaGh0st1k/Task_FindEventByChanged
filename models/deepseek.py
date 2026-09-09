import asyncio
from pathlib import Path

from parsers.deepseek_parser import DeepSeekParser


class DeepSeekModel:
    def __init__(self, prompt : str = None):
        self.parser = DeepSeekParser()
        self.prompt = self.__set_default_prompt() if prompt is None else prompt

    def __set_default_prompt(self):
        self.prompt = \
            "Ты специалист по добыче нейти." \
            "Задача слудующая - есть данные по добыче нефти, жидкости для определенной скважины." \
            "Необходимо по этим данным понять было ли произведено на скважине какое-либо мероприятие," \
            "связанное с дополнительной добычей нефти." \
            "Мы рассматриваем такие мероприятия как изменение частоты оборотов насоса, замена насоса." \
            "На выходе необходим ответ в формате JSON, а именно:" \
            "Ответ: { 'мероприятие' : 'да/нет', 'дата' : 'дд.мм.гггг'}"

    async def wait_for_stream(self, page, timeout: int = 120) -> str:
        """
        Динамически ожидает завершения генерации текста моделью DeepSeek,
        отслеживая остановку изменения DOM-элемента ответа.
        """

        start_time = asyncio.get_event_loop().time()
        last_text = ""
        unchanged_counter = 0

        print("Ждем генерацию ответа от DeepSeek...")

        while True:
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError("Превышено максимальное время ожидания ответа.")

            try:
                # Находим все блоки ответов с классом .ds-markdown
                responses = await page.select_all('.ds-markdown')
                if responses:
                    current_text = responses[-1].text
                    
                    # Если текст появился и не изменяется в течение 2 секунд (4 цикла по 0.5с)
                    if current_text and current_text == last_text:
                        unchanged_counter += 1
                        if unchanged_counter >= 4:
                            return current_text
                    else:
                        unchanged_counter = 0
                        last_text = current_text
            except Exception:
                # Игнорируем временные DOM-ошибки во время рендеринга
                pass

            await asyncio.sleep(0.5)


    async def send_promt_and_get_json(self, page, prompt_text : str = None, file_path : str = None) -> str: 
        """
        Прикрепляет файл (если указан), вводит промпт и возвращает распарсенный JSON.
        """

        if prompt_text is None:
            prompt_text = self.prompt

        # 1. Загрузка файла через hidden input (если файл передан)
        if file_path:
            abs_file_path = str(Path(file_path).resolve())
            print(f"Прикрепляем файл: {abs_file_path}")
            
            file_input = await page.select('input[type="file"]')
            await file_input.send_keys(abs_file_path)
            
            # Задержка на обработку/превью файла интерфейсом
            await page.sleep(2)

        # 2. Поиск поля ввода текста
        input_box = await page.select('#chat-input')
        await input_box.send_keys(prompt_text)
        await page.sleep(0.5)

        # 3. Отправка запроса (нажатие Enter)
        await input_box.send_keys('\n')
        print("Запрос отправлен.")

        # 4. Ожидание завершения ответа
        raw_response = await self.wait_forд_stream(page)

        # 5. Извлечение JSON
        
        parsed_json = self.parser.extract_json_from_text(raw_response)
        return parsed_json