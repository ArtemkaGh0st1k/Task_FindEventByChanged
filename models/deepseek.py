import asyncio
from pathlib import Path
import json

from models.base_model import BaseModel
from parsers.deepseek_parser import DeepSeekParser


class DeepSeekModel(BaseModel):
    def __init__(self, 
                 model_name : str = None,
                 base_prompt : str = None):
        super().__init__(model_name=model_name,
                        base_prompt=base_prompt,
                        parser=DeepSeekParser())

    async def _wait_for_element(self, page, selectors: list, timeout: int = 10):
        """Ожидает появления хотя бы одного из указанных элементов на странице."""
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            for selector in selectors:
                try:
                    element = await page.select(selector)
                    if element:
                        return element
                except Exception:
                    pass
            await asyncio.sleep(0.5)
        return None

    async def wait_for_deepseek_stream(self, page, timeout: int = 120) -> str:
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


    async def send_promt_and_get_json(self, page, prompt_text: str = None, file_path: str = None) -> dict:
        """
        Загружает файл (если передан), вводит промпт, отправляет запрос 
        и возвращает распарсенный результат в виде Python-словаря (dict).
        """

        if prompt_text is None:
            prompt_text = self.base_prompt
        
        # -------------------------------------------------------------
        # ШАГ 1: Загрузка файла через истинный элемент <input type="file">
        # -------------------------------------------------------------
        if file_path:
            abs_file_path = str(Path(file_path).resolve())
            print(f"Прикрепляем файл: {abs_file_path}")

            file_input = None
            for _ in range(10):
                try:
                    file_inputs = await page.select_all('input[type="file"]')
                    if file_inputs:
                        file_input = file_inputs[0]
                        break
                except Exception:
                    pass
                await asyncio.sleep(0.5)

            if file_input:
                await file_input.send_keys(abs_file_path)
                print("Файл успешно передан в файловый инпут.")
                # Пауза для обработки файла интерфейсом и отрисовки бейджа
                await page.sleep(3)
            else:
                print("[Предупреждение] Не удалось найти <input type='file'> на странице.")

        # -------------------------------------------------------------
        # ШАГ 2: Ввод промпта и активация React/Vue событий
        # -------------------------------------------------------------
        print("Поиск текстового поля ввода...")
        input_box = await self._wait_for_element(page, ['#chat-input', 'textarea'], timeout=10)

        if not input_box:
            raise RuntimeError("Не удалось найти текстовое поле (#chat-input / textarea).")

        if prompt_text:
            # Вставляем текст и генерируем события input/change через JS, 
            # чтобы веб-клиент разблокировал кнопку отправки
            escaped_prompt = json.dumps(prompt_text)
            await page.evaluate(f'''
                const el = document.querySelector('#chat-input') || document.querySelector('textarea');
                if (el) {{
                    el.value = {escaped_prompt};
                    el.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                }}
            ''')
            await page.sleep(1)

        # -------------------------------------------------------------
        # ШАГ 3: Точная отправка запроса (Без задевания меню слева)
        # -------------------------------------------------------------
        print("Отправка запроса...")

        # Поиск и клик по кнопке «Отправить» строго внутри контейнера поля ввода
        sent_via_js = await page.evaluate('''
            () => {
                const chatInput = document.querySelector('#chat-input') || document.querySelector('textarea');
                if (!chatInput) return false;

                // Поднимаемся до общего родительского контейнера зоны ввода
                const inputContainer = chatInput.closest('div[class*="input"]') || chatInput.parentElement.parentElement;
                if (!inputContainer) return false;

                // Находим все кнопки или элементы с роли button ВНУТРИ зоны ввода
                const buttons = Array.from(inputContainer.querySelectorAll('div[role="button"], button'));
                
                // Ищем кнопку отправки (содержит SVG и не является кнопкой прикрепления)
                const sendBtn = buttons.reverse().find(btn => {
                    const hasSvg = btn.querySelector('svg');
                    const isAttach = btn.getAttribute('aria-label')?.toLowerCase().includes('attach') || 
                                     btn.getAttribute('aria-label')?.toLowerCase().includes('file');
                    return hasSvg && !isAttach && btn.offsetWidth > 0;
                });

                if (sendBtn) {
                    sendBtn.click();
                    return true;
                }
                return false;
            }
        ''')

        if not sent_via_js:
            print("Кнопка не найдена через JS, отправка через эмуляцию Enter...")
            await input_box.send_keys('\n')

        print("Запрос отправлен. Ожидание ответа...")

        # -------------------------------------------------------------
        # ШАГ 4: Ожидание завершения ответа и парсинг JSON
        # -------------------------------------------------------------
        raw_response = await self.wait_for_deepseek_stream(page)
        return self.parser.extract_json_from_text(raw_response)