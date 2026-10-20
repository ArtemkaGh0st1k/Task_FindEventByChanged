import json
import re

from parsers.base_parser import BaseParser


class OllamaParser(BaseParser):

    def extract_json_from_text(self, raw_text: str) -> dict:
        """
        Извлекает JSON из ответа модели, вырезая блок рассуждений <think>...</think>
        и любые Markdown-оболочки.
        """
        # 1. Удаляем внутренний блок мыслей R1-модели <think>...</think>
        cleaned = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
        
        # 2. Очищаем markdown-блоки ```json ... ```
        cleaned = re.sub(r'```(?:json)?\s*|\s*```', '', cleaned).strip()
        
        # 3. Находим первую фигурную скобку { ... }
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        
        raise ValueError(f"Не удалось распарсить JSON из ответа: {raw_text}")