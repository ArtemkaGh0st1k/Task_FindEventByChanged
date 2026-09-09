import re
import json


class DeepSeekParser:

    def extract_json_from_text(self, raw_text: str) -> dict:
        """
        Извлекает и валидирует JSON из ответа модели, даже если он обернут 
        в Markdown-блоки (```json ... ```) или содержит текст до/после.
        """
        
    # Удаляем блоки кода markdown
        cleaned_text = re.sub(r'```(?:json)?\s*|\s*```', '', raw_text).strip()
        
        # Ищем любую структуру { ... }
        match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        
        raise ValueError(f"Не удалось найти валидный JSON в тексте: {raw_text}")