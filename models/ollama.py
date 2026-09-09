from pathlib import Path
import pandas as pd
import ollama

from models.base_model import BaseModel
from parsers.ollama_parser import OllamaParser


class OllamaModel(BaseModel):
    def __init__(self, 
                 model_name : str = None,
                 base_prompt : str = None):
        super().__init__(model_name=model_name,
                         base_prompt=base_prompt,
                         parser = OllamaParser())


    def read_file_content(self, file_path: str) -> str:
        """Считывает данные из XLSX, CSV или TXT в текстовый формат для промпта."""
        path = Path(file_path).resolve()
        
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")

        if path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(path)
            return df.to_csv(index=False)
        elif path.suffix == '.csv':
            df = pd.read_csv(path)
            return df.to_csv(index=False)
        else:
            return path.read_text(encoding='utf-8')

    def analyze_with_deepseek(self, file_path: str = None, user_prompt: str = "") -> dict:
        """Отправляет запрос в локальную модель Ollama и возвращает dict."""
        
        # 1. Если передан файл, считываем его содержимое
        context_text = ""
        if file_path:
            print(f"Считываем файл: {file_path}")
            context_text = f"\n\nСодержимое файла:\n{self.read_file_content(file_path)}"

        # 2. Формируем системное указание и данные
        full_prompt = (
            "Ты — аналитик данных. Проанализируй входящий текст/данные и определи, "
            "упоминается ли в них какое-либо мероприятие и его дата.\n"
            "Ответ выведи СТРОГО в формате JSON без каких-либо вводных слов или пояснений.\n\n"
            "Формат JSON:\n"
            "{\n"
            '  "мероприятие": "да/нет",\n'
            '  "название": "строка или null",\n'
            '  "дата": "ДД.ММ.ГГ или null"\n'
            "}\n"
            f"{context_text}\n\n"
            f"Дополнительное указание: {user_prompt}"
        )

        print("Отправка запроса в локальный DeepSeek...")
        
        # 3. Вызов API Ollama
        response = ollama.chat(
            model=self.model_name,
            messages=\
            [
                {
                    'role': 'user',
                    'content': full_prompt
                }
            ],
            options=
            {
                'temperature': 0.1  # Низкая температура для строго соблюдения JSON
            }
        )

        raw_content = response['message']['content']
        
        # 4. Извлечение и парсинг JSON
        return self.parser.extract_json_from_text(raw_content)