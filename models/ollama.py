from pathlib import Path
import pandas as pd
from os import getcwd
from os.path import join
import time
import ollama

from helpers.time import TimeHelper
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

    def analyze_with_deepseek(self,
                            file_path: str = None,
                            user_prompt: str = "",
                            save_result: bool = True) -> dict:
        """Отправляет запрос в локальную модель Ollama и возвращает dict."""
        
        # 1. Если передан файл, считываем его содержимое
        context_text = ""
        if file_path:
            print(f"Считываем файл: {file_path}")
            file_content = open(file_path, encoding='utf-8').read()
            context_text = f"\n\nСодержимое файла:\n{file_content}"

        # 2. Формируем системное указание и данные
        full_prompt = f"{self.base_prompt}\n" + \
            f"{context_text}\n" + \
            f"Дополнительное указание: {user_prompt}"

        print(f"Отправка запроса в локальную модель {self.model_name}...")
        
        start_time = time.time()

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

        end_time = time.time()
        elapsed_time = TimeHelper.get_elapsed_time(start_time, end_time)

        raw_content = response['message']['content']

        if save_result:
            self.parser.parse_by_model_name(model_name=self.model_name,
                                        raw_content=raw_content,
                                        elapsed_time=elapsed_time)

        print(f"Время выполнения: {elapsed_time}\n\n")
        print(f"\n\n {raw_content} \n\n")

        # 4. Извлечение и парсинг JSON
        return self.parser.extract_json_from_text(raw_content)