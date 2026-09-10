from abc import ABC, abstractmethod
import re


class BaseParser(ABC):

    @abstractmethod
    def extract_json_from_text(self, raw_text: str) -> dict:
        pass

    def parse_by_model_name(self,
                            model_name : str,
                            raw_content : str,
                            output_filename : str = None,
                            elapsed_time : float = None):
        if output_filename is None:
            output_filename = model_name.replace(".", "_") +".txt"
        match model_name:
            case "deepseek-r1:1.5b":
                self.__parse_and_save_deepseek(raw_content, output_filename, elapsed_time)
            case "llama3.2:1b":
                self.__parse_and_save_llama(raw_content, output_filename, elapsed_time)
            case "qwen2.5:0.5b":
                self.__parse_and_save_qwen(raw_content, output_filename, elapsed_time)
            case _:
                raise ValueError(f"Неизвестная модель: {model_name}")

        print(f"Результат сохранен в: {output_filename}")
        

    def __parse_and_save_deepseek(self, raw_content: str, output_filename: str = "deepseek_result.txt", elapsed_time: str = None):
        """
        Парсер для deepseek-r1:1.5b.
        Разделяет блок <think>...</think> и итоговый ответ.
        """
        # 1. Извлекаем мысли модели из тега <think>
        think_match = re.search(r'<think>(.*?)</think>', raw_content, re.DOTALL)
        think_part = think_match.group(1).strip() if think_match else "Блок рассуждений отсутствует."

        # 2. Извлекаем чистый ответ (все, что идет после </think>)
        answer_part = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
        
        # 3. Очищаем markdown-обертки кода (```json или ```), если они есть
        cleaned_answer = re.sub(r'```(?:json)?\s*|\s*```', '', answer_part).strip()

        # 4. Формируем текстовый файл
        with open(output_filename, "w", encoding="utf-8") as f:
            if elapsed_time is not None:
                f.write(elapsed_time)
            
            f.write("=== ХОД РАССУЖДЕНИЙ (THINK) ===\n")
            f.write(f"{think_part}\n\n")
            
            f.write("=== ИТОГОВЫЙ ОТВЕТ ===\n")
            f.write(f"{cleaned_answer}\n")


    def __parse_and_save_llama(self, raw_content: str, output_filename: str = "llama_result.txt", elapsed_time: str = None):
        """
        Парсер для llama3.2:1b.
        Очищает Markdown-разметку кода и сохраняет чистый ответ.
        """
        # Очищаем разметку кода, если модель обернула ответ в ```json ... ```
        cleaned_answer = re.sub(r'```(?:json)?\s*|\s*```', '', raw_content).strip()

        with open(output_filename, "w", encoding="utf-8") as f:
            if elapsed_time is not None:
                f.write(elapsed_time)
                
            f.write("=== ИТОГОВЫЙ ОТВЕТ (LLAMA 3.2) ===\n")
            f.write(f"{cleaned_answer}\n")


    def __parse_and_save_qwen(self, raw_content: str, output_filename: str = "qwen_result.txt", elapsed_time: str = None):
        """
        Парсер для qwen2.5:0.5b.
        Удаляет лишние служебные символы и сохраняет чистый ответ.
        """
        # Удаляем возможные Markdown-блоки
        cleaned_answer = re.sub(r'```(?:json)?\s*|\s*```', '', raw_content).strip()

        with open(output_filename, "w", encoding="utf-8") as f:
            if elapsed_time is not None:
                f.write(elapsed_time)
                
            f.write("=== ИТОГОВЫЙ ОТВЕТ (QWEN 2.5) ===\n")
            f.write(f"{cleaned_answer}\n")

    