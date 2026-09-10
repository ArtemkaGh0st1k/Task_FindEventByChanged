import json

from models.ollama import OllamaModel
from constants.path import Constant


if __name__ == "__main__":

    model_names = \
    [
        "qwen2.5:0.5b",
        "llama3.2:1b",
        "deepseek-r1:1.5b",
    ]

    for model_name in model_names:

        try:
            model = OllamaModel(model_name=model_name)

            result = model.analyze_with_deepseek\
            (
                file_path=Constant.PATH_DATASET_CSV
            )
            
            print("\n" + "="*40)
            print("УСПЕШНО ПОЛУЧЕН JSON (Python Dict):")
            print("="*40)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
            print(f"\nМероприятие: {result.get('мероприятие')}")
            print(f"Название: {result.get('название')}")
            print(f"Дата: {result.get('дата')}")

        except Exception as e:
            print(f"\n[Ошибка]: {e}")
            continue
