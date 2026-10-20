from os.path import join
from os import getcwd
from pathlib import Path


class Constant:
    PROFILE_DIR = None      # папка для сохранения профиля и cookie (абс.путь)
    DEEPSEEK_URL = "https://chat.deepseek.com/"  # URL для DeepSeek
    DIR_TO_UPDLOAD = join(getcwd(), "resources", "prompt_files")        # папка для загрузки файлов (абс.путь)
    PATH_DATASET_XLSX = r"C:\Users\Артем\Desktop\Job\Task_FindEventByChanged\resources\promt_files\dataset1.xlsx"
    PATH_DATASET_CSV = r"C:\Users\Артем\Desktop\Job\Task_FindEventByChanged\resources\promt_files\dataset1.csv"
    WEIGHT_PATH = join("gtm_detector", "models", "weights", "gtm_1dcnn_weights.pth")

    # Детектор «увеличение частоты F» (gtm_detector/increase_freq).
    # Пути считаются от корня проекта, а не от текущей папки.
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    INPUT_DATASET = PROJECT_ROOT / "resources" / "input_dataset.xlsx"  # замеры технологов
    OUTPUT_DATASET = PROJECT_ROOT / "resources" / "output_dataset.xlsm"  # расставленные даты
    INCREASE_FREQ_PARAMS = PROJECT_ROOT / "gtm_detector" / "increase_freq" / "params.json"
    INCREASE_FREQ_RESULTS_DIR = PROJECT_ROOT / "results" / "increase_freq"