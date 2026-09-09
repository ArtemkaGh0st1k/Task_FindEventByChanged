from os.path import join
from os import getcwd


class Constant:
    PROFILE_DIR = None      # папка для сохранения профиля и cookie (абс.путь)
    DEEPSEEK_URL = "https://chat.deepseek.com/"  # URL для DeepSeek
    DIR_TO_UPDLOAD = join(getcwd(), "resources", "prompt_files")        # папка для загрузки файлов (абс.путь)
    PATH_DATASET = r"C:\Users\Артем\Desktop\Job\Task_FindEventByChanged\resources\promt_files\dataset1.xlsx"