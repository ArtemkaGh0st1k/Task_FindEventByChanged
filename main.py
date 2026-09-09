import asyncio
import json
from os.path import join
from os import getcwd
from pathlib import Path
import subprocess
import nodriver as uc

from constants.path import Constant
from models.deepseek import DeepSeekModel
from helpers.converter import FileConverter

# Путь к исполняемому файлу Edge
EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PROFILE_DIR = Path("./edge_profile").resolve()  # Абсолютный путь к папке профиля Edge для сохранения сессии
CDP_PORT = 9222


async def main():

    first_run = not PROFILE_DIR.exists()  # Проверяем, существует ли папка профиля
    browser = None  # Инициализация переменной browser

    edge_cmd = \
        [
            EDGE_PATH,
            f"--remote-debugging-port={CDP_PORT}",
            f"--user-data-dir={PROFILE_DIR}",
            "--no-first-run",
            "--no-default-browser-check"
        ]

    print("Запуск процесса Microsoft Edge...")
    edge_process = subprocess.Popen(edge_cmd)

    await asyncio.sleep(10)  # Ждем, пока Edge запустится и откроется порт для удаленной отладки

    try:
        # 2. Подключаемся nodriver к запущенному порту Edge
        print(f"Подключение nodriver к 127.0.0.1:{CDP_PORT}...")
        
        config = uc.Config(host="127.0.0.1", port=CDP_PORT, 
                           browser_executable_path=EDGE_PATH)
        browser = await uc.Browser.create(config=config)
        # Переходим на сайт
        page = await browser.get(Constant.DEEPSEEK_URL)

        if first_run:
            print("\n" + "="*60)
            print("ПЕРВЫЙ ЗАПУСК:")
            print("1. Авторизуйтесь на сайте DeepSeek.")
            print("2. Пройдите капчу Cloudflare, если появится.")
            print("3. После появления интерфейса чата нажмите Enter в этой консоли...")
            print("="*60 + "\n")
            await asyncio.get_event_loop().run_in_executor(None, input)
            print("Сессия успешно сохранена!")
        else:
            print("Профиль загружен. Ожидание готовности интерфейса...")
            await page.sleep(4)

        deep_seek_model = DeepSeekModel()

        FILE_PATH = r"C:\Users\Артем\Desktop\Job\Task_FindEventByChanged\resources\promt_files\dataset1.xlsx"
        FILE_TO_UPLOAD = FileConverter.convert_xlsx_to_csv(FILE_PATH)  # Конвертируем XLSX в CSV

        # Выполняем отправку и получаем распарсенный результат
        result_data = await deep_seek_model.send_promt_and_get_json(
            page=page,  
            file_path=FILE_TO_UPLOAD
        )
        
        print("\n" + "="*40)
        print("УСПЕШНО ПОЛУЧЕН JSON (Python Dictionary):")
        print("="*40)
        print(json.dumps(result_data, ensure_ascii=False, indent=2))
        
        # Доступ к полям объекта
        print(f"\nМероприятие: {result_data.get('мероприятие')}")
        print(f"Дата: {result_data.get('дата')}")

    except Exception as e:
        print(f"\n[Ошибка]: {e}")
        
    finally:
        print("\nЗавершение работы браузера...")
        if browser:
            browser.stop()

if __name__ == "__main__":
    uc.loop().run_until_complete(main())
    