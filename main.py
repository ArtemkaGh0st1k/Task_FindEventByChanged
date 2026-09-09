import asyncio
import json
from os.path import join
from os import getcwd
import nodriver as uc

from constants.path import Constant
from models.deepseek import DeepSeekModel

async def main():

    first_run = not Constant.PROFILE_DIR

    # Запуск браузера с профилем
    browser = await uc.start\
    (
        headless=False,  # Оставляем False, чтобы пройти авторизацию и не вызывать подозрения у Cloudflare
        user_data_dir=str(Constant.PROFILE_DIR)
    )

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
    FILE_TO_UPLOAD = join(getcwd(), "resources", "prompt_files", "dataset1.xlsx")
    try:
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
        browser.stop()

if __name__ == "__main__":
    uc.loop().run_until_complete(main())
    