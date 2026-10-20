Проект создан для решения задад детекции начала `ГТМ`
и конца `ГТМ`.

Рабочий подход - `gtm_detector/increase_freq` (детектор ГТМ `увеличение частоты F`),
точка входа - `main.py`.

---

# Запуск

1) Окружение (один раз, нужен Python 3.11+):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r req.txt
```

2) Данные лежат в `resources/`: `input_dataset.xlsx` (замеры) и
`output_dataset.xlsm` (расставленные даты). Пути - в `constants/path.py`.

3) Полный прогон - поиск ГТМ, сравнение с расставленными датами, графики
(~1-2 мин):

```bash
python main.py
```

Всё складывается в папку прогона `results/increase_freq/<дата_время>/`:
- `events.xlsx` - найденные ГТМ, сравнение с разметкой, параметры;
- `summary.png` - сводка качества на одной странице;
- `wells/совпало`, `wells/ошибка_дат`, `wells/пропущено` - графики скважин
(F, Qж, Qн, обводнённость, простои; найденный и размеченный периоды).

Метрики совпадения с разметкой печатаются в консоль.

4) Другие режимы:

```bash
python main.py detect                 # только поиск ГТМ -> events.xlsx (когда разметки нет)
python main.py detect --plots         # поиск + графики всех скважин с найденными ГТМ
python main.py detect --results resources/output_dataset.xlsm --plots --all-wells
                                      # + графики скважин без разметки (wells/нет_в_разметке)
python main.py calibrate --cv 5       # подбор параметров по разметке -> params.json
python main.py detect --measurements путь\к\замерам.xlsx --out-dir путь\к\папке
python main.py --help
```

5) Тесты (по желанию): `pip install pytest`, затем
`python -m pytest gtm_detector/increase_freq/tests`.

В ходе было реализовано два подхода:

--- 

# prompt-enineering

1) Попытка дотянутся до сайта `deepseek` и скормить ему 
некий промпт, который содержал последовательную инструкцию,
дополнительную информацию и необходимые данные.
Далее после запроса мы должны были спарсить эти данные
в определённый формат.

---

--- 

# neural-network

2) Второй способ - сделать нейронную сеть.
Входной слой - данные замеров технологов.
Выходной слой - результат в виде дат начала и окончания ГТМ.
**Причем берем только ГТМ - увеличение частоты!**

---

--- 

# gtm_detector/increase_freq

3) Третий способ - детектор на правилах для ГТМ `увеличение частоты F`.
По замерам ищется скачок частоты, затем начало эффекта (рост Qж/Qн выше
режима «до») и его окончание (Qн вернулся к уровню «до» или остановка скважины).
Пороги правил подбираются по расставленным датам (`resources/output_dataset.xlsm`).

Запуск из корня проекта:

```bash
python -m gtm_detector.increase_freq detect --results resources/output_dataset.xlsm
```

Подробности - в `gtm_detector/increase_freq/README.md` и
`gtm_detector/increase_freq/Описание_работы_кода.docx`.

---

---

# dirs

`constants`:
- `path.py`: хранит класс констр.переменн (url, пути к сохр и тд),
в т.ч. пути к `resources/*` и `params.json` для `increase_freq`

`gtm_detector`:
- `increase_freq/`: детектор ГТМ `увеличение частоты F` на правилах

`helpers`:
- `converter.py`: вспомогательный класс для конвертации (не использовал)
- `time.py`: статичный класс для работы со временем; создаёт папки по датам

`resources`:
- `input_dataset.xlsx`: Данные с замеров по скважинам (по добыче нефти,
жидкости, давлений, частоты, состояний скважин)
- `output_dataset.xlsm`: Данные технологов с расставленными датами (26 год).
`results`: 
- `success_and_has_dates.csv`: Файлик с отфильтрованными скважинами (
но не только с `увеличением F`).

- `success_and_has_dates/`
    `all_gtms`: Все виды ГТМ с первоначального файлика Рината,
    которые успешны и имеют обе даты.
    `increase_f_gtm`: Только ГТМ с увеличением F
- `increase_freq/<дата_время>/`: прогоны детектора `gtm_detector/increase_freq`
(`events.xlsx`, `summary.png`, графики скважин в `wells/`)

---