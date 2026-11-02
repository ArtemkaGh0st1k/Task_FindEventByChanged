## @module  [d40d524d]
Zip-архив весов. Веса деревьев не помещаются в лимит GitHub (файл `extra_trees.joblib` — 178 МБ, а GitHub отклоняет файлы больше 100 МБ), но сжимаются примерно втрое. Поэтому в git лежит один `gtm_detector/ml_freq/weights/weights.zip` (79,4 МБ на пять моделей), а распакованные `.joblib` и `.pt` — только рабочие копии, они в `.gitignore`.

В цепочке «Excel → скважины → признаки → модель → интервалы» модуль стоит после модели: `cli.train` кладёт обученные веса в архив, `cli.predict` берёт их оттуда через `locate`, `cli.pack-weights` собирает архив из того, что лежит в папке. Архив — главный источник: рабочая копия, отличающаяся от него, заменяется.

## WeightsArchive  [26050f3c]
Архив `path` и папка `directory` с рабочими копиями весов. Публичные методы: `members`, `add`, `pack`, `locate`.

### Пример
```python
data = DataConfig()
archive = WeightsArchive(data.weights_archive, data.weights_dir)
archive.members()   # ['extra_trees.joblib', 'gboost.joblib', 'logreg.joblib', 'random_forest.joblib', 'transformer.pt']
```

## WeightsArchive.__init__  [d24e2365]
Запоминает пути `path` (файл `weights.zip`) и `directory` (папка `weights/`) как `Path`. Файловую систему не трогает: архива может ещё не быть.

```python
WeightsArchive(Path("weights/weights.zip"), Path("weights"))
```

## WeightsArchive.members  [8ab0a515]
Имена файлов внутри архива. Нет архива — пустой список (а не ошибка): так `locate` работает до первого `train`.

```python
archive.members()   # [] на чистой папке; 5 имён после pack-weights
```

## WeightsArchive.add  [bf6270c4]
Записывает `files` в архив, остальные модели оставляет как были. Одноимённый файл заменяется новым. Архив собирается заново во временном файле `weights.zip.tmp` и подменяет прежний только в конце, поэтому прерванный запуск не испортит старый архив. Сжатие `ZIP_DEFLATED`, уровень 9 (`COMPRESS_LEVEL`): архив пересобирается редко, а размер в git важнее времени. Пересборка пяти моделей занимает около 75 секунд — это добавка ко времени `train`.

Остальные модели переписываются из старого архива потоком по 1 МиБ (`CHUNK`), без распаковки на диск: после `train` на свежем клоне рабочих копий других моделей может не быть.

### Пример
```python
archive.add([Path("weights/gboost.joblib")])   # после train --model gboost
archive.members()                              # прочие четыре модели на месте
```
Сжатие на реальных весах: extra_trees 170,3 → 52,7 МБ, random_forest 81,0 → 26,2 МБ, gboost 1,2 → 0,5 МБ.

## WeightsArchive.pack  [08390eef]
Записывает в архив все файлы `.pt` и `.joblib` из папки (`SUFFIXES`) и возвращает их имена по алфавиту. Посторонние файлы (в том числе сам `weights.zip`) игнорируются. Нет ни одного файла — `FileNotFoundError("Нет весов в …: сначала выполните train")`. Вызывается командой `pack-weights`.

```python
archive.pack()   # ['extra_trees.joblib', 'gboost.joblib', 'logreg.joblib', 'random_forest.joblib', 'transformer.pt']
```

## WeightsArchive.locate  [dc6e4bfb]
Возвращает путь к весам модели `name` (`'gboost'`, `'transformer'`), при необходимости распаковав их. Для каждого суффикса из `SUFFIXES` (сначала `.pt`, затем `.joblib`):

1. файл есть в архиве — вызывается `_extract` (распакует, только если рабочей копии нет или она отличается), путь возвращается;
2. в архиве нет, но лежит в папке — возвращается рабочая копия (скопирована вручную, в архив ещё не добавлена);
3. нигде нет — `FileNotFoundError("Нет весов модели logreg в … : сначала выполните train")`.

Распаковывается только запрошенная модель: `predict --model logreg` не тратит время на 170 МБ extra_trees.

### Пример
```python
path = archive.locate("gboost")   # .../weights/gboost.joblib
model = load_model(path)
```
После `git clone` в `weights/` только `weights.zip`; первый `predict --model gboost` распакует один файл.

## WeightsArchive._extract  [cfd34ab7]
Приводит рабочую копию `file` в соответствие с архивом. Если файл существует, его размер равен `file_size` записи в архиве и CRC32 совпадает — ничего не делает (дата изменения файла не меняется). Иначе пишет файл во временный `*.tmp` и переименовывает: оборванная распаковка не оставит битый файл.

Сравнивается не только размер, но и CRC: после `git pull` с переобученным архивом размер новой модели может совпасть со старой, а содержимое — нет.

```python
archive._extract(Path("weights/gboost.joblib"))   # копия актуальна — пропуск; устарела — перезапись
```

## _crc32  [635e2ca7]
CRC32 файла, прочитанного кусками по `CHUNK` (1 МиБ), чтобы 178 МБ не читались в память целиком. Возвращает число, сравнимое с `ZipInfo.CRC`.

```python
_crc32(Path("weights/logreg.joblib")) == zipfile.ZipFile(path).getinfo("logreg.joblib").CRC   # True
```
