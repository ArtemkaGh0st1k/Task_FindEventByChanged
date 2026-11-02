## @module  [89450f18]
Тесты `WeightsArchive` (`models/archive.py`) на крошечных файлах-пустышках (`b"boost" * 1000`), без настоящих весов и без чтения Excel. Каждый тест имитирует ситуацию из жизни: `git clone` (только архив), `git pull` с новым архивом, `train` одной модели на свежем клоне. Всего в файле 8 тестов.

## archive  [09dff603]
Фикстура: временная папка `weights/` с двумя файлами, `gboost.joblib` (5000 байт) и `transformer.pt` (3000 байт), и объект `WeightsArchive(weights/weights.zip, weights/)`. Сам архив ещё не создан.

## test_pack_ignores_other_files  [5253da47]
Кладёт рядом `notes.txt` и проверяет, что `pack()` вернёт ровно `["gboost.joblib", "transformer.pt"]` и в архиве те же два имени. Поймал бы упаковку лишних файлов.

## test_pack_without_weights_fails  [06ac92ac]
В пустой папке `pack()` бросает `FileNotFoundError`. Поймал бы создание пустого архива, который потом «успешно» не найдёт ни одной модели.

## test_locate_extracts_missing_copy  [ee0690ed]
Как после `git clone`: упаковали, удалили рабочую копию `gboost.joblib`. `locate("gboost")` должен восстановить файл с прежним содержимым и не оставить `*.tmp`. Поймал бы отсутствие распаковки или мусор от неё.

## test_locate_replaces_stale_copy_of_same_size  [25b5a111]
Как после `git pull`: рабочая копия `transformer.pt` перезаписана `b"old" * 1000` — **того же размера**, что в архиве (3000 байт). `locate` обязан вернуть содержимое из архива. Поймал бы проверку «только по размеру», из-за которой `predict` работал бы со старыми весами.

## test_locate_keeps_matching_copy_untouched  [a1146383]
Копия совпадает с архивом: время изменения файла (`st_mtime_ns`) после `locate` не меняется. Поймал бы лишнюю перезапись 178 МБ при каждом `predict`.

## test_add_replaces_one_model_and_keeps_others  [fa2457fb]
Как `train gboost` на свежем клоне: рабочей копии `transformer.pt` нет, `gboost.joblib` переобучен (`b"retrained"`). После `add` в архиве оба имени, `gboost` отдаёт новое содержимое, `transformer` — прежнее. Поймал бы потерю остальных моделей при пересборке архива.

## test_locate_uses_copy_missing_from_archive  [6a1a9053]
Архива нет вовсе, файл лежит в папке: `locate("gboost")` возвращает его. Проверяет работу до первого `train`/`pack-weights` и вручную скопированные веса.

## test_locate_without_weights_fails  [5b340c32]
Модели `logreg` нет ни в архиве, ни в папке: `FileNotFoundError` с именем модели в сообщении (`match="logreg"`). Поймал бы возврат несуществующего пути, который упал бы позже с менее понятной ошибкой.
