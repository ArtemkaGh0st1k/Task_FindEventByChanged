"""Конфигурация ML-детектора мероприятий «увеличение частоты F»."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
RESOURCES = PROJECT_ROOT / "resources"


@dataclass(frozen=True)
class DataConfig:
    """Где лежат данные и куда писать результаты.

    ``dataset="merged"`` — объединённые файлы 2021–2026 (по умолчанию),
    ``dataset="test"`` — первые тестовые файлы из ``resources/test_datas``.
    """

    dataset: str = "merged"
    statistics: Path = RESOURCES / "Статистика_с_2021_2026_в_одном_файле.xlsx"
    labels: Path = RESOURCES / "Итоговый_файл_c_2021_2026.xlsx"  # «c» в имени — латинская
    # Исходная выгрузка за 2026 г.: только для проверки сдвигов, в обучение не идёт.
    reference: Path | None = RESOURCES / "test_datas" / "input_dataset.xlsx"
    measurements: Path = RESOURCES / "test_datas" / "input_dataset.xlsx"
    results: Path = RESOURCES / "test_datas" / "output_dataset.xlsm"
    output_dir: Path = PROJECT_ROOT / "results" / "ml_freq"
    weights_dir: Path = PACKAGE_DIR / "weights"


@dataclass(frozen=True)
class DecoderConfig:
    """Как превратить вероятности по дням в мероприятия."""

    threshold: float = 0.5  # день «в эффекте», если сглаженная вероятность выше
    smooth_days: int = 5  # окно скользящего среднего вероятностей
    max_gap: int = 3  # разрывы короче — склеиваются в один период
    min_length: int = 2  # более короткие периоды отбрасываются


# Дней до начала и после окончания размеченного мероприятия, на которых учатся модели.
# Остальные дни не используются: в разметке есть не все реальные подъёмы частоты.
CONTEXT_DAYS = 45


@dataclass(frozen=True)
class TransformerConfig:
    """Архитектура Transformer-энкодера для временных рядов."""

    # Небольшая модель (~20 тыс. параметров): на ~100 скважинах крупнее — хуже
    # (d_model=64, 3 слоя: точное начало 16% против 24% на перекрёстной проверке).
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 64
    dropout: float = 0.2
    conv_kernel: int = 5  # свёртка перед энкодером ловит локальные ступеньки
    max_length: int = 4096


@dataclass(frozen=True)
class TrainingConfig:
    """Обучение Transformer."""

    epochs: int = 80
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    crop_length: int = 128  # обучение на случайных окнах ряда — аугментация
    crops_per_well: int = 4
    seed: int = 0


@dataclass(frozen=True)
class ExperimentConfig:
    """Перекрёстная проверка по скважинам."""

    folds: int = 5
    seed: int = 0
    start_tolerance: int = 7  # дней: допуск при сопоставлении с разметкой
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
