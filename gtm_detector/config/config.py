from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class PipelineConfig:
    """Конфигурация параметров работы пайплайна."""

    q_base_window_days: int = 7     # Окно расчета базового дебита до ГТМ
    q_recovery_tolerance: float = 1.05  # Допуск возврата дебита к базовому (5%)
    idle_days_threshold: int = 2  # Порог дней простоя для квалификации замены насоса
    smoothing_window: int = 3  # Окно сглаживания дебита нефти
    gtm_keywords: list[str] = field(
        default_factory=lambda: [
            "ГРП",
            "ОПЗ",
            "СКО",
            "КРС",
            "ПРС",
            "перфорация",
            "перестрелка",
            "автооткл",
            "смена ЭЦН",
            "замена",
        ]
    )


@dataclass
class GTMVerdict:
    """Результат анализа сквадины"""

    well_id: str
    has_gtm: bool
    gtm_type: Optional[str] = None
    start_date: datetime
    end_date: datetime
    reasoning: str = ""     