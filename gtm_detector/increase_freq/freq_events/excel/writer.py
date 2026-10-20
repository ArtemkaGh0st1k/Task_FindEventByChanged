"""Выгрузка найденных мероприятий в Excel."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from freq_events.config import DetectorConfig
from freq_events.evaluation import EventMatcher
from freq_events.models import Event, Well

logger = logging.getLogger(__name__)


class EventsExcelWriter:
    """Пишет книгу из листов «Мероприятия», «Сравнение с разметкой» (если есть разметка)
    и «Параметры»."""

    EVENT_COLUMNS = {
        "number": "Скважина",
        "cluster": "Куст",
        "date_step": "Скачок частоты",
        "start": "Начало эффекта",
        "end": "Окончание эффекта",
        "end_reason": "Причина окончания",
        "f_before": "F до, Гц",
        "f_after": "F после, Гц",
        "delta_f": "ΔF, Гц",
        "n_steps": "Ступеней",
        "base_liquid": "Qж до",
        "liquid_at_start": "Qж в начале",
        "base_oil": "Qн до",
        "oil_at_start": "Qн в начале",
        "base_watercut": "Обв до, %",
        "responded": "Есть отклик",
        "after_restart": "После простоя",
        "confidence": "Уверенность",
    }

    def __init__(self, path: str | Path, matcher: EventMatcher | None = None) -> None:
        self.path = Path(path)
        self.matcher = matcher or EventMatcher()

    def write(
        self,
        wells: Sequence[Well],
        events_by_well: Mapping[str, Sequence[Event]],
        config: DetectorConfig,
    ) -> None:
        events = self._events_table(wells, events_by_well)
        comparison = self._comparison_table(wells, events_by_well)
        parameters = pd.Series(config.to_dict(), name="значение").map(str).to_frame()

        with pd.ExcelWriter(self.path) as writer:
            events.to_excel(writer, sheet_name="Мероприятия", index=False)
            if not comparison.empty:
                comparison.to_excel(writer, sheet_name="Сравнение с разметкой", index=False)
            parameters.to_excel(writer, sheet_name="Параметры")
        logger.info("Записано %d мероприятий в %s", len(events), self.path)

    def _events_table(
        self, wells: Sequence[Well], events_by_well: Mapping[str, Sequence[Event]]
    ) -> pd.DataFrame:
        rows = []
        for well in wells:
            for event in events_by_well.get(well.well_id, []):
                row = event.to_dict()
                row.update(number=well.number, cluster=well.cluster, end_reason=event.end_reason_title)
                rows.append(row)
        return pd.DataFrame(rows, columns=list(self.EVENT_COLUMNS)).rename(columns=self.EVENT_COLUMNS)

    def _comparison_table(
        self, wells: Sequence[Well], events_by_well: Mapping[str, Sequence[Event]]
    ) -> pd.DataFrame:
        rows = []
        for well in wells:
            if not well.labels:
                continue
            result = self.matcher.match(events_by_well.get(well.well_id, []), well.labels)
            for label, event in result.pairs:
                rows.append(
                    {
                        "Скважина": well.number,
                        "Куст": well.cluster,
                        "Дата выполнения": label.date_done,
                        "Начало (разметка)": label.start,
                        "Окончание (разметка)": label.end,
                        "Причина (разметка)": label.end_reason,
                        "Скачок F (найдено)": event.date_step if event else None,
                        "Начало (найдено)": event.start if event else None,
                        "Окончание (найдено)": event.end if event else None,
                        "Причина (найдено)": event.end_reason_title if event else None,
                        "Δ начала, дн": (event.start - label.start).days if event else None,
                        "Δ окончания, дн": (
                            (event.end - label.end).days if event and event.end and label.end else None
                        ),
                    }
                )
        return pd.DataFrame(rows)
