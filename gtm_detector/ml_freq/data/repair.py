"""Исправление сдвинутых блоков в склеенной статистике.

Макрос собирал файл из отдельных выгрузок и часть из них записал на 180 дней
позже (смещение на фиксированное число столбцов). Сдвиг видно двумя способами:

1. **Разметка.** На дату начала мероприятия частота должна подняться
   (:class:`FrequencyCheck`). В сдвинутом блоке подъём находится на 180 дней позже.
2. **Исходная выгрузка** (если есть): скважины блока дословно совпадают с ней
   при правильном сдвиге.

Календарь делится на сегменты — периоды с одним набором скважин (след одной
выгрузки). Для каждого сегмента проверяются сдвиги 0 и 180 дней и выбирается
подтверждённый; сегмент, где доказательства противоречат обоим, исключается.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SHIFT_DAYS = 180


class FrequencyCheck:
    """Подтверждают ли замеры подъём частоты с ``current`` до ``new`` Гц около дня ``i``.

    Подъём засчитывается, если медиана частоты за неделю после дня выросла
    относительно недели до него хотя бы на ``MIN_RISE`` и при этом

    * дошла до «Нового значения» (с допуском) — или
    * прирост не меньше половины запланированного (``new − current``): телеметрия
      бывает смещена относительно уставки технолога на несколько Гц, прирост — нет.

    Частоту часто поднимают ступенями за 1–2 недели, поэтому годится любое из двух.
    """

    WINDOW = 7  # дней до и после скачка
    TOLERANCE = 3  # дней: скачок может быть чуть раньше или позже даты технолога
    MIN_RISE = 0.3  # Гц
    LEVEL_TOLERANCE = 1.5  # Гц (или 3 % от «Нового значения», если больше)
    MIN_PLANNED_SHARE = 0.5
    MIN_WORKING = 10.0  # Гц: меньше — насос стоит или нет телеметрии
    MIN_VALUES = 3

    def confirms(self, frequency: np.ndarray, i: int, current: float, new: float) -> bool | None:
        """``True``/``False`` — подтверждён/нет; ``None`` — около дня ``i`` нет замеров."""
        planned = new - current if np.isfinite(new - current) else np.inf
        tolerance = max(self.LEVEL_TOLERANCE, 0.03 * new)
        tested = False
        for t in range(i - self.TOLERANCE, i + self.TOLERANCE + 1):
            before = self._working(frequency[max(t - self.WINDOW, 0) : max(t, 0)])
            after = self._working(frequency[max(t, 0) : max(t + self.WINDOW, 0)])
            if len(before) < self.MIN_VALUES or len(after) < self.MIN_VALUES:
                continue
            tested = True
            level = np.median(after)
            rise = level - np.median(before)
            if rise >= self.MIN_RISE and (
                abs(level - new) <= tolerance or rise >= self.MIN_PLANNED_SHARE * planned
            ):
                return True
        return False if tested else None

    def _working(self, values: np.ndarray) -> np.ndarray:
        return values[values > self.MIN_WORKING]


@dataclass
class Evidence:
    """Сколько объектов проверено (``tested``) и сколько подтвердило сдвиг (``confirmed``)."""

    confirmed: int = 0
    tested: int = 0

    @property
    def share(self) -> float:
        return self.confirmed / self.tested if self.tested else 0.0


@dataclass
class Segment:
    """Период с одним набором скважин — след одной выгрузки."""

    first: int  # позиции в календаре, включительно
    last: int
    start: date
    end: date
    wells: int
    cells: int  # ячеек с данными
    labels: dict[int, Evidence] = field(default_factory=dict)  # сдвиг -> мероприятия разметки
    reference: dict[int, Evidence] = field(default_factory=dict)  # сдвиг -> скважины исходной выгрузки
    shift: int | None = 0  # None — сегмент исключён
    verdict: str = ""
    collisions: int = 0  # ячеек Qж, куда сдвиг попал на уже занятые даты (оставлены прежние)

    @property
    def days(self) -> int:
        return self.last - self.first + 1

    @property
    def decision(self) -> str:
        if self.shift is None:
            return "исключён"
        return f"сдвиг −{self.shift} дн" if self.shift else "без сдвига"


@dataclass(frozen=True)
class LabelPoint:
    """Мероприятие разметки в координатах таблиц: строка скважины, день начала, частоты до/после."""

    row: int
    day: int
    current: float
    new: float


class BlockShiftRepair:
    """Находит сегменты, выбирает для каждого сдвиг и применяет его."""

    def __init__(
        self,
        shifts: Sequence[int] = (0, SHIFT_DAYS),
        min_flip: int = 5,
        seam_days: int = 3,
        min_confirmed: int = 3,
        min_share: float = 0.25,
        well_match: float = 0.9,
        reference_share: float = 0.5,
        min_reference_wells: int = 3,
        check: FrequencyCheck | None = None,
    ) -> None:
        self.shifts = tuple(shifts)
        self.min_flip = min_flip  # столько скважин должно появиться/пропасть, чтобы начался новый сегмент
        self.seam_days = seam_days  # сегменты короче — стыки выгрузок
        self.min_confirmed = min_confirmed  # по разметке: подтверждённых мероприятий не меньше
        self.min_share = min_share  # и их доля среди проверенных не меньше
        self.well_match = well_match  # скважина совпала с выгрузкой: равны не меньше этой доли ячеек
        self.reference_share = reference_share  # сегмент совпал: не меньше этой доли скважин
        self.min_reference_wells = min_reference_wells
        self.check = check or FrequencyCheck()

    def __repr__(self) -> str:  # входит в ключ кэша
        params = {k: v for k, v in vars(self).items() if k != "check"}
        check = {k: v for k, v in vars(type(self.check)).items() if k.isupper()}
        return f"BlockShiftRepair({params}, check={check})"

    def segments(self, presence: np.ndarray, dates: pd.DatetimeIndex) -> list[Segment]:
        """Делит календарь там, где набор скважин с данными резко меняется."""
        flips = (presence[:, 1:] != presence[:, :-1]).sum(axis=0)
        cuts = [0, *(np.flatnonzero(flips >= self.min_flip) + 1).tolist(), presence.shape[1]]
        return [
            Segment(
                first=a,
                last=b - 1,
                start=dates[a].date(),
                end=dates[b - 1].date(),
                wells=int(presence[:, a:b].any(axis=1).sum()),
                cells=int(presence[:, a:b].sum()),
            )
            for a, b in zip(cuts[:-1], cuts[1:], strict=True)
            if presence[:, a:b].any()
        ]

    def assess(
        self,
        segments: Sequence[Segment],
        presence: np.ndarray,
        frequency: np.ndarray,
        labels: Sequence[LabelPoint],
        reference: np.ndarray | None = None,
        observed: np.ndarray | None = None,
    ) -> None:
        """Заполняет доказательства и решение каждого сегмента.

        ``reference`` — исходная выгрузка в тех же координатах (NaN, где её нет),
        ``observed`` — тот же параметр склеенного файла.
        """
        for segment in segments:
            wells = presence[:, segment.first : segment.last + 1].any(axis=1)
            for shift in self.shifts:
                segment.labels[shift] = self._label_evidence(segment, shift, wells, frequency, labels)
                if reference is not None and observed is not None:
                    segment.reference[shift] = self._reference_match(segment, shift, reference, observed)
            segment.shift, segment.verdict = self._decide(segment)

        # Стык выгрузок (1–2 дня) проверить нечем — он принадлежит соседнему сегменту.
        for i in reversed(range(len(segments))):
            segment = segments[i]
            if segment.days >= self.seam_days or not segment.verdict.startswith("проверить нечем"):
                continue
            neighbour = segments[i + 1] if i + 1 < len(segments) else segments[i - 1] if i else None
            if neighbour is not None:
                segment.shift = neighbour.shift
                segment.verdict = f"стык выгрузок: как у сегмента {neighbour.start} — {neighbour.end}"

    def apply(
        self, tables: Mapping[str, pd.DataFrame], segments: Sequence[Segment]
    ) -> dict[str, pd.DataFrame]:
        """Таблицы после исправления. Сначала несдвинутые сегменты, затем сдвинутые —
        при наложении остаются несдвинутые данные (их место подтверждено)."""
        order = sorted((s for s in segments if s.shift is not None), key=lambda s: s.shift)
        repaired = {}
        for name, table in tables.items():
            source = table.to_numpy(dtype=float)
            target = np.full_like(source, np.nan)
            for segment in order:
                a, b = segment.first, segment.last + 1
                clip = max(0, segment.shift - a)  # сдвиг за начало календаря
                block = source[:, a + clip : b]
                place = target[:, a + clip - segment.shift : b - segment.shift]
                free = np.isnan(place)
                if name == "debit_liquid":
                    segment.collisions = int((~free & ~np.isnan(block)).sum())
                place[free] = block[free]
            repaired[name] = pd.DataFrame(target, index=table.index, columns=table.columns)
        return repaired

    def _label_evidence(
        self,
        segment: Segment,
        shift: int,
        wells: np.ndarray,
        frequency: np.ndarray,
        labels: Sequence[LabelPoint],
    ) -> Evidence:
        evidence = Evidence()
        for label in labels:
            day = label.day + shift  # где подъём должен быть в склеенном файле
            if not wells[label.row] or not segment.first <= day <= segment.last:
                continue
            series = frequency[label.row, segment.first : segment.last + 1]
            result = self.check.confirms(series, day - segment.first, label.current, label.new)
            if result is not None:
                evidence.tested += 1
                evidence.confirmed += int(result)
        return evidence

    def _reference_match(
        self, segment: Segment, shift: int, reference: np.ndarray, observed: np.ndarray
    ) -> Evidence:
        """Сколько скважин сегмента дословно совпадает с исходной выгрузкой при данном сдвиге."""
        a, b = segment.first, segment.last + 1
        evidence = Evidence()
        if a - shift < 0:
            return evidence
        mine = observed[:, a:b]
        theirs = reference[:, a - shift : b - shift]
        comparable = ~np.isnan(mine) & ~np.isnan(theirs) & (theirs != 0)
        for row in np.flatnonzero(comparable.any(axis=1)):
            cells = comparable[row]
            evidence.tested += 1
            evidence.confirmed += int(
                np.isclose(mine[row, cells], theirs[row, cells]).mean() >= self.well_match
            )
        return evidence

    def _decide(self, segment: Segment) -> tuple[int | None, str]:
        reference = {s: e for s, e in segment.reference.items() if e.tested >= self.min_reference_wells}
        matching = [s for s, e in reference.items() if e.share >= self.reference_share]
        if matching:
            shift = max(matching, key=lambda s: reference[s].share)
            e = reference[shift]
            return shift, f"совпадает с исходной выгрузкой: {e.confirmed} из {e.tested} скважин"

        confirmed = [
            s
            for s, e in segment.labels.items()
            if e.confirmed >= self.min_confirmed and e.share >= self.min_share
        ]
        if confirmed:
            shift = max(confirmed, key=lambda s: (segment.labels[s].share, segment.labels[s].confirmed))
            e = segment.labels[shift]
            return shift, f"разметка подтверждает: {e.confirmed} из {e.tested} мероприятий"

        if reference:
            parts = ", ".join(f"{s} дн — {e.confirmed}/{e.tested}" for s, e in sorted(reference.items()))
            return None, f"не совпадает с исходной выгрузкой ни при одном сдвиге (скважин: {parts})"
        tested = sum(e.tested for e in segment.labels.values())
        if tested < self.min_confirmed:
            return 0, f"проверить нечем (мероприятий: {tested}) — оставлен на месте"
        parts = ", ".join(f"{s} дн — {e.confirmed}/{e.tested}" for s, e in sorted(segment.labels.items()))
        return None, f"разметка не подтверждает ни один сдвиг ({parts})"
