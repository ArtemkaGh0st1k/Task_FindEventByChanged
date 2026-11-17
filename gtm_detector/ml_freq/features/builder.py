"""Признаки по дням для моделей.

* :meth:`FeatureBuilder.sequence` — матрица «день × канал» для Transformer:
  модель сама смотрит на соседние дни.
* :meth:`FeatureBuilder.tabular` — те же каналы плюс их сдвиги на ±1…14 дней
  и скользящие средние: классическим моделям каждый день виден отдельно,
  контекст нужно подать явно.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import WellSeries

CLIP = 3.0


class FeatureBuilder:
    """Строит признаки одной скважины. Без состояния — один экземпляр на все скважины."""

    MIN_WORKING_FREQ = 10.0  # Гц: ниже насос считается остановленным
    MIN_STEP = 0.1  # Гц: подъём частоты, после которого считаются признаки «от скачка»
    WINDOW = 30  # дней: «недавний» уровень параметров

    SEQUENCE_CHANNELS = \
    (
                    # Частота 
        "working",                  # 1/0, если скважина работает и частота выше 10 Гц
        "f_rel",                    # частота отн обычной для этой скважины. (част / медиана рабочей част - 1). 0 значит "как обычно", 0.03 это +3%
        "f_diff",                   # измение f_rel за сутки (* 10). Скачок вверх даёт полож число
        "f_above_min30",            # насколько частота выше минимума за последн. 30 дн. Показывает, что недавно был подъём
        "f_below_max30",            # насколько частота ниже максимума за последние 30 дней (число <=0). Показывает недавний спад
                    # Дебиты и давления отн "недавнего прошлого" (последн 30 раб дней)
        "liq_vs_30d",               # дебит жидк отн ее медианы за прошл 30 дн
        "oil_vs_30d",               # дебит нефти отн ее медианы за прошл 30 дн
        "wc_vs_30d",                # обв минус ее среднее за прошл 30 дн, делен на 10
        "pzab_vs_30d",              # заб давл отн прошл 30 дн
                    # Относительно последнего скачка частоты вверх
        "days_since_step",          # сколько дней прошло с посл момента скачка, делен на 120 (<= 1). Если = 1, то скачка не было 
        "step_size",                # величина посл скачка отн частоты до него
        "liq_vs_step",              # жидк отн значения за день до скачка: Qж / Qж_до - 1
        "oil_vs_step",              # нефть отн значения за день до скачка
        "wc_vs_step",               # обв минус значение за день до скачка, делен на 10
        "liq_missing",              # 1, если замера жидкости в этот день нет
                    # Возврат нефти к базе (база - медиана нефти за 3 дня перед скачком и сам день скачка;
                    # нужны для определения окончания эффекта)
        "oil_vs_base",              # нефть отн базы: нефть / база - 1
        "oil_below_base",           # 1, если нефть не выше базы, иначе 0
        "oil_days_below",           # сколько дней подряд нефть не выше базы, делен на 14 (< 1)
        "oil_drawdown",             # на сколько нефть упала от своего максимума после скачка в долях базы
        "oil_missing",              # 1, если замера нефти в этот день нет (иначе пропуск был бы похож на "как база")
    )

    # только для табличных моделей
    SHIFTED_CHANNELS = \
    (
        "working",                  # работает ли скважина вокруг этого дня
        "f_diff",                   # был ли скачок частоты рядом
        "f_above_min30",            # недавний подъём частоты 
        "liq_vs_30d",               # как менялась жидк воруг дня
        "oil_vs_30d",               # как нефть жидк воруг дня
        "wc_vs_30d",                # как обв жидк воруг дня
        "liq_vs_step",              # отклик дебита жидк на скачок
        "oil_vs_step",              # отклик дебита нефти на скачок
        "wc_vs_step",               # отклик дебита обв на скачок
        "oil_vs_base",              # вернулась ли нефть к базе, включая "ближайшие дни"
        "oil_below_base",           # вернулась ли нефть к базе, включая "ближайшие дни"
    )
    OFFSETS = (-14, -7, -3, -1, 1, 3, 7, 14)

    @property
    def sequence_size(self) -> int:
        return len(self.SEQUENCE_CHANNELS)

    @property
    def tabular_names(self) -> list[str]:
        names = list(self.SEQUENCE_CHANNELS)
        names += [f"{c}@{o:+d}" for c in self.SHIFTED_CHANNELS for o in self.OFFSETS]
        names += [f"{c}_mean_{side}7" for c in self.SHIFTED_CHANNELS for side in ("past", "next")]
        names += ["f_diff_max_pm3"]
        return names

    def sequence(self, well: WellSeries) -> np.ndarray:
        channels = self._channels(well)
        return channels[list(self.SEQUENCE_CHANNELS)].to_numpy(dtype=np.float32)

    def tabular(self, well: WellSeries) -> np.ndarray:
        channels = self._channels(well)
        parts = [channels[list(self.SEQUENCE_CHANNELS)]]
        shifted = channels[list(self.SHIFTED_CHANNELS)]
        for offset in self.OFFSETS:
            # offset > 0 — значение через offset дней (детекция офлайн, будущее известно).
            parts.append(shifted.shift(-offset).fillna(0.0).add_suffix(f"@{offset:+d}"))
        parts.append(shifted.rolling(7, min_periods=1).mean().add_suffix("_mean_past7"))
        parts.append(shifted[::-1].rolling(7, min_periods=1).mean()[::-1].add_suffix("_mean_next7"))
        parts.append(
            channels["f_diff"]
            .rolling(7, center=True, min_periods=1)
            .max()
            .rename("f_diff_max_pm3")
            .to_frame()
        )
        table = pd.concat(parts, axis=1)
        return table.to_numpy(dtype=np.float32)

    def _channels(self, well: WellSeries) -> pd.DataFrame:
        frame = well.frame
        working = (frame["state"] > 0) & (frame["frequency"] > self.MIN_WORKING_FREQ)
        frequency = frame["frequency"].where(working).ffill().bfill()
        reference = frequency[working].median() if working.any() else np.nan
        reference = reference if reference and not np.isnan(reference) else 1.0
        f_rel = (frequency / reference - 1).fillna(0.0)

        out = pd.DataFrame(index=frame.index)
        out["working"] = working.astype(float)
        out["f_rel"] = f_rel
        out["f_diff"] = f_rel.diff().fillna(0.0) * 10  # 1 Гц при 50 Гц -> 0.2
        out["f_above_min30"] = (f_rel - f_rel.rolling(self.WINDOW, min_periods=1).min()) * 10
        out["f_below_max30"] = (f_rel - f_rel.rolling(self.WINDOW, min_periods=1).max()) * 10
        out["liq_vs_30d"] = self._relative_to_past(frame["debit_liquid"], working)
        out["oil_vs_30d"] = self._relative_to_past(frame["debit_oil"], working)
        out["wc_vs_30d"] = self._difference_to_past(frame["watercut"], working) / 10
        out["pzab_vs_30d"] = self._relative_to_past(frame["pressure_zab"], working)
        out = out.join(self._since_last_step(frame, frequency, working))
        out["liq_missing"] = frame["debit_liquid"].isna().astype(float)
        return out.fillna(0.0).clip(-CLIP, CLIP)

    def _relative_to_past(self, series: pd.Series, working: pd.Series) -> pd.Series:
        past = series.where(working).shift(1).rolling(self.WINDOW, min_periods=3).median()
        return (series / past.replace(0, np.nan) - 1).where(working)

    def _difference_to_past(self, series: pd.Series, working: pd.Series) -> pd.Series:
        past = series.where(working).shift(1).rolling(self.WINDOW, min_periods=3).mean()
        return (series - past).where(working)

    def _since_last_step(self, frame: pd.DataFrame, frequency: pd.Series, working: pd.Series) -> pd.DataFrame:
        """Признаки относительно последнего скачка частоты вверх (как у детектора на правилах)."""
        smoothed = frequency.rolling(3, center=True, min_periods=1).median()
        is_step = (smoothed.diff() >= self.MIN_STEP) & working
        positions = np.flatnonzero(is_step.to_numpy())

        n = len(frame)
        result = {
            name: np.zeros(n)
            for name in ("days_since_step", "step_size", "liq_vs_step", "oil_vs_step", "wc_vs_step")
        }
        result["days_since_step"][:] = 1.0  # «давно не было скачка»
        liquid, oil = frame["debit_liquid"].to_numpy(), frame["debit_oil"].to_numpy()
        watercut, values = frame["watercut"].to_numpy(), smoothed.to_numpy()
        boundaries = list(positions) + [n]
        for step, next_step in zip(boundaries[:-1], boundaries[1:], strict=True):
            before = max(step - 1, 0)
            span = slice(step, next_step)
            days = np.arange(next_step - step)
            result["days_since_step"][span] = np.minimum(days, 120) / 120
            result["step_size"][span] = (values[step] - values[before]) / max(values[before], 1.0) * 10
            result["liq_vs_step"][span] = _ratio(liquid[span], liquid[before])
            result["oil_vs_step"][span] = _ratio(oil[span], oil[before])
            result["wc_vs_step"][span] = (watercut[span] - watercut[before]) / 10
        return pd.DataFrame(result, index=frame.index)


def _ratio(values: np.ndarray, base: float) -> np.ndarray:
    if base is None or np.isnan(base) or base == 0:
        return np.zeros_like(values)
    return values / base - 1
