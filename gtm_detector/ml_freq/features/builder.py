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
    # Окончание эффекта: технолог ставит его, когда дебит нефти вернулся к уровню до подъёма.
    # База — медиана за BASE_DAYS дней до скачка и сам день скачка: дебит одного дня
    # шумный, и «первый день не выше» срабатывает почти сразу (проверено на разметке:
    # 70 % против 27 % у дебита одного дня, ±7 дней). Те же признаки по дебиту жидкости
    # проверены на compare и убраны: устойчивого выигрыша они не дали.
    BASE_DAYS = 3
    BELOW_RUN_CAP = 14  # дней подряд ниже базы, при которых признак «насыщается»

    SEQUENCE_CHANNELS = (
        "working",
        "f_rel",
        "f_diff",
        "f_above_min30",
        "f_below_max30",
        "liq_vs_30d",
        "oil_vs_30d",
        "wc_vs_30d",
        "pzab_vs_30d",
        "days_since_step",
        "step_size",
        "liq_vs_step",
        "oil_vs_step",
        "wc_vs_step",
        "liq_missing",
        "oil_vs_base",
        "oil_below_base",
        "oil_days_below",
        "oil_drawdown",
        "oil_missing",
    )
    SHIFTED_CHANNELS = (
        "working",
        "f_diff",
        "f_above_min30",
        "liq_vs_30d",
        "oil_vs_30d",
        "wc_vs_30d",
        "liq_vs_step",
        "oil_vs_step",
        "wc_vs_step",
        "oil_vs_base",
        "oil_below_base",
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
        out["oil_missing"] = frame["debit_oil"].isna().astype(float)  # иначе пропуск = «как база»
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
            for name in (
                "days_since_step",
                "step_size",
                "liq_vs_step",
                "oil_vs_step",
                "wc_vs_step",
                "oil_vs_base",
                "oil_below_base",
                "oil_days_below",
                "oil_drawdown",
            )
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
            base = self._base_level(oil, step)
            if base is not None:
                for name, feature in self._return_features(oil[span], base).items():
                    result[f"oil_{name}"][span] = feature
        return pd.DataFrame(result, index=frame.index)

    def _base_level(self, debit: np.ndarray, step: int) -> float | None:
        """Уровень дебита до подъёма: медиана последних дней перед скачком; ``None`` — нет базы."""
        window = debit[max(step - self.BASE_DAYS, 0) : step + 1]
        window = window[np.isfinite(window)]
        base = float(np.median(window)) if len(window) else 0.0
        return base if base > 0 else None

    def _return_features(self, debit: np.ndarray, base: float) -> dict[str, np.ndarray]:
        """Дебит после скачка относительно базы: вернулся ли он к уровню до подъёма.

        День без замера не даёт сведений: отношение 0, он не считается «ниже базы» и
        обрывает серию дней подряд.
        """
        valid = np.isfinite(debit)
        below = valid & (debit <= base)
        run, count = np.zeros(len(debit)), 0
        for i, flag in enumerate(below):
            count = count + 1 if flag else 0
            run[i] = min(count, self.BELOW_RUN_CAP) / self.BELOW_RUN_CAP
        peak = np.maximum.accumulate(np.where(valid, debit, -np.inf))
        has_peak = valid & np.isfinite(peak)
        return {
            "vs_base": np.where(valid, debit / base - 1, 0.0),
            "below_base": below.astype(float),
            "days_below": run,
            "drawdown": np.where(has_peak, (peak - debit) / base, 0.0),
        }


def _ratio(values: np.ndarray, base: float) -> np.ndarray:
    if base is None or np.isnan(base) or base == 0:
        return np.zeros_like(values)
    return values / base - 1
