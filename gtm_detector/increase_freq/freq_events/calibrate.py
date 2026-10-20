"""Подбор параметров детектора по скважинам с проставленными датами мероприятий.

Пока разметка есть — подбираем пороги так, чтобы детектор воспроизводил даты
технологов. Когда разметки не станет — используем сохранённые параметры.
Перебор — покоординатный спуск (полная сетка слишком велика).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .detector import DetectorParams, detect_events
from .metrics import Score, match
from .schema import labels_of, well_to_frame

SEARCH_SPACE: dict[str, list] = {
    "smooth_window": [1, 3],
    "min_step": [0.1, 0.3, 0.5, 1.0],
    "min_total": [0.3, 0.5, 1.0],
    "max_gap": [0, 3, 7, 14],
    "hold_days": [1, 3, 5],
    "require_response": [False, True],
    "skip_after_restart": [False, True],
    "start_mode": ["change", "rise"],
    "start_same_day": [False, True],
    "response_days": [1, 3, 7, 14],
    "start_max_days": [3, 7, 14, 30],
    "start_default": [0, 1, 2, 3],
    "end_rules": [("liq",), ("wc",), ("oil",), ("liq", "wc"), ("liq", "wc", "oil")],
    "liq_tol": [0.0, 0.01, 0.02, 0.05],
    "wc_tol": [0.0, 0.5, 1.0, 2.0],
    "oil_tol": [0.0, 0.02, 0.05],
    "stop_days": [1, 2, 3, 5],
    "min_effect_days": [0, 3, 7, 14],
    "end_confirm_days": [1, 2, 3, 5, 7],
}


class Dataset:
    """Скважины с разметкой + заранее построенные DataFrame (для скорости)."""

    def __init__(self, wells: list[dict]):
        self.wells = wells
        self.frames = [well_to_frame(w) for w in wells]
        self.labels = [labels_of(w) for w in wells]
        self.neutral = [w.get("neutral_dates", []) for w in wells]

    def subset(self, idx) -> "Dataset":
        d = Dataset.__new__(Dataset)
        d.wells = [self.wells[i] for i in idx]
        d.frames = [self.frames[i] for i in idx]
        d.labels = [self.labels[i] for i in idx]
        d.neutral = [self.neutral[i] for i in idx]
        return d

    def __len__(self):
        return len(self.wells)


def evaluate(data: Dataset | list[dict], params: DetectorParams, tol: int = 7) -> Score:
    if not isinstance(data, Dataset):
        data = Dataset(data)
    total = Score()
    for w, f, lab, neu in zip(data.wells, data.frames, data.labels, data.neutral):
        total = total + match(detect_events(w, params, frame=f), lab, tol=tol, neutral_dates=neu)
    return total


CLIP = 14


def objective(s: Score, beta: float = 2.0) -> float:
    """Больше — лучше: F-beta + штраф ~0.01 за день ошибки дат (обрезанной на CLIP).

    beta=2 — полнота важнее точности: в разметке только мероприятия из списка
    рекомендаций, многие реальные подъёмы частоты в неё не попали и выглядят как FP.
    """
    if not s.tp:
        return -1.0
    p, r = s.precision, s.recall
    fb = (1 + beta**2) * p * r / (beta**2 * p + r) if p + r else 0.0
    # Ошибки дат обрезаем на CLIP днях: пара грубых промахов не должна перевешивать
    # долю точных попаданий. Незакрытый конец — как максимальная ошибка.
    start_pen = np.mean(np.minimum(np.abs(s.start_err), CLIP))
    end_pen = np.sum(np.minimum(np.abs(s.end_err), CLIP)) + CLIP * s.end_open
    end_pen /= max(1, len(s.end_err) + s.end_open)
    return fb - 0.01 * (start_pen + 0.5 * end_pen)


def calibrate(data: Dataset | list[dict], base: DetectorParams | None = None,
              space: dict[str, list] | None = None, rounds: int = 3, tol: int = 7,
              verbose: bool = False) -> tuple[DetectorParams, Score]:
    if not isinstance(data, Dataset):
        data = Dataset(data)
    space = space or SEARCH_SPACE
    best = base or DetectorParams()
    best_s = evaluate(data, best, tol)
    best_o = objective(best_s)
    for r in range(rounds):
        improved = False
        for k, values in space.items():
            for v in values:
                if getattr(best, k) == v:
                    continue
                p = best.with_(**{k: v})
                s = evaluate(data, p, tol)
                o = objective(s)
                if o > best_o + 1e-9:
                    best, best_s, best_o, improved = p, s, o, True
                    if verbose:
                        print(f"  [{r}] {k}={v!r}: obj={o:.4f} F1={s.f1:.3f} "
                              f"MAE нач={s.mae_start:.2f} кон={s.mae_end:.2f}")
        if not improved:
            break
    return best, best_s


def cross_validate(data: Dataset, k: int = 5, seed: int = 0, **kw) -> Score:
    """k-fold по скважинам: калибруем на k-1 частях, оцениваем на отложенной."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data))
    folds = np.array_split(order, k)
    total = Score()
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        p, _ = calibrate(data.subset(train), **kw)
        total = total + evaluate(data.subset(test), p)
    return total


def save_params(p: DetectorParams, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(p), ensure_ascii=False, indent=2), encoding="utf-8")


def load_params(path: str | Path) -> DetectorParams:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    d["end_rules"] = tuple(d.get("end_rules", ()))
    return DetectorParams(**d)
