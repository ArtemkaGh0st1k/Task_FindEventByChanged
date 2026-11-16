"""Собственная модель: Transformer-энкодер для суточных рядов скважины (PyTorch).

Архитектура::

    признаки дня (C) ─► Conv1d (локальные ступеньки) ─► + позиционное кодирование
                    ─► N × TransformerEncoderLayer (внимание по всему ряду)
                    ─► Linear ─► логит «день внутри эффекта»

Свёртка ловит форму скачка частоты и отклика дебита за несколько дней,
внимание связывает скачок с окончанием эффекта через недели. Модель
двунаправленная: детекция офлайн, весь ряд известен.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..config import TrainingConfig, TransformerConfig
from ..data import WellSeries
from ..features import FeatureBuilder
from .base import EventSegmenter

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Синусоидальное позиционное кодирование (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_length: int) -> None:
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        scale = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale)
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[: x.size(1)]


class TimeSeriesTransformer(nn.Module):
    """Сеть: (batch, дни, каналы) -> (batch, дни) логитов."""

    def __init__(self, n_channels: int, config: TransformerConfig) -> None:
        super().__init__()
        self.embed = nn.Conv1d(
            n_channels, config.d_model, kernel_size=config.conv_kernel, padding=config.conv_kernel // 2
        )
        self.position = PositionalEncoding(config.d_model, config.max_length)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1))

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.embed(x.transpose(1, 2)).transpose(1, 2)
        hidden = self.encoder(self.position(hidden), src_key_padding_mask=padding_mask)
        return self.head(hidden).squeeze(-1)


class TransformerSegmenter(EventSegmenter):
    """Обучение и применение :class:`TimeSeriesTransformer`."""

    name = "transformer"
    title = "Transformer (своя модель, PyTorch)"

    def __init__(
        self,
        config: TransformerConfig | None = None,
        training: TrainingConfig | None = None,
        features: FeatureBuilder | None = None,
    ) -> None:
        super().__init__(features)
        self.config = config or TransformerConfig()
        self.training = training or TrainingConfig()
        self.network: TimeSeriesTransformer | None = None

    @property
    def file_suffix(self) -> str:
        return ".pt"

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.network.parameters()) if self.network else 0

    def fit(self, wells: Sequence[WellSeries]) -> TransformerSegmenter:
        cfg = self.training
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        sequences = [self.features.sequence(well) for well in wells]
        targets = [well.target() for well in wells]
        masks = [well.training_mask(self.context_days) for well in wells]

        positives = float(sum(t[m].sum() for t, m in zip(targets, masks, strict=True)))
        negatives = float(sum(m.sum() for m in masks)) - positives
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(negatives / max(positives, 1.0)), reduction="none"
        )

        self.network = TimeSeriesTransformer(self.features.sequence_size, self.config)
        optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

        self.network.train()
        for epoch in range(cfg.epochs):
            crops = self._sample_crops(sequences, targets, masks, rng)
            order = rng.permutation(len(crops))
            total = 0.0
            for begin in range(0, len(order), cfg.batch_size):
                x, y, padding, weight = self._collate(
                    [crops[i] for i in order[begin : begin + cfg.batch_size]]
                )
                logits = self.network(x, padding_mask=padding)
                loss = (loss_fn(logits, y) * weight).sum() / weight.sum().clamp(min=1.0)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()
                total += loss.item()
            scheduler.step()
            if (epoch + 1) % 20 == 0:
                logger.debug("эпоха %d/%d: loss=%.4f", epoch + 1, cfg.epochs, total)
        self.network.eval()
        return self

    def predict_proba(self, well: WellSeries) -> np.ndarray:
        if self.network is None:
            raise RuntimeError("Transformer не обучен")
        x = torch.from_numpy(self.features.sequence(well)).unsqueeze(0)
        with torch.no_grad():
            return torch.sigmoid(self.network(x)).squeeze(0).numpy()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "training": asdict(self.training),
                "channels": list(self.features.SEQUENCE_CHANNELS),
                "state_dict": self.network.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> TransformerSegmenter:
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(TransformerConfig(**state["config"]), TrainingConfig(**state["training"]))
        if list(model.features.SEQUENCE_CHANNELS) != state["channels"]:
            raise ValueError("Набор признаков изменился — переобучите модель")
        model.network = TimeSeriesTransformer(model.features.sequence_size, model.config)
        model.network.load_state_dict(state["state_dict"])
        model.network.eval()
        return model

    def _sample_crops(
        self,
        sequences: list[np.ndarray],
        targets: list[np.ndarray],
        masks: list[np.ndarray],
        rng: np.random.Generator,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Случайные окна длиной crop_length вокруг размеченных мероприятий.

        Центр окна — случайный день из обучающей маски: окна не тратятся на дни,
        которые всё равно не входят в функцию потерь.
        """
        length = self.training.crop_length
        crops = []
        for x, y, m in zip(sequences, targets, masks, strict=True):
            allowed = np.flatnonzero(m)
            if not len(allowed):
                continue
            for _ in range(self.training.crops_per_well):
                center = int(rng.choice(allowed))
                begin = int(np.clip(center - length // 2, 0, max(0, len(x) - length)))
                window = slice(begin, begin + length)
                crops.append((x[window], y[window], m[window]))
        return crops

    @staticmethod
    def _collate(
        batch: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Батч: признаки, цели, маска паддинга (True — нет данных) и веса дней в потерях."""
        length = max(len(x) for x, _, _ in batch)
        channels = batch[0][0].shape[1]
        x = torch.zeros(len(batch), length, channels)
        y = torch.zeros(len(batch), length)
        weight = torch.zeros(len(batch), length)
        padding = torch.ones(len(batch), length, dtype=torch.bool)
        for i, (xi, yi, mi) in enumerate(batch):
            x[i, : len(xi)] = torch.from_numpy(xi)
            y[i, : len(yi)] = torch.from_numpy(yi)
            weight[i, : len(mi)] = torch.from_numpy(mi.astype(np.float32))
            padding[i, : len(xi)] = False
        return x, y, padding, weight
