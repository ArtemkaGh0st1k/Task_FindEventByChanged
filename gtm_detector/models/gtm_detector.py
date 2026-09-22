from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from gtm_detector.config.config import GTMVerdict, PipelineConfig
from gtm_detector.data.dto import BaseObjectDto, StateDto
from gtm_detector.parsers.note_parser import NoteParser


class GTM1DCNN(nn.Module):
  """1D-CNN нейросеть классификации временного ряда скважины по дням."""

  def __init__(self, in_channels: int = 5, num_classes: int = 3):
    super().__init__()
    self.conv_block = nn.Sequential(
        nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Conv1d(32, 64, kernel_size=5, padding=2),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Conv1d(64, 32, kernel_size=3, padding=1),
        nn.BatchNorm1d(32),
        nn.ReLU(),
    )
    self.classifier = nn.Conv1d(32, num_classes, kernel_size=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    features = self.conv_block(x)
    logits = self.classifier(features)
    return logits


class GTMDetector:
  """Единый сервис детекции ГТМ: локализует t_start нейросетью и вычисляет t_end."""

  def __init__(
      self,
      config: PipelineConfig,
      note_parser: NoteParser,
      model_path: Optional[str | Path] = None,
      confidence_threshold: float = 0.60,
  ):
    self.config = config
    self.note_parser = note_parser
    self.confidence_threshold = confidence_threshold
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.feature_cols = [
        "Q_n_smooth",
        "F_ecn",
        "Specific_Yield",
        "dF",
        "Is_Working",
    ]
    self.class_map = {
        1: "Изменение частоты вращения ЭЦН",
        2: "Замена насосного оборудования",
    }

    self.model = GTM1DCNN(
        in_channels=len(self.feature_cols), num_classes=3
    ).to(self.device)
    self.model.eval()

    if model_path and Path(model_path).exists():
      self.model.load_state_dict(
          torch.load(model_path, map_location=self.device)
      )

  def build_well_df(
      self,
      well_id: str | int,
      raw_data: dict[str, list[BaseObjectDto]],
      transformed_features: dict[str, list[BaseObjectDto]],
  ) -> pd.DataFrame:
    """Формирует единый DataFrame временного ряда для конкретной скважины из DTO."""
    q_smooth_dto = next(
        (
            d
            for d in transformed_features.get("Q_n_smooth", [])
            if str(d.well_id) == str(well_id)
        ),
        None,
    )
    f_ecn_dto = next(
        (
            d
            for d in raw_data.get("Fэцн ТМ", [])
            if str(d.well_id) == str(well_id)
        ),
        None,
    )
    spec_yield_dto = next(
        (
            d
            for d in transformed_features.get("Specific_Yield", [])
            if str(d.well_id) == str(well_id)
        ),
        None,
    )
    df_dto = next(
        (
            d
            for d in transformed_features.get("dF", [])
            if str(d.well_id) == str(well_id)
        ),
        None,
    )
    is_working_dto = next(
        (
            d
            for d in transformed_features.get("Is_Working", [])
            if str(d.well_id) == str(well_id)
        ),
        None,
    )

    if not q_smooth_dto:
      raise ValueError(
          f"Не найдены данные Q_n_smooth для скважины ID={well_id}"
      )

    dates = sorted(q_smooth_dto.data.keys())

    df_well = pd.DataFrame({
        "Date": dates,
        "Q_n_smooth": [
            float(q_smooth_dto.data.get(d, 0.0)) for d in dates
        ],
        "F_ecn": [
            float(f_ecn_dto.data.get(d, 0.0)) if f_ecn_dto else 0.0 for d in dates
        ],
        "Specific_Yield": [
            float(spec_yield_dto.data.get(d, 0.0)) if spec_yield_dto else 0.0
            for d in dates
        ],
        "dF": [
            float(df_dto.data.get(d, 0.0)) if df_dto else 0.0 for d in dates
        ],
        "Is_Working": [
            float(is_working_dto.data.get(d, 0.0)) if is_working_dto else 1.0
            for d in dates
        ],
    })

    df_well["Date"] = pd.to_datetime(df_well["Date"])
    return df_well

  def analyze_well_dtos(
      self,
      well_id: str | int,
      raw_data: dict[str, list[BaseObjectDto]],
      transformed_features: dict[str, list[BaseObjectDto]],
  ) -> GTMVerdict:
    """Анализирует скважину напрямую по переданным структурам DTO."""
    df_well = self.build_well_df(well_id, raw_data, transformed_features)

    # Поиск StateDto с примечаниями
    state_dto = next(
        (
            d
            for d in raw_data.get("Прим", [])
            if str(d.well_id) == str(well_id) and isinstance(d, StateDto)
        ),
        None,
    )

    return self.analyze_well(well_id, df_well, state_dto)

  def analyze_well(
      self,
      well_id: str | int,
      df_well: pd.DataFrame,
      state_dto: Optional[StateDto] = None,
  ) -> GTMVerdict:
    """Анализирует временной ряд одной скважины в формате DataFrame и возвращает GTMVerdict."""
    if len(df_well) < self.config.q_base_window_days:
      return GTMVerdict(
          well_id=str(well_id),
          has_gtm=False,
          start_date=datetime.now(),
          end_date=datetime.now(),
          reasoning="Недостаточно дней для анализа.",
      )

    # 1. Поиск старта ГТМ через 1D-CNN
    has_gtm, gtm_type, start_idx, start_date = self._detect_start(df_well)

    if not has_gtm or start_date is None:
      return GTMVerdict(
          well_id=str(well_id),
          has_gtm=False,
          start_date=df_well["Date"].iloc[0].to_pydatetime(),
          end_date=df_well["Date"].iloc[-1].to_pydatetime(),
          reasoning="Признаков проведения насосных ГТМ не зафиксировано.",
      )

    # 2. Извлечение сторонних ГТМ из примечаний StateDto
    next_gtm_dates = self.note_parser.extract_gtm_events(
        state_dto, self.config.gtm_keywords
    )

    # 3. Расчет t_end и базового дебита Q_base
    end_date, q_base = self._calculate_end(df_well, start_date, next_gtm_dates)

    reasoning = (
        f"Зафиксирован старт '{gtm_type}' от"
        f" {start_date.strftime('%d.%m.%Y')}. Базовый дебит Q_base ="
        f" {q_base:.1f} т/сут. Окончание эффекта:"
        f" {end_date.strftime('%d.%m.%Y')}."
    )

    return GTMVerdict(
        well_id=str(well_id),
        has_gtm=True,
        start_date=start_date.to_pydatetime(),
        end_date=end_date.to_pydatetime(),
        gtm_type=gtm_type,
        reasoning=reasoning,
    )

  def _prepare_tensor(self, df: pd.DataFrame) -> torch.Tensor:
    """Нормализует данные и формирует 3D-тензор формата [1, Channels, Time]."""
    features_data = df[self.feature_cols].fillna(0).values.astype(np.float32)

    mean = np.mean(features_data, axis=0)
    std = np.std(features_data, axis=0) + 1e-6
    norm_data = (features_data - mean) / std

    tensor_data = torch.tensor(norm_data, dtype=torch.float32).T.unsqueeze(0)
    return tensor_data.to(self.device)

  def _detect_start(
      self, df: pd.DataFrame
  ) -> Tuple[bool, Optional[str], Optional[int], Optional[pd.Timestamp]]:
    """Запускает нейросеть и определяет индекс и дату начала ГТМ."""
    input_tensor = self._prepare_tensor(df)

    with torch.no_grad():
      logits = self.model(input_tensor)
      probs = torch.softmax(logits, dim=1).squeeze(0)
      predictions = torch.argmax(probs, dim=0).cpu().numpy()

    for idx, pred_class in enumerate(predictions):
      if pred_class in (1, 2):
        confidence = probs[pred_class, idx].item()
        if confidence >= self.confidence_threshold:
          start_date = pd.Timestamp(df.loc[idx, "Date"])
          return True, self.class_map[pred_class], idx, start_date

    return False, None, None, None

  def _calculate_end(
      self,
      df: pd.DataFrame,
      start_date: pd.Timestamp,
      next_gtm_dates: list[pd.Timestamp],
  ) -> Tuple[pd.Timestamp, float]:
    """Расчет даты окончания эффекта: min(t_natural, t_next_gtm)."""
    # Расчет Q_base за окно до ГТМ
    pre_mask = (
        df["Date"]
        >= start_date - pd.Timedelta(days=self.config.q_base_window_days)
    ) & (df["Date"] < start_date)
    pre_df = df[pre_mask]

    if not pre_df.empty:
      q_base = float(pre_df["Q_n_smooth"].median())
    else:
      earlier = df[df["Date"] < start_date]
      q_base = (
          float(earlier["Q_n_smooth"].median()) if not earlier.empty else 0.0
      )

    # Поиск естественного падения дебита t_natural
    post_df = df[df["Date"] > start_date]
    threshold_q = q_base * self.config.q_recovery_tolerance
    natural_end_rows = post_df[post_df["Q_n_smooth"] <= threshold_q]

    if not natural_end_rows.empty:
      t_natural = pd.Timestamp(natural_end_rows.iloc[0]["Date"])
    else:
      t_natural = pd.Timestamp(df["Date"].max())

    # Ограничение по дате следующего ГТМ из комментариев
    future_gtms = [d for d in next_gtm_dates if d > start_date]
    t_next_gtm = min(future_gtms) if future_gtms else pd.Timestamp.max

    return min(t_natural, t_next_gtm), q_base