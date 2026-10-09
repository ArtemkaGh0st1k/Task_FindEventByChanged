from pathlib import Path
from typing import Dict, Optional

from gtm_detector.config.config import GTMVerdict, PipelineConfig
from gtm_detector.data.excel.config import WellConfig
from gtm_detector.data.excel.in_loader import InnerWellDataLoader
from gtm_detector.features.feature_engineering import FeatureEngineer
from gtm_detector.models.gtm_detector import GTMDetector
from gtm_detector.parsers.note_parser import NoteParser


class GTMPipeline:

  def __init__(
      self,
      pipeline_config: Optional[PipelineConfig] = None,
      model_path: Optional[str | Path] = None,
  ):
    self.config = pipeline_config or PipelineConfig()
    self.note_parser = NoteParser()
    self.feature_engineer = FeatureEngineer(
        config=self.config, note_parser=self.note_parser
    )
    self.detector = GTMDetector(
        config=self.config,
        note_parser=self.note_parser,
        model_path=model_path,
    )

  def run(
      self, excel_file_path: str | Path, well_config: WellConfig
  ) -> Dict[str, GTMVerdict]:
    """Сквозной запуск: Загрузка Excel -> Feature Engineering -> Детекция ГТМ."""
    # 1. Загрузка данных
    loader = InnerWellDataLoader(excel_file_path)
    raw_data = loader.load_wells_by_sheets(well_config)

    # 2. Трансформация признаков
    transformed_features = self.feature_engineer.transform(raw_data)

    # 3. Детекция ГТМ по всем скважинам
    results: Dict[str, GTMVerdict] = {}
    well_ids = [str(dto.well_id) for dto in raw_data.get("Qн", [])]

    for well_id in well_ids:
      verdict = self.detector.analyze_well_dtos(
          well_id, raw_data, transformed_features
      )
      results[well_id] = verdict

    return results