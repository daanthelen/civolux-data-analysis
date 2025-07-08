import pandas as pd
import logging
import asyncio
import functools
from typing import List
from models import Dataset

from analysis_engine import analysis_engine
from data_preparation_engine import data_preparation_engine

logger = logging.getLogger(__name__)

class DatasetManager:
  def __init__(self):
    self.datasets: List[Dataset] = []
  
  async def load_datasets_on_startup(self) -> bool:
    try:
      logger.info(f"Loading datasets")

      loop = asyncio.get_running_loop()
      read_buildings_dataset = functools.partial(pd.read_csv, 'datasets/Heerlen_dataset.csv', dtype={'pand_id': str})
      buildings_dataset = await loop.run_in_executor(None, read_buildings_dataset)

      read_material_dataset = functools.partial(pd.read_csv, 'datasets/materialen.csv')
      material_dataset = await loop.run_in_executor(None, read_material_dataset)

      buildings_dataset = data_preparation_engine.prepare_buildings_dataset(buildings_dataset)
      buildings_dataset = analysis_engine.predict_demolitions_for_all_buildings(buildings_dataset)

      self.datasets.append(
        Dataset (
          name='buildings',
          dataset=buildings_dataset
        )
      )
      self.datasets.append(
        Dataset (
          name='materials',
          dataset=material_dataset
        )
      )

      logger.info(f"Successfully loaded datasets")
      return True
    
    except Exception as e:
      logger.error(f"Failed to load dataset: {str(e)}")
      return False
  
  def get_dataset(self, name) -> pd.DataFrame:
    return next((dataset.dataset for dataset in self.datasets if dataset.name == name), None)
  
  def clear_cache(self):
    self.datasets = None
    logger.info("Cleared datasets")

dataset_manager = DatasetManager()