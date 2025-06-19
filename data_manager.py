import pandas as pd
import logging
import asyncio

logger = logging.getLogger(__name__)

class DatasetManager:
  def __init__(self):
    self.dataset: pd.DataFrame = None
  
  async def load_dataset_on_startup(self, file_path: str) -> bool:
    try:
      logger.info(f"Loading dataset from {file_path}")

      loop = asyncio.get_event_loop()
      self.dataset = await loop.run_in_executor(None, pd.read_csv, file_path)
      logger.info(f"Successfully loaded dataset")
      return True
    
    except Exception as e:
      logger.error(f"Failed to load dataset: {str(e)}")
      return False
  
  def get_dataset(self) -> pd.DataFrame:
    return self.dataset
  
  def clear_cache(self):
    self.dataset = None
    logger.info("Cleared dataset")

dataset_manager = DatasetManager()