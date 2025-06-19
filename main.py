from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from models import Address
import uvicorn
import time

from data_manager import dataset_manager
from analysis_engine import analysis_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
  logger.info('Starting up FastAPI service...')
  start_time = time.time()

  data_dir = Path('Heerlen_dataset.csv')
  if data_dir.exists():
    success = await dataset_manager.load_dataset_on_startup(data_dir)
    if not success:
      logger.warning("Failed to load dataset")
  
  logger.info(f"FastAPI service startup complete in {time.time() - start_time:.2f} seconds")
  yield

  logger.info("Shutting down FastAPI service")
  dataset_manager.clear_cache()
  logger.info("FastAPI service shutdown complete")

app = FastAPI(
  title="Python Service for Building Predictions",
  description="FastAPI service for making various predictions on building and material information",
  version="1.0.0",
  lifespan=lifespan
)

app.add_middleware(
  CORSMiddleware,
  allow_origins=["http://localhost:3000", "http://192.168.178.105:3000"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

@app.get("/health")
async def health_check():
  return {
    "status": "healthy",
    "service": "Python FastAPI Data Service",
    "dataset_loaded": dataset_manager.dataset is not None  
  }

@app.post("/predict")
async def predictDemolish(address: Address):
  try:
    logger.info(f"Predicting demolition for address: {address}")

    df = dataset_manager.get_dataset()
    if df is None:
      raise HTTPException(status_code=404, detail="Dataset not found")

    demolition_prediction = analysis_engine.predict_demolish(df, address)

    return demolition_prediction
  
  except HTTPException:
    raise
  except Exception as e:
    logger.error(f"Error in demolish prediction: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))
  
if __name__ == "__main__":
  uvicorn.run(
    "main:app",
    host="0.0.0.0",
    port=8000,
    reload=True,
    log_level="info"
  )