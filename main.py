from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager
from models import Address, AddressSearchQuery
import uvicorn
import time
from typing import Set

from data_manager import dataset_manager
from analysis_engine import analysis_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
  logger.info('Starting up FastAPI service...')
  start_time = time.time()

  success = await dataset_manager.load_datasets_on_startup()
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
    "datasets_loaded": dataset_manager.datasets is not None  
  }

@app.post("/addresses")
async def get_addresses(address: AddressSearchQuery):
  try:
    logger.info('Retrieving all addresses')

    df = dataset_manager.get_dataset('buildings')
    if df is None:
      raise HTTPException(status_code=404, detail="Dataset not found")
    
    unique_addresses: Set[Address] = set()

    df['huisnummer'] = df['huisnummer'].astype(str)
    df['huisletter'] = df['huisletter'].fillna('')
    df['adres'] = (
      df['openbareruimtenaam'] + ' ' +
      df['huisnummer'] +
      df['huisletter']
    )
    df['adres'] = df['adres'].str.strip()

    filtered_df = df[df['adres'].str.contains(address.searchQuery, case=False, na=False)]

    for row in filtered_df.itertuples(index=False):
      try:
        address = Address(
          street=row.openbareruimtenaam,
          house_number=row.huisnummer,
          house_number_addition=row.huisletter
        )
        unique_addresses.add(address)
      except Exception as e:
        logger.error(e)

    addresses = list(unique_addresses)

    return sorted(addresses, key=lambda addr: (addr.street, addr.house_number, addr.house_number_addition))
  
  except HTTPException:
    raise
  except KeyError as e:
    raise HTTPException(status_code=500, detail=str(e))
  except Exception as e:
    logger.error(f"Error in demolish prediction: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_demolish")
async def predict_demolish(address: Address):
  try:
    logger.info(f"Predicting demolition for address: {address}")

    df = dataset_manager.get_dataset('buildings')
    if df is None:
      raise HTTPException(status_code=404, detail="Dataset not found")

    demolition_prediction = analysis_engine.predict_demolish(df, address)

    return demolition_prediction
  
  except HTTPException:
    raise
  except KeyError as e:
    raise HTTPException(status_code=500, detail=str(e))
  except Exception as e:
    logger.error(f"Error in demolish prediction: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))
  
@app.get("/predict_clusters")
async def cluster():
  try:
    logger.info(f"Starting clustering algorithm")

    df = dataset_manager.get_dataset('buildings')
    if df is None:
      raise HTTPException(status_code=404, detail="Dataset not found")
    
    cluster_result = analysis_engine.predict_clusters(df)

    return cluster_result
  
  except HTTPException:
    raise
  except KeyError as e:
    raise HTTPException(status_code=500, detail=str(e))
  except Exception as e:
    logger.error(f"Error in clustering algorithm: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))
  
@app.post('/predict_twins')
async def predict_twins(address: Address):
  try:
    logger.info(f"Starting twin prediction")

    df = dataset_manager.get_dataset('buildings')
    if df is None:
      raise HTTPException(status_code=404, detail="Dataset not found")

    twin_result = analysis_engine.predict_twins(df, address)

    return twin_result

  except HTTPException:
    raise
  except KeyError as e:
    raise HTTPException(status_code=500, detail=str(e))
  except Exception as e:
    logger.error(f"Error in twin prediction: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))
  
if __name__ == "__main__":
  uvicorn.run(
    "main:app",
    host="0.0.0.0",
    port=8000,
    reload=True,
    log_level="info"
  )