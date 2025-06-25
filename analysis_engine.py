import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from models import Address, Building, DemolishPrediction, ClusterPrediction
from typing import List

from data_preparation_engine import data_preparation_engine

logger = logging.getLogger(__name__)

class DataAnalysisEngine:
  def predict_demolish(self, df: pd.DataFrame, address: Address) -> DemolishPrediction:
    try:
      df_prepared = data_preparation_engine.prepare_for_demolition_prediction(df)

      model = self._run_rfc(df_prepared)

      building = data_preparation_engine.construct_building(df_prepared, address)

      pred, prob = self._make_prediction(model, building)

    except KeyError as e:
      error_message = f'Could not find column {e} in dataset.'
      logger.error(error_message)
      raise
    except Exception as e:
      logger.error(e, exc_info=True)
      raise
    
    return DemolishPrediction(
      address=f"{address.street} {address.house_number}{address.house_number_addition}",
      build_year=building.build_year,
      building_type=building.building_type,
      age=building.age,
      relative_age=round(building.relative_age * 100, 2),
      predicted_lifespan=building.predicted_lifespan,
      area=building.area,
      area_ratio=building.area_ratio,
      prediction=pred,
      demolition_probability=round(prob * 100, 2)
    )
  
  def predict_clusters(self, df: pd.DataFrame) -> List[ClusterPrediction]:
    try:
      df_prepared = data_preparation_engine.prepare_for_clustering(df)

      features = df_prepared[['lon', 'lat', 'leeftijd']]
      scaler = StandardScaler()
      df_scaled = scaler.fit_transform(features)

      kmeans = KMeans(n_clusters=5)
      df_prepared['cluster'] = kmeans.fit_predict(df_scaled)

      clusters = df_prepared.apply(data_preparation_engine.construct_cluster_prediction, axis=1).tolist()
      return clusters
    
    except KeyError as e:
      error_message = f'Could not find column {e} in dataset.'
      logger.error(error_message)
      raise
    except Exception as e:
      logger.error(e, exc_info=True)
      raise
  
  def _predict_lifespan(self, building_type: str):
      if building_type in ['Appartement', 'Tussenwoning', 'Hoekwoning', 'Vrijstaande woning', 'Woonhuis']:
        return 75
      elif building_type in ['Kantoor', 'Winkel', 'Bedrijfspand']:
        return 50
      else:
        return 60
  
  def _run_rfc(self, df: pd.DataFrame) -> RandomForestClassifier:
    logger.info("Applying RandomForestClassifier")

    X = df[['bouwjaar', 'woningtype_categorie', 'verhouding_opp_vbo_opp_pnd', 'opp_adresseerbaarobject_m2', 'relatieve_ouderdom']]
    y = df['sloopkans']

    try:
      model = RandomForestClassifier()
      model.fit(X, y)

    except Exception as e:
      logger.error(f"Error in RandomForestClassifier: {str(e)}")
      raise

    logger.info("Successfully applied RandomForestClassifier")

    return model
  
  def _make_prediction(self, model: RandomForestClassifier, building: Building):
    logger.info(f"Making prediction for building: {building}")

    try:
      X_input = np.array([[
        building.build_year,
        building.building_type_idx,
        building.relative_age,
        building.area,
        building.area_ratio
      ]])
      pred = model.predict(X_input)

      if isinstance(pred, np.ndarray):
        if pred.size == 1:
          pred = pred.item()
        else:
          pred = pred[0]

      prob = model.predict_proba(X_input)[0][1]

    except Exception as e:
      logger.error(f"Error while making prediction: {str(e)}")
      return {"error": str(e)}

    return pred, prob

analysis_engine = DataAnalysisEngine()