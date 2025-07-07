import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from models import Address, Building, DemolishPrediction, BuildingResponse, TwinBuildingPrediction, Cluster
from typing import List

from data_preparation_engine import data_preparation_engine

logger = logging.getLogger(__name__)

class DataAnalysisEngine:
  def predict_demolish(self, df: pd.DataFrame, address: Address) -> DemolishPrediction:
    try:
      model = self._run_rfc(df)

      building = data_preparation_engine.construct_building(df, address)

      if building is None:
        raise Exception("Address not found.")

      pred, prob = self._calculate_demolish_prob(model, building)

    except KeyError as e:
      error_message = f'Could not find column {e} in dataset.'
      logger.error(error_message)
      raise KeyError(error_message) from e
    except Exception as e:
      logger.error(e, exc_info=True)
      raise

    return data_preparation_engine.construct_demolish_prediction(building, address, pred, prob)
  
  def predict_clusters(self, df: pd.DataFrame) -> List[Cluster]:
    try:
      df_prepared = data_preparation_engine.prepare_for_clustering(df)

      features = df_prepared[['lon', 'lat', 'leeftijd']]
      scaler = StandardScaler()
      df_scaled = scaler.fit_transform(features)

      kmeans = KMeans(n_clusters=5)
      df_prepared['cluster'] = kmeans.fit_predict(df_scaled)

      clusters_buildings = df_prepared.apply(data_preparation_engine.construct_cluster_building, axis=1).tolist()
      clusters = data_preparation_engine.construct_cluster_prediction(clusters_buildings, df_prepared)

      return clusters
    
    except KeyError as e:
      error_message = f'Could not find column {e} in dataset.'
      logger.error(error_message)
      raise KeyError(error_message) from e
    except Exception as e:
      logger.error(e, exc_info=True)
      raise

  def predict_twins(self, df: pd.DataFrame, address: Address) -> TwinBuildingPrediction:
    try:
      df_prepared = data_preparation_engine.prepare_for_twin_prediction(df)

      reference_building = data_preparation_engine.construct_building(df_prepared, address)

      if reference_building is None:
        raise Exception("Address not found.")

      twins = self._find_twins(df, reference_building, 3, 0.15)

      if twins.empty:
        raise Exception("No twin buildings found.")
      
      twins = twins.apply(data_preparation_engine.construct_building_response, axis=1).tolist()

      twin_building_prediction = TwinBuildingPrediction(
        reference_building=BuildingResponse(
          id=reference_building.id,
          longitude=reference_building.longitude,
          latitude=reference_building.latitude,
          address=f"{address.street} {address.house_number}{address.house_number_addition}",
          build_year=reference_building.build_year,
          area=reference_building.area,
          building_type=reference_building.building_type,
        ),
        twin_buildings=twins,
      )
      logger.info(twin_building_prediction)
      return twin_building_prediction
    
    except KeyError as e:
      error_message = f'Could not find column {e} in dataset.'
      logger.error(error_message)
      raise KeyError(error_message) from e
    except Exception as e:
      logger.error(e, exc_info=True)
      raise
  
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
  
  def _calculate_demolish_prob(self, model: RandomForestClassifier, building: Building):
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
  
  def _find_twins(self, df: pd.DataFrame, building: Building, years_tolerance: int, area_tolerance):
    logger.info(f"Finding twins for building: {building}")

    query = (
      (df['pand_id'] != building.id) &
      (abs(df['bouwjaar'] - building.build_year) <= years_tolerance) &
      (abs(df['opp_pand'] - building.area) / building.area <= area_tolerance) &
      (df['woningtype'] == building.building_type) &
      (df['woonfunctie'] == building.has_residential_func) &
      (df['kantoorfunctie'] == building.has_office_func) &
      (df['winkelfunctie'] == building.has_shop_func)
    )

    return df[query]

analysis_engine = DataAnalysisEngine()