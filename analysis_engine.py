import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from models import Address, Building, DemolishPrediction

logger = logging.getLogger(__name__)

class DataAnalysisEngine:
  def predict_demolish(self, df: pd.DataFrame, address: Address) -> DemolishPrediction:
    try:
      df_prepared = self._prepare_dataset(df)

      model = self._run_rfc(df_prepared)

      building = self._construct_building(df_prepared, address)

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
  
  def _prepare_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Preparing dataset')

    try:
      df_prepared = df[['huisnummer', 'bouwjaar', 'pandstatus', 'woningtype', 'opp_adresseerbaarobject_m2', 'verhouding_opp_vbo_opp_pnd']].copy()

      # Gebouwen met woning type 'NULL' zijn geen verblijfsobjecten
      df_prepared['woningtype'] = df_prepared['woningtype'].fillna('Niet bewoonbaar')

      df_prepared['leeftijd'] = datetime.now().year - df_prepared['bouwjaar']
      
      df_prepared['openbareruimtenaam'] = df['openbareruimtenaam'].astype(str).str.strip().str.lower()

      df_prepared['huisletter'] = df['huisletter'].fillna('').astype(str).str.strip().str.upper()

      df_prepared['levensduur'] = df_prepared['woningtype'].apply(self._predict_lifespan)
      df_prepared['relatieve_ouderdom'] = df_prepared['leeftijd'] / df_prepared['levensduur']

      df_prepared['pandstatus'] = df_prepared['pandstatus'].astype('category').cat.codes
      df_prepared['woningtype'] = df_prepared['woningtype'].astype('category')
      df_prepared['woningtype_categorie'] = df_prepared['woningtype'].cat.codes
      
      df_prepared['sloopkans'] = (df_prepared['bouwjaar'] < 1970).astype(int)

      logger.info("Successfully prepared dataset")

      return df_prepared
    
    except KeyError as e:
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
  
  def _construct_building(self, df: pd.DataFrame, address: Address) -> Building:
    logger.info(f"Constructing building object for address: {address}")

    query = (
      (df['openbareruimtenaam'] == address.street) &
      (df['huisnummer'] == address.house_number) &
      (df['huisletter'] == address.house_number_addition)
    )

    match = df[query]
    if match.empty:
      return None
    
    row = match.iloc[0]

    build_year = int(row['bouwjaar'])

    building_type = row['woningtype']
    if pd.isna(building_type):
      building_type = 'Niet bewoonbaar'
    building_type = str(building_type)

    area = float(row['opp_adresseerbaarobject_m2'])
    area_ratio = float(row['verhouding_opp_vbo_opp_pnd'])

    age = datetime.now().year - build_year
    lifespan = self._predict_lifespan(building_type)
    relative_age = age / lifespan

    try:
      building_type_encoded = df['woningtype'].cat.categories.get_loc(building_type)
    except KeyError:
      error_message = f"Building type {building_type} is not a valid building type."
      logger.error(error_message)
      return {"error": error_message}
    
    return Building(
      build_year=build_year,
      building_type=building_type,
      building_type_idx=building_type_encoded,
      age=age,
      relative_age=relative_age,
      predicted_lifespan=lifespan,
      area=area,
      area_ratio=area_ratio
    )

analysis_engine = DataAnalysisEngine()