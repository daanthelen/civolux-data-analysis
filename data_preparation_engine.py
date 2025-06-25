import pandas as pd
import logging
from datetime import datetime
from models import Address, Building, ClusterPrediction

logger = logging.getLogger(__name__)

class DataPreparationEngine:
  def prepare_for_demolition_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Preparing dataset for demolition prediction')

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

      logger.info('Successfully prepared dataset')

      return df_prepared
    
    except KeyError as e:
      raise
  
  def prepare_for_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Preparing dataset for clustering')

    try:
      df_prepared = df[['lon', 'lat', 'bouwjaar']].dropna()

      df_prepared['leeftijd'] = datetime.now().year - df_prepared['bouwjaar']

      logger.info('Successfully prepared dataset')

      return df_prepared
    
    except KeyError as e:
      raise

  def construct_building(self, df: pd.DataFrame, address: Address) -> Building:
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
  
  def construct_cluster_prediction(self, row: pd.Series) -> ClusterPrediction:
    return ClusterPrediction(
      longtitude=row['lon'],
      latitude=row['lat'],
      build_year=row['bouwjaar'],
      age=row['leeftijd'],
      cluster=row['cluster']
    )

data_preparation_engine = DataPreparationEngine()