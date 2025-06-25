import pandas as pd
import logging
from datetime import datetime
from models import Address, Building, ClusterPrediction, DemolishPrediction, TwinBuilding

logger = logging.getLogger(__name__)

class DataPreparationEngine:
  def prepare_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Preparing dataset')

    # Gebouwen met woning type 'NULL' zijn geen verblijfsobjecten
    df['woningtype'] = df['woningtype'].fillna('Niet bewoonbaar')

    df['leeftijd'] = datetime.now().year - df['bouwjaar']
    
    df['openbareruimtenaam'] = df['openbareruimtenaam'].astype(str).str.strip().str.lower()

    df['huisletter'] = df['huisletter'].fillna('').astype(str).str.strip().str.upper()

    df['levensduur'] = df['woningtype'].apply(self._predict_lifespan)
    df['relatieve_ouderdom'] = df['leeftijd'] / df['levensduur']

    df['pandstatus'] = df['pandstatus'].astype('category').cat.codes
    df['woningtype'] = df['woningtype'].astype('category')
    df['woningtype_categorie'] = df['woningtype'].cat.codes
    
    df['sloopkans'] = (df['bouwjaar'] < 1970).astype(int)

    logger.info('Successfully prepared dataset')

    return df
  
  def prepare_for_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Preparing dataset for clustering')

    df_prepared = df[['lon', 'lat', 'bouwjaar', 'leeftijd']].dropna()

    logger.info('Successfully prepared dataset')

    return df_prepared
  
  def prepare_for_twin_prediction(self, df: pd.DataFrame):
    logger.info('Preparing dataset for twin prediction')

    df_prepared = df.dropna(subset=['bouwjaar', 'opp_pand', 'lon', 'lat'])

    return df_prepared

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
      id=row['pand_id'],
      build_year=build_year,
      building_type=building_type,
      building_type_idx=building_type_encoded,
      has_residential_func=bool(row['woonfunctie']),
      has_office_func=bool(row['kantoorfunctie']),
      has_shop_func=bool(row['winkelfunctie']),
      age=age,
      relative_age=relative_age,
      predicted_lifespan=lifespan,
      area=float(row['opp_adresseerbaarobject_m2']),
      area_ratio=float(row['verhouding_opp_vbo_opp_pnd'])
    )
  
  def construct_demolish_prediction(self, building: Building, address: Address, pred: bool, prob: float) -> DemolishPrediction:
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
  
  def construct_cluster_prediction(self, row: pd.Series) -> ClusterPrediction:
    return ClusterPrediction(
      longitude=row['lon'],
      latitude=row['lat'],
      build_year=row['bouwjaar'],
      age=row['leeftijd'],
      cluster=row['cluster']
    )
  
  def construct_twin_building(self, row: pd.Series) -> TwinBuilding:
    return TwinBuilding(
      longitude=row['lon'],
      latitude=row['lat'],
      build_year=row['bouwjaar'],
      area=row['opp_pand'],
      building_type=row['woningtype']
    )
  
  def _predict_lifespan(self, building_type: str):
    if building_type in ['Appartement', 'Tussenwoning', 'Hoekwoning', 'Vrijstaande woning', 'Woonhuis']:
      return 75
    elif building_type in ['Kantoor', 'Winkel', 'Bedrijfspand']:
      return 50
    else:
      return 60

data_preparation_engine = DataPreparationEngine()