import pandas as pd
import numpy as np
import logging
from datetime import datetime
from models import Address, Building, DemolishPrediction, BuildingResponse, Cluster, ClusterBuilding, Material
from typing import List

logger = logging.getLogger(__name__)

class DataPreparationEngine:
  def prepare_buildings_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Preparing buildings dataset')

    # Gebouwen met woning type 'NULL' zijn geen verblijfsobjecten
    df['woningtype'] = df['woningtype'].fillna('Niet bewoonbaar')

    df.dropna(subset=['bouwjaar'], inplace=True)

    df['leeftijd'] = datetime.now().year - df['bouwjaar']
    
    df['openbareruimtenaam'] = df['openbareruimtenaam'].astype(str).str.strip()
    df['huisletter'] = df['huisletter'].fillna('').astype(str).str.strip().str.upper()

    df['levensduur'] = df['woningtype'].apply(self._predict_lifespan)
    df['relatieve_ouderdom'] = df['leeftijd'] / df['levensduur']

    df['pandstatus'] = df['pandstatus'].astype('category').cat.codes
    df['woningtype'] = df['woningtype'].astype('category')
    df['woningtype_categorie'] = df['woningtype'].cat.codes
  
    df['sloopkans'] = (df['bouwjaar'] < 1970).astype(int)

    conditions = [
      df['woonfunctie'] == 1,
      df['celfunctie'] == 1,
      df['gezondheidszorgfunctie'] == 1,
      df['industriefunctie'] == 1,
      df['kantoorfunctie'] == 1,
      df['logiesfunctie'] == 1,
      df['onderwijsfunctie'] == 1,
      df['sportfunctie'] == 1,
      df['winkelfunctie'] == 1,
    ]

    building_goals = [
      'Woning',
      'Gevangenis',
      'Zorg',
      'Industrie',
      'Kantoor',
      'Hotel',
      'Onderwijs',
      'Sporthal',
      'Winkel',
    ]

    df['gebouw_doel'] = np.select(conditions, building_goals, default='Overig')

    logger.info('Successfully prepared dataset')

    return df
  
  def prepare_materials_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Preparing materials dataset')

    df['functie'] = df['functie'].str.lower().str.strip()
  
  def prepare_for_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Preparing dataset for clustering')

    df.dropna(subset=['lon', 'lat', 'bouwjaar'], inplace=True)

    df['leeftijd'] = datetime.now().year - df['bouwjaar']

    logger.info('Successfully prepared dataset')

    return df
  
  def prepare_for_twin_prediction(self, df: pd.DataFrame):
    logger.info('Preparing dataset for twin prediction')

    df_prepared = df.dropna(subset=['bouwjaar', 'opp_pand', 'lon', 'lat'])

    return df_prepared

  def construct_building(self, df: pd.DataFrame, address: Address) -> Building:
    logger.info(f"Constructing building object for address: {address}")

    query = (
      (df['openbareruimtenaam'] == address.street) &
      (df['huisnummer'].astype(str) == str(address.house_number)) &
      (df['huisletter'] == address.house_number_addition)
    )

    match = df[query]
    if match.empty:
      return None
    
    row = match.iloc[0]

    return self.construct_building_from_row(row)
  
  def construct_building_from_row(self, row: pd.Series) -> Building:
    build_year = int(row['bouwjaar'])

    building_type = row['woningtype']
    if pd.isna(building_type):
      building_type = 'Niet bewoonbaar'
    building_type = str(building_type)

    age = datetime.now().year - build_year
    lifespan = self._predict_lifespan(building_type)
    relative_age = age / lifespan

    address = Address (
      street=row['openbareruimtenaam'],
      house_number=row['huisnummer'],
      house_number_addition=row['huisletter']
    )
    
    return Building(
      id=row['uniq_key'],
      longitude=row['lon'],
      latitude=row['lat'],
      address=address,
      build_year=build_year,
      building_type=building_type,
      building_type_idx=row['woningtype_categorie'],
      has_residential_func=bool(row['woonfunctie']),
      has_office_func=bool(row['kantoorfunctie']),
      has_shop_func=bool(row['winkelfunctie']),
      age=age,
      relative_age=relative_age,
      predicted_lifespan=lifespan,
      area=float(row['opp_adresseerbaarobject_m2']),
      area_ratio=float(row['verhouding_opp_vbo_opp_pnd'])
    )
  
  def construct_demolish_prediction(self, row: pd.Series, df_materials: pd.DataFrame) -> DemolishPrediction:
    building = self.construct_building_from_row(row)

    materials = self.get_materials(row, df_materials)

    return DemolishPrediction(
      id=building.id,
      longitude=building.longitude,
      latitude=building.latitude,
      address=f"{building.address.street} {building.address.house_number}{building.address.house_number_addition}",
      build_year=building.build_year,
      area=building.area,
      building_type=building.building_type,
      age=building.age,
      relative_age=building.relative_age,
      predicted_lifespan=building.predicted_lifespan,
      area_ratio=building.area_ratio,
      prediction=row['demolish_prediction'],
      demolition_probability=row['demolish_probability'],
      materials=materials
    )
  
  def get_materials(self, row: pd.Series, df_materials: pd.DataFrame) -> List[Material]:
    woningtype = row['woningtype']

    building_category = None
    if woningtype.lower() == 'vrijstaande woning' or woningtype.lower() == 'tweeonder1kap':
      building_category = 'vrijstaand'
    elif woningtype.lower() == 'tussen of geschakelde woning' or woningtype.lower() == 'hoekwoning':
      building_category = 'serieel'
    elif woningtype.lower() == 'appartement':
      building_category = 'appartement'

    if building_category is None:
      if row['winkelfunctie'] == 1:
        building_category = 'winkel'
      elif row['kantoorfunctie'] == 1:
        building_category = 'kantoor'
      elif row['industriefunctie'] == 1:
        building_category = 'bedrijfshal'
      elif row['gezondheidszorgfunctie'] == 1:
        building_category = 'zorg'
      elif row['onderwijsfunctie'] == 1:
        building_category = 'onderwijs'
    
    if row['bouwjaar'] < 1945:
        building_year_interval = '<1945'
    elif 1945 <= row['bouwjaar'] <= 1970:
        building_year_interval = '1945-1970'
    elif 1970 < row['bouwjaar'] <= 2000:
        building_year_interval = '1970-2000'
    elif row['bouwjaar'] > 2000:
        building_year_interval = '>2000'

    materials = df_materials[
      (df_materials['functie'] == building_category) &
      (df_materials['bouwjaar'] == building_year_interval)
    ]

    if materials is None or materials.empty:
      return []
    
    material_weights = materials.iloc[0].drop(['functie', 'bouwjaar']).astype(float)
    calculated_materials = { Material(name=material, quantity=round(weight_per_m2 * row['opp_pand'], 2) if pd.notna(weight_per_m2) else 0) for material, weight_per_m2 in material_weights.items() }

    return calculated_materials
  
  def construct_cluster_building(self, row: pd.Series) -> ClusterBuilding:
    building = self.construct_building_response(row)

    return ClusterBuilding(
      cluster=row['cluster'],
      building=building
    )
  
  def construct_cluster_prediction(self, buildings: List[ClusterBuilding], df: pd.DataFrame) -> List[Cluster]:
    clusters: List[Cluster] = []

    average_cluster_ages = df.groupby('cluster')['leeftijd'].mean().round(1)

    for building in buildings:
      cluster = next((cluster for cluster in clusters if cluster.id == building.cluster), None)

      if cluster:
        cluster.buildings.append(building.building)
      else:
        new_cluster = Cluster(
          id=building.cluster,
          buildings=[building.building],
          average_age=average_cluster_ages[building.cluster]
        )

        clusters.append(new_cluster)
    
    return clusters
        
  
  def construct_building_response(self, row: pd.Series) -> BuildingResponse:
    return BuildingResponse(
      id=row['uniq_key'],
      longitude=row['lon'],
      latitude=row['lat'],
      address=f"{row['openbareruimtenaam']} {row['huisnummer']}{row['huisletter']}",
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