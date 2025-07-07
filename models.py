from pydantic import BaseModel, field_validator
from typing import Optional, List
import pandas as pd

class Dataset(BaseModel):
  name: str
  dataset: pd.DataFrame

  class Config:
    arbitrary_types_allowed = True

class Address(BaseModel):
  street: str
  house_number: int
  house_number_addition: Optional[str] = ''

  @field_validator('street')
  @classmethod
  def set_street(cls, v: str) -> str:
    return v.strip()
  
  @field_validator('house_number_addition')
  @classmethod
  def set_house_number_addition(cls, v: str = '') -> str:
    return v.strip().upper() if v else ''
  
  class Config:
    frozen = True

class Building(BaseModel):
  id: str
  longitude: float
  latitude: float
  address: Address
  build_year: int
  building_type: str
  building_type_idx: int
  has_residential_func: bool
  has_office_func: bool
  has_shop_func: bool
  age: int
  relative_age: float
  predicted_lifespan: int
  area: float
  area_ratio: float

class BuildingResponse(BaseModel):
  id: str
  longitude: float
  latitude: float
  address: str
  build_year: int
  area: float
  building_type: str

  @field_validator('address')
  @classmethod
  def set_address(cls, v: str) -> str:
    words = v.split()
    formatted_words = []
    for word in words:
      if any(char.isdigit() for char in word) and any(char.isalpha() for char in word):
        numeric_part = ""
        alpha_part = ""
        for char in word:
          if char.isdigit():
            numeric_part += char
          else:
            alpha_part += char
        formatted_words.append(numeric_part + alpha_part.upper())
      else:
        formatted_words.append(word.title())
    return " ".join(formatted_words)

class ClusterBuilding(BaseModel):
  cluster: int
  building: BuildingResponse

class TwinBuildingPrediction(BaseModel):
  reference_building: BuildingResponse
  twin_buildings: List[BuildingResponse]

class Cluster(BaseModel):
  id: int
  buildings: List[BuildingResponse]
  average_age: float

class Material(BaseModel):
  name: str
  quantity: int

class DemolishPrediction(BuildingResponse):
  age: int
  relative_age: float
  predicted_lifespan: int
  area_ratio: float
  prediction: bool
  demolition_probability: float
  materials: List[Material]