from pydantic import BaseModel, field_validator
from typing import Optional

class Address(BaseModel):
  street: str
  house_number: int
  house_number_addition: Optional[str] = ''

  @field_validator('street')
  @classmethod
  def set_street(cls, v: str) -> str:
    return v.strip().lower()
  
  @field_validator('house_number_addition')
  @classmethod
  def set_house_number_addition(cls, v: str = '') -> str:
    return v.strip().upper() if v else ''

class Building(BaseModel):
  id: float
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

class DemolishPrediction(BaseModel):
  address: str
  build_year: int
  building_type: str
  age: int
  relative_age: float
  predicted_lifespan: int
  area: float
  area_ratio: float
  prediction: bool
  demolition_probability: float

class ClusterPrediction(BaseModel):
  longitude: float
  latitude: float
  build_year: int
  age: int
  cluster: int