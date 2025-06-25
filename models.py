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
    return v.strip().lower() if v else ''

class Building(BaseModel):
  build_year: int
  building_type: str
  building_type_idx: int
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
  longtitude: float
  latitude: float
  build_year: int
  age: int
  cluster: int