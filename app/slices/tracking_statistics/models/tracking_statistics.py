from pydantic import BaseModel
from typing import List

class TrackingStatistics(BaseModel):
    number_of_pedestrian: int
    number_of_vehicle: int
    accel: float
    speed: float
    brake_pressure: float
    timestamp: str

class TrackingStatisticsResponse(BaseModel):
    statistics: List[TrackingStatistics]
    count: int 