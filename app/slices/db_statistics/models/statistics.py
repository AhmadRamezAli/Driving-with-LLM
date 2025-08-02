from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime


class SystemHealthStats(BaseModel):
    total_entries: int
    database_size_mb: float
    entries_over_time: Dict[str, int]  # timestamp -> count


class ModelPerformanceStats(BaseModel):
    time_taken_distribution: Dict[str, int]  # range -> count
    accelerate_distribution: Dict[str, float]  # statistical values
    brake_distribution: Dict[str, float]  # statistical values
    steering_distribution: Dict[str, float]  # statistical values
    common_caption_words: List[Dict[str, Any]]  # word -> count


class SceneCompositionStats(BaseModel):
    situation_frequencies: Dict[str, int]  # situation type -> count
    vehicles_distribution: Dict[str, int]  # range -> count
    pedestrians_distribution: Dict[str, int]  # range -> count
    road_features_frequency: Dict[str, int]  # feature -> count


class EgoActionCorrelationStats(BaseModel):
    speed_distribution: Dict[str, int]  # range -> count
    speed_vs_accelerate: List[Dict[str, float]]  # list of {speed, accelerate} pairs
    speed_vs_steering: List[Dict[str, float]]  # list of {speed, steering} pairs
    tl_stop_vs_brake: Dict[str, Dict[str, float]]  # Has TL -> brake stats


class DatabaseStatistics(BaseModel):
    system_health: SystemHealthStats
    model_performance: ModelPerformanceStats
    scene_composition: SceneCompositionStats
    ego_action_correlation: EgoActionCorrelationStats
    time_range: Dict[str, datetime]  # from_time, to_time 