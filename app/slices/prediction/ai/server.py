# --------- imports at the top stay the same ----------
import json
from pathlib import Path
from typing import Dict, List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer

from app.slices.prediction.ai.utils.model_utils import load_model
from app.slices.prediction.ai.utils.vector_utils.randomize_utils import VectorObservation
from app.slices.prediction.ai.utils.vector_utils.modular_vector_representation import EgoField, VehicleField, PedestrianField, RouteField
from app.slices.prediction.ai.action_parser import parse_actions

_EGO_ORDER        = [f.name for f in EgoField]          # 31 names
_VEHICLE_ORDER    = [f.name for f in VehicleField]      # 33 names
_PEDESTRIAN_ORDER = [f.name for f in PedestrianField]   #  9 names
_ROUTE_ORDER      = [f.name for f in RouteField]        # 17 names




def _pad(rows: list[list[float]], want: int, width: int):
    rows = rows[:want] + [[0.0]*width]*(want-len(rows))
    return rows

class PacketModel(BaseModel):
    ego_vehicle_descriptor:  List[float]
    vehicle_descriptors:     List[List[float]]
    pedestrian_descriptors:  List[List[float]]
    route_descriptors:       List[List[float]]
class ActionOut(BaseModel):
    accelerator_percent: float
    brake_percent: float
    steer_percent: float
