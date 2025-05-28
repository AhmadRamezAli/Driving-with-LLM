import math
import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import utils.vector_utils.utils_function as utils_function
from utils.vector_utils.modular_vector_representation import *
# Randomization utils
class Randomizable:
    ENUM: Any = None
    FIELD_TYPES_RANGES: Dict[str, Tuple[Any, Any]] = {}
    FIELD_PROB: Dict[str, float] = {}

    @classmethod
    def randomize(cls, vector: np.array):
        randomize_enum(cls.ENUM, vector, cls.FIELD_TYPES_RANGES, cls.FIELD_PROB)


def randomize_enum(enum_cls, vector: np.array, field_types_ranges, field_probs):
    for field_name in field_types_ranges:
        idx = getattr(enum_cls, field_name)
        field_type_range = field_types_ranges[field_name]
        field_prob = field_probs.get(field_name, 0.5)
        vector[idx] = random_value(field_type_range, field_prob)


def random_value(
    field_type_range: Tuple[type, Tuple[float, float]],
    field_prob: float = 0.5,
) -> Union[int, float]:
    field_type, field_range = field_type_range
    if field_type == bool:
        return 1 if random.random() < field_prob else 0
    if field_type == float:
        return random.uniform(*field_range)
    raise ValueError(f"Unsupported field type: {field_type}")


class VehicleFieldRandom(Randomizable):
    ENUM = VehicleField
    FIELD_TYPES_RANGES: Dict[str, Any] = {
        "ACTIVE": (bool, (0.0, 1.0)),
        "DYNAMIC": (bool, (0.0, 1.0)),
        "SPEED": (float, (-1.4e-09, 2.2e00)),
        "X": (float, (-9.9e00, 9.9e00)),
        "Y": (float, (-9.9e00, 9.9e00)),
        "Z": (float, (0.0, 0.0)),
        "DX": (float, (-1.0, 1.0)),
        "DY": (float, (-1.0, 1.0)),
        "PITCH": (float, (0.0, 0.0)),
        "HALF_LENGTH": (float, (0.0, 2.4e00)),
        "HALF_WIDTH": (float, (0.0, 9.6e-01)),
        "HALF_HEIGHT": (float, (0.0, 8.9e-01)),
        "UNSPECIFIED_12": (bool, (0.0, 1.0)),
        "UNSPECIFIED_13": (bool, (0.0, 1.0)),
        "UNSPECIFIED_14": (bool, (0.0, 1.0)),
        "UNSPECIFIED_15": (bool, (0.0, 1.0)),
        "UNSPECIFIED_16": (bool, (0.0, 1.0)),
        "UNSPECIFIED_17": (bool, (0.0, 1.0)),
        "UNSPECIFIED_18": (float, (-1.0e01, 1.0e01)),
        "UNSPECIFIED_19": (float, (-1.0e01, 1.0e01)),
        "UNSPECIFIED_20": (float, (-1.0e01, 1.0e01)),
        "UNSPECIFIED_21": (float, (-1.0e01, 1.0e01)),
        "UNSPECIFIED_22": (float, (-1.0e01, 1.0e01)),
        "UNSPECIFIED_23": (float, (-1.0e01, 1.0e01)),
        "UNSPECIFIED_24": (float, (-1.0e01, 1.0e01)),
        "UNSPECIFIED_25": (float, (-1.0e01, 1.0e01)),
        "UNSPECIFIED_26": (float, (-1.0e-09, 2.0e00)),
        "UNSPECIFIED_27": (float, (-1.0e-09, 2.0e00)),
        "UNSPECIFIED_28": (float, (-1.0e-10, 2.0e00)),
        "UNSPECIFIED_29": (float, (-2.0e-10, 4.0e00)),
        "UNSPECIFIED_30": (bool, (0.0, 1.0)),
        "UNSPECIFIED_31": (bool, (0.0, 1.0)),
        "UNSPECIFIED_32": (bool, (0.0, 1.0)),
    }

    FIELD_PROB: Dict[str, float] = {"ACTIVE": 0.8}


class PedestrianFieldRandom(Randomizable):
    ENUM = PedestrianField

    FIELD_TYPES_RANGES: Dict[str, Tuple[Any, Any]] = {
        "ACTIVE": (bool, (0.0, 1.0)),
        "SPEED": (float, (0.0, 1.7933)),
        "X": (float, (0.0, 4.9982)),
        "Y": (float, (-4.9940, 4.9829)),
        "Z": (float, (0.0, 0.1992)),
        "DX": (float, (-1.0, 1.0)),
        "DY": (float, (-1.0, 1.0)),
        "CROSSING": (bool, (0.0, 1.0)),
    }
    FIELD_PROB: Dict[str, float] = {"ACTIVE": 0.9}


class RouteFieldRandom:
    FIELD_TYPES_RANGES: Dict[str, Tuple[Any, Any]] = {
        "X": (float, (-4.2200, 6.0593)),
        "Y": (float, (-5.7505, 5.7946)),
        "Z": (float, (0.0, 0.0)),
        "TANGENT_DX": (float, (-1.0, 1.0)),
        "TANGENT_DY": (float, (-1.0, 1.0)),
        "PITCH": (float, (0.0, 0.0)),
        "SPEED_LIMIT": (float, (1.7882, 2.6822)),
        "HAS_JUNCTION": (bool, (0.0, 1.0)),
        "ROAD_WIDTH0": (float, (0.1275, 1.0)),
        "ROAD_WIDTH1": (float, (0.1708, 1.0)),
        "HAS_TL": (bool, (0.0, 1.0)),
        "TL_GO": (bool, (0.0, 1.0)),
        "TL_GOTOSTOP": (bool, (0.0, 1.0)),
        "TL_STOP": (bool, (0.0, 1.0)),
        "TL_STOPTOGO": (bool, (0.0, 1.0)),
        "IS_GIVEWAY": (bool, (0.0, 1.0)),
        "IS_ROUNDABOUT": (bool, (0.0, 1.0)),
    }

    @staticmethod
    def randomize_route_field(field_types_ranges, vector: np.array, has_tl=None):
        if has_tl is None:
            has_tl = random.random() < 0.5
        for field_name in field_types_ranges:
            idx = getattr(RouteField, field_name)
            field_type_range = field_types_ranges[field_name]
            vector[idx] = random_value(field_type_range)
        # reset the traffic light state
        if has_tl is None:
            has_tl = random.random() < 0.75
        vector[RouteField.HAS_TL] = 1 if has_tl else 0
        vector[RouteField.TL_GO : RouteField.TL_STOPTOGO + 1] = (
            vector[RouteField.TL_GO : RouteField.TL_STOPTOGO + 1] * 0.0
        )
        if vector[RouteField.HAS_TL] == 1:
            tl_state = random.randint(0, 3)
            vector[RouteField.TL_GO + tl_state] = 1

    @classmethod
    def randomize(cls, vector: np.array, has_tl=None):
        cls.randomize_route_field(cls.FIELD_TYPES_RANGES, vector, has_tl)


class EgoFieldRandom(Randomizable):
    ENUM = EgoField
    FIELD_TYPES_RANGES: Dict[str, Tuple[Any, Any]] = {
        "ACCEL": (float, (-4.1078, 1.9331)),
        "SPEED": (float, (-3.2891e-10, 2.1762)),
        "BRAKE_PRESSURE": (float, (0.0, 0.89103)),
        "STEERING_ANGLE": (float, (-7.8079, 7.3409)),
        "PITCH": (float, (0.0, 0.0)),
        "HALF_LENGTH": (float, (0.0, 2.3250)),
        "HALF_WIDTH": (float, (0.0, 1.0050)),
        "HALF_HEIGHT": (float, (0.0, 0.7800)),
        "UNSPECIFIED_8": (bool, (0.0, 1.0)),
        "CLASS_START": (bool, (0.0, 0.0)),
        "CLASS_END": (bool, (0.0, 0.0)),
        "DYNAMICS_START": (bool, (0.0, 1.0)),
        "DYNAMICS_END": (bool, (0.0, 0.0)),
        "PREV_ACTION_START": (float, (0.0, 0.99949)),
        "UNSPECIFIED_17": (float, (-0.98492, 0.91599)),
        "PREV_ACTION_END": (float, (0.0, 0.99169)),
        "RAYS_LEFT_START": (bool, (0.0, 1.0)),
        "UNSPECIFIED_20": (bool, (0.0, 1.0)),
        "UNSPECIFIED_21": (bool, (0.0, 1.0)),
        "UNSPECIFIED_22": (bool, (0.0, 1.0)),
        "UNSPECIFIED_23": (bool, (0.0, 1.0)),
        "RAYS_LEFT_END": (bool, (0.0, 1.0)),
        "RAYS_RIGHT_START": (bool, (0.0, 1.0)),
        "UNSPECIFIED_26": (bool, (0.0, 1.0)),
        "UNSPECIFIED_27": (bool, (0.0, 1.0)),
        "UNSPECIFIED_28": (bool, (0.0, 1.0)),
        "UNSPECIFIED_29": (bool, (0.0, 1.0)),
        "RAYS_RIGHT_END": (bool, (0.0, 1.0)),
    }


class LiableVehiclesRandom(Randomizable):
    ENUM = LiableVechicleField
    FIELD_TYPES_RANGES: Dict[str, Tuple[Any, Any]] = {
        "VEHICLE_0": (bool, (0.0, 1.0)),
        "VEHICLE_1": (bool, (0.0, 1.0)),
        "VEHICLE_2": (bool, (0.0, 1.0)),
        "VEHICLE_3": (bool, (0.0, 1.0)),
        "VEHICLE_4": (bool, (0.0, 1.0)),
        "VEHICLE_5": (bool, (0.0, 1.0)),
        "VEHICLE_6": (bool, (0.0, 1.0)),
        "VEHICLE_7": (bool, (0.0, 1.0)),
        "VEHICLE_8": (bool, (0.0, 1.0)),
        "VEHICLE_9": (bool, (0.0, 1.0)),
        "VEHICLE_13": (bool, (0.0, 1.0)),
        "VEHICLE_14": (bool, (0.0, 1.0)),
        "VEHICLE_15": (bool, (0.0, 1.0)),
        "VEHICLE_16": (bool, (0.0, 1.0)),
        "VEHICLE_17": (bool, (0.0, 1.0)),
        "VEHICLE_18": (bool, (0.0, 1.0)),
        "VEHICLE_19": (bool, (0.0, 1.0)),
        "VEHICLE_20": (bool, (0.0, 1.0)),
        "VEHICLE_21": (bool, (0.0, 1.0)),
        "VEHICLE_22": (bool, (0.0, 1.0)),
        "VEHICLE_23": (bool, (0.0, 1.0)),
        "VEHICLE_24": (bool, (0.0, 1.0)),
        "VEHICLE_25": (bool, (0.0, 1.0)),
        "VEHICLE_26": (bool, (0.0, 1.0)),
        "VEHICLE_27": (bool, (0.0, 1.0)),
        "VEHICLE_28": (bool, (0.0, 1.0)),
        "VEHICLE_29": (bool, (0.0, 1.0)),
    }
    

@dataclass
class VectorObservation:
    """
    Vectorized representation
    It stores information about the environment in the float torch tensors, coding flags and properties
    about the route, nearby vehicles, pedestrians etc.

    Arrays of dynamic number of objects, such as nearby pedestrians use the following scheme:
     - an array is preallocated for a max capacity and initialized with 0s
     - every found object sets first number to 1
    Example: we allocate 20 rows to describe pedestrians, but if there are 5 pedestrians around, then
        only first 5 rows of that array will have 1 in the first column. Others would be 0s
    All objects like that are ordered by distance, such that the closest object goes in the row 0 and so on.

    """

    ROUTE_DIM = 17
    VEHICLE_DIM = 33
    PEDESTRIAN_DIM = 9
    EGO_DIM = 31

    # A 2d array per ego vehicle describing the route to follow.
    # It finds route points for each vehicle. Each point goes into a new row, then for each point:
    # - (x, y, z) of a point
    # - its direction
    # - pitch
    # - speed limit
    # - is junction?
    # - width of the road
    # - is traffic light and its state
    # - is give way?
    # - is a roundabout?
    route_descriptors: torch.FloatTensor

    # A 2d array per ego vehicle describing nearby vehicles.
    # First, if finds nearby vehicle in the neighbourhood of the car.
    # Then allocates an array of zeros a fixed max size (about 30).
    # There is a logic that tries to fit dynamic and static vehicles into rows of that array.
    # For every vehicle:
    # - "1" for marking that a vehicle is found in the row (empty rows will have this still "0")
    # - Is it dynamic or static (parked) vehicle
    # - its speed
    # - its relative position in the ego coordinates
    # - its relative orientation
    # - its pitch
    # - its size
    # - vehicle class
    # - positions of its 4 corners
    vehicle_descriptors: torch.FloatTensor

    # A 2d array per ego vehicle describing pedestrians.
    # First, if finds nearby pedestrians in the neighbourhood of the car.
    # Then allocates an array of zeros a fixed max size (about 20).
    # Then every found pedestrian is described in a row of that array:
    # - "1" for marking that a pedestrian is found in the row (empty rows will have this still "0")
    # - ped. speed
    # - its relative position in the ego coordinates (x, y, z)
    # - its relative orientation
    # - pedestrian type
    # - intent of crossing the road
    pedestrian_descriptors: torch.FloatTensor

    # A 1D array per ego vehicle describing its state. Specifically,
    # - VehicleDynamicsState (acc, steering, pitch ..)
    # - Vehicle size
    # - Vehicle class
    # - Vehicle dynamics type
    # - Previous action
    # - 2 lidar distance arrays, placed on the front corner of the vehicle
    ego_vehicle_descriptor: torch.FloatTensor

    # Deprecated
    liable_vehicles: Optional[torch.FloatTensor] = None


class VectorObservationConfig:
    num_route_points: int = 30
    num_vehicle_slots: int = 30
    num_pedestrian_slots: int = 20

    radius_m: float = 100.0
    pedestrian_radius_m: float = 50.0
    pedestrian_angle_threshold_rad: float = math.pi / 2
    route_spacing_m: float = 2.0
    num_max_static_vehicles: int = 10

    # Only include vehicles and pedestrians that have line of sight inside footpath and road navmesh
    line_of_sight: bool = False

