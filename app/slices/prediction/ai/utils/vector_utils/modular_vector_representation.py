import math
import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

# Setting up scale factors
METRES_M_SCALE = 10.0
MS_TO_MPH = 2.23694
VELOCITY_MS_SCALE = 5.0


class VehicleField(IntEnum):
    ACTIVE = 0
    DYNAMIC = 1
    SPEED = 2
    X = 3
    Y = 4
    Z = 5
    DX = 6
    DY = 7
    PITCH = 8
    HALF_LENGTH = 9
    HALF_WIDTH = 10
    HALF_HEIGHT = 11
    UNSPECIFIED_12 = 12
    UNSPECIFIED_13 = 13
    UNSPECIFIED_14 = 14
    UNSPECIFIED_15 = 15
    UNSPECIFIED_16 = 16
    UNSPECIFIED_17 = 17
    UNSPECIFIED_18 = 18
    UNSPECIFIED_19 = 19
    UNSPECIFIED_20 = 20
    UNSPECIFIED_21 = 21
    UNSPECIFIED_22 = 22
    UNSPECIFIED_23 = 23
    UNSPECIFIED_24 = 24
    UNSPECIFIED_25 = 25
    UNSPECIFIED_26 = 26
    UNSPECIFIED_27 = 27
    UNSPECIFIED_28 = 28
    UNSPECIFIED_29 = 29
    UNSPECIFIED_30 = 30
    UNSPECIFIED_31 = 31
    UNSPECIFIED_32 = 32


class PedestrianField(IntEnum):
    ACTIVE = 0
    SPEED = 1
    X = 2
    Y = 3
    Z = 4
    DX = 5
    DY = 6
    CROSSING = 8


class RouteField(IntEnum):
    X = 0
    Y = 1
    Z = 2
    TANGENT_DX = 3
    TANGENT_DY = 4
    PITCH = 5
    SPEED_LIMIT = 6
    HAS_JUNCTION = 7
    ROAD_WIDTH0 = 8
    ROAD_WIDTH1 = 9
    HAS_TL = 10
    TL_GO = 11
    TL_GOTOSTOP = 12
    TL_STOP = 13
    TL_STOPTOGO = 14
    IS_GIVEWAY = 15
    IS_ROUNDABOUT = 16


class EgoField(IntEnum):
    ACCEL = 0
    SPEED = 1
    BRAKE_PRESSURE = 2
    STEERING_ANGLE = 3
    PITCH = 4
    HALF_LENGTH = 5
    HALF_WIDTH = 6
    HALF_HEIGHT = 7
    UNSPECIFIED_8 = 8
    CLASS_START = 9
    CLASS_END = 13
    DYNAMICS_START = 14
    DYNAMICS_END = 15
    PREV_ACTION_START = 16
    UNSPECIFIED_17 = 17
    PREV_ACTION_END = 18
    RAYS_LEFT_START = 19
    UNSPECIFIED_20 = 20
    UNSPECIFIED_21 = 21
    UNSPECIFIED_22 = 22
    UNSPECIFIED_23 = 23
    RAYS_LEFT_END = 24
    RAYS_RIGHT_START = 25
    UNSPECIFIED_26 = 26
    UNSPECIFIED_27 = 27
    UNSPECIFIED_28 = 28
    UNSPECIFIED_29 = 29
    RAYS_RIGHT_END = 30


class LiableVechicleField(IntEnum):
    VEHICLE_0 = 0
    VEHICLE_1 = 1
    VEHICLE_2 = 2
    VEHICLE_3 = 3
    VEHICLE_4 = 4
    VEHICLE_5 = 5
    VEHICLE_6 = 6
    VEHICLE_7 = 7
    VEHICLE_8 = 8
    VEHICLE_9 = 9
    VEHICLE_13 = 13
    VEHICLE_14 = 14
    VEHICLE_15 = 15
    VEHICLE_16 = 16
    VEHICLE_17 = 17
    VEHICLE_18 = 18
    VEHICLE_19 = 19
    VEHICLE_20 = 20
    VEHICLE_21 = 21
    VEHICLE_22 = 22
    VEHICLE_23 = 23
    VEHICLE_24 = 24
    VEHICLE_25 = 25
    VEHICLE_26 = 26
    VEHICLE_27 = 27
    VEHICLE_28 = 28
    VEHICLE_29 = 29
