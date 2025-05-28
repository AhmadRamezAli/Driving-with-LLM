from utils.vector_utils.modular_vector_representation import *
import math
import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

# Utility functions
def xy_from_vehicle_desc(vehicle_array):
    x = vehicle_array[:, VehicleField.X]
    y = vehicle_array[:, VehicleField.Y]
    return torch.vstack((x, y)).T * METRES_M_SCALE


def traveling_angle_deg_from_vehicle_desc(vehicle_array):
    dx = vehicle_array[:, VehicleField.DX]
    dy = vehicle_array[:, VehicleField.DY]
    return direction_to_angle_deg(dx, dy)


def speed_mph_from_vehicle_desc(vehicle_array):
    return vehicle_array[:, VehicleField.SPEED] * VELOCITY_MS_SCALE * MS_TO_MPH


def xy_from_pedestrian_desc(pedestrian_array):
    x = pedestrian_array[:, PedestrianField.X]
    y = pedestrian_array[:, PedestrianField.Y]
    return torch.vstack((x, y)).T * METRES_M_SCALE


def traveling_angle_deg_from_pedestrian_desc(pedestrian_array):
    dx = pedestrian_array[:, PedestrianField.DX]
    dy = pedestrian_array[:, PedestrianField.DY]
    return direction_to_angle_deg(dx, dy)


def xy_from_route_desc(route_array):
    x = route_array[:, RouteField.X]
    y = route_array[:, RouteField.Y]
    return torch.vstack((x, y)).T * METRES_M_SCALE


def route_angles_from_route_desc(route_array):
    dirx, diry = (
        route_array[:, RouteField.TANGENT_DX],
        route_array[:, RouteField.TANGENT_DY],
    )
    return direction_to_angle_deg(dirx, diry)


def flags_in_fov(xy_coords, fov_degrees=60, max_distance=40):
    distances, angular = angles_deg_and_distances(xy_coords)
    return (
        (xy_coords[:, 0] > 0)
        & (-fov_degrees < angular)
        & (angular < fov_degrees)
        & (distances <= max_distance)
    )


def angles_deg_and_distances(xy_coords):
    distances = torch.linalg.norm(xy_coords, axis=1)
    angular = direction_to_angle_deg(xy_coords[:, 0], xy_coords[:, 1])
    return distances, angular


def direction_to_angle_deg(dirx, diry):
    return torch.atan2(-diry, dirx) * 180.0 / np.pi


def sort_angular(distances, angular):
    angular_order = np.argsort(angular)
    return distances[angular_order], angular[angular_order]


def vehicle_filter_flags(vehicle_descriptors):
    active_flags = vehicle_descriptors[:, VehicleField.ACTIVE] == 1
    fov_flags = flags_in_fov(xy_from_vehicle_desc(vehicle_descriptors), max_distance=40)
    return active_flags & fov_flags


def pedestrian_filter_flags(pedestrian_descriptors):
    active_flags = pedestrian_descriptors[:, PedestrianField.ACTIVE] == 1
    fov_flags = flags_in_fov(
        xy_from_pedestrian_desc(pedestrian_descriptors), max_distance=30
    )
    return active_flags & fov_flags


def distance_to_junction(route_descriptor):
    is_junction = route_descriptor[:, RouteField.HAS_JUNCTION] > 0.0
    if is_junction[0]:
        return 0
    elif torch.all(~is_junction):
        return torch.inf
    else:
        distances, angular = angles_deg_and_distances(
            xy_from_route_desc(route_descriptor)
        )
        return torch.amin(distances[is_junction])


def get_tl_state(route_descriptor):
    is_tl = route_descriptor[:, RouteField.HAS_TL] > 0.0
    if torch.all(~is_tl):
        return None, None

    distances, angular = angles_deg_and_distances(xy_from_route_desc(route_descriptor))

    tl_index = np.where(is_tl)[0][0]
    tl_flags = route_descriptor[tl_index, RouteField.TL_GO : RouteField.TL_GO + 4] > 0.0
    index_on = np.where(tl_flags)[0][0]

    return ["green", "yellow", "red", "red+yellow"][index_on], distances[tl_index]


def object_direction(angle_deg):
    if abs(angle_deg) < 45:
        return "same direction as me"
    elif abs(abs(angle_deg) - 180) < 45:
        return "opposite direction from me"
    elif abs(angle_deg - 90) < 45:
        return "from left to right"
    elif abs(angle_deg + 90) < 45:
        return "from right to left"
    return f"{angle_deg} degrees"


def side(angle_deg):
    return "left" if angle_deg < 0 else "right"


def control_to_pedals(control_longitudinal):
    x = 2.0 * control_longitudinal - 1.0
    accelerator_pedal_pct = np.clip(x, 0.0, 1.0)
    brake_pressure_pct = np.clip(-x, 0.0, 1.0)
    return accelerator_pedal_pct, brake_pressure_pct


def determine_roundabout(route_descriptors):
    route_angles = route_angles_from_route_desc(route_descriptors)
    angle_diffs = torch.diff(route_angles)
    angle_diffs[angle_diffs > 180] -= 360
    angle_diffs[angle_diffs < -180] += 360
    total_turn_right = torch.sum(angle_diffs[angle_diffs > 0])
    total_turn_left = torch.sum(angle_diffs[angle_diffs < 0])
    return abs(total_turn_left) > 30 and abs(total_turn_right) > 30

