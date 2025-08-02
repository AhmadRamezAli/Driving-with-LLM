from pydantic import BaseModel, Field
from typing import List


class Situation(BaseModel):
    """Represents a single situation in the scene."""
    collection: str = Field(..., description="The collection name.")


class Vehicle(BaseModel):
    """Represents a single vehicle in the scene."""

    active: float = Field(..., description="Indicates if the vehicle is active.")
    dynamic: float = Field(..., description="Indicates if the vehicle is dynamic.")
    speed: float = Field(..., description="The speed of the vehicle.")
    x: float = Field(..., description="The x-coordinate of the vehicle.")
    y: float = Field(..., description="The y-coordinate of the vehicle.")
    z: float = Field(..., description="The z-coordinate of the vehicle.")
    dx: float = Field(..., description="The change in x-coordinate.")
    dy: float = Field(..., description="The change in y-coordinate.")
    pitch: float = Field(..., description="The pitch of the vehicle.")
    half_length: float = Field(..., description="Half the length of the vehicle.")
    half_width: float = Field(..., description="Half the width of the vehicle.")
    half_height: float = Field(..., description="Half the height of the vehicle.")


class Pedestrian(BaseModel):
    """Represents a single pedestrian in the scene."""

    active: float = Field(..., description="Indicates if the pedestrian is active.")
    speed: float = Field(..., description="The speed of the pedestrian.")
    x: float = Field(..., description="The x-coordinate of the pedestrian.")
    y: float = Field(..., description="The y-coordinate of the pedestrian.")
    z: float = Field(..., description="The z-coordinate of the pedestrian.")
    dx: float = Field(..., description="The change in x-coordinate.")
    dy: float = Field(..., description="The change in y-coordinate.")
    crossing: float = Field(..., description="Indicates if the pedestrian is crossing.")


class Route(BaseModel):
    """Represents a single point on a route."""

    x: float = Field(..., description="The x-coordinate of the route point.")
    y: float = Field(..., description="The y-coordinate of the route point.")
    z: float = Field(..., description="The z-coordinate of the route point.")
    tangent_dx: float = Field(..., description="The tangent change in x-coordinate.")
    tangent_dy: float = Field(..., description="The tangent change in y-coordinate.")
    pitch: float = Field(..., description="The pitch of the route point.")
    speed_limit: float = Field(..., description="The speed limit at this point.")
    has_junction: float = Field(..., description="Indicates if there is a junction.")
    road_width0: float = Field(..., description="The road width at this point.")
    road_width1: float = Field(..., description="The road width at this point.")
    has_tl: float = Field(..., description="Indicates if there is a traffic light.")
    tl_go: float = Field(..., description="Indicates if the traffic light is green.")
    tl_gotostop: float = Field(
        ..., description="Indicates if the traffic light is changing to red."
    )
    tl_stop: float = Field(..., description="Indicates if the traffic light is red.")
    tl_stoptogo: float = Field(
        ..., description="Indicates if the traffic light is changing to green."
    )
    is_giveway: float = Field(..., description="Indicates if there is a give way sign.")
    is_roundabout: float = Field(..., description="Indicates if there is a roundabout.")


class Ego(BaseModel):
    """Represents the ego vehicle."""

    accel: float = Field(..., description="The acceleration of the ego vehicle.")
    speed: float = Field(..., description="The speed of the ego vehicle.")
    brake_pressure: float = Field(
        ..., description="The brake pressure of the ego vehicle."
    )
    steering_angle: float = Field(
        ..., description="The steering angle of the ego vehicle."
    )
    pitch: float = Field(..., description="The pitch of the ego vehicle.")
    half_length: float = Field(..., description="Half the length of the ego vehicle.")
    half_width: float = Field(..., description="Half the width of the ego vehicle.")
    half_height: float = Field(..., description="Half the height of the ego vehicle.")
    class_start: float = Field(..., description="Start of the class.")
    class_end: float = Field(..., description="End of the class.")
    dynamics_start: float = Field(..., description="Start of the dynamics.")
    dynamics_end: float = Field(..., description="End of the dynamics.")
    prev_action_start: float = Field(..., description="Start of the previous action.")
    prev_action_end: float = Field(..., description="End of the previous action.")
    rays_left_start: float = Field(..., description="Start of the left rays.")
    rays_left_end: float = Field(..., description="End of the left rays.")
    rays_right_start: float = Field(..., description="Start of the right rays.")
    rays_right_end: float = Field(..., description="End of the right rays.")


class Scene(BaseModel):
    """Represents the entire scene, containing vehicles, pedestrians, routes, and the ego vehicle."""

    vehicles: List[Vehicle] = Field(..., description="A list of vehicles in the scene.")
    pedestrians: List[Pedestrian] = Field(
        ..., description="A list of pedestrians in the scene."
    )
    routes: List[Route] = Field(..., description="A list of routes in the scene.")
    ego: Ego = Field(..., description="The ego vehicle.")
    situation: Situation = Field(..., description="The situation of the prediction.")
