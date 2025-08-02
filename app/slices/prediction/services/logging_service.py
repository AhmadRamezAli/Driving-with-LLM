import datetime
from app.slices.prediction.db.database import collection
# Scene-related models
from app.slices.prediction.models.scene import (
    Scene,
    Vehicle,
    Pedestrian,
    Route,
    Ego,
    Situation,
)
# External models / enums
from app.slices.prediction.models.prediction import Prediction
from app.slices.prediction.ai.server import PacketModel
from app.slices.prediction.ai.utils.vector_utils.modular_vector_representation import (
    VehicleField,
    PedestrianField,
    RouteField,
    EgoField,
)

# -------------------------------------------------------------
# Helper : convert PacketModel  →  Scene
# -------------------------------------------------------------


def _packet_to_scene(packet: PacketModel) -> Scene:
    """Transform the raw numeric packet into the structured *Scene* model."""

    # -------------------- Ego ----------------------------------------
    ed = list(packet.ego_vehicle_descriptor)
    if len(ed) <= max(EgoField):
        ed += [0.0] * (max(EgoField) + 1 - len(ed))
    ego = Ego(
        accel=ed[EgoField.ACCEL],
        speed=ed[EgoField.SPEED],
        brake_pressure=ed[EgoField.BRAKE_PRESSURE],
        steering_angle=ed[EgoField.STEERING_ANGLE],
        pitch=ed[EgoField.PITCH],
        half_length=ed[EgoField.HALF_LENGTH],
        half_width=ed[EgoField.HALF_WIDTH],
        half_height=ed[EgoField.HALF_HEIGHT],
        class_start=ed[EgoField.CLASS_START],
        class_end=ed[EgoField.CLASS_END],
        dynamics_start=ed[EgoField.DYNAMICS_START],
        dynamics_end=ed[EgoField.DYNAMICS_END],
        prev_action_start=ed[EgoField.PREV_ACTION_START],
        prev_action_end=ed[EgoField.PREV_ACTION_END],
        rays_left_start=ed[EgoField.RAYS_LEFT_START],
        rays_left_end=ed[EgoField.RAYS_LEFT_END],
        rays_right_start=ed[EgoField.RAYS_RIGHT_START],
        rays_right_end=ed[EgoField.RAYS_RIGHT_END],
    )

    # -------------------- Vehicles -----------------------------------
    vehicles = []
    for vd in packet.vehicle_descriptors:
        vd = list(vd)
        if len(vd) <= max(VehicleField):
            vd += [0.0] * (max(VehicleField) + 1 - len(vd))
        vehicles.append(
            Vehicle(
                active=vd[VehicleField.ACTIVE],
                dynamic=vd[VehicleField.DYNAMIC],
                speed=vd[VehicleField.SPEED],
                x=vd[VehicleField.X],
                y=vd[VehicleField.Y],
                z=vd[VehicleField.Z],
                dx=vd[VehicleField.DX],
                dy=vd[VehicleField.DY],
                pitch=vd[VehicleField.PITCH],
                half_length=vd[VehicleField.HALF_LENGTH],
                half_width=vd[VehicleField.HALF_WIDTH],
                half_height=vd[VehicleField.HALF_HEIGHT],
            )
        )

    # -------------------- Pedestrians --------------------------------
    pedestrians = []
    for pd in packet.pedestrian_descriptors:
        pd = list(pd)
        if len(pd) <= max(PedestrianField):
            pd += [0.0] * (max(PedestrianField) + 1 - len(pd))
        pedestrians.append(
            Pedestrian(
                active=pd[PedestrianField.ACTIVE],
                speed=pd[PedestrianField.SPEED],
                x=pd[PedestrianField.X],
                y=pd[PedestrianField.Y],
                z=pd[PedestrianField.Z],
                dx=pd[PedestrianField.DX],
                dy=pd[PedestrianField.DY],
                crossing=pd[PedestrianField.CROSSING],
            )
        )

    # -------------------- Route --------------------------------------
    routes = []
    for rd in packet.route_descriptors:
        rd = list(rd)
        if len(rd) <= max(RouteField):
            rd += [0.0] * (max(RouteField) + 1 - len(rd))
        routes.append(
            Route(
                x=rd[RouteField.X],
                y=rd[RouteField.Y],
                z=rd[RouteField.Z],
                tangent_dx=rd[RouteField.TANGENT_DX],
                tangent_dy=rd[RouteField.TANGENT_DY],
                pitch=rd[RouteField.PITCH],
                speed_limit=rd[RouteField.SPEED_LIMIT],
                has_junction=rd[RouteField.HAS_JUNCTION],
                road_width0=rd[RouteField.ROAD_WIDTH0],
                road_width1=rd[RouteField.ROAD_WIDTH1],
                has_tl=rd[RouteField.HAS_TL],
                tl_go=rd[RouteField.TL_GO],
                tl_gotostop=rd[RouteField.TL_GOTOSTOP],
                tl_stop=rd[RouteField.TL_STOP],
                tl_stoptogo=rd[RouteField.TL_STOPTOGO],
                is_giveway=rd[RouteField.IS_GIVEWAY],
                is_roundabout=rd[RouteField.IS_ROUNDABOUT],
            )
        )

    # -------------------- Assemble Scene -----------------------------
    scene = Scene(
        vehicles=vehicles,
        pedestrians=pedestrians,
        routes=routes,
        ego=ego,
        situation=Situation(collection="prediction")
    )

    return scene


import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_prediction(packet: PacketModel, prediction: Prediction):
    """Persist the prediction together with the **original request** in structured form."""
    print("--------------------------------")
    print("--------------------------------")
    print("--------------------------------")
    print("--------------------------------")
    print("start logging")
    print("--------------------------------")
    print("--------------------------------")
    print("--------------------------------")
    print("--------------------------------")
    # 1️⃣ convert raw packet → structured scene ------------------------
    scene = _packet_to_scene(packet)
    print("--------------------------------")
    print("--------------------------------")
    print("--------------------------------")
    print("end logging")
    print("--------------------------------")
    print("--------------------------------")
    print("--------------------------------")
    print("--------------------------------")
    log_entry = {
        "request": scene.model_dump(),
        "timestamp": datetime.datetime.now().isoformat(),
        "accelerate": prediction.accelerate,
        "brake": prediction.brake,
        "steering": prediction.steering,
        "caption": prediction.caption,
        "time_taken": prediction.time_taken,
    }
    print("--------------------------------")
    print("--------------------------------")
    print("log_entry")
    print(log_entry)
    print("--------------------------------")
    print("--------------------------------")
    try:
        collection.insert_one(log_entry)
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")
