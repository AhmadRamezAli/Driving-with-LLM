from __future__ import annotations

import time
from typing import List

import torch
from transformers import AutoTokenizer

from app.slices.prediction.ai.utils.vector_utils.modular_vector_representation import (
    EgoField,
    VehicleField,
    PedestrianField,
    RouteField,
)
from app.slices.prediction.ai.utils.vector_utils.randomize_utils import VectorObservation
from app.slices.prediction.ai.action_parser import parse_actions
from app.slices.prediction.ai.utils.model_utils import load_model  # type: ignore
from app.slices.prediction.models.scene import Scene
from app.slices.prediction.models.prediction import Prediction
from app.slices.prediction.services.prediction_service import PredictionService
from app.slices.prediction.ai.server import PacketModel

# ------------------------------------------------------------------
# 1️⃣  Device, model & tokenizer – loaded once at module import
# ------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# NOTE: Tune these values to match your fine-tuned checkpoint
_MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"
_CHECKPOINT = "last"  # or an absolute / relative path

model = (
    load_model(
        base_model=_MODEL_NAME,
        resume_from_checkpoint=_CHECKPOINT,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        lora_target_modules=("q_proj", "v_proj"),
        load_in_8bit=False,
    )
    .eval()
    .to(DEVICE)
)

tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)

# ------------------------------------------------------------------
# 2️⃣  Column orders derived from Enum classes (lower-case names)
# ------------------------------------------------------------------
_EGO_ORDER: List[str] = [f.name.lower() for f in EgoField]
_VEHICLE_ORDER: List[str] = [f.name.lower() for f in VehicleField]
_PEDESTRIAN_ORDER: List[str] = [f.name.lower() for f in PedestrianField]
_ROUTE_ORDER: List[str] = [f.name.lower() for f in RouteField]


# ------------------------------------------------------------------
# 3️⃣  Helper utilities
# ------------------------------------------------------------------

def _row(src: dict | list[float], order: list[str]) -> list[float]:
    """Convert an individual descriptor (dict or ordered list) into a flat list.

    • Dict – missing keys are filled with 0.0
    • List – assumed already in correct order, padded / trimmed to length
    """
    # Unwrap singleton list e.g. [ { … } ] → { … }
    if isinstance(src, list) and len(src) == 1 and isinstance(src[0], dict):
        src = src[0]

    if isinstance(src, dict):
        return [float(src.get(k, 0.0)) for k in order]

    # Already a list / tuple of values
    return (list(map(float, src)) + [0.0] * len(order))[: len(order)]


# def _pad(rows: list[list[float]], want: int, width: int) -> list[list[float]]:
#     """Pad / trim *rows* to *want* rows, each of length *width*"""
#     return rows[:want] + [[0.0] * width] * (want - len(rows))

def _pad(rows: list[list[float]], want: int, width: int):
    rows = rows[:want] + [[0.0]*width]*(want-len(rows))
    return rows
# ------------------------------------------------------------------
# 4️⃣  Conversion Scene ➔ VectorObservation
# ------------------------------------------------------------------


def _scene_to_vector_obs(packet: PacketModel) -> VectorObservation:
    """Transform a pydantic *Scene* into VectorObservation (batched)."""

    # ── Ego (1 × 31) ────────────────────────────────────────────────
    ego = torch.tensor(
        _pad([packet.ego_vehicle_descriptor], 1, len(_EGO_ORDER)),
        dtype=torch.float32, device=DEVICE
    )

    veh = torch.tensor(
        _pad(packet.vehicle_descriptors, 30, len(_VEHICLE_ORDER)),
        dtype=torch.float32, device=DEVICE
    )

    ped = torch.tensor(
        _pad(packet.pedestrian_descriptors, 20, len(_PEDESTRIAN_ORDER)),
        dtype=torch.float32, device=DEVICE
    )

    rte = torch.tensor(
        _pad(packet.route_descriptors, 30, len(_ROUTE_ORDER)),
        dtype=torch.float32, device=DEVICE
    )

    # ── Wrap into VectorObservation (adds batch dim) ───────────────
    return VectorObservation(
        route_descriptors=rte.unsqueeze(0),  # (1,30,17)
        vehicle_descriptors=veh.unsqueeze(0),  # (1,30,33)
        pedestrian_descriptors=ped.unsqueeze(0),  # (1,20,9)
        ego_vehicle_descriptor=ego.unsqueeze(0),  # (1,1,31)
    )


# ------------------------------------------------------------------
# 5️⃣  Service implementation
# ------------------------------------------------------------------

class AiPredictionService(PredictionService):
    """Prediction service that invokes the fine-tuned Vector-LM model."""

    def predict(self, packet: PacketModel) -> Prediction:  # type: ignore[override]
        start = time.time()

        obs = _scene_to_vector_obs(packet)

        # System prompt
        instruction = (
            "You are a certified professional driving instructor. "
            "Based only on the sensor inputs, first describe the scene, "
            "then recommend an action."
        )
        header = (
            "### Instruction:\n"
            f"{instruction}\n"
            "### Input:\n"
            "<VECTOR>\n"
            "### Response:\n"
        )

        enc = tokenizer(header, add_special_tokens=False, return_tensors="pt").to(DEVICE)
        print(enc)
        print(obs)


        # Generation with proper context and tensor dimensions
        with torch.no_grad():
            gen_ids = model.generate(
                user_input_ids=enc.input_ids,
                user_attention_mask=enc.attention_mask,
                route_descriptors=obs.route_descriptors,
                vehicle_descriptors=obs.vehicle_descriptors,
                pedestrian_descriptors=obs.pedestrian_descriptors,
                ego_vehicle_descriptor=obs.ego_vehicle_descriptor,
                max_length=600,
            )
        output_ids = gen_ids[0][enc.input_ids.shape[1]:]  # slice out only the new tokens
        answer: str = tokenizer.decode(output_ids, skip_special_tokens=True)

        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")

        print(gen_ids)
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")

        print("************************************************")
        print("************************************************")
        print("************************************************")
        print("************************************************")
        print("************************************************")
        print("************************************************")
        print("************************************************")
        print(answer)
        print("************************************************")
        print("************************************************")
        print("************************************************")
        print("************************************************")
        print("************************************************")
        print("************************************************")
        elapsed = time.time() - start

        actions = parse_actions(answer)
        print("///////////////////////////////////////////////")
        print("///////////////////////////////////////////////")
        print("///////////////////////////////////////////////")
        print("///////////////////////////////////////////////")
        print(actions)
        print("///////////////////////////////////////////////")
        print("///////////////////////////////////////////////")
        print("///////////////////////////////////////////////")
        print("///////////////////////////////////////////////")
        print("///////////////////////////////////////////////")
        predict = Prediction(
            caption=answer.strip(),
            accelerate=actions.get("accelerator_percent", 0.0),
            brake=actions.get("brake_percent", 0.0),
            steering=actions.get("steer_percent", 0.0),
            time_taken=elapsed
        ) 
        print("2222222222222222222222222222")
        print("2222222222222222222222222222")
        print("2222222222222222222222222222")
        print("2222222222222222222222222222")
        print(predict)
        print("2222222222222222222222222222")
        print("2222222222222222222222222222")
        print("2222222222222222222222222222")
        print("2222222222222222222222222222")
            
        return predict