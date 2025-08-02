from fastapi import APIRouter, Query
from app.slices.prediction.models.scene import Scene
from app.slices.prediction.models.prediction import Prediction
# from app.slices.prediction.services.dummy_prediction_service import DummyPredictionService
from app.slices.prediction.services.ai_prediction_service import AiPredictionService
from app.slices.prediction.services.logging_service import log_prediction
from app.slices.prediction.db.database import collection
from typing import Optional
from fastapi.responses import JSONResponse
from app.slices.prediction.ai.server import PacketModel

router = APIRouter(tags=["Prediction"])
# prediction_service = DummyPredictionService()
# LLM-powered service
ai_prediction_service = AiPredictionService()


# @router.post("/predict", response_model=Prediction)
# async def predict(scene: Scene):
#     prediction = prediction_service.predict(scene)
#     log_prediction(scene, prediction)
#     return prediction


# -----------------------------------------------------------------------------
# 🧠  New Vector-LM powered endpoint
# -----------------------------------------------------------------------------


@router.post("/predict", summary="LLM-powered prediction")
async def predict_ai(packet: PacketModel):
    """Generate a prediction using the fine-tuned Vector-LM model."""
    prediction = ai_prediction_service.predict(packet)
    log_prediction(packet, prediction)
    print("--------------------------------")
    print("--------------------------------")
    print("finish predict")
    print("--------------------------------")
    print("--------------------------------")
    return {
        "accelerator_percent": prediction.accelerate,
        "brake_percent":       prediction.brake,
        "steer_percent":       prediction.steering,
    }



@router.get("/prediction-logs")
async def get_prediction_logs(
    limit: int = Query(10, ge=1, le=100),
    skip: int = Query(0, ge=0),
    sort_by: Optional[str] = Query(None, description="Field to sort by"),
    sort_order: int = Query(
        1, ge=-1, le=1, description="Sort order: 1 for ascending, -1 for descending"
    ),
):
    """
    Retrieve prediction logs from the database.
    """
    try:
        # Define sorting
        sort_options = {}
        if sort_by:
            sort_options[sort_by] = sort_order

        # Get total count first
        total_count = collection.count_documents({})

        # Query the database
        cursor = collection.find({}, {"_id": 0})

        # Apply sorting if specified
        if sort_options:
            cursor = cursor.sort(list(sort_options.items()))

        # Apply pagination
        cursor = cursor.skip(skip).limit(limit)

        # Convert to list
        logs = list(cursor)

        return {"logs": logs, "count": total_count, "skip": skip, "limit": limit}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error retrieving prediction logs: {str(e)}"},
        )
