from fastapi import APIRouter, Query, HTTPException
from app.slices.tracking_statistics.services.tracking_service import TrackingService
from app.slices.tracking_statistics.models.tracking_statistics import TrackingStatisticsResponse
from typing import Optional
from fastapi.responses import JSONResponse

router = APIRouter(tags=["Tracking Statistics"])
tracking_service = TrackingService()

@router.get("/statistics", response_model=TrackingStatisticsResponse)
async def get_tracking_statistics(
    start_timestamp: str = Query(..., description="Start timestamp in ISO format (YYYY-MM-DDTHH:MM:SS.sssZ)"),
    end_timestamp: str = Query(..., description="End timestamp in ISO format (YYYY-MM-DDTHH:MM:SS.sssZ)")
):
    """
    Retrieve tracking statistics between two timestamps.
    Returns a list of objects containing:
    - number of Pedestrians
    - number of Vehicles
    - acceleration
    - speed
    - brake pressure
    
    Data is retrieved from the prediction_log collection and ordered by timestamp.
    """
    try:
        # Get statistics from service
        statistics = tracking_service.get_statistics_between_timestamps(start_timestamp, end_timestamp)
        
        # Create response
        response = TrackingStatisticsResponse(
            statistics=statistics,
            count=len(statistics)
        )
        
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving tracking statistics: {str(e)}"
        ) 