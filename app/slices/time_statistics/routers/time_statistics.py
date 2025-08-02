from fastapi import APIRouter, Query, HTTPException
from app.slices.time_statistics.models.time_stat import TimeStatisticsResponse, PieChartResponse
from app.slices.time_statistics.services.time_statistics_service import TimeStatisticsService
from typing import Optional
from datetime import datetime

router = APIRouter(tags=["Time Statistics"])
time_statistics_service = TimeStatisticsService()


@router.get("/time-statistics", response_model=TimeStatisticsResponse)
async def get_time_statistics(
    from_timestamp: Optional[str] = Query(
        None, description="Starting timestamp in ISO format (YYYY-MM-DDTHH:MM:SS.mmmZ)"
    ),
    to_timestamp: Optional[str] = Query(
        None, description="Ending timestamp in ISO format (YYYY-MM-DDTHH:MM:SS.mmmZ)"
    ),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
):
    """
    Retrieve time statistics from prediction logs between two timestamps, ordered by time taken.
    
    This endpoint allows you to query prediction execution times within a specified time range.
    Results are returned in ascending order by the time taken for each prediction.
    """
    # Validate timestamps if provided
    if from_timestamp:
        try:
            datetime.fromisoformat(from_timestamp.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid from_timestamp format")
            
    if to_timestamp:
        try:
            datetime.fromisoformat(to_timestamp.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid to_timestamp format")
    
    # Get statistics from service
    time_stats = time_statistics_service.get_time_statistics_between(
        from_timestamp, to_timestamp, limit, skip
    )
    
    # Format response
    response = {
        "stats": time_stats,
        "count": len(time_stats),
        "from_timestamp": from_timestamp,
        "to_timestamp": to_timestamp
    }
    
    return response


@router.get("/time-statistics/pie-chart", response_model=PieChartResponse)
async def get_time_statistics_pie_chart(
    segment_size: float = Query(
        0.5, gt=0, description="Size of each segment in seconds (e.g., 0.5 for half-second segments)"
    ),
    from_timestamp: Optional[str] = Query(
        None, description="Starting timestamp in ISO format (YYYY-MM-DDTHH:MM:SS.mmmZ)"
    ),
    to_timestamp: Optional[str] = Query(
        None, description="Ending timestamp in ISO format (YYYY-MM-DDTHH:MM:SS.mmmZ)"
    ),
):
    """
    Retrieve time statistics as pie chart data, segmented by time taken.
    
    This endpoint returns counts of predictions grouped by time taken intervals.
    For example, with segment_size=0.5, it will return counts for 0-0.5s, 0.5-1.0s, etc.
    """
    # Validate timestamps if provided
    if from_timestamp:
        try:
            datetime.fromisoformat(from_timestamp.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid from_timestamp format")
            
    if to_timestamp:
        try:
            datetime.fromisoformat(to_timestamp.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid to_timestamp format")
    
    # Validate segment size
    if segment_size <= 0:
        raise HTTPException(status_code=400, detail="segment_size must be greater than 0")
    
    # Get pie chart data from service
    pie_chart_data = time_statistics_service.get_pie_chart_data(
        segment_size, from_timestamp, to_timestamp
    )
    
    return pie_chart_data 