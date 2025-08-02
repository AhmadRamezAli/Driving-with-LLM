from fastapi import APIRouter, Depends, HTTPException, Query
from datetime import datetime, timedelta
import logging
from typing import Optional
from ..services import StatisticsService
from ..models import DatabaseStatistics
from ..db import DatabaseClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/db_statistics",
    tags=["db_statistics"],
    responses={404: {"description": "Not found"}},
)

def get_db_client():
    """Dependency to get database client"""
    connection_string = "mongodb://localhost:27017"  # Configure as needed
    return DatabaseClient(connection_string)

def get_statistics_service(db_client: DatabaseClient = Depends(get_db_client)):
    """Dependency to get statistics service"""
    return StatisticsService(db_client)

@router.get("/", response_model=DatabaseStatistics)
async def get_database_statistics(
    from_time: Optional[datetime] = Query(
        None,
        description="Start time for statistics (ISO format). If not provided, defaults to 7 days ago."
    ),
    to_time: Optional[datetime] = Query(
        None,
        description="End time for statistics (ISO format). If not provided, defaults to current time."
    ),
    service: StatisticsService = Depends(get_statistics_service)
):
    """
    Get comprehensive database statistics between two time points.
    
    This API returns detailed statistics about:
    - System health (entry counts, database size)
    - Model performance (prediction times, control outputs)
    - Scene composition (situation types, object counts)
    - Ego-vehicle and prediction correlations
    """
    # Set default time range if not provided
    if from_time is None:
        # Default to 7 days ago instead of 24 hours
        from_time = datetime.now() - timedelta(days=7)
        
    if to_time is None:
        # Default to current time
        to_time = datetime.now()
    
    logger.info(f"Getting statistics from {from_time} to {to_time}")
    
    try:
        statistics = await service.get_database_statistics(from_time, to_time)
        return statistics
    except Exception as e:
        logger.error(f"Failed to retrieve database statistics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve database statistics: {str(e)}"
        ) 