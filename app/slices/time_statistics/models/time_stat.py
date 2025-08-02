from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Dict, Optional


class TimeStatistic(BaseModel):
    """Represents a time statistic entry from a prediction log."""

    timestamp: str = Field(..., description="The timestamp of the prediction")
    time_taken: float = Field(
        ..., description="The time taken to process the prediction"
    )


class TimeStatisticsResponse(BaseModel):
    """Response model for time statistics queries."""

    stats: List[TimeStatistic] = Field(
        default_factory=list, description="List of time statistics"
    )
    count: int = Field(..., description="Total count of records returned")
    from_timestamp: str = Field(None, description="Start timestamp of the query range")
    to_timestamp: str = Field(None, description="End timestamp of the query range")


class PieChartSegment(BaseModel):
    """Represents a segment in a pie chart for time statistics."""
    
    label: str = Field(..., description="Label for the segment (e.g., '0-0.5s')")
    count: int = Field(..., description="Number of records in this segment")
    min_value: float = Field(..., description="Minimum value for this segment")
    max_value: Optional[float] = Field(None, description="Maximum value for this segment (None for open-ended segments)")


class PieChartResponse(BaseModel):
    """Response model for pie chart data."""
    
    segments: List[PieChartSegment] = Field(..., description="Segments for the pie chart")
    total_count: int = Field(..., description="Total number of records")
    segment_size: float = Field(..., description="Size of each segment")
    from_timestamp: Optional[str] = Field(None, description="Start timestamp of the query range")
    to_timestamp: Optional[str] = Field(None, description="End timestamp of the query range")
