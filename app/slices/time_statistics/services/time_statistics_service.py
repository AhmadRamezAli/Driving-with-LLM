from app.slices.time_statistics.db.database import collection
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeStatisticsService:
    """Service for retrieving time statistics from prediction logs."""

    def get_time_statistics_between(
        self,
        from_timestamp: Optional[str] = None,
        to_timestamp: Optional[str] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Get time statistics between two timestamps, ordered by time taken.

        Args:
            from_timestamp: Start timestamp (ISO format)
            to_timestamp: End timestamp (ISO format)
            limit: Maximum number of records to return
            skip: Number of records to skip

        Returns:
            List of time statistic records
        """
        try:
            # Build query
            query = {}
            if from_timestamp or to_timestamp:
                query["timestamp"] = {}

            if from_timestamp:
                query["timestamp"]["$gte"] = from_timestamp

            if to_timestamp:
                query["timestamp"]["$lte"] = to_timestamp

            # Execute query and get results
            cursor = (
                collection.find(query, {"timestamp": 1, "time_taken": 1, "_id": 0})
                .sort("timestamp", 1)
                .skip(skip)
                .limit(limit)
            )

            results = list(cursor)
            logger.info(f"Retrieved {len(results)} time statistics records")
            return results

        except Exception as e:
            logger.error(f"Error retrieving time statistics: {str(e)}")
            return []
            
    def get_pie_chart_data(
        self,
        segment_size: float,
        from_timestamp: Optional[str] = None,
        to_timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get time statistics grouped into segments for pie chart visualization.
        
        Args:
            segment_size: Size of each segment in seconds (e.g., 0.5)
            from_timestamp: Start timestamp (ISO format)
            to_timestamp: End timestamp (ISO format)
            
        Returns:
            Dictionary with pie chart segment data
        """
        try:
            # Build query
            query = {}
            if from_timestamp or to_timestamp:
                query["timestamp"] = {}
                
            if from_timestamp:
                query["timestamp"]["$gte"] = from_timestamp
                
            if to_timestamp:
                query["timestamp"]["$lte"] = to_timestamp
            
            # Get all time_taken values within the time range
            cursor = collection.find(query, {"time_taken": 1, "_id": 0})
            all_times = [doc["time_taken"] for doc in cursor]
            
            if not all_times:
                logger.warning("No time statistics found for pie chart")
                return {
                    "segments": [],
                    "total_count": 0,
                    "segment_size": segment_size,
                    "from_timestamp": from_timestamp,
                    "to_timestamp": to_timestamp
                }
            
            # Find the maximum time to determine the number of segments needed
            max_time = max(all_times)
            num_segments = math.ceil(max_time / segment_size)
            
            # Initialize segment counters
            segments = []
            for i in range(num_segments):
                min_value = i * segment_size
                max_value = (i + 1) * segment_size
                
                # For the last segment, make sure it includes the maximum value
                if i == num_segments - 1:
                    count = sum(1 for t in all_times if t >= min_value)
                    label = f"{min_value:.1f}s+"
                    segments.append({
                        "label": label,
                        "count": count,
                        "min_value": min_value,
                        "max_value": None  # Open-ended for the last segment
                    })
                else:
                    count = sum(1 for t in all_times if min_value <= t < max_value)
                    label = f"{min_value:.1f}s-{max_value:.1f}s"
                    segments.append({
                        "label": label,
                        "count": count,
                        "min_value": min_value,
                        "max_value": max_value
                    })
            
            # Remove any empty segments at the end
            segments = [s for s in segments if s["count"] > 0]
            
            logger.info(f"Created pie chart with {len(segments)} segments from {len(all_times)} records")
            return {
                "segments": segments,
                "total_count": len(all_times),
                "segment_size": segment_size,
                "from_timestamp": from_timestamp,
                "to_timestamp": to_timestamp
            }
            
        except Exception as e:
            logger.error(f"Error creating pie chart data: {str(e)}")
            return {
                "segments": [],
                "total_count": 0,
                "segment_size": segment_size,
                "from_timestamp": from_timestamp,
                "to_timestamp": to_timestamp
            }
