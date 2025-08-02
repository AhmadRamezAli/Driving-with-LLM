from app.slices.tracking_statistics.db.database import collection
from app.slices.tracking_statistics.models.tracking_statistics import TrackingStatistics
from typing import List, Optional
from datetime import datetime


class TrackingService:
    @staticmethod
    def get_statistics_between_timestamps(
        start_timestamp: str, end_timestamp: str
    ) -> List[TrackingStatistics]:
        """
        Fetch statistics data from prediction_log between two timestamps

        Args:
            start_timestamp: Start timestamp in ISO format
            end_timestamp: End timestamp in ISO format

        Returns:
            List of TrackingStatistics objects
        """
        # Create query to filter by timestamp range
        query = {"timestamp": {"$gte": start_timestamp, "$lte": end_timestamp}}

        # Get records from database
        cursor = collection.find(query).sort("timestamp", 1)

        # Convert database records to TrackingStatistics objects
        statistics = []
        for record in cursor:
            # Extract scene data from the request field
            request_data = record.get("request", {})

            # Count pedestrians and vehicles from objects in the scene
            num_pedestrians = sum(
                1 for obj in request_data.get("pedestrians", [])
            )
            num_vehicles = sum(
                1 for obj in request_data.get("vehicles", [])
            )

            # Extract metrics from ego_vehicle in the request
            ego_vehicle = request_data.get("ego", {})
            accel = ego_vehicle.get("accel", 0.0)
            speed = ego_vehicle.get("speed", 0.0)
            brake_pressure = ego_vehicle.get("brake_pressure", 0.0)

            # Create statistics object
            stat = TrackingStatistics(
                number_of_pedestrian=num_pedestrians,
                number_of_vehicle=num_vehicles,
                accel=accel,
                speed=speed,
                brake_pressure=brake_pressure,
                timestamp=record.get("timestamp", ""),
            )

            statistics.append(stat)

        return statistics
