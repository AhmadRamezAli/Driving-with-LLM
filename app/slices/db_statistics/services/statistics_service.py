from datetime import datetime
from typing import Dict, List, Any
import pymongo
from collections import Counter
import re
import logging
from ..models.statistics import (
    SystemHealthStats,
    ModelPerformanceStats,
    SceneCompositionStats,
    EgoActionCorrelationStats,
    DatabaseStatistics,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticsService:
    def __init__(self, db_client):
        self.db_client = db_client

    async def get_database_statistics(
        self, from_time: datetime, to_time: datetime
    ) -> DatabaseStatistics:
        """
        Get comprehensive statistics about the database between the specified time range
        """
        # Create filter for time range
        time_filter = {}
        if from_time and to_time:
            # Use direct datetime objects for MongoDB's ISODate compatibility
            # instead of string format
            time_filter = {
                "timestamp": {"$gte": from_time.isoformat(), "$lte": to_time.isoformat()}
            }
            logger.info(f"Using time filter: {time_filter}")

        # Get system health statistics
        system_health = await self._get_system_health_stats(time_filter)

        # Get model performance statistics
        model_performance = await self._get_model_performance_stats(time_filter)

        # Get scene composition statistics
        scene_composition = await self._get_scene_composition_stats(time_filter)

        # Get ego-vehicle and action correlation statistics
        ego_action_correlation = await self._get_ego_action_correlation_stats(
            time_filter
        )

        return DatabaseStatistics(
            system_health=system_health,
            model_performance=model_performance,
            scene_composition=scene_composition,
            ego_action_correlation=ego_action_correlation,
            time_range={"from_time": from_time, "to_time": to_time},
        )

    async def _get_system_health_stats(self, time_filter: Dict) -> SystemHealthStats:
        """Get system health statistics"""
        collection = self.db_client.get_prediction_logs_collection()

        # Debug: Check if we can get any data from the collection
        total_docs = await collection.count_documents({})
        logger.info(f"Total documents in collection (no filter): {total_docs}")

        # Count total entries with time filter
        total_entries = await collection.count_documents(time_filter)
        logger.info(f"Total entries with time filter: {total_entries}")

        # Get collection size in MB
        db_stats = await self.db_client.get_database_stats()
        database_size_mb = db_stats.get("size", 0) / (1024 * 1024)
        logger.info(f"Database size in MB: {database_size_mb}")

        # Get entries over time (aggregated by day)
        pipeline = [
            {"$match": time_filter},
            {
                "$project": {
                    "date": {"$substr": ["$timestamp", 0, 10]}  # Extract YYYY-MM-DD
                }
            },
            {"$group": {"_id": "$date", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
        ]

        entries_over_time = {}
        async for doc in collection.aggregate(pipeline):
            entries_over_time[doc["_id"]] = doc["count"]

        logger.info(f"Entries over time: {entries_over_time}")

        return SystemHealthStats(
            total_entries=total_entries,
            database_size_mb=database_size_mb,
            entries_over_time=entries_over_time,
        )

    async def _get_model_performance_stats(
        self, time_filter: Dict
    ) -> ModelPerformanceStats:
        """Get model performance statistics"""
        collection = self.db_client.get_prediction_logs_collection()

        # Get time taken distribution
        time_taken_ranges = {
            "0-50ms": {"$lte": 50},
            "51-100ms": {"$gt": 50, "$lte": 100},
            "101-200ms": {"$gt": 100, "$lte": 200},
            "201-500ms": {"$gt": 200, "$lte": 500},
            "500ms+": {"$gt": 500},
        }

        time_taken_distribution = {}
        for range_name, range_filter in time_taken_ranges.items():
            filter_query = time_filter.copy()

            if isinstance(range_filter, dict):
                for op, val in range_filter.items():
                    if "time_taken" not in filter_query:
                        filter_query["time_taken"] = {}
                    filter_query["time_taken"][op] = val
            else:
                filter_query["time_taken"] = range_filter

            count = await collection.count_documents(filter_query)
            time_taken_distribution[range_name] = count

        # Get accelerate, brake, and steering distributions
        action_stats = {}
        for action in ["accelerate", "brake", "steering"]:
            pipeline = [
                {"$match": time_filter},
                {
                    "$group": {
                        "_id": None,
                        "min": {"$min": f"${action}"},
                        "max": {"$max": f"${action}"},
                        "avg": {"$avg": f"${action}"},
                        "stdDev": {"$stdDevPop": f"${action}"},
                    }
                },
            ]

            stats = await collection.aggregate(pipeline).to_list(1)
            if stats and stats[0] and None not in stats[0]:
                action_stats[action] = {
                    "min": stats[0].get("min", 0),
                    "max": stats[0].get("max", 0),
                    "avg": stats[0].get("avg", 0),
                    "stdDev": stats[0].get("stdDev", 0),
                }
            else:
                action_stats[action] = {"min": 0, "max": 0, "avg": 0, "stdDev": 0}

        # Get common words from captions
        caption_docs = await collection.find(
            {**time_filter, "caption": {"$exists": True, "$ne": ""}},
            {"caption": 1}
        ).to_list(1000)  # Limit to 1000 for performance

        all_words = []
        for doc in caption_docs:
            if doc.get("caption"):
                words = re.findall(r"\b\w+\b", doc["caption"].lower())
                all_words.extend(words)

        word_counts = Counter(all_words).most_common(50)  # Get top 50 words
        common_caption_words = [
            {"word": word, "count": count} for word, count in word_counts
        ]

        return ModelPerformanceStats(
            time_taken_distribution=time_taken_distribution,
            accelerate_distribution=action_stats.get("accelerate", {"min": 0, "max": 0, "avg": 0, "stdDev": 0}),
            brake_distribution=action_stats.get("brake", {"min": 0, "max": 0, "avg": 0, "stdDev": 0}),
            steering_distribution=action_stats.get("steering", {"min": 0, "max": 0, "avg": 0, "stdDev": 0}),
            common_caption_words=common_caption_words,
        )

    async def _get_scene_composition_stats(
        self, time_filter: Dict
    ) -> SceneCompositionStats:
        """Get scene composition statistics"""
        collection = self.db_client.get_prediction_logs_collection()

        # Get situation frequencies - updated to match actual data structure
        pipeline_situations = [
            {"$match": time_filter},
            {"$unwind": {"path": "$request.situation", "preserveNullAndEmptyArrays": True}},
            {"$group": {"_id": "$request.situation.collection", "count": {"$sum": 1}}},
        ]

        situation_frequencies = {}
        async for doc in collection.aggregate(pipeline_situations):
            if doc["_id"]:
                situation_frequencies[doc["_id"]] = doc["count"]

        # Get vehicle and pedestrian distributions
        vehicles_distribution = {}
        pedestrians_distribution = {}

        # Check if any documents have the expected structure
        sample_doc = await collection.find_one(time_filter)
        logger.info(f"Sample document structure: {sample_doc}")

        for entity, distribution in [
            ("vehicles", vehicles_distribution),
            ("pedestrians", pedestrians_distribution),
        ]:
            ranges = {
                "0": 0,
                "1-3": {"$gt": 0, "$lte": 3},
                "4-8": {"$gt": 3, "$lte": 8},
                "9-15": {"$gt": 8, "$lte": 15},
                "16+": {"$gt": 15},
            }

            for range_name, range_filter in ranges.items():
                filter_query = time_filter.copy()
                count_field = f"request.{entity}"

                # Check if we need to use $size operator or just field presence
                if range_name == "0":
                    filter_query["$or"] = [
                        {count_field: {"$exists": False}},
                        {count_field: {"$size": 0}}
                    ]
                else:
                    if isinstance(range_filter, dict):
                        if "$or" in filter_query:
                            del filter_query["$or"]
                            
                        for op, val in range_filter.items():
                            filter_query[f"{count_field}.{op}"] = val
                    else:
                        filter_query[count_field] = range_filter

                count = await collection.count_documents(filter_query)
                distribution[range_name] = count

        # Get road features frequency with adjusted query paths based on actual structure
        road_features = ["has_tl", "has_junction", "is_roundabout"]
        road_features_frequency = {}

        for feature in road_features:
            # Try both direct field and nested structure
            filter_query = {
                "$or": [
                    {**time_filter, f"request.routes.{feature}": True},
                    {**time_filter, f"request.{feature}": True}
                ]
            }
            count = await collection.count_documents(filter_query)
            road_features_frequency[feature] = count

        return SceneCompositionStats(
            situation_frequencies=situation_frequencies,
            vehicles_distribution=vehicles_distribution,
            pedestrians_distribution=pedestrians_distribution,
            road_features_frequency=road_features_frequency,
        )

    async def _get_ego_action_correlation_stats(
        self, time_filter: Dict
    ) -> EgoActionCorrelationStats:
        """Get ego-vehicle and action correlation statistics"""
        collection = self.db_client.get_prediction_logs_collection()

        # Get ego speed distribution with adjusted query paths
        speed_ranges = {
            "0-10 km/h": {"$lte": 10},
            "11-30 km/h": {"$gt": 10, "$lte": 30},
            "31-50 km/h": {"$gt": 30, "$lte": 50},
            "51-80 km/h": {"$gt": 50, "$lte": 80},
            "81+ km/h": {"$gt": 80},
        }

        speed_distribution = {}
        for range_name, range_filter in speed_ranges.items():
            filter_query = time_filter.copy()

            # Try different field paths
            if isinstance(range_filter, dict):
                filter_query["$or"] = []
                for field_path in ["request.ego.speed", "request.speed"]:
                    condition = {}
                    for op, val in range_filter.items():
                        condition[f"{field_path}{op}"] = val
                    filter_query["$or"].append(condition)
            else:
                filter_query["$or"] = [
                    {"request.ego.speed": range_filter},
                    {"request.speed": range_filter}
                ]

            count = await collection.count_documents(filter_query)
            speed_distribution[range_name] = count

        # Get speed vs. accelerate and speed vs. steering correlations
        # Use more flexible query to match actual document structure
        pipeline_speed_actions = [
            {
                "$match": {
                    **time_filter, 
                    "accelerate": {"$exists": True}, 
                    "steering": {"$exists": True},
                    "$or": [
                        {"request.ego.speed": {"$exists": True}},
                        {"request.speed": {"$exists": True}}
                    ]
                }
            },
            {
                "$project": {
                    "speed": {
                        "$ifNull": ["$request.ego.speed", "$request.speed"]
                    },
                    "accelerate": "$accelerate",
                    "steering": "$steering",
                }
            },
        ]

        speed_vs_accelerate = []
        speed_vs_steering = []

        async for doc in collection.aggregate(pipeline_speed_actions):
            if "speed" in doc and "accelerate" in doc:
                speed_vs_accelerate.append(
                    {"speed": doc["speed"], "accelerate": doc["accelerate"]}
                )

            if "speed" in doc and "steering" in doc:
                speed_vs_steering.append(
                    {"speed": doc["speed"], "steering": doc["steering"]}
                )

        # Get traffic light stop vs. brake correlation with adjusted query paths
        pipeline_tl_brake = [
            {
                "$match": {
                    **time_filter, 
                    "brake": {"$exists": True},
                    "$or": [
                        {"request.routes.tl_stop": {"$exists": True}},
                        {"request.tl_stop": {"$exists": True}}
                    ]
                }
            },
            {
                "$project": {
                    "has_red_light": {
                        "$cond": [
                            {
                                "$or": [
                                    {"$gt": [{"$ifNull": ["$request.routes.tl_stop", 0]}, 0]},
                                    {"$gt": [{"$ifNull": ["$request.tl_stop", 0]}, 0]}
                                ]
                            },
                            True,
                            False
                        ]
                    },
                    "brake": "$brake",
                }
            },
            {
                "$group": {
                    "_id": "$has_red_light",
                    "avg_brake": {"$avg": "$brake"},
                    "min_brake": {"$min": "$brake"},
                    "max_brake": {"$max": "$brake"},
                    "count": {"$sum": 1},
                }
            },
        ]

        tl_stop_vs_brake = {
            "true": {"avg": 0, "min": 0, "max": 0, "count": 0},
            "false": {"avg": 0, "min": 0, "max": 0, "count": 0},
        }

        try:
            async for doc in collection.aggregate(pipeline_tl_brake):
                key = "true" if doc["_id"] else "false"
                tl_stop_vs_brake[key] = {
                    "avg": doc.get("avg_brake", 0),
                    "min": doc.get("min_brake", 0),
                    "max": doc.get("max_brake", 0),
                    "count": doc.get("count", 0),
                }
        except Exception as e:
            logger.error(f"Error in tl_stop_vs_brake aggregation: {e}")

        return EgoActionCorrelationStats(
            speed_distribution=speed_distribution,
            speed_vs_accelerate=speed_vs_accelerate[:100],  # Limit to 100 points for frontend
            speed_vs_steering=speed_vs_steering[:100],  # Limit to 100 points for frontend
            tl_stop_vs_brake=tl_stop_vs_brake,
        )
