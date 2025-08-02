import motor.motor_asyncio
from typing import Dict, Any


class DatabaseClient:
    """MongoDB database client for the statistics API"""
    
    def __init__(self, connection_string: str, database_name: str = "prediction_db"):
        """Initialize the database client
        
        Args:
            connection_string: MongoDB connection string
            database_name: Name of the database to connect to
        """
        self.client = motor.motor_asyncio.AsyncIOMotorClient(connection_string)
        self.db = self.client[database_name]
    
    def get_prediction_logs_collection(self):
        """Get the prediction logs collection"""
        return self.db["prediction_log"]
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the database
        
        Returns:
            Dictionary with database statistics
        """
        return await self.db.command("dbStats") 