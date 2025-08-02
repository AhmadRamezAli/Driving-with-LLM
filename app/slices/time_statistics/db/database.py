import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["prediction_db"]
collection = db["prediction_log"]  # Using the same collection as prediction slice 