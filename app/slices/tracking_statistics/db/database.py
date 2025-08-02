import pymongo

# We'll use the same collection as prediction since we're accessing prediction_log data
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["prediction_db"]
collection = db["prediction_log"] 