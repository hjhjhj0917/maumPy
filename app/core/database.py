import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "maum_db")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]

diary_logs_collection = db["DIARY_LOGS"]