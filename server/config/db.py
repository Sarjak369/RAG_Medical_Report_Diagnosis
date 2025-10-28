import os
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables from .env file
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "rag_medical_diagnosis")


# Create MongoDB client and connect to the database
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

users_collection = db["users"]
reports_collection = db["reports"]
diagnosis_collection = db["diagnosis_history"]
