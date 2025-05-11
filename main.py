from fastapi import FastAPI
from pymongo import MongoClient

app = FastAPI()

# MongoDB setup
connection_string = "mongodb+srv://lounisaya01:4jbuG89czpaEkvSw@cluster0.int5had.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(connection_string)
db = client["video_analysis_db"]
collection = db["violations"]

@app.get("/")
def read_root():
    # Test DB connection
    try:
        db.list_collection_names()  # Just a basic call
        return {"message": "MongoDB connected!"}
    except Exception as e:
        return {"error": str(e)}
