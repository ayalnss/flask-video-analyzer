from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from datetime import datetime
from pymongo import MongoClient
import cv2
import numpy as np
import tempfile
import shutil
from paddleocr import PaddleOCR
import os

app = FastAPI()

# ---------- MongoDB Setup ----------
connection_string = "mongodb+srv://lounisaya01:4jbuG89czpaEkvSw@cluster0.int5had.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(connection_string)
db = client['video_analysis_db']
collection = db['violations']

# ---------- OCR Function ----------
def perform_ocr(image_array):
    ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Create OCR model
    results = ocr.ocr(image_array, rec=True)
    text = ' '.join([result[1][0] for result in results[0]] if results[0] else "")
    return text.strip()

# ---------- Root Route (for browser) ----------
@app.get("/")
def read_root():
    return {"message": "FastAPI Video Analyzer is running. Use POST /analyze-video to upload videos."}

# ---------- Video Analysis Route ----------
@app.post("/analyze-video")  # ✅ This line connects the function to the endpoint
async def analyze_video(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    detected_plates = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % 30 != 0:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        text = perform_ocr(rgb_frame)

        if text:
            current_time = datetime.now()
            record = {
                "date": current_time.strftime("%Y-%m-%d"),
                "time": current_time.strftime("%H:%M:%S"),
                "frame_id": frame_id,
                "text_detected": text
            }

            collection.insert_one(record)
            detected_plates.append({
                "text_detected": text,
                "time": record["time"],
                "date": record["date"]
            })

    cap.release()
    os.remove(video_path)

    return JSONResponse(content=detected_plates)
