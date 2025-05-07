from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from datetime import datetime
from pymongo import MongoClient
import cv2
import numpy as np
import tempfile
import shutil
from ultralytics import YOLO
from paddleocr import PaddleOCR
import os

app = FastAPI()

# ---------- MongoDB Setup ----------
connection_string = "mongodb+srv://lounisaya01:4jbuG89czpaEkvSw@cluster0.int5had.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(connection_string)
db = client['video_analysis_db']
collection = db['violations']

# ---------- Load Models ----------
plate_model = YOLO("best.pt")  # Only the plate detection model
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# ---------- OCR Function ----------
def perform_ocr(image_array):
    results = ocr.ocr(image_array, rec=True)
    text = ' '.join([result[1][0] for result in results[0]] if results[0] else "")
    return text.strip()

# ---------- API Endpoint ----------
@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    # Save uploaded video temporarily
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

        # Plate Detection
        plate_results = plate_model.predict(frame, conf=0.5, verbose=False)

        for result in plate_results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                plate_crop = frame[y1:y2, x1:x2]

                if plate_crop.size == 0:
                    continue

                rgb_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                plate_number = perform_ocr(rgb_crop)

                if plate_number:
                    current_time = datetime.now()
                    record = {
                        "date": current_time.strftime("%Y-%m-%d"),
                        "time": current_time.strftime("%H:%M:%S"),
                        "frame_id": frame_id,
                        "class_name": "vehicle",
                        "numberplate": plate_number
                    }

                    collection.insert_one(record)
                    detected_plates.append({
                        "numberplate": plate_number,
                        "time": record["time"],
                        "date": record["date"]
                    })

    cap.release()
    os.remove(video_path)

    return JSONResponse(content=detected_plates)
