from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
import cv2
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import os

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Local MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client['ems']  # Database name
collection = db['violations']  # Collection name

# Image upload folder
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# OCR Setup
ocr = PaddleOCR(use_angle_cls=True, lang='en')

@app.get("/")
def read_root():
    return {"message": "OCR API Ready"}

@app.post("/ocr-image")
async def ocr_image(file: UploadFile = File(...)):
    contents = await file.read()
    filename = f"{datetime.utcnow().timestamp()}_{file.filename}"
    image_path = os.path.join(UPLOAD_DIR, filename)
    with open(image_path, "wb") as f:
        f.write(contents)

    # Decode image
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run OCR
    results = ocr.ocr(img_rgb, rec=True)
    text_detected = ' '.join([res[1][0] for res in results[0]]) if results and results[0] else "NOT FOUND"

    now = datetime.now()
    violation = {
        "numberplate": text_detected.strip(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "frame_id": 0,  # You can update this based on your use case
        "class_name": "vehicle",  # Fixed value unless you want to detect class
        "image_path": image_path,
        "created_at": now
    }
    collection.insert_one(violation)

    return JSONResponse(content={
        "message": "Violation saved",
        "text_detected": text_detected,
        "image_path": image_path
    })

@app.get("/violations")
def get_violations():
    violations = list(collection.find({}, {'_id': 0}))  # Hide _id
    return violations
