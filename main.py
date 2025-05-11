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

# Allow frontend CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB (local) setup
client = MongoClient("mongodb://localhost:27017/")
db = client['ems']
collection = db['violations']

# Image upload folder
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# OCR engine
ocr = PaddleOCR(use_angle_cls=True, lang='en')

@app.get("/")
def root():
    return {"message": "OCR API Connected to Local MongoDB"}

@app.post("/ocr-image")
async def ocr_image(file: UploadFile = File(...)):
    contents = await file.read()

    # Save uploaded image to disk
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

    # Current time details
    now = datetime.now()
    violation = {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "frame_id": 180,  # You can update this dynamically if needed
        "class_name": "vehicle",
        "numberplate": text_detected.strip()
    }

    # Save to MongoDB
    collection.insert_one(violation)

    return JSONResponse(content={
        "message": "Violation saved to local MongoDB",
        "text_detected": text_detected,
        "stored_data": violation
    })

@app.get("/violations")
def get_violations():
    violations = list(collection.find({}, {"_id": 0}))  # Hide MongoDB internal _id
    return violations
