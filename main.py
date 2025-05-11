from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
import cv2
import numpy as np
from pymongo import MongoClient
from datetime import datetime

# Initialize FastAPI app
app = FastAPI()

# MongoDB Setup
connection_string = "mongodb+srv://lounisaya01:4jbuG89czpaEkvSw@cluster0.int5had.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(connection_string)
db = client['video_analysis_db']
collection = db['ocr_results']  # This will store OCR results

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

@app.get("/")
def read_root():
    return {"message": "OCR test ready!"}

@app.post("/ocr-image")
async def ocr_image(file: UploadFile = File(...)):
    # Read image from the uploaded file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform OCR
    results = ocr.ocr(img_rgb, rec=True)
    text = ' '.join([res[1][0] for res in results[0]]) if results and results[0] else ""

    # Save the OCR result to MongoDB
    if text:
        current_time = datetime.now()
        record = {
            "date": current_time.strftime("%Y-%m-%d"),
            "time": current_time.strftime("%H:%M:%S"),
            "text_detected": text
        }
        collection.insert_one(record)  # Insert OCR result into MongoDB

    return JSONResponse(content={"text_detected": text, "status": "OCR result saved to MongoDB!"})
