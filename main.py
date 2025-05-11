from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
import cv2
import numpy as np

app = FastAPI()

ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Load once at startup

@app.get("/")
def read_root():
    return {"message": "OCR test ready!"}

@app.post("/ocr-image")
async def ocr_image(file: UploadFile = File(...)):
    # Read the uploaded image into OpenCV format
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run OCR
    results = ocr.ocr(img_rgb, rec=True)
    text = ' '.join([res[1][0] for res in results[0]]) if results and results[0] else ""

    return JSONResponse(content={"text_detected": text})
