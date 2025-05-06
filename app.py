from flask import Flask, request, jsonify
import cv2
import numpy as np
from datetime import datetime
from paddleocr import PaddleOCR
from ultralytics import YOLO
from pymongo import MongoClient
import tempfile
import os
app = Flask(__name__)

# ---------- Connect to MongoDB ----------
client = MongoClient("mongodb+srv://lounisaya01:4jbuG89czpaEkvSw@cluster0.int5had.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client['video_analysis_db']
collection = db['violations']

# ---------- Models ----------
ocr = PaddleOCR(use_angle_cls=True, lang='en')
plate_model = YOLO('best.pt')
vehicle_model = YOLO('yolov8n.pt')

# ---------- OCR Function ----------
def perform_ocr(image_array):
    results = ocr.ocr(image_array, rec=True)
    text = ' '.join([result[1][0] for result in results[0]] if results[0] else "")
    return text.strip()

# ---------- Route to process uploaded video ----------
@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video_file = request.files['video']
    temp_video_path = tempfile.NamedTemporaryFile(delete=False).name
    video_file.save(temp_video_path)

    cap = cv2.VideoCapture(temp_video_path)
    frame_id = 0
    detected_plates = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % 30 == 0:
            vehicle_results = vehicle_model.predict(frame, conf=0.5, verbose=False)

            for result in vehicle_results:
                class_ids = result.boxes.cls.cpu().numpy()

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
                        timestamp = datetime.now()
                        document = {
                            "date": timestamp.strftime("%Y-%m-%d"),
                            "time": timestamp.strftime("%H:%M:%S"),
                            "frame_id": frame_id,
                            "numberplate": plate_number
                        }
                        collection.insert_one(document)
                        detected_plates.append(document)

    cap.release()
    return jsonify({"status": "Done", "results": detected_plates})



if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
