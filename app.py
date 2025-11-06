import io
import json
import random
import time
from datetime import datetime
from typing import Tuple

from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
from PIL import Image
import cv2
import os

# Optional: load Keras model if exists
USE_MODEL = False
MODEL = None
MODEL_INPUT = (48, 48)  # default expected input (grayscale 48x48)
MODEL_PATH = "model.h5"

try:
    if os.path.exists(MODEL_PATH):
        from tensorflow.keras.models import load_model
        MODEL = load_model(MODEL_PATH)
        USE_MODEL = True
        print("Loaded model:", MODEL_PATH)
except Exception as e:
    print("Model load failed:", e)

app = Flask(__name__, static_folder="static", template_folder="templates")

# Haar cascade shipped with opencv-python — use it for face detection.
# If this fails on some systems, provide a path to haarcascade_frontalface_default.xml
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

EMOTIONS = ["neutral", "happy", "sad", "stressed"]

def predict_emotion_from_face(face_img: np.ndarray) -> Tuple[str, float]:
    """
    face_img: cropped face as numpy array (BGR)
    returns (label, confidence)
    """
    if USE_MODEL and MODEL is not None:
        # Example pipeline for a grayscale 48x48 model:
        # convert -> resize -> normalize -> model.predict
        img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, MODEL_INPUT)
        x = img_resized.astype("float32") / 255.0
        x = x.reshape((1, MODEL_INPUT[0], MODEL_INPUT[1], 1))
        preds = MODEL.predict(x)
        idx = int(np.argmax(preds))
        confidence = float(np.max(preds))
        return EMOTIONS[idx % len(EMOTIONS)], confidence
    else:
        # Dummy fallback: random label biased by simple heuristic (mouth openness)
        h, w = face_img.shape[:2]
        mouth_region = face_img[int(h*0.6):h, int(w*0.2):int(w*0.8)]
        mouth_gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
        mouth_mean = mouth_gray.mean() / 255.0
        # heuristic: brighter mouth -> maybe smiling/happy
        if mouth_mean > 0.6:
            label = "happy"
            confidence = 0.7 + 0.3 * random.random()
        else:
            label = random.choice(EMOTIONS)
            confidence = 0.4 + 0.6 * random.random()
        return label, float(confidence)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects form-data: 'frame' = image/jpeg bytes
    Returns JSON: {label, confidence, timestamp}
    """
    if "frame" not in request.files:
        return jsonify({"error": "no frame"}), 400
    file = request.files["frame"]
    image_bytes = file.read()
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(pil)[:, :, ::-1]  # RGB to BGR for OpenCV

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))
    if len(faces) == 0:
        # no face
        return jsonify({"label": "no_face", "confidence": 0.0, "timestamp": time.time()})

    # choose the largest face
    x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
    pad = int(0.15 * w)
    x1, y1 = max(0, x-pad), max(0, y-pad)
    x2, y2 = min(img.shape[1], x+w+pad), min(img.shape[0], y+h+pad)
    face = img[y1:y2, x1:x2]

    label, conf = predict_emotion_from_face(face)
    # simple logging (append to file)
    with open("history.log", "a") as f:
        f.write(json.dumps({
            "t": datetime.utcnow().isoformat(),
            "label": label,
            "conf": conf
        }) + "\n")

    return jsonify({"label": label, "confidence": float(conf), "timestamp": time.time()})

# static and templates served automatically by Flask. Add a small health endpoint
@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": USE_MODEL})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
