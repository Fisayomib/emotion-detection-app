# app.py
from pathlib import Path
from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import tensorflow as tf
import csv, json
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))

import csv
import json
import os
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "emotion_model.h5"
LABELS_PATH = BASE_DIR / "labels.json"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
DB_PATH = BASE_DIR / "database.csv"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Load model + labels
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH) as f:
    LABELS = json.load(f)

IMG_SIZE = (48, 48)

app = Flask(__name__)

def log_result(name, filename, emotion):
    header = ["Name", "Image", "Emotion", "Date"]
    row = [name, filename, emotion, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    file_exists = DB_PATH.exists()
    with open(DB_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)

def predict_emotion(image_path):
    img = Image.open(image_path).convert("L").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))  # (1, 48, 48, 1)
    preds = model.predict(arr)
    idx = int(np.argmax(preds, axis=1)[0])
    return LABELS[idx]

@app.route("/", methods=["GET", "POST"])
def index():
    emotion = None
    img_url = None
    name = ""

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        file = request.files.get("file")
        if file and file.filename:
            save_path = UPLOAD_DIR / file.filename
            file.save(save_path)
            emotion = predict_emotion(save_path)
            img_url = f"/static/uploads/{file.filename}"
            if name:
                log_result(name, file.filename, emotion)

    return render_template("index.html", emotion=emotion, img_url=img_url)

if __name__ == "__main__":
    app.run(debug=True)
