import os
import uuid
import joblib
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   
import pandas as pd

from flask import Flask, render_template, request, jsonify
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

app = Flask(__name__)

# ✅ FIX: Use current directory (works in Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

UPLOAD_DIR = os.path.join(BASE_DIR, "static/uploads")
PLOT_DIR = os.path.join(BASE_DIR, "static/plots")
REPORT_DIR = os.path.join(BASE_DIR, "static/reports")
ANALYTICS_DIR = os.path.join(BASE_DIR, "static/analytics")

# create folders
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(ANALYTICS_DIR, exist_ok=True)

# load model
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model loaded")
except Exception as e:
    print("⚠️ Model load error:", e)
    model = None
    scaler = None


def extract_features(file_path, sound_type):
    y, sr = librosa.load(file_path, duration=5, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features = np.hstack([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
    features = np.append(features, 0 if sound_type == "Heart" else 1)
    return features, y, sr, mfcc


@app.route("/")
def home():
    return render_template("index.html")   # MUST be inside templates/


@app.route("/predict", methods=["POST"])
def predict():
    if "audioFile" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["audioFile"]
    sound_type = request.form.get("soundType", "Heart")

    filename = str(uuid.uuid4()) + ".wav"
    file_path = os.path.join(UPLOAD_DIR, filename)
    file.save(file_path)

    features, y, sr, mfcc = extract_features(file_path, sound_type)

    result = "Model not available"
    if model and scaler:
        features_scaled = scaler.transform([features])
        pred = model.predict(features_scaled)[0]
        result = "Normal ❤️" if pred == 1 else "Abnormal ⚠️"

    # plots
    wf_name = str(uuid.uuid4()) + ".png"
    mfcc_name = str(uuid.uuid4()) + ".png"

    wf_path = os.path.join(PLOT_DIR, wf_name)
    mfcc_path = os.path.join(PLOT_DIR, mfcc_name)

    plt.figure()
    librosa.display.waveshow(y, sr=sr)
    plt.savefig(wf_path)
    plt.close()

    plt.figure()
    librosa.display.specshow(mfcc, sr=sr)
    plt.savefig(mfcc_path)
    plt.close()

    return jsonify({
        "result": result,
        "waveform_url": "/static/plots/" + wf_name,
        "mfcc_url": "/static/plots/" + mfcc_name
    })


@app.route("/analytics")
def analytics():
    return render_template("analytics.html")


# ✅ IMPORTANT for Render
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
