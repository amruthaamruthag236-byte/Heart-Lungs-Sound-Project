import os
import uuid
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

from flask import Flask, render_template, request, jsonify

# ---------------- APP SETUP ----------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "cnn_model.h5")

UPLOAD_DIR = os.path.join(BASE_DIR, "static/uploads")
PLOT_DIR = os.path.join(BASE_DIR, "static/plots")
REPORT_DIR = os.path.join(BASE_DIR, "static/reports")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ---------------- LOAD MODEL ----------------
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ CNN Model loaded")
except Exception as e:
    print("❌ Model load error:", e)
    model = None


# ---------------- FEATURE EXTRACTION ----------------
def extract_spectrogram(file_path):
    y, sr = librosa.load(file_path, duration=5, sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.resize(mel_db, (128, 128))
    return mel_db.reshape(1, 128, 128, 1), y, sr


# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ✅ FIXED PREDICT
@app.route("/predict", methods=["POST"])
def predict():
    if "audioFile" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["audioFile"]

    filename = str(uuid.uuid4()) + ".wav"
    file_path = os.path.join(UPLOAD_DIR, filename)
    file.save(file_path)

    # Extract features
    spec, y, sr = extract_spectrogram(file_path)

    prediction_label = "Model not available"

    if model:
        pred = model.predict(spec)
        pred_class = np.argmax(pred)
        prediction_label = "Normal ❤️" if pred_class == 1 else "Abnormal ⚠️"

    # ---------------- PLOTS ----------------
    wf_name = str(uuid.uuid4()) + ".png"
    spec_name = str(uuid.uuid4()) + ".png"

    wf_path = os.path.join(PLOT_DIR, wf_name)
    spec_path = os.path.join(PLOT_DIR, spec_name)

    # Waveform
    plt.figure()
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.savefig(wf_path)
    plt.close()

    # Spectrogram
    plt.figure()
    librosa.display.specshow(spec[0][:, :, 0], sr=sr)
    plt.title("Mel Spectrogram")
    plt.colorbar()
    plt.savefig(spec_path)
    plt.close()

    return jsonify({
        "result": prediction_label,
        "waveform_url": "/static/plots/" + wf_name,
        "mfcc_url": "/static/plots/" + spec_name
    })


# ✅ SEPARATE REPORT ROUTE (IMPORTANT)
@app.route("/download_report", methods=["POST"])
def download_report():
    prediction = request.form.get("prediction", "N/A")

    filename = "report_" + str(uuid.uuid4()) + ".pdf"
    filepath = os.path.join(REPORT_DIR, filename)

    c = canvas.Canvas(filepath, pagesize=A4)
    c.drawString(100, 800, "Heart & Lung Sound Report")
    c.drawString(100, 750, f"Prediction: {prediction}")
    c.save()

    return jsonify({
        "report_url": "/static/reports/" + filename
    })


@app.route("/analytics")
def analytics():
    return render_template("analytics.html")


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
