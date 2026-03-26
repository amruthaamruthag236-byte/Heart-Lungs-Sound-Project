import os
import uuid
import joblib
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd

from flask import Flask, render_template, request, jsonify
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

app = Flask(__name__)

BASE_DIR = r"C:\Users\Amrutha\Downloads\Heart_Lung_Sound_Project"

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
HS_DIR = os.path.join(DATASET_DIR, "HS")   
LS_DIR = os.path.join(DATASET_DIR, "LS")

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
PLOT_DIR = os.path.join(BASE_DIR, "static", "plots")
REPORT_DIR = os.path.join(BASE_DIR, "static", "reports")
ANALYTICS_DIR = os.path.join(BASE_DIR, "static", "analytics")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(ANALYTICS_DIR, exist_ok=True)


try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model & scaler loaded")
except Exception as e:
    print("⚠️ Could not load model/scaler:", e)
    model = None
    scaler = None


def extract_features(file_path, sound_type):
    """Return (features, y, sr, mfcc) for uploaded .wav."""
    y, sr = librosa.load(file_path, duration=5, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features = np.hstack([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
    
    features = np.append(features, 0 if sound_type == "Heart" else 1)
    return features, y, sr, mfcc



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "audioFile" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["audioFile"]
    sound_type = request.form.get("soundType", "Heart")

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    
    filename = f"{uuid.uuid4()}.wav"
    file_path = os.path.join(UPLOAD_DIR, filename)
    file.save(file_path)

    # extract features
    try:
        features, y, sr, mfcc = extract_features(file_path, sound_type)
    except Exception as e:
        return jsonify({"error": f"Error reading audio: {e}"}), 500

    # prediction
    prediction_label = "Model not available"
    if model is not None and scaler is not None:
        try:
            features_scaled = scaler.transform([features])
            pred_raw = model.predict(features_scaled)[0]
            prediction_label = "Normal ❤️" if pred_raw == 1 else "Abnormal ⚠️"
        except Exception as e:
            prediction_label = f"Prediction error: {e}"

    # plots
    wf_plot = f"{uuid.uuid4()}_wave.png"
    mfcc_plot = f"{uuid.uuid4()}_mfcc.png"
    wf_path = os.path.join(PLOT_DIR, wf_plot)
    mfcc_path = os.path.join(PLOT_DIR, mfcc_plot)

    plt.figure(figsize=(8, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"{sound_type} Sound - Waveform")
    plt.tight_layout()
    plt.savefig(wf_path)
    plt.close()

    plt.figure(figsize=(8, 4))
    librosa.display.specshow(mfcc, x_axis="time", sr=sr)
    plt.title(f"{sound_type} Sound - MFCC Spectrogram")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(mfcc_path)
    plt.close()

    return jsonify(
        {
            "result": f"{sound_type} Sound → {prediction_label}",
            "waveform_url": f"/static/plots/{wf_plot}",
            "mfcc_url": f"/static/plots/{mfcc_plot}",
            "prediction_raw": prediction_label,
            "sound_type": sound_type,
        }
    )


def _abs_from_rel(url):
    """Convert '/static/plots/x.png' to absolute OS path."""
    rel = url.lstrip("/") if url.startswith("/") else url
    return os.path.join(BASE_DIR, rel.replace("/", os.sep))


@app.route("/download_report", methods=["POST"])
def download_report():
    prediction = request.form.get("prediction", "N/A")
    sound_type = request.form.get("sound_type", "N/A")
    wf_rel = request.form.get("waveform", "")
    mfcc_rel = request.form.get("mfcc", "")

    wf_abs = _abs_from_rel(wf_rel) if wf_rel else None
    mfcc_abs = _abs_from_rel(mfcc_rel) if mfcc_rel else None

    report_name = f"Report_{uuid.uuid4()}.pdf"
    report_path = os.path.join(REPORT_DIR, report_name)

    c = canvas.Canvas(report_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "Heart & Lung Sound Classification Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 90, f"Sound Type: {sound_type}")
    c.drawString(50, height - 110, f"Prediction: {prediction}")

    y_pos = height - 150

    if wf_abs and os.path.exists(wf_abs):
        c.drawString(50, y_pos, "Waveform Plot:")
        y_pos -= 20
        c.drawImage(wf_abs, 50, y_pos - 250, width=350, height=250)
        y_pos -= 270

    if mfcc_abs and os.path.exists(mfcc_abs):
        c.drawString(50, y_pos, "MFCC Spectrogram:")
        y_pos -= 20
        c.drawImage(mfcc_abs, 50, y_pos - 250, width=350, height=250)
        y_pos -= 270

    c.save()
    return jsonify({"report_url": f"/static/reports/{report_name}"})



def generate_analytics():
    """Read HS.csv & LS.csv and save charts into static/analytics."""
    heart_csv = os.path.join(BASE_DIR, "HS.csv")
    lung_csv = os.path.join(BASE_DIR, "LS.csv")

    hs_df = pd.read_csv(heart_csv)
    ls_df = pd.read_csv(lung_csv)

    # HEART
    hs_df["Condition"] = hs_df["Heart Sound Type"].apply(
        lambda x: "Normal" if str(x).strip().lower() == "normal" else "Abnormal"
    )

    plt.figure()
    hs_df["Gender"].value_counts().plot.pie(autopct="%1.1f%%")
    plt.title("Heart: Gender Distribution")
    plt.savefig(os.path.join(ANALYTICS_DIR, "heart_gender.png"))
    plt.close()

    plt.figure()
    hs_df["Condition"].value_counts().plot.pie(autopct="%1.1f%%")
    plt.title("Heart: Condition Distribution")
    plt.savefig(os.path.join(ANALYTICS_DIR, "heart_condition.png"))
    plt.close()

    plt.figure(figsize=(9, 4))
    hs_df[hs_df["Condition"] == "Abnormal"]["Heart Sound Type"].value_counts().plot.bar()
    plt.title("Heart: Abnormal Types")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYTICS_DIR, "heart_abnormal.png"))
    plt.close()

    # LUNG
    ls_df["Condition"] = ls_df["Lung Sound Type"].apply(
        lambda x: "Normal" if str(x).strip().lower() == "normal" else "Abnormal"
    )

    plt.figure()
    ls_df["Gender"].value_counts().plot.pie(autopct="%1.1f%%")
    plt.title("Lung: Gender Distribution")
    plt.savefig(os.path.join(ANALYTICS_DIR, "lung_gender.png"))
    plt.close()

    plt.figure()
    ls_df["Condition"].value_counts().plot.pie(autopct="%1.1f%%")
    plt.title("Lung: Condition Distribution")
    plt.savefig(os.path.join(ANALYTICS_DIR, "lung_condition.png"))
    plt.close()

    plt.figure(figsize=(9, 4))
    ls_df[ls_df["Condition"] == "Abnormal"]["Lung Sound Type"].value_counts().plot.bar()
    plt.title("Lung: Abnormal Types")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYTICS_DIR, "lung_abnormal.png"))
    plt.close()

    # counts for template
    return len(hs_df), len(ls_df)


@app.route("/analytics")
def analytics():
    try:
        hs_total, ls_total = generate_analytics()
    except Exception as e:
        return f"Analytics generation failed: {e}", 500
    return render_template("analytics.html", hs_total=hs_total, ls_total=ls_total)


# ----------------- run -----------------
if __name__ == "__main__":
    print("Starting Flask app at http://127.0.0.1:5000")
    app.run(debug=True)
