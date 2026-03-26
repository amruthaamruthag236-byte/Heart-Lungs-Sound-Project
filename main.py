# main.py (corrected training script)

import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings("ignore")

DATA_ROOT = r"C:\Users\Amrutha\Downloads\Heart_Lung_Sound_Project"
HS_DIR = os.path.join(DATA_ROOT, "dataset", "HS")
LS_DIR = os.path.join(DATA_ROOT, "dataset", "LS")

def extract_features(file_path, label, sound_type):
    try:
        y, sr = librosa.load(file_path, duration=5, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        features = np.hstack([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
        features = np.append(features, 0 if sound_type == "Heart" else 1)
        return features, label
    except Exception as e:
        print("Error:", file_path, e)
        return None, None

data, labels = [], []

print("\n🔹 Extracting Heart Sound features...")
if os.path.exists(HS_DIR):
    for f in tqdm(os.listdir(HS_DIR)):
        if f.lower().endswith(".wav"):
            # better filename detection for label
            lab = "normal" if any(k in f.lower() for k in ["normal", "_n_", "-n-"]) else "abnormal"
            feat, lab2 = extract_features(os.path.join(HS_DIR, f), lab, "Heart")
            if feat is not None:
                data.append(feat)
                labels.append(lab2)

print("\n🔹 Extracting Lung Sound features...")
if os.path.exists(LS_DIR):
    for f in tqdm(os.listdir(LS_DIR)):
        if f.lower().endswith(".wav"):
            lab = "normal" if any(k in f.lower() for k in ["normal", "_n_", "-n-"]) else "abnormal"
            feat, lab2 = extract_features(os.path.join(LS_DIR, f), lab, "Lung")
            if feat is not None:
                data.append(feat)
                labels.append(lab2)

if len(data) == 0:
    raise SystemExit("No audio features extracted. Check HS_DIR and LS_DIR paths and .wav files.")

df = pd.DataFrame(data)
df["label"] = labels

print("\n✅ Total extracted samples:", df.shape)

X = df.drop("label", axis=1)
y = LabelEncoder().fit_transform(df["label"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

model = RandomForestClassifier(n_estimators=400, random_state=42)
model.fit(X_train_bal, y_train_bal)

y_pred = model.predict(X_test)
print("\n🎯 Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print(classification_report(y_test, y_pred))

MODEL_PATH = os.path.join(DATA_ROOT, "model.pkl")
SCALER_PATH = os.path.join(DATA_ROOT, "scaler.pkl")

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("\n✅ Model saved to:", MODEL_PATH)
print("✅ Scaler saved to:", SCALER_PATH)
print("🚀 Training complete!")
