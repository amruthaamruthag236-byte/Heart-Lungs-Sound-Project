import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
import joblib

DATASET_PATH = "dataset"

X = []
y = []

def extract_features(file_path):
    y_audio, sr = librosa.load(file_path, duration=5)
    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

for label, folder in enumerate(["normal", "abnormal"]):
    path = os.path.join(DATASET_PATH, folder)
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        try:
            features = extract_features(file_path)
            X.append(features)
            y.append(label)
        except:
            continue

X = np.array(X)
y = np.array(y)

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "rf_model.pkl")

print("✅ Random Forest model saved!")
