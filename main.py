import os
import numpy as np
import librosa
import tensorflow as tf
from tqdm import tqdm

DATA_ROOT = "dataset"
HS_DIR = os.path.join(DATA_ROOT, "HS")
LS_DIR = os.path.join(DATA_ROOT, "LS")

X = []
y = []

def extract_spectrogram(file_path):
    try:
        y_audio, sr = librosa.load(file_path, duration=5, sr=22050)
        mel = librosa.feature.melspectrogram(y=y_audio, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = np.resize(mel_db, (128, 128))
        return mel_db
    except:
        return None

print("🔹 Processing Heart Sounds...")
for f in tqdm(os.listdir(HS_DIR)):
    if f.endswith(".wav"):
        label = 0 if "abnormal" in f.lower() else 1
        spec = extract_spectrogram(os.path.join(HS_DIR, f))
        if spec is not None:
            X.append(spec)
            y.append(label)

print("🔹 Processing Lung Sounds...")
for f in tqdm(os.listdir(LS_DIR)):
    if f.endswith(".wav"):
        label = 0 if "abnormal" in f.lower() else 1
        spec = extract_spectrogram(os.path.join(LS_DIR, f))
        if spec is not None:
            X.append(spec)
            y.append(label)

X = np.array(X).reshape(-1, 128, 128, 1)
y = np.array(y)

print("✅ Total samples:", len(X))

# CNN MODEL
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=10, batch_size=16)

model.save("cnn_model.h5")

print("✅ CNN model saved as cnn_model.h5")
