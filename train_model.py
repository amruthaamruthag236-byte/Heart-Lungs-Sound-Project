import os
import numpy as np
import librosa
import tensorflow as tf

# Dataset paths
HS_DIR = os.path.join("dataset", "HS")
LS_DIR = os.path.join("dataset", "LS")

X = []
y = []

# 🔊 Convert audio → spectrogram
def extract_spectrogram(file_path):
    y_audio, sr = librosa.load(file_path, duration=5, sr=22050)
    mel = librosa.feature.melspectrogram(y=y_audio, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Resize to fixed shape
    mel_db = np.resize(mel_db, (128, 128))
    return mel_db

# 🏷️ Label function
def get_label(filename):
    return 1 if "normal" in filename.lower() else 0


# ✅ Check folders exist
if not os.path.exists(HS_DIR) or not os.path.exists(LS_DIR):
    print("❌ Dataset folders not found!")
    print("Expected:")
    print("dataset/HS and dataset/LS")
    exit()


print("🔹 Loading Heart Sounds...")
for file in os.listdir(HS_DIR):
    if file.endswith(".wav"):
        file_path = os.path.join(HS_DIR, file)
        try:
            spec = extract_spectrogram(file_path)
            label = get_label(file)
            X.append(spec)
            y.append(label)
        except Exception as e:
            print("Error:", file, e)


print("🔹 Loading Lung Sounds...")
for file in os.listdir(LS_DIR):
    if file.endswith(".wav"):
        file_path = os.path.join(LS_DIR, file)
        try:
            spec = extract_spectrogram(file_path)
            label = get_label(file)
            X.append(spec)
            y.append(label)
        except Exception as e:
            print("Error:", file, e)


# Convert to numpy
X = np.array(X).reshape(-1, 128, 128, 1)
y = np.array(y)

print("✅ Total samples:", len(X))

# 🚀 Build CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X, y, epochs=10, batch_size=16)

# Save model
model.save("cnn_model.h5")

print("🎉 Model saved as cnn_model.h5")
