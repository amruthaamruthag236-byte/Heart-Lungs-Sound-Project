import os
import numpy as np
import librosa
import tensorflow as tf

DATASET_PATH = "dataset"  # your dataset folder

X = []
y = []

def extract_spectrogram(file_path):
    y_audio, sr = librosa.load(file_path, duration=5, sr=22050)
    mel = librosa.feature.melspectrogram(y=y_audio, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.resize(mel_db, (128, 128))
    return mel_db

# Load dataset
for label, category in enumerate(["normal", "abnormal"]):
    folder = os.path.join(DATASET_PATH, category)

    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            spec = extract_spectrogram(file_path)
            X.append(spec)
            y.append(label)
        except:
            continue

X = np.array(X).reshape(-1, 128, 128, 1)
y = np.array(y)

# Build CNN model
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

# Save model
model.save("cnn_model.h5")

print("✅ Model saved as cnn_model.h5")
