# Task 2: Emotion Recognition from Speech
# Install: pip install librosa scikit-learn tensorflow keras numpy pandas matplotlib seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical

# ── 1. Feature Extraction (MFCCs) ────────────────────────────────
def extract_mfcc(file_path, n_mfcc=40):
    """Extract MFCC features from audio file"""
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# ── 2. Simulate Dataset (replace with RAVDESS/TESS/EMO-DB) ───────
# In real project: loop through audio files and extract MFCCs
# Emotions: happy, angry, sad, neutral, fearful, disgusted, surprised
np.random.seed(42)
n_samples = 500
n_features = 40  # MFCC features
emotions = ['happy', 'angry', 'sad', 'neutral', 'fearful']

X = np.random.randn(n_samples, n_features)
y_labels = np.random.choice(emotions, n_samples)

print("Dataset shape:", X.shape)
print("Emotion distribution:")
print(pd.Series(y_labels).value_counts())

# ── 3. Encode Labels ─────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y_labels)
y_categorical = to_categorical(y_encoded)
n_classes = len(emotions)

# ── 4. Train/Test Split + Scaling ────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Reshape for CNN/LSTM input
X_train_rs = X_train_sc.reshape(X_train_sc.shape[0], X_train_sc.shape[1], 1)
X_test_rs  = X_test_sc.reshape(X_test_sc.shape[0], X_test_sc.shape[1], 1)

# ── 5. Build CNN Model ───────────────────────────────────────────
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(n_features, 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── 6. Train Model ───────────────────────────────────────────────
history = model.fit(
    X_train_rs, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ── 7. Evaluate Model ────────────────────────────────────────────
loss, accuracy = model.evaluate(X_test_rs, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test Loss: {loss:.4f}")

y_pred = model.predict(X_test_rs)
y_pred_labels = le.inverse_transform(np.argmax(y_pred, axis=1))
y_test_labels = le.inverse_transform(np.argmax(y_test, axis=1))

print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred_labels))

# ── 8. Plot Training History ─────────────────────────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.show()

# ── 9. Confusion Matrix ──────────────────────────────────────────
cm = confusion_matrix(y_test_labels, y_pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=emotions, yticklabels=emotions)
plt.title('Confusion Matrix — Emotion Recognition')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

print("\nDone! Model trained for Emotion Recognition from Speech.")