import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from preprocess import full_pipeline

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("=" * 50)
print("  LSTM Model Training")
print("=" * 50)

X_train, X_test, y_train_ohe, y_test_ohe, y_train, y_test, scaler = full_pipeline()

model = Sequential([
    Input(shape=(187, 1)),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(5, activation="softmax"),
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint("models/lstm_model.h5", save_best_only=True, monitor="val_accuracy", verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
]

print("\n[LSTM] Starting training ...")
history = model.fit(
    X_train,
    y_train_ohe,
    epochs=10,
    batch_size=128,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1,
)

loss, acc = model.evaluate(X_test, y_test_ohe, verbose=0)
print(f"\n[LSTM] Test Accuracy: {acc*100:.2f}%  |  Test Loss: {loss:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history["accuracy"], label="Train Acc", color="#3182ce")
ax1.plot(history.history["val_accuracy"], label="Val Acc", color="#e53e3e")
ax1.set_title("LSTM Accuracy")
ax1.set_xlabel("Epoch")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history["loss"], label="Train Loss", color="#3182ce")
ax2.plot(history.history["val_loss"], label="Val Loss", color="#e53e3e")
ax2.set_title("LSTM Loss")
ax2.set_xlabel("Epoch")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/lstm_training_history.png", dpi=150, bbox_inches="tight")
plt.close()
print("[LSTM] Training history saved to outputs/lstm_training_history.png")
print("[LSTM] Model saved to models/lstm_model.h5")
print("LSTM training complete!")
