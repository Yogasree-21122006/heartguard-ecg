import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("reports", exist_ok=True)


def butter_lowpass_filter(signal, cutoff=40, fs=360, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, signal)


def denoise_signal(signal):
    return butter_lowpass_filter(np.array(signal, dtype=np.float32))


def preprocess_single(signal_array, scaler):
    signal = np.array(signal_array, dtype=np.float32).reshape(1, -1)
    signal_denoised = np.array([denoise_signal(signal[0])])
    signal_scaled = scaler.transform(signal_denoised)
    return signal_scaled.reshape(1, 187, 1)


def generate_synthetic_ecg(class_idx, noise=0.02):
    t = np.linspace(0, 1, 187)
    signal = np.zeros(187)

    if class_idx == 0:
        signal += 0.15 * np.exp(-0.5 * ((t - 0.22) / 0.03) ** 2)
        signal[75:80] += [-0.15, -0.3, 1.0, 0.8, 0.1]
        signal += 0.3 * np.exp(-0.5 * ((t - 0.65) / 0.06) ** 2)
        signal += 0.05 * np.sin(2 * np.pi * 0.8 * t)

    elif class_idx == 1:
        signal += 0.06 * np.sin(2 * np.pi * 6 * t) + 0.04 * np.sin(2 * np.pi * 9 * t)
        for pos in [60, 125]:
            if pos - 2 >= 0 and pos + 3 <= 187:
                signal[pos - 2 : pos + 3] += [-0.1, 0.5, 1.0, 0.5, -0.1]

    elif class_idx == 2:
        vals = [0, -0.1, -0.2, 0.1, 0.4, 0.8, 1.2, 1.0, 0.7, 0.3, 0, -0.3, -0.5, -0.6, -0.4, -0.2, -0.1, 0, 0, 0]
        signal[80:100] += vals

    elif class_idx == 3:
        signal += 0.1 * np.sin(2 * np.pi * 1.1 * t)
        vals = [0, 0.1, 0.3, 0.6, 0.9, 0.7, 0.4, 0.2, -0.1, -0.3, -0.2, -0.1, 0]
        signal[85:98] += vals

    elif class_idx == 4:
        signal[40] = 1.5
        vals = [0.2, 0.6, 1.0, 0.8, 0.4, -0.2, -0.4, -0.3, -0.1, 0]
        signal[44:54] += vals

    signal += noise * np.random.randn(187)
    signal = np.clip(signal, -0.5, 1.5)
    mn, mx = signal.min(), signal.max()
    return ((signal - mn) / (mx - mn + 1e-9)).astype(np.float32)


def full_pipeline(csv_path="dataset/mitbih_train.csv", n_samples=15000):
    print(f"[preprocess] Loading {csv_path} ...")
    df = pd.read_csv(csv_path, header=None)
    print(f"[preprocess] Loaded shape: {df.shape}")

    y_col = df.columns[-1]
    X_raw = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = df.iloc[:, -1].values.astype(int)

    samples_per_class = n_samples // 5
    indices = []
    for cls in range(5):
        cls_idx = np.where(y_raw == cls)[0]
        n = min(samples_per_class, len(cls_idx))
        chosen = np.random.choice(cls_idx, size=n, replace=False)
        indices.extend(chosen.tolist())

    np.random.shuffle(indices)
    X_raw = X_raw[indices]
    y_raw = y_raw[indices]
    print(f"[preprocess] Sampled {len(y_raw)} rows. Class dist: {np.bincount(y_raw)}")

    print("[preprocess] Applying Butterworth low-pass filter ...")
    X_denoised = np.array([denoise_signal(row) for row in X_raw], dtype=np.float32)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_denoised, y_raw, test_size=0.2, stratify=y_raw, random_state=42
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    joblib.dump(scaler, "models/scaler.pkl")
    print("[preprocess] Scaler saved to models/scaler.pkl")

    X_train = X_train_scaled.reshape(-1, 187, 1)
    X_test = X_test_scaled.reshape(-1, 187, 1)

    y_train_ohe = to_categorical(y_train, num_classes=5)
    y_test_ohe = to_categorical(y_test, num_classes=5)

    print(f"[preprocess] X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train_ohe, y_test_ohe, y_train, y_test, scaler


if __name__ == "__main__":
    X_train, X_test, y_train_ohe, y_test_ohe, y_train, y_test, scaler = full_pipeline()
    print("Preprocessing complete.")
