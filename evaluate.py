import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize

from preprocess import full_pipeline

os.makedirs("outputs", exist_ok=True)

CLASS_NAMES = ["Normal", "AFib", "VTach", "Fusion", "Paced"]

print("=" * 55)
print("  ECG Arrhythmia Model Evaluation")
print("=" * 55)

X_train, X_test, y_train_ohe, y_test_ohe, y_train, y_test, scaler = full_pipeline()

results = {}
histories = {}

for name in ["lstm", "gru", "cnn"]:
    path = f"models/{name}_model.h5"
    if not os.path.exists(path):
        print(f"[{name.upper()}] Model not found at {path} — skipping.")
        continue

    print(f"\n[{name.upper()}] Evaluating ...")
    model = tf.keras.models.load_model(path)

    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    y_test_bin = label_binarize(y_test, classes=list(range(5)))
    try:
        auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class="ovr", average="weighted")
    except Exception:
        auc = 0.0

    results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}

    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall   : {rec*100:.2f}%")
    print(f"  F1-Score : {f1*100:.2f}%")
    print(f"  AUC-ROC  : {auc:.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(f"{name.upper()} Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"outputs/confusion_matrix_{name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#3182ce", "#38a169", "#e53e3e", "#d69e2e", "#718096"]
    for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        ax.plot(fpr, tpr, label=f"{cls_name}", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_title(f"{name.upper()} ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"outputs/roc_curve_{name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[{name.upper()}] Saved confusion matrix and ROC curve to outputs/")

if results:
    df = pd.DataFrame(results).T.reset_index()
    df.columns = ["model", "accuracy", "precision", "recall", "f1", "auc"]
    df.to_csv("outputs/model_comparison_results.csv", index=False)
    print("\nModel comparison saved to outputs/model_comparison_results.csv")

    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    x = np.arange(len(metrics))
    width = 0.25
    colors = ["#3182ce", "#38a169", "#e53e3e"]
    model_names = list(results.keys())

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (mname, color) in enumerate(zip(model_names, colors)):
        vals = [results[mname][m] for m in metrics]
        bars = ax.bar(x + i * width, [v * 100 for v in vals], width, label=mname.upper(), color=color, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score (%)")
    ax.set_title("LSTM vs GRU vs CNN — Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 115)
    plt.tight_layout()
    plt.savefig("outputs/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Model comparison chart saved to outputs/model_comparison.png")

print("\nEvaluation complete!")
