# AI-Powered ECG Arrhythmia Detection System
### Using LSTM, GRU, and CNN Networks
**Kongu Engineering College — Industry-Level Deep Learning Healthcare Project**

---

## Quick Start (Run Order)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place dataset
#    Download mitbih_train.csv from:
#    https://www.kaggle.com/datasets/shayanfazeli/heartbeat
#    and place it in:  dataset/mitbih_train.csv

# 3. Train models (fastest first)
python train_cnn.py       # ~3-5 min on CPU
python train_gru.py       # ~12-15 min on CPU
python train_lstm.py      # ~15-20 min on CPU

# 4. Evaluate all models
python evaluate.py        # ~2 min

# 5. Launch dashboard
streamlit run app.py
```

---

## Project Structure

```
ecg-arrhythmia-project/
├── app.py                  ← Streamlit dashboard (8 pages)
├── preprocess.py           ← Data loading + preprocessing pipeline
├── train_lstm.py           ← Bidirectional LSTM training
├── train_gru.py            ← GRU training
├── train_cnn.py            ← 1D CNN training
├── evaluate.py             ← Metrics, plots, comparison
├── utils.py                ← Shared helpers, PDF generator
├── requirements.txt
├── README.md
├── dataset/
│   └── mitbih_train.csv    ← Download from Kaggle
├── models/
│   ├── lstm_model.h5
│   ├── gru_model.h5
│   ├── cnn_model.h5
│   └── scaler.pkl
├── outputs/                ← Auto-generated charts
└── reports/                ← Auto-generated PDF reports
```

---

## Dataset

- **Source:** MIT-BIH Arrhythmia Dataset (Kaggle)
- **URL:** https://www.kaggle.com/datasets/shayanfazeli/heartbeat
- **File needed:** `mitbih_train.csv`
- **Samples used:** 15,000 (3,000 per class, stratified)
- **Classes:** Normal, AFib, VTach/PVC, Fusion Beat, Paced

---

## Models

| Model | Architecture | Expected Time (CPU) |
|-------|-------------|---------------------|
| CNN   | Conv1D × 3 + GAP | 3–5 min |
| GRU   | GRU(64) → GRU(32) | 12–15 min |
| LSTM  | BiLSTM(64) → LSTM(32) | 15–20 min |

**Total training time: ~30–40 min on a modern CPU laptop**

---

## Dashboard Pages

1. **Dashboard** — Live ECG simulation, stats, model status
2. **Predict ECG** — Upload CSV or use demo synthetic signal
3. **Batch Analysis** — Predict many signals at once
4. **Model Metrics** — Accuracy, F1, ROC curves, confusion matrices
5. **ECG Beat Counter** — R-peak detection, HRV analysis
6. **ECG Education** — Learn about arrhythmia types
7. **AI Assistant** — Chatbot with 15+ medical topics
8. **Training Guide** — Step-by-step training instructions

---

## Speed Tips

- Train CNN first (fastest) to verify the pipeline works
- Models use `EarlyStopping(patience=5)` — training may stop before 10 epochs if converged
- Only 15,000 samples are used (not the full 87K dataset)
- No class balancing → 4× faster preprocessing

---

## Notes

- Predictions use **synthetic ECG signals** — no need for test CSV files
- PDF reports are generated via ReportLab
- All Plotly charts use white backgrounds (professional medical look)
- The app works even without trained models (shows "Not trained" status)

---

*⚠ Disclaimer: This project is for educational and research purposes only. Not for clinical use.*
