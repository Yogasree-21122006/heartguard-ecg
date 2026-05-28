import os
import io
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks
import joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

CLASS_INFO = {
    0: {
        "name": "Normal Sinus Rhythm",
        "short": "Normal",
        "emoji": "✅",
        "risk": "LOW",
        "color": "#38a169",
        "risk_color": "#c6f6d5",
        "risk_border": "#38a169",
        "description": "Normal heart rhythm with regular P waves, QRS complexes, and T waves. Heart rate between 60-100 BPM.",
        "ecg_features": "Regular P-P and R-R intervals, normal QRS duration <120ms, upright P waves in lead II",
        "symptoms": "No symptoms — normal heart function",
        "action": "No treatment required. Continue regular health check-ups.",
    },
    1: {
        "name": "Supraventricular (AFib)",
        "short": "AFib",
        "emoji": "⚠️",
        "risk": "MEDIUM",
        "color": "#d69e2e",
        "risk_color": "#fefcbf",
        "risk_border": "#d69e2e",
        "description": "Atrial fibrillation — chaotic electrical signals in the atria causing irregular heartbeat.",
        "ecg_features": "Absent P waves, irregular R-R intervals, fibrillatory baseline, narrow QRS",
        "symptoms": "Palpitations, shortness of breath, fatigue, dizziness, chest discomfort",
        "action": "Refer to cardiologist. May require rate control, rhythm control, or anticoagulation.",
    },
    2: {
        "name": "Ventricular (VTach/PVC)",
        "short": "VTach",
        "emoji": "🚨",
        "risk": "HIGH",
        "color": "#dd6b20",
        "risk_color": "#feebc8",
        "risk_border": "#dd6b20",
        "description": "Ventricular tachycardia or premature ventricular contractions — abnormal signals from ventricles.",
        "ecg_features": "Wide bizarre QRS >120ms, no preceding P wave, T wave opposite to QRS, compensatory pause",
        "symptoms": "Palpitations, dizziness, syncope, chest pain, sudden cardiac arrest risk",
        "action": "Urgent cardiology referral. May require antiarrhythmic drugs, ICD, or ablation.",
    },
    3: {
        "name": "Fusion Beat",
        "short": "Fusion",
        "emoji": "🔶",
        "risk": "MEDIUM",
        "color": "#d69e2e",
        "risk_color": "#fefcbf",
        "risk_border": "#d69e2e",
        "description": "Fusion of normal supraventricular and ventricular impulses — hybrid QRS morphology.",
        "ecg_features": "QRS morphology intermediate between normal and PVC, occurs during ventricular ectopy",
        "symptoms": "Usually asymptomatic, occasional palpitations",
        "action": "Monitor closely. Evaluate for underlying ventricular ectopy. Cardiology consultation recommended.",
    },
    4: {
        "name": "Unknown / Paced",
        "short": "Paced",
        "emoji": "🔲",
        "risk": "LOW",
        "color": "#718096",
        "risk_color": "#e2e8f0",
        "risk_border": "#718096",
        "description": "Pacemaker-induced beat or unclassified rhythm. Often seen in patients with implanted pacemakers.",
        "ecg_features": "Pacing spike before QRS, wide QRS complex, fixed pacing rate",
        "symptoms": "Depends on underlying condition; device-controlled rhythm",
        "action": "Routine pacemaker check. Ensure proper pacemaker function and settings.",
    },
}


def get_risk_level(class_idx, confidence=1.0):
    risk_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "MEDIUM", 4: "LOW"}
    base = risk_map.get(class_idx, "LOW")
    if class_idx == 2 and confidence > 0.85:
        return "EMERGENCY"
    return base


def estimate_heart_rate(signal, fs=360):
    signal = np.array(signal, dtype=np.float32)
    peaks, _ = find_peaks(signal, height=0.3, distance=int(fs * 0.5))
    if len(peaks) < 2:
        return 75.0
    rr = np.diff(peaks) / fs
    bpm = 60.0 / np.mean(rr)
    return round(float(np.clip(bpm, 30, 250)), 1)


def signal_quality(signal):
    signal = np.array(signal, dtype=np.float32)
    snr = float(np.max(signal) - np.min(signal))
    noise = float(np.std(np.diff(signal)))
    if snr > 0.6 and noise < 0.05:
        return "Excellent"
    elif snr > 0.4 and noise < 0.1:
        return "Good"
    elif snr > 0.2:
        return "Fair"
    return "Poor"


def load_model(name):
    import tensorflow as tf
    path = f"models/{name}_model.h5"
    if os.path.exists(path):
        try:
            return tf.keras.models.load_model(path)
        except Exception:
            return None
    return None


def load_scaler():
    path = "models/scaler.pkl"
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


def run_prediction(model, tensor):
    import time
    t0 = time.time()
    probs = model.predict(tensor, verbose=0)[0]
    elapsed_ms = round((time.time() - t0) * 1000, 1)
    class_idx = int(np.argmax(probs))
    confidence = float(probs[class_idx])
    info = CLASS_INFO[class_idx]
    risk = get_risk_level(class_idx, confidence)
    return {
        "class_idx": class_idx,
        "class_name": info["name"],
        "short": info["short"],
        "emoji": info["emoji"],
        "risk": risk,
        "color": info["color"],
        "confidence": confidence,
        "probs": probs.tolist(),
        "inference_ms": elapsed_ms,
        "description": info["description"],
        "ecg_features": info["ecg_features"],
        "symptoms": info["symptoms"],
        "action": info["action"],
    }


def ecg_chart(signal, title="ECG Signal", color="#3182ce", height=250, annotate=False):
    signal = np.array(signal, dtype=np.float32)
    x = np.arange(len(signal))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=signal, mode="lines",
        line=dict(color=color, width=1.5),
        name="ECG"
    ))
    if annotate:
        peaks, _ = find_peaks(signal, height=0.4, distance=30)
        if len(peaks):
            fig.add_trace(go.Scatter(
                x=peaks, y=signal[peaks], mode="markers",
                marker=dict(color="red", size=8, symbol="circle"),
                name="R-peak"
            ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#2d3748", size=14)),
        height=height,
        margin=dict(l=40, r=20, t=40, b=30),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f7fafc",
        xaxis=dict(title="Sample", gridcolor="#e2e8f0", color="#2d3748"),
        yaxis=dict(title="Amplitude", gridcolor="#e2e8f0", color="#2d3748"),
        showlegend=False,
    )
    return fig


def prob_bar_chart(probs_list):
    names = [CLASS_INFO[i]["short"] for i in range(5)]
    colors = [CLASS_INFO[i]["color"] for i in range(5)]
    fig = go.Figure(go.Bar(
        x=names,
        y=[round(p * 100, 1) for p in probs_list],
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in probs_list],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text="Class Probability Distribution", font=dict(color="#2d3748", size=14)),
        height=300,
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f7fafc",
        xaxis=dict(gridcolor="#e2e8f0", color="#2d3748"),
        yaxis=dict(title="Probability (%)", gridcolor="#e2e8f0", color="#2d3748", range=[0, 115]),
        showlegend=False,
    )
    return fig


def confidence_gauge(confidence):
    pct = round(confidence * 100, 1)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        title={"text": "Confidence", "font": {"color": "#2d3748", "size": 14}},
        number={"suffix": "%", "font": {"color": "#2d3748"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4a5568"},
            "bar": {"color": "#3182ce"},
            "steps": [
                {"range": [0, 50], "color": "#fed7d7"},
                {"range": [50, 75], "color": "#fefcbf"},
                {"range": [75, 100], "color": "#c6f6d5"},
            ],
            "threshold": {"line": {"color": "#e53e3e", "width": 3}, "value": 90},
        },
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=30, r=30, t=50, b=20),
        paper_bgcolor="#ffffff",
    )
    return fig


def detect_r_peaks(signal, fs=360):
    signal = np.array(signal, dtype=np.float32)
    peaks, _ = find_peaks(signal, height=0.3, distance=int(fs * 0.4))
    return peaks.tolist()


def compute_rr_intervals(peaks, fs=360):
    peaks = np.array(peaks)
    if len(peaks) < 2:
        return np.array([])
    return np.diff(peaks) / fs


def rhythm_regularity(rr_intervals):
    if len(rr_intervals) < 2:
        return "Unknown"
    cv = np.std(rr_intervals) / (np.mean(rr_intervals) + 1e-9)
    return "Regular" if cv < 0.1 else "Irregular"


def generate_pdf_report(prediction, signal, patient_name, age, gender, notes, model_name):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.units import cm
    import datetime

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm,
                             leftMargin=2*cm, rightMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle("title", parent=styles["Title"],
                                  fontSize=16, textColor=colors.HexColor("#1a202c"),
                                  spaceAfter=4)
    sub_style = ParagraphStyle("sub", parent=styles["Normal"],
                                fontSize=10, textColor=colors.HexColor("#4a5568"),
                                spaceAfter=12)
    head_style = ParagraphStyle("head", parent=styles["Heading2"],
                                 fontSize=12, textColor=colors.HexColor("#2b6cb0"),
                                 spaceBefore=12, spaceAfter=6)

    story.append(Paragraph("AI-Powered ECG Arrhythmia Detection Report", title_style))
    story.append(Paragraph("Kongu Engineering College — Deep Learning Healthcare Project", sub_style))
    story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", sub_style))

    story.append(Paragraph("Patient Information", head_style))
    pat_data = [
        ["Name", patient_name or "—"],
        ["Age", str(age)],
        ["Gender", gender],
        ["Model Used", model_name.upper()],
        ["Clinical Notes", notes or "—"],
    ]
    t = Table(pat_data, colWidths=[5*cm, 12*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#ebf8ff")),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor("#1a202c")),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#bee3f8")),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#f0f4f8")]),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(t)

    story.append(Paragraph("Prediction Result", head_style))
    risk_color = {"LOW": "#c6f6d5", "MEDIUM": "#fefcbf", "HIGH": "#feebc8", "EMERGENCY": "#fed7d7"}
    res_data = [
        ["Class", prediction["class_name"]],
        ["Confidence", f"{prediction['confidence']*100:.1f}%"],
        ["Risk Level", prediction["risk"]],
        ["Inference Time", f"{prediction['inference_ms']} ms"],
        ["Recommended Action", prediction["action"]],
    ]
    t2 = Table(res_data, colWidths=[5*cm, 12*cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#ebf8ff")),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor("#1a202c")),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#bee3f8")),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#f0f4f8")]),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(t2)

    story.append(Paragraph("Class Probabilities", head_style))
    prob_data = [["Class", "Probability"]] + [
        [CLASS_INFO[i]["name"], f"{prediction['probs'][i]*100:.1f}%"] for i in range(5)
    ]
    t3 = Table(prob_data, colWidths=[10*cm, 7*cm])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2b6cb0")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#bee3f8")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f0f4f8")]),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(t3)

    story.append(Spacer(1, 0.5*cm))
    disclaimer = ("⚠ DISCLAIMER: This AI prediction is for research and educational purposes only. "
                  "It does not constitute medical advice. Always consult a qualified cardiologist "
                  "for clinical diagnosis and treatment decisions.")
    story.append(Paragraph(disclaimer, ParagraphStyle("disc", parent=styles["Normal"],
                                                       fontSize=8, textColor=colors.HexColor("#718096"))))
    doc.build(story)
    return buf.getvalue()
