import os
import time
import random
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import find_peaks

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from utils import (
    CLASS_INFO, get_risk_level, estimate_heart_rate, signal_quality,
    load_model, load_scaler, run_prediction, ecg_chart, prob_bar_chart,
    confidence_gauge, generate_pdf_report, detect_r_peaks,
    compute_rr_intervals, rhythm_regularity,
)
from preprocess import generate_synthetic_ecg, preprocess_single

st.set_page_config(
    page_title="ECG Arrhythmia Detection",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

LIGHT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background-color: #f0f4f8;
    color: #1a202c;
}
.stApp { background-color: #f0f4f8; }
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}
section[data-testid="stSidebar"] * { color: #1a202c !important; }

.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 18px 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    text-align: center;
}
.metric-card .label { color: #718096; font-size: 12px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-card .value { color: #1a202c; font-size: 26px; font-weight: 700; margin-top: 4px; }
.metric-card .sub { color: #4a5568; font-size: 11px; margin-top: 2px; }

.section-header {
    font-size: 20px; font-weight: 700; color: #1a202c;
    border-bottom: 2px solid #3182ce; padding-bottom: 6px; margin-bottom: 18px;
}

.arch-card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.arch-card h4 { color: #2b6cb0; margin: 0 0 8px 0; font-size: 15px; }
.arch-card p { color: #4a5568; font-size: 12px; line-height: 1.6; margin: 0; }

.risk-badge {
    display: inline-block; padding: 4px 12px;
    border-radius: 20px; font-weight: 700; font-size: 13px;
    border: 2px solid;
}
.risk-LOW    { background:#c6f6d5; color:#276749; border-color:#38a169; }
.risk-MEDIUM { background:#fefcbf; color:#744210; border-color:#d69e2e; }
.risk-HIGH   { background:#feebc8; color:#7b341e; border-color:#dd6b20; }
.risk-EMERGENCY { background:#fed7d7; color:#742a2a; border-color:#e53e3e; }

.alert-box {
    border-left: 4px solid;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 12px 0;
    font-size: 14px;
}
.alert-LOW    { background:#f0fff4; border-color:#38a169; color:#276749; }
.alert-MEDIUM { background:#fffff0; border-color:#d69e2e; color:#744210; }
.alert-HIGH   { background:#fffaf0; border-color:#dd6b20; color:#7b341e; }
.alert-EMERGENCY { background:#fff5f5; border-color:#e53e3e; color:#742a2a; }

.status-ok  { color:#38a169; font-weight:600; }
.status-err { color:#e53e3e; font-weight:600; }

.chat-user { background:#ebf8ff; border-radius:10px; padding:10px 14px; margin:6px 0; }
.chat-bot  { background:#f7fafc; border:1px solid #e2e8f0; border-radius:10px; padding:10px 14px; margin:6px 0; }

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #3182ce 0%, #2b6cb0 100%);
    color: white; border: none; border-radius: 8px;
    padding: 8px 20px; font-weight: 600; font-size: 14px;
    transition: all 0.2s ease;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #2b6cb0 0%, #2c5282 100%);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(49,130,206,0.35);
}
</style>
"""
st.markdown(LIGHT_CSS, unsafe_allow_html=True)

if "pred_history" not in st.session_state:
    st.session_state.pred_history = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ecg_live" not in st.session_state:
    st.session_state.ecg_live = False
if "live_buffer" not in st.session_state:
    st.session_state.live_buffer = list(generate_synthetic_ecg(0))


def model_status_badge(name):
    path = f"models/{name}_model.h5"
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024
        return f'<span class="status-ok">✅ Ready ({size:.0f} KB)</span>'
    return '<span class="status-err">❌ Not trained</span>'


with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:10px 0 6px 0;'>
        <div style='font-size:30px'>🫀</div>
        <div style='font-size:15px; font-weight:700; color:#1a202c;'>ECG Arrhythmia AI</div>
        <div style='font-size:11px; color:#718096;'>Kongu Engineering College</div>
    </div>
    <hr style='border-color:#e2e8f0;'>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🏥 Dashboard",
        "📊 Predict ECG",
        "📁 Batch Analysis",
        "📈 Model Metrics",
        "💡 ECG Beat Counter",
        "📚 ECG Education",
        "🤖 AI Assistant",
        "⚙️ Training Guide",
    ])

    st.markdown("<hr style='border-color:#e2e8f0;'>", unsafe_allow_html=True)
    model_choice = st.selectbox("Model", ["LSTM", "GRU", "CNN"], key="sidebar_model")
    demo_class = st.selectbox("Demo Class", [f"{i}: {CLASS_INFO[i]['short']}" for i in range(5)], key="sidebar_class")
    demo_class_idx = int(demo_class.split(":")[0])

    st.markdown("<hr style='border-color:#e2e8f0;'>", unsafe_allow_html=True)
    st.markdown("**Model Status**", unsafe_allow_html=False)
    for m in ["lstm", "gru", "cnn"]:
        st.markdown(f"{m.upper()}: {model_status_badge(m)}", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#e2e8f0;'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;color:#718096;">Dataset: MIT-BIH Arrhythmia<br>Samples: 15,000 | Classes: 5<br>Framework: TensorFlow/Keras</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 1: Dashboard
# ─────────────────────────────────────────────
if page == "🏥 Dashboard":
    st.markdown('<h2 style="color:#1a202c;margin-bottom:2px;">🏥 ECG Arrhythmia Detection Dashboard</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#4a5568;margin-bottom:20px;">AI-Powered Cardiac Monitoring — Kongu Engineering College</p>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        ("Models", "3", "LSTM · GRU · CNN"),
        ("Classes", "5", "Arrhythmia Types"),
        ("Dataset", "MIT-BIH", "Kaggle Benchmark"),
        ("Framework", "TF/Keras", "TensorFlow 2.x"),
        ("Samples", "15K", "Stratified Sample"),
    ]
    for col, (label, value, sub) in zip([c1,c2,c3,c4,c5], cards):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{value}</div>
                <div class="sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Live ECG Simulation</div>', unsafe_allow_html=True)

    b1, b2, _, hr_col, sq_col = st.columns([1, 1, 3, 2, 2])
    with b1:
        if st.button("▶ Start"):
            st.session_state.ecg_live = True
    with b2:
        if st.button("⏹ Stop"):
            st.session_state.ecg_live = False

    ecg_placeholder = st.empty()
    hr_placeholder = hr_col.empty()
    sq_placeholder = sq_col.empty()

    base_signal = generate_synthetic_ecg(demo_class_idx, noise=0.02)
    current_buf = list(st.session_state.live_buffer)

    if st.session_state.ecg_live:
        for _ in range(6):
            new_samples = base_signal[(_ * 31) % 187 : (_ * 31) % 187 + 31]
            current_buf.extend(new_samples.tolist())
            if len(current_buf) > 374:
                current_buf = current_buf[-374:]
            st.session_state.live_buffer = current_buf
            disp = np.array(current_buf)
            fig = go.Figure(go.Scatter(y=disp, mode="lines",
                                       line=dict(color="#3182ce", width=1.5)))
            fig.update_layout(
                height=200, margin=dict(l=30,r=20,t=20,b=20),
                paper_bgcolor="#ffffff", plot_bgcolor="#f7fafc",
                xaxis=dict(showticklabels=False, gridcolor="#e2e8f0"),
                yaxis=dict(gridcolor="#e2e8f0", color="#2d3748"),
                showlegend=False,
            )
            ecg_placeholder.plotly_chart(fig, use_container_width=True, key=f"live_{_}")
            hr_val = estimate_heart_rate(disp)
            sq_val = signal_quality(disp)
            hr_placeholder.metric("❤️ Heart Rate", f"{hr_val} BPM")
            sq_placeholder.metric("📶 Signal Quality", sq_val)
            time.sleep(0.15)
    else:
        disp = np.array(current_buf if current_buf else base_signal)
        fig = ecg_chart(disp, title="ECG Preview (Press Start)", color="#3182ce", height=200)
        ecg_placeholder.plotly_chart(fig, use_container_width=True)
        hr_placeholder.metric("❤️ Heart Rate", f"{estimate_heart_rate(disp)} BPM")
        sq_placeholder.metric("📶 Signal Quality", signal_quality(disp))

    st.markdown('<div class="section-header">Recent Prediction History</div>', unsafe_allow_html=True)
    if st.button("🗑 Clear History"):
        st.session_state.pred_history = []
        st.rerun()

    if st.session_state.pred_history:
        rows = []
        for p in st.session_state.pred_history[-10:][::-1]:
            rows.append({
                "Time": p.get("time", "—"),
                "Model": p.get("model", "—"),
                "Class": p.get("class_name", "—"),
                "Confidence": f"{p.get('confidence',0)*100:.1f}%",
                "Risk": p.get("risk", "—"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No predictions yet. Go to Predict ECG to analyse a signal.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Model Architectures</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    arch = [
        ("🧠 LSTM", "Bidirectional LSTM",
         "Input(187,1) → BiLSTM(64, return_seq) → Dropout(0.3) → LSTM(32) → Dropout(0.3) → Dense(32,relu) → Dense(5,softmax)"),
        ("⚡ GRU", "Gated Recurrent Unit",
         "Input(187,1) → GRU(64, return_seq) → Dropout(0.3) → GRU(32) → Dropout(0.3) → Dense(32,relu) → Dense(5,softmax)"),
        ("🔍 CNN", "1D Convolutional Network",
         "Input(187,1) → Conv1D(32,k=5) → BN → MaxPool(2) → Conv1D(64,k=3) → BN → MaxPool(2) → Conv1D(128,k=3) → GAP → Dense(64,relu) → Dropout → Dense(5,softmax)"),
    ]
    for col, (name, sub, layers) in zip([a1, a2, a3], arch):
        with col:
            st.markdown(f"""
            <div class="arch-card">
                <h4>{name} — {sub}</h4>
                <p>{layers}</p>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 2: Predict ECG
# ─────────────────────────────────────────────
elif page == "📊 Predict ECG":
    st.markdown('<h2 style="color:#1a202c;">📊 Predict ECG Signal</h2>', unsafe_allow_html=True)

    tab_upload, tab_demo = st.tabs(["📂 Upload CSV", "🔬 Demo Signal"])

    signal_to_predict = None

    with tab_upload:
        uploaded = st.file_uploader("Upload ECG CSV (rows = samples, cols = 187 features)", type=["csv"])
        if uploaded:
            df_up = pd.read_csv(uploaded, header=None)
            st.info(f"Loaded: {df_up.shape[0]} rows × {df_up.shape[1]} columns")
            row_idx = st.slider("Select row index", 0, len(df_up) - 1, 0)
            sig = df_up.iloc[row_idx, :187].values.astype(np.float32)
            st.plotly_chart(ecg_chart(sig, title=f"Uploaded ECG — Row {row_idx}", annotate=True), use_container_width=True)
            signal_to_predict = sig

    with tab_demo:
        d1, d2 = st.columns(2)
        with d1:
            cls_sel = st.selectbox("Arrhythmia Class", [f"{i}: {CLASS_INFO[i]['name']}" for i in range(5)])
            cls_idx = int(cls_sel.split(":")[0])
        with d2:
            noise_val = st.slider("Noise Level", 0.0, 0.1, 0.02, 0.005)
        demo_sig = generate_synthetic_ecg(cls_idx, noise=noise_val)
        st.plotly_chart(ecg_chart(demo_sig, title=f"Synthetic ECG — {CLASS_INFO[cls_idx]['name']}", annotate=True), use_container_width=True)
        signal_to_predict = demo_sig

    st.markdown('<div class="section-header">Patient Information</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        patient_name = st.text_input("Patient Name", placeholder="e.g. John Doe")
    with p2:
        patient_age = st.number_input("Age", min_value=1, max_value=120, value=45)
    with p3:
        patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    clinical_notes = st.text_area("Clinical Notes", placeholder="Enter relevant clinical information...")

    sel_model = st.selectbox("Select Model", ["LSTM", "GRU", "CNN"], key="pred_model")

    if st.button("🔍 Analyse ECG Signal", use_container_width=True):
        if signal_to_predict is None:
            st.error("No signal selected. Please upload a CSV or use the Demo Signal tab.")
        else:
            prog = st.progress(0)
            status = st.empty()

            status.text("Loading model...")
            prog.progress(15)
            model = load_model(sel_model.lower())
            scaler = load_scaler()

            if model is None:
                prog.empty()
                st.error(f"❌ {sel_model} model not found. Please run `python train_{sel_model.lower()}.py` first.")
            elif scaler is None:
                prog.empty()
                st.error("❌ Scaler not found. Please run the preprocessing pipeline first.")
            else:
                status.text("Preprocessing signal...")
                prog.progress(40)
                tensor = preprocess_single(signal_to_predict, scaler)

                status.text("Running inference...")
                prog.progress(70)
                result = run_prediction(model, tensor)
                result["model"] = sel_model
                result["time"] = time.strftime("%H:%M:%S")

                status.text("Generating results...")
                prog.progress(90)
                st.session_state.pred_history.append(result)

                prog.progress(100)
                status.empty()
                prog.empty()

                st.markdown("---")
                st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)

                rc1, rc2, rc3 = st.columns([2, 1, 1])
                with rc1:
                    st.markdown(f"""
                    <div class="metric-card" style="text-align:left;">
                        <div style="font-size:36px;">{result['emoji']}</div>
                        <div style="font-size:22px;font-weight:700;color:#1a202c;">{result['class_name']}</div>
                        <div style="font-size:15px;color:#4a5568;">Confidence: {result['confidence']*100:.1f}%</div>
                        <div style="margin-top:8px;">
                            <span class="risk-badge risk-{result['risk']}">{result['risk']} RISK</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with rc2:
                    hr_bpm = estimate_heart_rate(signal_to_predict)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="label">Heart Rate</div>
                        <div class="value">{hr_bpm}</div>
                        <div class="sub">BPM</div>
                    </div>
                    """, unsafe_allow_html=True)
                with rc3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="label">Inference</div>
                        <div class="value">{result['inference_ms']}</div>
                        <div class="sub">ms</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                ch1, ch2 = st.columns(2)
                with ch1:
                    st.plotly_chart(prob_bar_chart(result["probs"]), use_container_width=True)
                with ch2:
                    st.plotly_chart(confidence_gauge(result["confidence"]), use_container_width=True)

                risk = result["risk"]
                st.markdown(f"""
                <div class="alert-box alert-{risk}">
                    <strong>{result['emoji']} {result['class_name']} — {risk} RISK</strong><br>
                    {result['action']}
                </div>
                """, unsafe_allow_html=True)

                with st.expander("📉 Raw vs Processed Signal Comparison"):
                    e1, e2 = st.columns(2)
                    with e1:
                        st.plotly_chart(ecg_chart(signal_to_predict, title="Raw Signal", color="#718096"), use_container_width=True)
                    with e2:
                        processed = tensor.reshape(187)
                        st.plotly_chart(ecg_chart(processed, title="Processed Signal", color="#3182ce"), use_container_width=True)

                with st.expander("📋 Full Medical Information"):
                    st.markdown(f"**Description:** {result['description']}")
                    st.markdown(f"**ECG Features:** {result['ecg_features']}")
                    st.markdown(f"**Symptoms:** {result['symptoms']}")
                    st.markdown(f"**Recommended Action:** {result['action']}")

                pdf_bytes = generate_pdf_report(
                    result, signal_to_predict,
                    patient_name, patient_age, patient_gender, clinical_notes, sel_model
                )
                st.download_button(
                    "📄 Download PDF Medical Report",
                    data=pdf_bytes,
                    file_name=f"ecg_report_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )


# ─────────────────────────────────────────────
# PAGE 3: Batch Analysis
# ─────────────────────────────────────────────
elif page == "📁 Batch Analysis":
    st.markdown('<h2 style="color:#1a202c;">📁 Batch ECG Analysis</h2>', unsafe_allow_html=True)

    uploaded_batch = st.file_uploader("Upload CSV with multiple ECG signals", type=["csv"])
    max_rows = st.slider("Max rows to analyse", 1, 500, 50)
    batch_model = st.selectbox("Model", ["LSTM", "GRU", "CNN"], key="batch_model")

    if uploaded_batch and st.button("🚀 Run Batch Prediction", use_container_width=True):
        df_batch = pd.read_csv(uploaded_batch, header=None)
        n = min(max_rows, len(df_batch))
        model = load_model(batch_model.lower())
        scaler = load_scaler()

        if model is None:
            st.error(f"❌ {batch_model} model not trained yet.")
        elif scaler is None:
            st.error("❌ Scaler not found. Run preprocessing first.")
        else:
            prog = st.progress(0)
            results_list = []
            for i in range(n):
                sig = df_batch.iloc[i, :187].values.astype(np.float32)
                tensor = preprocess_single(sig, scaler)
                res = run_prediction(model, tensor)
                results_list.append({
                    "Row": i,
                    "Class": res["class_name"],
                    "Confidence": f"{res['confidence']*100:.1f}%",
                    "Risk": res["risk"],
                    "HR (BPM)": estimate_heart_rate(sig),
                })
                prog.progress((i + 1) / n)

            prog.empty()
            df_res = pd.DataFrame(results_list)
            st.success(f"✅ Analysed {n} signals")
            st.dataframe(df_res, use_container_width=True, hide_index=True)

            c1, c2 = st.columns(2)
            with c1:
                risk_counts = df_res["Risk"].value_counts()
                colors = {"LOW": "#38a169", "MEDIUM": "#d69e2e", "HIGH": "#dd6b20", "EMERGENCY": "#e53e3e"}
                fig_pie = go.Figure(go.Pie(
                    labels=risk_counts.index.tolist(),
                    values=risk_counts.values.tolist(),
                    marker_colors=[colors.get(r, "#718096") for r in risk_counts.index],
                    hole=0.35,
                ))
                fig_pie.update_layout(title="Risk Distribution", paper_bgcolor="#ffffff",
                                      title_font_color="#2d3748", height=300)
                st.plotly_chart(fig_pie, use_container_width=True)

            with c2:
                cls_counts = df_res["Class"].value_counts()
                fig_bar = go.Figure(go.Bar(
                    x=cls_counts.index.tolist(),
                    y=cls_counts.values.tolist(),
                    marker_color="#3182ce",
                ))
                fig_bar.update_layout(
                    title="Class Distribution", paper_bgcolor="#ffffff",
                    plot_bgcolor="#f7fafc", title_font_color="#2d3748",
                    xaxis=dict(color="#2d3748"), yaxis=dict(color="#2d3748"),
                    height=300,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            csv_out = df_res.to_csv(index=False).encode()
            st.download_button("⬇ Download Results CSV", data=csv_out,
                               file_name="batch_results.csv", mime="text/csv", use_container_width=True)
    elif not uploaded_batch:
        st.info("Upload a CSV file to begin batch analysis. Each row should contain 187 ECG feature values.")


# ─────────────────────────────────────────────
# PAGE 4: Model Metrics
# ─────────────────────────────────────────────
elif page == "📈 Model Metrics":
    st.markdown('<h2 style="color:#1a202c;">📈 Model Performance Metrics</h2>', unsafe_allow_html=True)

    csv_path = "outputs/model_comparison_results.csv"
    if os.path.exists(csv_path):
        df_metrics = pd.read_csv(csv_path)
        st.markdown('<div class="section-header">Comparison Table</div>', unsafe_allow_html=True)
        st.dataframe(
            df_metrics.style.format({c: "{:.4f}" for c in df_metrics.columns if c != "model"})
                            .background_gradient(cmap="Blues", subset=["accuracy", "f1", "auc"]),
            use_container_width=True, hide_index=True,
        )

        metrics = ["accuracy", "precision", "recall", "f1", "auc"]
        model_names = df_metrics["model"].tolist()
        x = list(range(len(metrics)))
        colors_m = ["#3182ce", "#38a169", "#e53e3e"]
        fig = go.Figure()
        for i, row in df_metrics.iterrows():
            vals = [row[m] * 100 for m in metrics]
            fig.add_trace(go.Bar(
                name=row["model"].upper(),
                x=[m.upper() for m in metrics],
                y=vals,
                text=[f"{v:.1f}%" for v in vals],
                textposition="outside",
                marker_color=colors_m[i % len(colors_m)],
            ))
        fig.update_layout(
            title="LSTM vs GRU vs CNN — All Metrics",
            barmode="group", height=400,
            paper_bgcolor="#ffffff", plot_bgcolor="#f7fafc",
            title_font_color="#2d3748",
            xaxis=dict(color="#2d3748", gridcolor="#e2e8f0"),
            yaxis=dict(color="#2d3748", gridcolor="#e2e8f0", range=[0, 115]),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No metrics found. Run `python evaluate.py` after training all models to generate metrics.")

    st.markdown('<div class="section-header">Saved Output Charts</div>', unsafe_allow_html=True)
    output_pngs = sorted([f for f in os.listdir("outputs") if f.endswith(".png")]) if os.path.exists("outputs") else []

    if output_pngs:
        for i in range(0, len(output_pngs), 2):
            c1, c2 = st.columns(2)
            with c1:
                from PIL import Image
                img = Image.open(f"outputs/{output_pngs[i]}")
                st.image(img, caption=output_pngs[i], use_column_width=True)
            if i + 1 < len(output_pngs):
                with c2:
                    img2 = Image.open(f"outputs/{output_pngs[i+1]}")
                    st.image(img2, caption=output_pngs[i+1], use_column_width=True)
    else:
        st.info("No chart images found. Run `python evaluate.py` to generate them.")


# ─────────────────────────────────────────────
# PAGE 5: ECG Beat Counter
# ─────────────────────────────────────────────
elif page == "💡 ECG Beat Counter":
    st.markdown('<h2 style="color:#1a202c;">💡 ECG Beat Counter & HRV Analysis</h2>', unsafe_allow_html=True)

    tab_gen, tab_up = st.tabs(["🔬 Generate Signal", "📂 Upload Signal"])

    with tab_gen:
        b1, b2 = st.columns(2)
        with b1:
            beat_cls = st.selectbox("Arrhythmia Class", [f"{i}: {CLASS_INFO[i]['name']}" for i in range(5)], key="beat_cls")
            beat_cls_idx = int(beat_cls.split(":")[0])
        with b2:
            num_beats = st.slider("Number of Beats", 3, 10, 5)

        full_sig = np.concatenate([
            generate_synthetic_ecg(beat_cls_idx, noise=0.02) for _ in range(num_beats)
        ])
        peaks = detect_r_peaks(full_sig, fs=360)
        rr = compute_rr_intervals(peaks, fs=360)
        hr_bpm = round(60.0 / np.mean(rr), 1) if len(rr) > 0 else 75.0
        mean_rr_ms = round(np.mean(rr) * 1000, 1) if len(rr) > 0 else 0
        rr_var_ms = round(np.std(rr) * 1000, 1) if len(rr) > 0 else 0
        rhythm = rhythm_regularity(rr)

        m1, m2, m3, m4, m5 = st.columns(5)
        for col, (lbl, val, sub) in zip([m1, m2, m3, m4, m5], [
            ("Beats Found", len(peaks), "R-peaks"),
            ("Heart Rate", f"{hr_bpm}", "BPM"),
            ("Mean RR", f"{mean_rr_ms}", "ms"),
            ("RR Variability", f"{rr_var_ms}", "ms"),
            ("Rhythm", rhythm, ""),
        ]):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">{lbl}</div>
                    <div class="value">{val}</div>
                    <div class="sub">{sub}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        fig_ecg = go.Figure()
        fig_ecg.add_trace(go.Scatter(y=full_sig, mode="lines",
                                      line=dict(color="#3182ce", width=1.5), name="ECG"))
        if peaks:
            fig_ecg.add_trace(go.Scatter(
                x=peaks, y=[full_sig[p] for p in peaks if p < len(full_sig)],
                mode="markers", marker=dict(color="red", size=9, symbol="circle"),
                name="R-peaks"
            ))
        fig_ecg.update_layout(
            title="Multi-Beat ECG with R-Peak Detection",
            height=280, margin=dict(l=40,r=20,t=50,b=30),
            paper_bgcolor="#ffffff", plot_bgcolor="#f7fafc",
            xaxis=dict(title="Sample", gridcolor="#e2e8f0", color="#2d3748"),
            yaxis=dict(title="Amplitude", gridcolor="#e2e8f0", color="#2d3748"),
        )
        st.plotly_chart(fig_ecg, use_container_width=True)

        if len(rr) > 1:
            fig_rr = go.Figure(go.Scatter(
                x=list(range(1, len(rr)+1)), y=(rr * 1000).tolist(),
                mode="lines+markers",
                line=dict(color="#e53e3e", width=2),
                marker=dict(size=7, color="#e53e3e"),
                name="RR Interval (ms)"
            ))
            fig_rr.update_layout(
                title="RR Interval Variability",
                height=250, margin=dict(l=40,r=20,t=50,b=30),
                paper_bgcolor="#ffffff", plot_bgcolor="#f7fafc",
                xaxis=dict(title="Beat #", gridcolor="#e2e8f0", color="#2d3748"),
                yaxis=dict(title="RR (ms)", gridcolor="#e2e8f0", color="#2d3748"),
            )
            st.plotly_chart(fig_rr, use_container_width=True)

    with tab_up:
        upl = st.file_uploader("Upload ECG segment CSV (single row of 187 values)", type=["csv"], key="beat_up")
        if upl:
            df_u = pd.read_csv(upl, header=None)
            sig_u = df_u.iloc[0, :187].values.astype(np.float32)
            peaks_u = detect_r_peaks(sig_u)
            rr_u = compute_rr_intervals(peaks_u)
            hr_u = round(60.0 / np.mean(rr_u), 1) if len(rr_u) > 0 else 0
            st.plotly_chart(ecg_chart(sig_u, title="Uploaded Signal with R-peaks", annotate=True), use_container_width=True)
            st.metric("Beats Found", len(peaks_u))
            st.metric("Heart Rate", f"{hr_u} BPM")
            st.metric("Rhythm", rhythm_regularity(rr_u))


# ─────────────────────────────────────────────
# PAGE 6: ECG Education
# ─────────────────────────────────────────────
elif page == "📚 ECG Education":
    st.markdown('<h2 style="color:#1a202c;">📚 ECG Arrhythmia Education</h2>', unsafe_allow_html=True)

    for idx, info in CLASS_INFO.items():
        with st.expander(f"{info['emoji']} {info['name']}"):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown(f"**Description:** {info['description']}")
                st.markdown(f"**ECG Features:** {info['ecg_features']}")
                st.markdown(f"**Symptoms:** {info['symptoms']}")
                st.markdown(f"**Action:** {info['action']}")
                st.markdown(f"<span class='risk-badge risk-{info['risk']}'>{info['risk']} RISK</span>", unsafe_allow_html=True)
            with c2:
                demo_s = generate_synthetic_ecg(idx, noise=0.01)
                st.plotly_chart(ecg_chart(demo_s, title="Demo ECG", color=info["color"], height=200), use_container_width=True)

    st.markdown('<div class="section-header">How the AI Works — 6 Steps</div>', unsafe_allow_html=True)
    steps = [
        ("1. Signal Acquisition", "ECG signal is captured as a 187-sample time series at 360 Hz sampling rate."),
        ("2. Preprocessing", "Butterworth low-pass filter removes noise. Min-Max scaling normalizes amplitude."),
        ("3. Feature Extraction", "LSTM/GRU capture temporal patterns; CNN extracts local morphological features."),
        ("4. Classification", "Softmax output gives probability across 5 arrhythmia classes."),
        ("5. Risk Assessment", "Risk level (LOW/MEDIUM/HIGH/EMERGENCY) assigned based on class and confidence."),
        ("6. Report Generation", "Full PDF report with patient info, metrics, and clinical recommendations."),
    ]
    s_cols = st.columns(3)
    for i, (title, desc) in enumerate(steps):
        with s_cols[i % 3]:
            st.markdown(f"""
            <div class="arch-card" style="margin-bottom:12px;">
                <h4 style="color:#2b6cb0;">{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Class Comparison Table</div>', unsafe_allow_html=True)
    comp = []
    for idx, info in CLASS_INFO.items():
        comp.append({
            "Class": info["name"],
            "Risk": info["risk"],
            "Key ECG Feature": info["ecg_features"][:60] + "...",
            "Primary Symptom": info["symptoms"][:50] + "...",
        })
    st.dataframe(pd.DataFrame(comp), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# PAGE 7: AI Assistant
# ─────────────────────────────────────────────
elif page == "🤖 AI Assistant":
    st.markdown('<h2 style="color:#1a202c;">🤖 ECG AI Assistant</h2>', unsafe_allow_html=True)

    KB = {
        "afib": "Atrial fibrillation (AFib) is a common arrhythmia where the atria beat chaotically and irregularly. On ECG, it shows absent P waves and an irregularly irregular rhythm. It increases stroke risk 5x. Treatment includes rate control, rhythm control, and anticoagulation.",
        "vtach": "Ventricular tachycardia (VTach) is a serious arrhythmia originating from the ventricles. ECG shows wide QRS complexes >120ms, rate >100 BPM. It can degenerate into ventricular fibrillation. Requires urgent treatment — antiarrhythmics, cardioversion, or ICD.",
        "pvc": "Premature Ventricular Contractions (PVCs) are early heartbeats from the ventricles. They appear as wide, bizarre QRS complexes with a compensatory pause. Occasional PVCs are benign; frequent ones need evaluation.",
        "normal": "Normal Sinus Rhythm has a rate of 60-100 BPM, regular P-P and R-R intervals, normal P wave (upright in lead II), and QRS duration <120ms. No treatment required.",
        "lstm": "LSTM (Long Short-Term Memory) networks are a type of recurrent neural network that excels at learning long-term dependencies in sequential data. They use memory cells and gating mechanisms to retain important information across many time steps — ideal for ECG temporal patterns.",
        "gru": "GRU (Gated Recurrent Unit) is a simplified version of LSTM with fewer parameters, making it faster to train while maintaining good accuracy. It uses reset and update gates. In this project it achieves similar accuracy to LSTM in ~30% less time.",
        "cnn": "1D CNN (Convolutional Neural Network) applies convolutional filters to detect local patterns in ECG waveforms. It's the fastest model in this project (3-5 min training) and excels at detecting morphological features like QRS width and P-wave shape.",
        "dataset": "The MIT-BIH Arrhythmia Dataset from PhysioNet is the gold standard for ECG research. It contains 48 half-hour ECG recordings, digitized at 360 Hz. The Kaggle version provides pre-segmented beats in CSV format. Download from: kaggle.com/datasets/shayanfazeli/heartbeat",
        "train": "To train models: 1) Download mitbih_train.csv from Kaggle, 2) Place in dataset/ folder, 3) Run: python train_cnn.py (5 min), python train_gru.py (15 min), python train_lstm.py (20 min). Total ~40 min on CPU.",
        "stop": "If training is interrupted, just re-run the training script. ModelCheckpoint saves the best weights automatically. The EarlyStopping callback (patience=5) will also stop training early if validation loss stops improving, saving time.",
        "accuracy": "Expected accuracy: CNN ~95%, GRU ~96%, LSTM ~96% on the MIT-BIH dataset with 15,000 samples. These are strong results for a 10-epoch, CPU-friendly model. Full dataset training typically achieves 97-99%.",
        "risk": "Risk levels: LOW (Normal, Paced) = routine monitoring. MEDIUM (AFib, Fusion) = cardiology referral recommended. HIGH (VTach) = urgent evaluation. EMERGENCY (VTach with >85% confidence) = immediate medical attention.",
        "heart rate": "Normal heart rate is 60-100 BPM. Bradycardia (<60 BPM) and tachycardia (>100 BPM) are abnormal. The AI estimates HR by detecting R-peaks and computing mean R-R intervals: HR = 60 / mean_RR_seconds.",
        "preprocessing": "ECG preprocessing pipeline: 1) Stratified sampling (3000/class), 2) Butterworth low-pass filter (cutoff=40Hz, fs=360Hz, order=4), 3) Min-Max normalization (fit on train only), 4) 80/20 train-test split, 5) Reshape to (N,187,1), 6) One-hot encoding.",
        "fusion": "Fusion beats occur when a supraventricular impulse and a ventricular impulse activate the ventricles simultaneously. The QRS morphology is intermediate between normal and PVC morphology. Usually benign but indicates underlying ventricular ectopy.",
        "paced": "Paced rhythms originate from an implanted pacemaker. ECG shows a pacing spike (vertical line) followed by a wide QRS. The rate is typically fixed by the device. Routine pacemaker checks ensure proper function.",
    }

    quick_qs = ["What is AFib?", "Explain VTach", "How does LSTM work?", "Where to download dataset?"]
    q_cols = st.columns(4)
    for col, q in zip(q_cols, quick_qs):
        with col:
            if st.button(q, use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": q})
                key_map = {"What is AFib?": "afib", "Explain VTach": "vtach",
                           "How does LSTM work?": "lstm", "Where to download dataset?": "dataset"}
                ans = KB.get(key_map.get(q, ""), "Please ask a more specific question.")
                st.session_state.chat_history.append({"role": "assistant", "content": ans})

    st.markdown("---")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask about ECG, arrhythmias, models, or training...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        lower = user_input.lower()
        reply = None
        for kw, ans in KB.items():
            if kw in lower:
                reply = ans
                break
        if reply is None:
            reply = ("I can answer questions about: AFib, VTach, PVC, Normal rhythm, Fusion beats, Paced rhythm, "
                     "LSTM, GRU, CNN models, the MIT-BIH dataset, training procedure, accuracy, risk levels, "
                     "heart rate estimation, preprocessing, and more. Please try rephrasing your question.")
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# ─────────────────────────────────────────────
# PAGE 8: Training Guide
# ─────────────────────────────────────────────
elif page == "⚙️ Training Guide":
    st.markdown('<h2 style="color:#1a202c;">⚙️ Training Guide</h2>', unsafe_allow_html=True)

    st.warning("⚠️ Do NOT close the terminal during training. ModelCheckpoint saves best weights automatically, so interrupted training loses only progress after the last saved epoch.")

    st.markdown('<div class="section-header">Step-by-Step Setup</div>', unsafe_allow_html=True)
    steps = [
        ("Step 1: Install Dependencies", "pip install -r requirements.txt", "Install all required Python packages. Requires Python 3.8+ and ~3GB disk space for TensorFlow."),
        ("Step 2: Download Dataset", "# Visit: kaggle.com/datasets/shayanfazeli/heartbeat\n# Download mitbih_train.csv\n# Place in: dataset/mitbih_train.csv", "You need a free Kaggle account. The file is ~34MB."),
        ("Step 3: Train CNN (Fastest)", "python train_cnn.py", "Trains the 1D CNN model. Takes ~3-5 minutes on CPU. Run this first to verify your setup."),
        ("Step 4: Train GRU", "python train_gru.py", "Trains the GRU model. Takes ~12-15 minutes on CPU."),
        ("Step 5: Train LSTM", "python train_lstm.py", "Trains the Bidirectional LSTM model. Takes ~15-20 minutes on CPU."),
        ("Step 6: Evaluate All Models", "python evaluate.py", "Generates confusion matrices, ROC curves, and model comparison charts in outputs/."),
        ("Step 7: Launch Dashboard", "streamlit run app.py", "Opens the dashboard at http://localhost:8501 — all 8 pages and features are ready."),
    ]
    for i, (title, cmd, desc) in enumerate(steps):
        with st.container():
            st.markdown(f"""
            <div class="arch-card" style="margin-bottom:10px;">
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
            st.code(cmd, language="bash")

    st.markdown('<div class="section-header">Speed Optimization Tips</div>', unsafe_allow_html=True)
    tips = [
        ("Tip 1: Smaller sample size", "Reduce n_samples in preprocess.py (e.g. 10000) for even faster training."),
        ("Tip 2: Fewer epochs", "Set epochs=5 in training scripts for a quick test run."),
        ("Tip 3: Larger batch size", "Increase batch_size to 256 if you have >8GB RAM — speeds up training 30%."),
        ("Tip 4: TF warnings", 'os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" — already set in all scripts.'),
        ("Tip 5: Train CNN only", "For a demo, train only CNN — it gives ~95% accuracy in just 5 minutes."),
    ]
    for t_title, t_desc in tips:
        st.markdown(f"""
        <div class="arch-card" style="margin-bottom:8px;">
            <h4>{t_title}</h4>
            <p>{t_desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Current Model Status</div>', unsafe_allow_html=True)
    for m in ["lstm", "gru", "cnn"]:
        path = f"models/{m}_model.h5"
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            st.success(f"✅ {m.upper()}: models/{m}_model.h5 — {size:.0f} KB")
        else:
            st.error(f"❌ {m.upper()}: Not trained yet — run `python train_{m}.py`")

    scaler_path = "models/scaler.pkl"
    if os.path.exists(scaler_path):
        st.success("✅ Scaler: models/scaler.pkl — Ready")
    else:
        st.error("❌ Scaler: Not found — will be created when any training script runs")
