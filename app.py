"""
app.py
------
Streamlit web application for the AI Brain Tumor Detection System.
Run with:  streamlit run app.py
"""

import os
import sys
import time
import base64
from io import BytesIO

import streamlit as st
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import BrainTumorPredictor, PredictionResult, create_demo_model
from utils.preprocessing import enhance_mri_display, load_image_from_bytes

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
MODEL_PATH   = "model/brain_tumor_model.pt"
APP_TITLE    = "AI Brain Tumor Detection System"
APP_SUBTITLE = "Deep Learning–Powered MRI Analysis"

# ──────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS — clean medical/clinical aesthetic
# ──────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Grotesk:wght@400;600;700&display=swap');

/* ── Root palette ── */
:root {
  --bg:         #0a0f1e;
  --surface:    #111827;
  --surface2:   #1a2235;
  --border:     #1e2d45;
  --accent:     #00d4ff;
  --accent2:    #7c3aed;
  --danger:     #ef4444;
  --safe:       #10b981;
  --warning:    #f59e0b;
  --text:       #e2e8f0;
  --muted:      #64748b;
  --card-shadow: 0 8px 32px rgba(0,0,0,0.4);
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}

/* ── Header ── */
.app-header {
  text-align: center;
  padding: 2.5rem 1rem 1rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 2rem;
}
.app-header h1 {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 2.4rem;
  font-weight: 700;
  color: var(--accent);
  margin: 0 0 0.3rem;
  letter-spacing: -0.5px;
}
.app-header p {
  color: var(--muted);
  font-size: 1rem;
  margin: 0;
}

/* ── Cards ── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.6rem;
  box-shadow: var(--card-shadow);
  margin-bottom: 1.2rem;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
  background: var(--surface2) !important;
  border: 2px dashed var(--border) !important;
  border-radius: 16px !important;
  transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--accent) !important;
}

/* ── Result cards ── */
.result-tumor {
  background: linear-gradient(135deg, #2d0a0a 0%, #1a0505 100%);
  border: 1px solid #ef4444aa;
  border-radius: 16px;
  padding: 2rem;
  text-align: center;
  animation: fadeInUp 0.6s ease;
}
.result-no-tumor {
  background: linear-gradient(135deg, #062d1a 0%, #031a10 100%);
  border: 1px solid #10b981aa;
  border-radius: 16px;
  padding: 2rem;
  text-align: center;
  animation: fadeInUp 0.6s ease;
}
.result-icon {
  font-size: 3.5rem;
  display: block;
  margin-bottom: 0.5rem;
}
.result-label {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.8rem;
  font-weight: 700;
  margin: 0 0 0.4rem;
}
.result-label-tumor    { color: var(--danger); }
.result-label-safe     { color: var(--safe); }
.confidence-badge {
  display: inline-block;
  background: rgba(255,255,255,0.08);
  border-radius: 999px;
  padding: 0.3rem 1rem;
  font-size: 0.95rem;
  color: var(--text);
  margin-top: 0.4rem;
}

/* ── Probability bars ── */
.prob-row {
  display: flex;
  align-items: center;
  gap: 0.8rem;
  margin-bottom: 0.6rem;
}
.prob-label { width: 120px; font-size: 0.85rem; color: var(--muted); }
.prob-bar-wrap {
  flex: 1;
  background: var(--surface2);
  border-radius: 999px;
  height: 8px;
  overflow: hidden;
}
.prob-bar {
  height: 100%;
  border-radius: 999px;
  transition: width 1s ease;
}
.prob-val { width: 42px; text-align: right; font-size: 0.85rem; color: var(--text); }

/* ── Disclaimer ── */
.disclaimer {
  background: rgba(245, 158, 11, 0.08);
  border: 1px solid rgba(245, 158, 11, 0.3);
  border-radius: 12px;
  padding: 1rem 1.2rem;
  font-size: 0.82rem;
  color: #fbbf24;
  margin-top: 1.5rem;
}

/* ── Sidebar elements ── */
.sidebar-section {
  font-size: 0.85rem;
  color: var(--muted);
  line-height: 1.7;
}
.step-badge {
  display: inline-block;
  background: var(--accent2);
  color: white;
  border-radius: 50%;
  width: 22px;
  height: 22px;
  text-align: center;
  line-height: 22px;
  font-size: 0.75rem;
  font-weight: 700;
  margin-right: 6px;
}

/* ── Metric tiles ── */
.metric-grid { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-tile {
  flex: 1;
  min-width: 110px;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1rem;
  text-align: center;
}
.metric-value {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--accent);
}
.metric-label { font-size: 0.75rem; color: var(--muted); margin-top: 2px; }

/* ── Animations ── */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(20px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-ring {
  0%   { box-shadow: 0 0 0 0 rgba(0,212,255,0.4); }
  70%  { box-shadow: 0 0 0 14px rgba(0,212,255,0); }
  100% { box-shadow: 0 0 0 0 rgba(0,212,255,0); }
}
.pulse { animation: pulse-ring 2s infinite; }

/* ── Button overrides ── */
.stButton > button {
  background: linear-gradient(135deg, var(--accent2), #4f46e5) !important;
  color: white !important;
  border: none !important;
  border-radius: 10px !important;
  padding: 0.6rem 1.5rem !important;
  font-weight: 600 !important;
  font-family: 'DM Sans', sans-serif !important;
  transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_predictor() -> BrainTumorPredictor | None:
    """Load the predictor once and cache it."""
    if not os.path.isfile(MODEL_PATH):
        return None
    try:
        return BrainTumorPredictor(MODEL_PATH, device="cpu")
    except Exception:
        return None


def pil_to_b64(img: Image.Image, fmt="JPEG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def render_prob_bars(tumor_p: float, no_tumor_p: float):
    tumor_w    = int(tumor_p * 100)
    no_tumor_w = int(no_tumor_p * 100)
    st.markdown(f"""
    <div style="margin-top:1rem;">
      <div class="prob-row">
        <span class="prob-label">🔴 Tumor</span>
        <div class="prob-bar-wrap">
          <div class="prob-bar" style="width:{tumor_w}%; background:#ef4444;"></div>
        </div>
        <span class="prob-val">{tumor_p:.1%}</span>
      </div>
      <div class="prob-row">
        <span class="prob-label">🟢 No Tumor</span>
        <div class="prob-bar-wrap">
          <div class="prob-bar" style="width:{no_tumor_w}%; background:#10b981;"></div>
        </div>
        <span class="prob-val">{no_tumor_p:.1%}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_result_card(result: PredictionResult):
    if result.is_tumor:
        st.markdown(f"""
        <div class="result-tumor">
          <span class="result-icon">⚠️</span>
          <div class="result-label result-label-tumor">{result.label}</div>
          <span class="confidence-badge">Confidence: {result.confidence:.1%}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-no-tumor">
          <span class="result-icon">✅</span>
          <div class="result-label result-label-safe">{result.label}</div>
          <span class="confidence-badge">Confidence: {result.confidence:.1%}</span>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("### 🧠 Brain Tumor Detection")
        st.markdown("---")

        st.markdown("""
        <div class="sidebar-section">
        <b>How to use this tool</b><br><br>
        <span class="step-badge">1</span> Upload an MRI brain scan (JPG, PNG, or JPEG).<br><br>
        <span class="step-badge">2</span> The AI model analyses the scan automatically.<br><br>
        <span class="step-badge">3</span> View the prediction result and confidence score.<br><br>
        <span class="step-badge">4</span> Always follow up with a medical professional.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div class="sidebar-section">
        <b>About the Model</b><br><br>
        Architecture: Custom CNN (4 Conv blocks + GAP)<br>
        Input size: 224 × 224 px<br>
        Classes: Tumour / No Tumour<br>
        Framework: PyTorch (CPU)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div class="sidebar-section">
        <b>Supported formats</b><br>
        JPG &nbsp;·&nbsp; JPEG &nbsp;·&nbsp; PNG
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div class="sidebar-section" style="color:#475569; font-size:0.78rem;">
        B.Tech Computer Science – Capstone Project<br>
        AI-Powered Medical Imaging · 2024
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────

def main():
    render_sidebar()

    # ── Header ────────────────────────────────
    st.markdown(f"""
    <div class="app-header">
      <h1>🧠 {APP_TITLE}</h1>
      <p>{APP_SUBTITLE} &nbsp;·&nbsp; PyTorch + Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Model status ──────────────────────────
    predictor = get_predictor()

    if predictor is None:
        st.warning(
            "⚠️  **Model not found.** "
            "Run `python train.py` to train the model first, "
            "or run `python predict.py --image demo_mri.jpg` to create a demo model. "
            "A demo (untrained) model will be generated now for UI preview.",
            icon="⚠️"
        )
        # Auto-generate demo model so the UI is still usable
        create_demo_model(MODEL_PATH)
        predictor = get_predictor()

    # ── Metric strip ──────────────────────────
    st.markdown("""
    <div class="metric-grid">
      <div class="metric-tile">
        <div class="metric-value">CNN</div>
        <div class="metric-label">Architecture</div>
      </div>
      <div class="metric-tile">
        <div class="metric-value">224px</div>
        <div class="metric-label">Input Size</div>
      </div>
      <div class="metric-tile">
        <div class="metric-value">2</div>
        <div class="metric-label">Classes</div>
      </div>
      <div class="metric-tile">
        <div class="metric-value">CPU</div>
        <div class="metric-label">Device</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Layout: upload | result ────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📤 Upload MRI Scan")
        uploaded = st.file_uploader(
            "Drag & drop or click to browse",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded:
            img_bytes = uploaded.read()
            try:
                pil_img = load_image_from_bytes(img_bytes)
            except ValueError as e:
                st.error(f"❌ {e}")
                st.stop()

            # ── Display MRI image ──────────────
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 🔬 Uploaded MRI Scan")

            tab1, tab2 = st.tabs(["Original", "Enhanced (CLAHE)"])
            with tab1:
                st.image(pil_img, use_container_width=True, caption="Original MRI")
            with tab2:
                enhanced = enhance_mri_display(pil_img)
                st.image(enhanced, use_container_width=True, caption="Contrast Enhanced")

            st.markdown(
                f"<p style='color:var(--muted); font-size:0.8rem;'>"
                f"File: {uploaded.name} &nbsp;·&nbsp; "
                f"Size: {pil_img.size[0]}×{pil_img.size[1]} px</p>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Results column ──────────────────────
    with col_right:
        if uploaded and predictor:
            # Analyse button
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 🤖 AI Analysis")

            if st.button("🔍 &nbsp; Analyse MRI Scan", use_container_width=True):
                with st.spinner("Analysing scan with deep learning model…"):
                    # Simulate a short loading delay for UX
                    progress = st.progress(0, text="Preprocessing image…")
                    time.sleep(0.3)
                    progress.progress(33, text="Running CNN inference…")
                    time.sleep(0.4)
                    progress.progress(66, text="Computing probabilities…")

                    try:
                        result = predictor.predict_from_bytes(img_bytes)
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                        st.stop()

                    progress.progress(100, text="Done!")
                    time.sleep(0.3)
                    progress.empty()

                # ── Store in session state ──
                st.session_state["result"] = result

            st.markdown("</div>", unsafe_allow_html=True)

            # ── Show cached result ───────────
            if "result" in st.session_state:
                res: PredictionResult = st.session_state["result"]

                render_result_card(res)

                # Probability breakdown
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### 📊 Probability Breakdown")
                render_prob_bars(res.tumor_prob, res.no_tumor_prob)
                st.markdown("</div>", unsafe_allow_html=True)

                # Explanation
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### 💡 Interpretation")
                if res.is_tumor:
                    st.markdown(f"""
                    The model detected **patterns consistent with a brain tumour** 
                    in this MRI scan with a confidence of **{res.confidence:.1%}**.

                    Potential indicators include abnormal tissue density, asymmetry, 
                    or contrast differences in the scan region.

                    > **Next steps:** Please consult a qualified neurologist or 
                    radiologist for a professional diagnosis. This tool is intended 
                    for educational and screening assistance only.
                    """)
                else:
                    st.markdown(f"""
                    The model found **no significant tumour-like patterns** in this 
                    MRI scan, with a confidence of **{res.confidence:.1%}**.

                    The scan appears to show normal brain tissue distribution 
                    without notable anomalies detected by the model.

                    > **Note:** Even a negative result should be reviewed by a 
                    medical professional if you have clinical symptoms.
                    """)
                st.markdown("</div>", unsafe_allow_html=True)

        elif not uploaded:
            st.markdown("""
            <div class="card" style="text-align:center; padding:3rem 1.5rem; color:var(--muted);">
              <div style="font-size:3rem; margin-bottom:1rem;">🧬</div>
              <div style="font-size:1.1rem; font-weight:500; margin-bottom:0.5rem;">
                Awaiting MRI Scan
              </div>
              <div style="font-size:0.85rem;">
                Upload an MRI image on the left to begin analysis
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <b>Medical Disclaimer:</b> This application is a research and educational prototype 
    built for a B.Tech capstone project. It is <b>NOT</b> a certified medical device and must 
    not be used for clinical diagnosis. Always consult a qualified medical professional 
    for diagnosis and treatment decisions. The developers assume no liability for 
    any clinical decisions made based on this tool.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
