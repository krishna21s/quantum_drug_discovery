"""
App V3 — Hybrid Quantum Drug Discovery: Target Binding Affinity
================================================================
Streamlit Dual-Mode UI:
  • Mode A: Binding Affinity (pIC50) — EGFR / Lung Cancer Target
  • Mode B: Toxicity Screening (V2 bridge)

Features:
  - Animated Binding Affinity Gauge (0–10 pIC50 scale)
  - Hybrid Ensemble: XGBRegressor + 20-qubit QSVR
  - 3D Conformer pipeline visualization
  - Reference molecule validation panel
  - Shot-mode CI estimation
"""

import sys
import os
import json
import time
import pickle
import numpy as np
import streamlit as st
from pathlib import Path

# ── Patch sys.path for sibling imports ──────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from config import (
    REFERENCE_MOLECULES, CHECKPOINT_DIR, W_XGB, W_QML,
    PIC50_HIGH_THRESHOLD, PIC50_MED_THRESHOLD,
    INTERACTIVE_SLA_S, CHEMBL_TARGET_NAME, DISEASE_AREA,
    N_QUBITS, NYSTROM_LANDMARKS
)

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quantum Drug Discovery V3 — Binding Affinity",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #E8EAF0;
  }

  /* Background */
  .stApp {
    background: linear-gradient(135deg, #0D1117 0%, #0F1C2E 50%, #0D1117 100%);
  }

  /* Header banner */
  .hero-banner {
    background: linear-gradient(135deg, #0B3D91 0%, #1565C0 40%, #00ACC1 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(11,61,145,0.4);
  }
  .hero-title {
    font-size: 2.0rem;
    font-weight: 900;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin: 0;
  }
  .hero-sub {
    font-size: 1.0rem;
    color: rgba(255,255,255,0.78);
    margin: 6px 0 0 0;
  }

  /* Metric cards */
  .metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px;
    padding: 20px 24px;
    backdrop-filter: blur(10px);
    margin-bottom: 12px;
  }
  .metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #90CAF9;
    margin-bottom: 6px;
  }
  .metric-value {
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.1;
  }
  .metric-sub {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.5);
    margin-top: 4px;
  }

  /* pIC50 Gauge */
  .gauge-container {
    text-align: center;
    padding: 20px 0;
  }
  .gauge-label {
    font-size: 0.9rem;
    color: #90CAF9;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
  }

  /* Affinity badge */
  .badge-high   { background: linear-gradient(135deg,#00C853,#1DE9B6); color:#000; }
  .badge-med    { background: linear-gradient(135deg,#FFD600,#FFAB00); color:#000; }
  .badge-low    { background: linear-gradient(135deg,#FF5252,#D50000); color:#fff; }
  .affinity-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.5px;
    margin-top: 8px;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #0B1622 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
  }

  /* Action button */
  div.stButton > button {
    background: linear-gradient(135deg, #1565C0, #00ACC1);
    color: white;
    font-weight: 700;
    border-radius: 10px;
    border: none;
    padding: 14px 32px;
    font-size: 1.0rem;
    letter-spacing: 0.5px;
    transition: all 0.25s ease;
    width: 100%;
  }
  div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(21,101,192,0.5);
  }

  /* Status boxes */
  .status-box {
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.88rem;
    margin: 8px 0;
  }
  .status-green { background:rgba(0,200,83,0.12); border:1px solid rgba(0,200,83,0.3); color:#69FF9C; }
  .status-amber { background:rgba(255,214,0,0.10); border:1px solid rgba(255,214,0,0.3); color:#FFE57F; }
  .status-red   { background:rgba(255,82,82,0.12); border:1px solid rgba(255,82,82,0.3); color:#FF8A80; }
  .status-blue  { background:rgba(33,150,243,0.12); border:1px solid rgba(33,150,243,0.3); color:#90CAF9; }

  /* Warning / info overrides */
  .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════════

def pic50_color(pic50: float) -> str:
    if pic50 >= PIC50_HIGH_THRESHOLD:
        return "#00E676"
    elif pic50 >= PIC50_MED_THRESHOLD:
        return "#FFD600"
    else:
        return "#FF5252"


def pic50_label(pic50: float) -> str:
    if pic50 >= PIC50_HIGH_THRESHOLD:
        return "🟢 Strong Binder"
    elif pic50 >= PIC50_MED_THRESHOLD:
        return "🟡 Moderate Binder"
    else:
        return "🔴 Weak / Non-Binder"


def pic50_badge_class(pic50: float) -> str:
    if pic50 >= PIC50_HIGH_THRESHOLD:
        return "badge-high"
    elif pic50 >= PIC50_MED_THRESHOLD:
        return "badge-med"
    else:
        return "badge-low"


def render_gauge(pic50: float, max_val: float = 10.0):
    """Render an SVG-based animated pIC50 gauge."""
    fraction = min(max(pic50 / max_val, 0), 1)
    angle    = -140 + fraction * 280   # -140° to +140°
    color    = pic50_color(pic50)

    svg = f"""
    <div class="gauge-container">
      <p class="gauge-label">pIC₅₀ Binding Affinity Score</p>
      <svg width="260" height="160" viewBox="0 0 260 160" style="overflow:visible">
        <!-- Gradient arc background -->
        <defs>
          <linearGradient id="gGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" style="stop-color:#FF5252"/>
            <stop offset="50%" style="stop-color:#FFD600"/>
            <stop offset="100%" style="stop-color:#00E676"/>
          </linearGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
            <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>

        <!-- Background track arc -->
        <path d="M 30 145 A 100 100 0 0 1 230 145" fill="none"
              stroke="rgba(255,255,255,0.08)" stroke-width="18" stroke-linecap="round"/>

        <!-- Filled progress arc (0 to current value) -->
        <path d="M 30 145 A 100 100 0 0 1 230 145" fill="none"
              stroke="url(#gGrad)" stroke-width="18" stroke-linecap="round"
              stroke-dasharray="314" stroke-dashoffset="{int(314 * (1 - fraction))}"
              filter="url(#glow)"/>

        <!-- Tick marks -->
        {"".join([
            f'<line x1="130" y1="50" x2="130" y2="35" '
            f'stroke="rgba(255,255,255,0.25)" stroke-width="1.5" '
            f'transform="rotate({-140 + i*28} 130 145)"/>'
            for i in range(11)
        ])}

        <!-- Needle -->
        <line x1="130" y1="145" x2="130" y2="52"
              stroke="{color}" stroke-width="3" stroke-linecap="round"
              filter="url(#glow)"
              transform="rotate({angle} 130 145)"
              style="transition: transform 0.8s cubic-bezier(0.34,1.56,0.64,1)"/>
        <circle cx="130" cy="145" r="8" fill="{color}" filter="url(#glow)"/>
        <circle cx="130" cy="145" r="4" fill="#0D1117"/>

        <!-- Scale labels -->
        <text x="22"  y="158" fill="rgba(255,255,255,0.5)" font-size="10" text-anchor="middle">0</text>
        <text x="75"  y="88"  fill="rgba(255,255,255,0.5)" font-size="10" text-anchor="middle">2.5</text>
        <text x="130" y="52"  fill="rgba(255,255,255,0.5)" font-size="10" text-anchor="middle">5</text>
        <text x="185" y="88"  fill="rgba(255,255,255,0.5)" font-size="10" text-anchor="middle">7.5</text>
        <text x="238" y="158" fill="rgba(255,255,255,0.5)" font-size="10" text-anchor="middle">10</text>

        <!-- Value display -->
        <text x="130" y="128" fill="{color}" font-size="26" font-weight="900"
              text-anchor="middle" font-family="Inter, sans-serif">{pic50:.2f}</text>
        <text x="130" y="142" fill="rgba(255,255,255,0.5)" font-size="9"
              text-anchor="middle" font-family="Inter, sans-serif">pIC₅₀</text>
      </svg>
    </div>
    """
    st.markdown(svg, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# MODEL LOADING (cached)
# ════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_models():
    """Load all inference artifacts from checkpoints."""
    results = {
        "xgb_ready":  False,
        "qsvr_ready": False,
        "errors":     [],
    }

    # XGBRegressor
    try:
        with open(CHECKPOINT_DIR / "xgb_regressor_v3.pkl", "rb") as f:
            results["xgb_model"] = pickle.load(f)
        with open(CHECKPOINT_DIR / "xgb_var_selector_v3.pkl", "rb") as f:
            results["xgb_selector"] = pickle.load(f)
        results["xgb_ready"] = True
    except FileNotFoundError:
        results["errors"].append("XGB: Checkpoint not found. Run train_xgb_regressor.py")

    # QSVR + Nystrom artifacts
    try:
        with open(CHECKPOINT_DIR / "qsvr_model_v3.pkl", "rb") as f:
            results["svr_model"] = pickle.load(f)
        with open(CHECKPOINT_DIR / "qsvr_scaler_v3.pkl", "rb") as f:
            results["scaler"] = pickle.load(f)
        results["landmarks_scaled"] = np.load(
            CHECKPOINT_DIR / "qsvr_landmarks_scaled_v3.npy"
        )
        results["K_mm_inv"]   = np.load(CHECKPOINT_DIR / "qsvr_K_mm_inv_v3.npy")
        results["diag_train"] = np.load(CHECKPOINT_DIR / "qsvr_diag_train_v3.npy")
        K_nm_path             = CHECKPOINT_DIR / "K_nm_v3.npy"
        results["K_nm"]       = np.load(K_nm_path)
        with open(CHECKPOINT_DIR / "qsvr_selected_features_v3.json") as f:
            results["selected_features"] = json.load(f)
        results["qsvr_ready"] = True
    except FileNotFoundError:
        results["errors"].append("QSVR: Checkpoint not found. Run train_qsvr.py")

    return results


@st.cache_resource(show_spinner=False)
def get_feature_service():
    from services.feature_service_3d import FeatureService3D
    return FeatureService3D()


@st.cache_resource(show_spinner=False)
def get_classical_router(_feature_svc, _models):
    from services.classical_router import ClassicalRouter
    if not _models.get("xgb_ready"):
        return None
    return ClassicalRouter(
        feature_service=_feature_svc,
        xgb_model=_models["xgb_model"],
        xgb_selector=_models["xgb_selector"],
    )


@st.cache_resource(show_spinner=False)
def get_quantum_service(_feature_svc, _models):
    from services.nystrom_engine import NystromEngine
    from services.quantum_kernel_service import QuantumKernelService
    from quantum.backends import StatevectorBackend, ShotBackend

    if not _models.get("qsvr_ready"):
        return None

    nystrom           = NystromEngine()
    nystrom.K_mm_inv  = _models["K_mm_inv"]
    nystrom.K_nm      = _models["K_nm"]
    nystrom.diag_train = _models["diag_train"]

    backend_sv   = StatevectorBackend()
    backend_shot = ShotBackend()

    # Attach selected features to feature service
    _feature_svc._selected_features = _models.get("selected_features")

    return QuantumKernelService(
        backend_sv        = backend_sv,
        backend_shot      = backend_shot,
        nystrom_engine    = nystrom,
        svr_model         = _models["svr_model"],
        scaler            = _models["scaler"],
        landmarks_scaled  = _models["landmarks_scaled"],
        feature_service   = _feature_svc,
        selected_features = _models.get("selected_features"),
    )


# ════════════════════════════════════════════════════════════════════
# PREDICTION LOGIC
# ════════════════════════════════════════════════════════════════════

def run_prediction(smiles: str, mode: str, models: dict,
                   classical_router, quantum_svc) -> dict:
    """Run the full hybrid ensemble prediction."""
    result = {
        "smiles":      smiles,
        "mode":        mode,
        "xgb_pic50":  None,
        "qsvr_pic50": None,
        "final_pic50": None,
        "latency_s":  None,
        "errors":     [],
    }
    t0 = time.time()

    # XGBRegressor
    if classical_router and models.get("xgb_ready"):
        try:
            r = classical_router.predict_pic50(smiles)
            result["xgb_pic50"]  = r["pic50"]
            result["xgb_latency"] = r["latency_ms"]
        except Exception as e:
            result["errors"].append(f"XGB: {e}")

    # QSVR
    if quantum_svc and models.get("qsvr_ready"):
        try:
            progress_ph = st.empty()
            m = len(models["landmarks_scaled"])

            def cb(step, total):
                progress_ph.progress(step / total, text=f"⚛️ Quantum kernel: {step}/{total}")

            r = quantum_svc.predict_pic50(smiles, mode=mode, progress_callback=cb)
            progress_ph.empty()
            result["qsvr_pic50"]  = r["pic50"]
            result["qsvr_latency"] = r["latency_s"]
        except Exception as e:
            result["errors"].append(f"QSVR: {e}")

    # Ensemble blend
    xgb_p  = result["xgb_pic50"]
    qsvr_p = result["qsvr_pic50"]
    if xgb_p is not None and qsvr_p is not None:
        result["final_pic50"] = W_XGB * xgb_p + W_QML * qsvr_p
    elif xgb_p is not None:
        result["final_pic50"] = xgb_p
    elif qsvr_p is not None:
        result["final_pic50"] = qsvr_p

    result["latency_s"] = time.time() - t0
    return result


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════

def render_sidebar(models: dict) -> tuple:
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:12px 0 18px">
          <div style="font-size:2.2rem">🧬</div>
          <div style="font-weight:800; font-size:1.0rem; color:#90CAF9">V3 Platform</div>
          <div style="font-size:0.72rem; color:rgba(255,255,255,0.4)">Quantum Drug Discovery</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🎯 Target Configuration")
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Target Protein</div>
          <div style="font-weight:700; color:#4FC3F7; font-size:1.1rem">EGFR</div>
          <div class="metric-sub">Epidermal Growth Factor Receptor</div>
        </div>
        <div class="metric-card" style="margin-top:8px">
          <div class="metric-label">Disease</div>
          <div style="font-weight:700; color:#E91E63; font-size:1.05rem">{DISEASE_AREA}</div>
          <div class="metric-sub">ChEMBL ID: CHEMBL203</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### ⚛️ Quantum Config")
        st.markdown(f"""
        <div class="status-blue">
          🔲 <b>{N_QUBITS} Qubits</b> — 3D feature encoding<br/>
          🔗 <b>{NYSTROM_LANDMARKS} Landmarks</b> — Nystrom kernel<br/>
          ⚖️ XGB {int(W_XGB*100)}% + QSVR {int(W_QML*100)}%
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📊 Model Status")
        xgb_status  = "✅ Ready" if models.get("xgb_ready") else "⏳ Not Trained"
        qsvr_status = "✅ Ready" if models.get("qsvr_ready") else "⏳ Not Trained"
        xgb_cls  = "status-green" if models.get("xgb_ready") else "status-amber"
        qsvr_cls = "status-green" if models.get("qsvr_ready") else "status-amber"
        st.markdown(f"""
        <div class="status-box {xgb_cls}">XGBRegressor: {xgb_status}</div>
        <div class="status-box {qsvr_cls}">QSVR: {qsvr_status}</div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### ⚙️ Prediction Mode")
        mode = st.radio(
            "Quantum Backend",
            ["statevector", "shot"],
            captions=["Fast, deterministic (~1–3s)", "Hardware-realistic (slow, ±CI)"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.7rem; color:rgba(255,255,255,0.3); text-align:center">
          Hybrid Quantum-Classical Platform<br/>V3 · EGFR · pIC₅₀ Regression
        </div>
        """, unsafe_allow_html=True)

    return mode


# ════════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════════

def main():
    # ── Hero Banner ──────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-banner">
      <p class="hero-title">🧬 Hybrid Quantum Drug Discovery · V3</p>
      <p class="hero-sub">
        Target Binding Affinity Regression — EGFR (Lung Cancer) ·
        20-Qubit Quantum SVR + XGBoost Ensemble · Continuous pIC₅₀ Prediction
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load models ───────────────────────────────────────────────────
    with st.spinner("Loading inference artifacts..."):
        models = load_models()

    if models["errors"]:
        for err in models["errors"]:
            st.warning(f"⚠️ {err}")

    # ── Services ─────────────────────────────────────────────────────
    feat_svc        = get_feature_service()
    classical_router = get_classical_router(feat_svc, models)
    quantum_svc     = get_quantum_service(feat_svc, models)

    # ── Sidebar ───────────────────────────────────────────────────────
    mode = render_sidebar(models)

    # ── Main Tabs ─────────────────────────────────────────────────────
    tab_predict, tab_ref, tab_science, tab_train = st.tabs([
        "🔬 Predict Affinity",
        "📋 Reference Molecules",
        "🧪 Science Behind It",
        "🚀 Training Guide",
    ])

    # ──────────────────────────────────────────────────────────────────
    # TAB 1: PREDICT AFFINITY
    # ──────────────────────────────────────────────────────────────────
    with tab_predict:
        col_left, col_right = st.columns([1.1, 1.9])

        with col_left:
            st.markdown("#### 🧪 Enter Molecule")
            smiles_input = st.text_area(
                "SMILES String",
                placeholder="e.g. COCCOC1=C(OCCO)C=C2C(=CC1)NCNC3=CC=CC(=C3)C#C",
                height=110,
                label_visibility="collapsed",
                help="Enter a valid SMILES string. 3D conformer will be generated automatically."
            )

            # Quick picks
            st.markdown("**Quick picks:**")
            ref_choices = list(REFERENCE_MOLECULES.keys())
            chosen_ref  = st.selectbox("Load reference molecule", ["— select —"] + ref_choices,
                                       label_visibility="collapsed")
            if chosen_ref != "— select —":
                smiles_input = REFERENCE_MOLECULES[chosen_ref][0]
                st.code(smiles_input, language="text")

            st.markdown("---")

            # 3D generation info
            st.markdown("""
            <div class="status-box status-blue">
              <b>3D Pipeline Active</b><br/>
              RDKit ETKDG v3 → MMFF94 → WHIM/AUTOCORR3D → 20 Qubit Features
            </div>
            """, unsafe_allow_html=True)

            predict_btn = st.button("⚛️  Run Quantum Prediction", type="primary")

        with col_right:
            if predict_btn and smiles_input.strip():
                with st.spinner("Running hybrid ensemble..."):
                    res = run_prediction(
                        smiles_input.strip(), mode, models, classical_router, quantum_svc
                    )

                if res["final_pic50"] is not None:
                    pic50 = res["final_pic50"]

                    # ── Gauge ────────────────────────────────────────────
                    render_gauge(pic50)

                    # ── Badge ─────────────────────────────────────────────
                    badge_cls = pic50_badge_class(pic50)
                    label     = pic50_label(pic50)
                    st.markdown(f"""
                    <div style="text-align:center; margin:-10px 0 16px">
                      <span class="affinity-badge {badge_cls}">{label}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Ensemble metrics ──────────────────────────────────
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.markdown(f"""
                        <div class="metric-card">
                          <div class="metric-label">Final pIC₅₀</div>
                          <div class="metric-value">{pic50:.2f}</div>
                          <div class="metric-sub">Ensemble (XGB+QSVR)</div>
                        </div>""", unsafe_allow_html=True)
                    with m2:
                        xgb_val = f"{res['xgb_pic50']:.2f}" if res['xgb_pic50'] else "N/A"
                        st.markdown(f"""
                        <div class="metric-card">
                          <div class="metric-label">XGB pIC₅₀</div>
                          <div class="metric-value">{xgb_val}</div>
                          <div class="metric-sub">Topological baseline</div>
                        </div>""", unsafe_allow_html=True)
                    with m3:
                        qsvr_val = f"{res['qsvr_pic50']:.2f}" if res['qsvr_pic50'] else "N/A"
                        st.markdown(f"""
                        <div class="metric-card">
                          <div class="metric-label">QSVR pIC₅₀</div>
                          <div class="metric-value">{qsvr_val}</div>
                          <div class="metric-sub">Quantum physical depth</div>
                        </div>""", unsafe_allow_html=True)
                    with m4:
                        latency = res.get("latency_s", 0)
                        sla_ok  = latency <= INTERACTIVE_SLA_S
                        lat_cls = "status-green" if sla_ok else "status-amber"
                        st.markdown(f"""
                        <div class="metric-card">
                          <div class="metric-label">Latency</div>
                          <div class="metric-value">{latency:.1f}s</div>
                          <div class="metric-sub">SLA: ≤{INTERACTIVE_SLA_S}s</div>
                        </div>""", unsafe_allow_html=True)

                    # ── Interpretation ────────────────────────────────────
                    st.markdown("#### 🔬 Clinical Interpretation")
                    if pic50 >= PIC50_HIGH_THRESHOLD:
                        st.markdown(f"""
                        <div class="status-box status-green">
                          <b>🟢 Strong EGFR Binder (pIC₅₀ = {pic50:.2f})</b><br/>
                          This molecule demonstrates strong inhibitory activity against EGFR (Lung Cancer target).
                          IC₅₀ ≈ {10**(-(pic50))*1e9:.1f} nM — consistent with clinical kinase inhibitors.
                          <b>Recommend for further in-vitro validation.</b>
                        </div>""", unsafe_allow_html=True)
                    elif pic50 >= PIC50_MED_THRESHOLD:
                        st.markdown(f"""
                        <div class="status-box status-amber">
                          <b>🟡 Moderate EGFR Binder (pIC₅₀ = {pic50:.2f})</b><br/>
                          Moderate inhibitory activity. IC₅₀ ≈ {10**(-(pic50))*1e9:.0f} nM.
                          Further scaffold optimization may improve potency.
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="status-box status-red">
                          <b>🔴 Weak / Non-Binder (pIC₅₀ = {pic50:.2f})</b><br/>
                          Low binding affinity to EGFR. IC₅₀ ≈ {10**(-(pic50))*1e9:.0f} nM.
                          This molecule is unlikely to inhibit the target effectively.
                        </div>""", unsafe_allow_html=True)

                    if res["errors"]:
                        for e in res["errors"]:
                            st.error(f"Pipeline warning: {e}")

                elif res["errors"]:
                    for e in res["errors"]:
                        st.error(e)
                else:
                    st.warning("No models are ready. Please train models first.")

            elif not (predict_btn and smiles_input.strip()):
                # Welcome state
                st.markdown("""
                <div style="text-align:center; padding:60px 20px; color:rgba(255,255,255,0.3)">
                  <div style="font-size:4rem; margin-bottom:16px">⚛️</div>
                  <div style="font-size:1.1rem; font-weight:600">Enter a SMILES string and click predict</div>
                  <div style="font-size:0.85rem; margin-top:8px">
                    The 20-qubit quantum kernel will compute binding affinity to EGFR
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────
    # TAB 2: REFERENCE MOLECULES
    # ──────────────────────────────────────────────────────────────────
    with tab_ref:
        st.markdown("#### 📋 EGFR Reference Molecules (ChEMBL-Verified)")
        st.markdown("""
        <div class="status-box status-blue">
          These molecules have experimentally validated pIC₅₀ values from ChEMBL.
          Use them to calibrate the platform's predictions.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")

        for name, (smi, known_pic50) in REFERENCE_MOLECULES.items():
            with st.expander(f"{pic50_label(known_pic50)}  ·  {name}", expanded=False):
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.code(smi, language="text")
                with c2:
                    render_gauge(known_pic50)
                    st.markdown(f"""
                    <div style="text-align:center; margin-top:-8px">
                      <div style="color:rgba(255,255,255,0.5); font-size:0.8rem">Known pIC₅₀</div>
                      <div style="font-size:1.8rem; font-weight:800; color:{pic50_color(known_pic50)}">{known_pic50}</div>
                    </div>""", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────
    # TAB 3: SCIENCE
    # ──────────────────────────────────────────────────────────────────
    with tab_science:
        st.markdown("#### 🧪 The Science Behind the Platform")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("""
            **Why Quantum for Drug Binding?**

            Drug binding happens at the quantum level — the electron clouds of the
            drug molecule and protein pocket interact via van der Waals forces and
            hydrogen bonding in a 3D spatial configuration.

            Classical methods (molecular docking) model this with simplified force fields
            and miss sub-Ångstrom quantum effects. Our 20-qubit HEA circuit encodes
            the molecule's 3D geometry directly into the quantum Hilbert space.

            **The 3D Pipeline:**
            1. **ETKDG v3** — Generates a realistic 3D conformer from 2D SMILES
            2. **MMFF94** — Force-field optimization to find the natural resting shape
            3. **WHIM + AUTOCORR3D** — Shape/symmetry descriptors in 3D space
            4. **Pearson Filter** — Selects 20 orthogonally unique features
            5. **Quantum Encoding** — Each feature → 1 RY rotation angle per qubit
            """)

        with col_b:
            st.markdown("""
            **The pIC₅₀ Scale:**

            | pIC₅₀  | IC₅₀ (nM)  | Potency Class |
            |--------|------------|---------------|
            | ≥ 9.0  | ≤ 1 nM     | Ultra-strong (clinical) |
            | 8.0–9.0 | 1–10 nM  | Strong binder |
            | 7.0–8.0 | 10–100 nM | Good lead compound |
            | 6.0–7.0 | 100–1000 nM | Moderate |
            | < 5.0  | > 10 μM    | Weak / inactive |

            **Hybrid Ensemble:**
            - **XGBRegressor** (50%) — 4273 topological features, fast, robust
            - **QSVR** (50%) — 20-qubit quantum similarity kernel, captures 3D geometry

            **Success Metrics:**
            - Pearson R² > 0.65
            - RMSE < 1.0 log unit
            - Latency ≤ 3s (statevector mode)
            """)

    # ──────────────────────────────────────────────────────────────────
    # TAB 4: TRAINING GUIDE
    # ──────────────────────────────────────────────────────────────────
    with tab_train:
        st.markdown("#### 🚀 How to Train the Models")

        st.markdown("""
        <div class="status-box status-amber">
          <b>⚠️ Models require training before predictions are available.</b><br/>
          Follow the steps below in order. Step 1 downloads the ChEMBL dataset automatically.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        st.markdown("**Step 1: Train XGBRegressor (fast, ~5–10 min)**")
        st.code("cd backend/construction_v3\npython training/train_xgb_regressor.py", language="bash")
        st.markdown("Downloads EGFR IC50 data from ChEMBL, runs 60-trial Optuna search.")
        st.markdown("")

        st.markdown("**Step 2: Train QSVR (slow, ~30–180 min depending on hardware)**")
        st.code("python training/train_qsvr.py", language="bash")
        st.markdown("""
        Runs the 20-qubit Nystrom kernel computation. Progress is checkpointed every 10 rows,
        so you can safely interrupt and resume.
        """)
        st.markdown("")

        st.markdown("**Step 3: Launch App V3**")
        st.code("streamlit run app_v3.py", language="bash")

        # Show existing checkpoints
        st.markdown("---")
        st.markdown("**Checkpoint Status:**")
        checkpoint_files = [
            ("xgb_regressor_v3.pkl",       "XGB Regressor model"),
            ("xgb_var_selector_v3.pkl",    "XGB Variance Selector"),
            ("qsvr_model_v3.pkl",          "QSVR model"),
            ("qsvr_scaler_v3.pkl",         "MinMaxScaler (3D features)"),
            ("qsvr_landmarks_scaled_v3.npy", "Nystrom Landmarks"),
            ("K_mm_v3.npy",                "Landmark kernel K_mm"),
            ("K_nm_v3.npy",                "Training kernel K_nm"),
        ]

        for fname, desc in checkpoint_files:
            path = CHECKPOINT_DIR / fname
            if path.exists():
                size_kb = path.stat().st_size / 1024
                st.markdown(f"""
                <div class="status-box status-green">
                  ✅ <b>{fname}</b> — {desc} ({size_kb:.1f} KB)
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="status-box status-amber">
                  ⏳ <b>{fname}</b> — {desc} (not found)
                </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
