"""
Quantum Drug Discovery Platform V2 — Streamlit App
====================================================
Production-grade hybrid quantum-classical toxicity screening.

Preserves ALL V1 features:
  ✓ Single-molecule analysis with XGBoost + Quantum Oracle
  ✓ Hybrid ensemble with conservative max-alert policy
  ✓ Batch CSV validation with Brier/AUC metrics
  ✓ Calibration curves and scatter plots
  ✓ Model disagreement table
  ✓ IBM Quantum Hardware certificate
  ✓ Downloadable JSON reports
  ✓ Sidebar examples (Aspirin, Phenanthrene)

New V2 features:
  + Non-blocking progress indicators
  + "Hardware-realistic final check" toggle with CI display
  + Two-output display: fast estimate + CI-equipped shot-based update
  + Model comparison dashboard
  + Enhanced architecture info
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CHECKPOINT_DIR,
    N_QUBITS,
    N_SHOTS,
    W_XGB,
    W_QML,
    ALERT_THRESHOLD,
    REFERENCE_MOLECULES,
)
from services.feature_service import FeatureService
from services.nystrom_engine import NystromEngine
from services.classical_router import ClassicalRouter
from services.quantum_kernel_service import QuantumKernelService
from services.calibration import CalibrationService
from quantum.backends import StatevectorBackend, ShotBackend
from pipeline.orchestrator import InferencePipeline
from pipeline.pipeline_config import PipelineConfig

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


# ================================================================
# PAGE CONFIGURATION
# ================================================================
st.set_page_config(
    page_title="Quantum Drug Discovery V2",
    layout="wide",
    page_icon="⚛️",
)

st.title("⚛️ Hybrid Quantum-Classical Drug Screening Platform — V2")
st.markdown("""
Welcome to the **production-grade V2** toxicity screening pipeline.

**Classical Router:** High-Throughput XGBoost (4278-d multi-fingerprint)
**Quantum Oracle:** 20-Qubit Hardware-Efficient QSVM (Nystrom kernel)
**New:** Hardware-realistic final evaluation with confidence intervals, progressive results, and enhanced validation.
""")


# ================================================================
# CACHED BACKEND ENGINE
# ================================================================
@st.cache_resource
def load_pipeline():
    """Initialize the full V2 inference pipeline."""
    ckpt = str(CHECKPOINT_DIR)

    # 1. Feature Service
    feature_svc = FeatureService()

    # 2. Classical Router (XGBoost from checkpoint)
    classical_router = ClassicalRouter.from_checkpoints(feature_svc, ckpt)

    # 3. Load selected features
    with open(os.path.join(ckpt, "selected_features.json"), "r") as f:
        selected_features = json.load(f)

    # 4. Rebuild dataset scaler (same as V1)
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
    df = pd.read_csv(url).dropna(subset=["NR-AR"])
    toxic = df[df["NR-AR"] == 1].head(250)
    safe = df[df["NR-AR"] == 0].head(250)
    train_df = pd.concat([toxic, safe]).sample(frac=1, random_state=42)
    y_train = train_df["NR-AR"].values

    X_train_raw = np.array(
        [
            feature_svc.extract_orthogonal_descriptors(s, selected_features)
            for s in train_df["smiles"]
        ]
    )
    scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    scaler.fit(X_train_raw)

    # 5. Nystrom engine (load checkpoints)
    nystrom = NystromEngine(ckpt)
    nystrom.load_checkpoints()
    K_train, K_mm_inv, diag_train = nystrom.reconstruct_kernel()

    # 6. Train SVM
    svm_model = SVC(
        kernel="precomputed",
        probability=True,
        class_weight="balanced",
        C=20.0,
    )
    svm_model.fit(K_train, y_train)

    # 7. Prepare landmarks
    m = len(nystrom.K_mm)
    landmark_idx = np.linspace(0, 499, m, dtype=int)
    landmarks_raw = np.array(
        [
            feature_svc.extract_orthogonal_descriptors(s, selected_features)
            for s in train_df.iloc[landmark_idx]["smiles"]
        ]
    )
    landmarks_scaled = np.nan_to_num(scaler.transform(landmarks_raw))

    # 8. Quantum backends
    backend_sv = StatevectorBackend()
    backend_shot = ShotBackend()

    # 9. Quantum Kernel Service
    quantum_svc = QuantumKernelService(
        backend_sv=backend_sv,
        backend_shot=backend_shot,
        nystrom_engine=nystrom,
        svm_model=svm_model,
        scaler=scaler,
        landmarks_scaled=landmarks_scaled,
        feature_service=feature_svc,
        selected_features=selected_features,
    )

    # 10. Pipeline config
    pipeline_config = PipelineConfig()

    # 11. Orchestrator
    pipeline = InferencePipeline(
        feature_service=feature_svc,
        classical_router=classical_router,
        quantum_service=quantum_svc,
        pipeline_config=pipeline_config,
    )

    return pipeline, feature_svc, train_df, m


with st.spinner("Loading Quantum Checkpoints and Initializing V2 Pipeline..."):
    pipeline, feature_svc, train_df, m = load_pipeline()


# ================================================================
# SIDEBAR
# ================================================================
st.sidebar.header("🧪 Molecule Input & Validation")
st.sidebar.markdown("**Try these SMILES:**")
st.sidebar.code("CC(=O)OC1=CC=CC=C1C(=O)O  # Aspirin (Safe)")
st.sidebar.code("C1=CC=C2C(=C1)C=CC3=CC=CC=C32  # Phenanthrene (Toxic)")

smiles_input = st.sidebar.text_input(
    "Enter custom SMILES string:", "CC(=O)OC1=CC=CC=C1C(=O)O"
)

# V2 Feature: Hardware-realistic toggle
st.sidebar.markdown("---")
enable_final_check = st.sidebar.checkbox(
    "🔬 Enable Hardware-Realistic Final Check",
    value=False,
    help="Run shot-based quantum evaluation with bootstrap CI. Takes 15-120s per molecule.",
)

n_bootstrap = st.sidebar.slider(
    "Bootstrap repeats (for CI)",
    min_value=3,
    max_value=20,
    value=5,
    disabled=not enable_final_check,
)

# Upload CSV for batch validation
st.sidebar.markdown("---")
st.sidebar.markdown("**Upload ground-truth CSV for batch validation:**")
st.sidebar.markdown("CSV must contain columns: `smiles`, `experimental` (0/1).")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])


# ================================================================
# SINGLE MOLECULE ANALYSIS
# ================================================================
if st.sidebar.button("🚀 Run Hybrid Analysis"):
    col1, col2 = st.columns(2)

    # --- CLASSICAL ROUTER ---
    with col1:
        st.subheader("💻 Classical XGBoost Router")
        st.caption("Using Morgan r2+r3 + MACCS Keys + RDKit FP + PhysChem (V2)")

        start_time = time.time()
        xgb_result = pipeline.classical.predict_xgb(smiles_input)
        xgb_prob = xgb_result["probability"]
        xgb_time = time.time() - start_time

        st.metric(label="Toxicity Probability", value=f"{xgb_prob:.2%}")
        st.write(f"⏱️ **Inference Time:** {xgb_time:.4f} seconds")
        st.progress(float(xgb_prob))

    # --- QUANTUM ORACLE ---
    with col2:
        st.subheader("⚛️ 20-Qubit Quantum Oracle")
        st.caption("Using Nystrom Approximation on Orthogonal Descriptors")

        progress_bar = st.progress(0)
        status_text = st.empty()

        def quantum_progress(step, total):
            progress_bar.progress(step / total)
            status_text.text(f"Simulating circuit {step}/{total}...")

        start_time = time.time()
        q_result = pipeline.quantum.predict(
            smiles_input,
            mode="statevector",
            progress_callback=quantum_progress,
        )
        svm_prob = q_result["probability"]
        q_time = time.time() - start_time

        status_text.text(f"✅ Complete ({q_time:.1f}s)")
        st.metric(label="Toxicity Probability", value=f"{svm_prob:.2%}")
        st.write(f"⏱️ **Inference Time:** {q_time:.2f} seconds")
        if q_result.get("cached"):
            st.info("⚡ Kernel rows served from cache")

    # ================================================================
    # HYBRID ENSEMBLE
    # ================================================================
    st.divider()
    st.subheader("🧬 Hybrid Ensemble Verdict")
    st.caption(
        "Combines Topological (XGBoost) and Quantum Phase-Space (QSVM) signals "
        "using a conservative max-alert drug safety policy"
    )

    ensemble_prob = pipeline._compute_ensemble(xgb_prob, svm_prob)

    ecol1, ecol2, ecol3 = st.columns(3)
    with ecol1:
        st.metric("XGBoost (Topology)", f"{xgb_prob:.2%}")
    with ecol2:
        st.metric("Quantum (Physics)", f"{svm_prob:.2%}")
    with ecol3:
        st.metric("Ensemble Verdict", f"{ensemble_prob:.2%}")

    st.progress(float(np.clip(ensemble_prob, 0, 1)))

    # Baseline heuristic
    baseline_score = feature_svc.baseline_rule_score(smiles_input)
    st.write("Baseline heuristic toxicity score:", f"{baseline_score:.2%}")

    # ================================================================
    # V2: SHOT-BASED FINAL CHECK (if enabled)
    # ================================================================
    shot_result = None
    if enable_final_check:
        st.divider()
        st.subheader("🔬 Hardware-Realistic Final Evaluation")
        st.caption(f"Shot-based quantum evaluation with {n_bootstrap}× bootstrap CI")

        shot_progress = st.progress(0)
        shot_status = st.empty()

        def shot_progress_cb(step, total):
            shot_progress.progress(step / total)
            shot_status.text(f"Bootstrap run {step}/{total}...")

        shot_result = pipeline.quantum.predict_with_ci(
            smiles_input,
            n_bootstrap=n_bootstrap,
            progress_callback=shot_progress_cb,
        )

        shot_status.text(f"✅ Complete ({shot_result['latency_s']:.1f}s)")

        scol1, scol2, scol3 = st.columns(3)
        with scol1:
            st.metric("Shot-Based Probability", f"{shot_result['probability']:.2%}")
        with scol2:
            st.metric("Standard Deviation", f"{shot_result['std']:.4f}")
        with scol3:
            ci_str = f"[{shot_result['ci_lower']:.2%}, {shot_result['ci_upper']:.2%}]"
            st.metric("95% Confidence Interval", ci_str)

        # Update ensemble with shot-based quantum
        ensemble_shot = pipeline._compute_ensemble(xgb_prob, shot_result["probability"])
        st.write(f"**Shot-adjusted Ensemble:** {ensemble_shot:.2%}")

    # ================================================================
    # DOWNLOADABLE REPORT
    # ================================================================
    result_detail = {
        "smiles": smiles_input,
        "xgb_prob": xgb_prob,
        "quantum_prob_statevector": svm_prob,
        "ensemble_prob": ensemble_prob,
        "baseline_rule_score": baseline_score,
        "timings": {"xgb_s": xgb_time, "quantum_s": q_time},
        "mode": "fast",
    }
    if shot_result:
        result_detail["quantum_prob_shot"] = shot_result["probability"]
        result_detail["quantum_ci"] = {
            "lower": shot_result["ci_lower"],
            "upper": shot_result["ci_upper"],
            "std": shot_result["std"],
            "n_bootstrap": n_bootstrap,
        }
        result_detail["mode"] = "full"

    st.download_button(
        "📥 Download per-molecule JSON",
        json.dumps(result_detail, indent=2),
        file_name="prediction_detail_v2.json",
    )

    if ensemble_prob > 0.5:
        st.error(f"⚠️ HIGH TOXICITY RISK — Confidence: {ensemble_prob:.1%}")
    else:
        st.success(f"✅ LOW TOXICITY RISK — Confidence: {1 - ensemble_prob:.1%} safe.")


# ================================================================
# BATCH VALIDATION
# ================================================================
if uploaded is not None:
    st.header("📊 Batch Validation")
    gt_df = pd.read_csv(uploaded)
    if "smiles" not in gt_df.columns or "experimental" not in gt_df.columns:
        st.error("CSV must contain 'smiles' and 'experimental' columns.")
    else:
        preds = []
        pbar = st.progress(0)
        total = len(gt_df)
        t0 = time.time()

        for i, row in gt_df.reset_index(drop=True).iterrows():
            s = row["smiles"]
            result = pipeline.predict_fast(s)
            result["experimental"] = float(row["experimental"])
            preds.append(result)
            pbar.progress((i + 1) / total)

        t_elapsed = time.time() - t0
        preds_df = pd.DataFrame(preds)

        st.write(f"Completed {total} predictions in {t_elapsed:.1f}s")

        # Metrics
        brier_xgb = brier_score_loss(preds_df["experimental"], preds_df["xgb_prob"])
        brier_q = brier_score_loss(preds_df["experimental"], preds_df["quantum_prob"])
        brier_ens = brier_score_loss(
            preds_df["experimental"], preds_df["ensemble_prob"]
        )
        brier_base = brier_score_loss(
            preds_df["experimental"], preds_df["baseline_score"]
        )

        st.subheader("Metrics")
        st.write(
            f"Brier score — XGBoost: {brier_xgb:.4f}, Quantum: {brier_q:.4f}, "
            f"Ensemble: {brier_ens:.4f}, Baseline: {brier_base:.4f}"
        )

        # ROC AUC
        try:
            uniq = set(preds_df["experimental"].unique())
            if uniq.issubset({0.0, 1.0, 0, 1}):
                auc_xgb = roc_auc_score(preds_df["experimental"], preds_df["xgb_prob"])
                auc_q = roc_auc_score(
                    preds_df["experimental"], preds_df["quantum_prob"]
                )
                auc_ens = roc_auc_score(
                    preds_df["experimental"], preds_df["ensemble_prob"]
                )
                st.write(
                    f"ROC AUC — XGBoost: {auc_xgb:.4f}, Quantum: {auc_q:.4f}, "
                    f"Ensemble: {auc_ens:.4f}"
                )
        except Exception:
            st.info("ROC AUC not computed: requires binary labels.")

        # Calibration curves
        st.subheader("Calibration (reliability) curves")
        fig, ax = plt.subplots(figsize=(6, 4))
        for label, col, style in [
            ("XGBoost", "xgb_prob", "-"),
            ("Quantum", "quantum_prob", "--"),
            ("Ensemble", "ensemble_prob", ":"),
            ("Baseline", "baseline_score", "-."),
        ]:
            try:
                prob_true, prob_pred = calibration_curve(
                    preds_df["experimental"],
                    preds_df[col],
                    n_bins=10,
                    strategy="uniform",
                )
                ax.plot(prob_pred, prob_true, marker="o", linestyle=style, label=label)
            except Exception:
                pass
        ax.plot([0, 1], [0, 1], "k:", label="ideal")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction positive (observed)")
        ax.legend()
        ax.set_title("Reliability Diagram")
        st.pyplot(fig)

        # Scatter plot
        st.subheader("Predicted vs Experimental (scatter)")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.scatter(
            preds_df["experimental"],
            preds_df["ensemble_prob"],
            alpha=0.7,
            label="Ensemble",
        )
        ax2.scatter(
            preds_df["experimental"],
            preds_df["xgb_prob"],
            alpha=0.5,
            label="XGBoost",
            marker="x",
        )
        ax2.scatter(
            preds_df["experimental"],
            preds_df["quantum_prob"],
            alpha=0.5,
            label="Quantum",
            marker="^",
        )
        ax2.set_xlabel("Experimental (ground truth)")
        ax2.set_ylabel("Predicted probability")
        ax2.legend()
        st.pyplot(fig2)

        # Disagreements
        preds_df["abs_diff_x_q"] = np.abs(
            preds_df["xgb_prob"] - preds_df["quantum_prob"]
        )
        top_disagree = preds_df.sort_values("abs_diff_x_q", ascending=False).head(10)
        st.subheader("Top 10 model disagreements (|XGB - Quantum|)")
        display_cols = [
            "smiles",
            "experimental",
            "xgb_prob",
            "quantum_prob",
            "ensemble_prob",
            "baseline_score",
        ]
        st.dataframe(
            top_disagree[[c for c in display_cols if c in top_disagree.columns]]
        )

        # Download report
        report = {
            "summary": {
                "n_samples": int(total),
                "brier": {
                    "xgb": float(brier_xgb),
                    "quantum": float(brier_q),
                    "ensemble": float(brier_ens),
                    "baseline": float(brier_base),
                },
            },
            "predictions": preds_df.to_dict(orient="records"),
        }
        st.download_button(
            "📥 Download validation report (JSON)",
            json.dumps(report, default=str),
            file_name="validation_report_v2.json",
        )


# ================================================================
# HARDWARE CERTIFICATE (preserved from V1)
# ================================================================
st.divider()
st.subheader("🏆 IBM Quantum Hardware Verification")
st.info("""
**Verified Execution on Physical Quantum Matter**
* **Hardware:** `ibm_fez` (156-Qubit IBM Heron r2 Processor)
* **Job ID:** `d6deb9954hss73b9lc40`
* **Hardware Fidelity:** 98.2% Diagonal Self-Similarity
* **Status:** Verified. Physical-to-Phase mapping successfully compiled and executed with tunable-coupler crosstalk mitigation.
""")


# ================================================================
# V2: ARCHITECTURE INFO
# ================================================================
with st.expander("ℹ️ V2 Architecture"):
    st.markdown("""
    **Pipeline Flow:**
    ```
    SMILES → Feature Service → Classical Router (XGBoost ≤50ms)
                              → Quantum Kernel Service (Statevector ≤3s)
                              → [Optional] Shot-Based Final Check (15-120s)
                              → Calibrated Ensemble → Result
    ```

    **Components:**
    - `FeatureService`: Unified descriptor extraction with caching
    - `ClassicalRouter`: XGBoost + future GNN classifier
    - `QuantumKernelService`: Two-mode (statevector / shot)
    - `NystromEngine`: Improved Nystrom with SVD+PSD+Cosine+Clip
    - `CalibrationService`: Platt/isotonic per-model calibration
    - `UncertaintyEstimator`: Bootstrap CI for shot-based predictions
    """)
