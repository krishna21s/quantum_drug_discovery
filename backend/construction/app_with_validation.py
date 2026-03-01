# save as app_with_validation.py (overwrite your current app if desired)
import os
import json
import time
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import brier_score_loss, roc_auc_score
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

RDLogger.DisableLog("rdApp.*")

# ================================================================
# MULTI-FINGERPRINT FEATURE EXTRACTOR (mirrors train_xgb_v2.py)
# ================================================================
PHYSCHEM_DESCS = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumRotatableBonds",
    "NumHAcceptors",
    "NumHDonors",
    "NumAromaticRings",
    "RingCount",
    "FractionCSP3",
    "HeavyAtomCount",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "BalabanJ",
    "BertzCT",
    "Chi0",
]


def extract_xgb_features(smiles: str) -> np.ndarray:
    """Multi-fingerprint vector: Morgan r2+r3 + MACCS + RDKit + PhysChem (4278-d)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(4278, dtype=np.float32)
    fp_m2 = np.array(
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024),
        dtype=np.float32,
    )
    fp_m3 = np.array(
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024),
        dtype=np.float32,
    )
    fp_maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
    fp_rdk = np.array(Chem.RDKFingerprint(mol, fpSize=2048), dtype=np.float32)
    desc = Descriptors.CalcMolDescriptors(mol)
    phys = np.array([float(desc.get(d, 0.0)) for d in PHYSCHEM_DESCS], dtype=np.float32)
    phys = np.nan_to_num(phys, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([fp_m2, fp_m3, fp_maccs, fp_rdk, phys])


# ================================================================
# PAGE CONFIGURATION
# ================================================================
st.set_page_config(
    page_title="Quantum Drug Discovery (with validation)", layout="wide", page_icon="⚛️"
)
st.title("⚛️ Hybrid Quantum-Classical Drug Screening Platform — Validation Edition")
st.markdown(
    """
Welcome to the production-grade toxicity screening pipeline. 
This system utilizes a **High-Throughput Classical XGBoost Router** and a **20-Qubit Hardware-Efficient Quantum SVM** Oracle for physicochemical edge cases.

**New:** upload ground-truth CSV or validate single molecules against a simple rule-based baseline and compute calibration metrics.
"""
)


# ================================================================
# CACHED BACKEND ENGINE (Loads instantly using your checkpoints)
# returns additional objects to support batch validation & reuse
# ================================================================
@st.cache_resource
def load_backend_engine():
    """Loads datasets, checkpoints, and pre-trains models for instant inference."""
    CHECKPOINT_DIR = "./checkpoints"

    if not os.path.exists(f"{CHECKPOINT_DIR}/K_mm.npy"):
        raise FileNotFoundError(
            "Checkpoints not found! Run core_engine.py first and ensure ./checkpoints exists."
        )

    # 1. Load the 20 selected orthogonal features
    with open(f"{CHECKPOINT_DIR}/selected_features.json", "r") as f:
        selected_features = json.load(f)

    # 2. Rebuild the dataset for the quantum SVM scaler
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
    df = pd.read_csv(url).dropna(subset=["NR-AR"])
    toxic = df[df["NR-AR"] == 1].head(250)
    safe = df[df["NR-AR"] == 0].head(250)
    train_df = pd.concat([toxic, safe]).sample(frac=1, random_state=42)
    y_train = train_df["NR-AR"].values

    # --- CLASSICAL XGBOOST SETUP (load from checkpoint) ---
    xgb_ckpt = f"{CHECKPOINT_DIR}/xgb_model_v2.pkl"
    xgb_sel_ckpt = f"{CHECKPOINT_DIR}/xgb_var_selector.pkl"
    if not os.path.exists(xgb_ckpt) or not os.path.exists(xgb_sel_ckpt):
        raise FileNotFoundError(
            "XGBoost V2 checkpoint not found! Run: python train_xgb_v2.py"
        )
    with open(xgb_ckpt, "rb") as f:
        xgb_model = pickle.load(f)
    with open(xgb_sel_ckpt, "rb") as f:
        xgb_var_selector = pickle.load(f)

    # --- QUANTUM 20-QUBIT SETUP ---
    def get_orthogonal_features(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(20)
        desc_dict = Descriptors.CalcMolDescriptors(mol)
        return np.array([desc_dict.get(f, 0.0) for f in selected_features])

    X_train_raw = np.array([get_orthogonal_features(s) for s in train_df["smiles"]])
    scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    scaler.fit(X_train_raw)

    # Load Nystrom Checkpoints
    K_mm = np.load(f"{CHECKPOINT_DIR}/K_mm.npy")
    K_nm = np.load(f"{CHECKPOINT_DIR}/K_nm.npy")

    # --- Robust Nystrom Reconstruction (SVD + PSD + Cosine + Clip) ---
    m = len(K_mm)

    # FIX 1a: SVD-truncated pseudoinverse
    U, s, Vt = np.linalg.svd(K_mm, full_matrices=False)
    threshold = 0.10 * s[0]
    s_inv = np.where(s > threshold, 1.0 / s, 0.0)
    K_mm_inv = Vt.T @ np.diag(s_inv) @ U.T

    K_train = K_nm @ K_mm_inv @ K_nm.T
    np.fill_diagonal(K_train, 1.0)
    K_train = (K_train + K_train.T) / 2.0

    # FIX 1b: PSD projection
    eigvals, eigvecs = np.linalg.eigh(K_train)
    eigvals = np.maximum(eigvals, 0)
    K_train = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # FIX 1c: Cosine normalization
    diag_train = np.sqrt(np.maximum(np.diag(K_train), 1e-12))
    K_train = K_train / np.outer(diag_train, diag_train)

    # FIX 1d: Clip to [0, 1]
    K_train = np.clip(K_train, 0, 1)
    np.fill_diagonal(K_train, 1.0)

    svm_model = SVC(
        kernel="precomputed", probability=True, class_weight="balanced", C=20.0
    )
    svm_model.fit(K_train, y_train)

    # Rebuild the Qiskit Circuit for live inference
    sim = AerSimulator(method="statevector")

    def compute_single_fidelity(x1, x2):
        qc = QuantumCircuit(20)
        for i in range(20):
            qc.ry(float(x1[i]), i)
        for i in range(0, 19, 2):
            qc.cx(i, i + 1)
        for i in range(1, 19, 2):
            qc.cx(i, i + 1)
        for i in range(1, 19, 2)[::-1]:
            qc.cx(i, i + 1)
        for i in range(0, 19, 2)[::-1]:
            qc.cx(i, i + 1)
        for i in range(20):
            qc.ry(-float(x2[i]), i)
        qc.measure_all()
        counts = sim.run(qc, shots=1024).result().get_counts()
        return counts.get("0" * 20, 0) / 1024.0

    # Precompute scaled landmarks (so quantum_predict can be fast)
    landmark_idx = np.linspace(0, 499, m, dtype=int)
    landmarks_raw = np.array(
        [get_orthogonal_features(s) for s in train_df.iloc[landmark_idx]["smiles"]]
    )
    landmarks_scaled = np.nan_to_num(scaler.transform(landmarks_raw))

    return (
        xgb_model,
        xgb_var_selector,
        svm_model,
        scaler,
        K_mm_inv,
        K_nm,
        np.sqrt(np.maximum(np.diag(K_train), 1e-12)),  # diag_train
        compute_single_fidelity,
        get_orthogonal_features,
        train_df,
        landmarks_scaled,
        m,
    )


with st.spinner("Loading Quantum Checkpoints and Initializing Models..."):
    (
        xgb_model,
        xgb_var_selector,
        svm_model,
        scaler,
        K_mm_inv,
        K_nm,
        diag_train,
        compute_single_fidelity,
        get_orthogonal_features,
        train_df,
        landmarks_scaled,
        m,
    ) = load_backend_engine()


# ================================================================
# Helper prediction wrappers (clean API for batch validation)
# ================================================================
def classical_predict(smiles: str):
    raw_feat = extract_xgb_features(smiles).reshape(1, -1)
    sel_feat = xgb_var_selector.transform(raw_feat)
    prob = float(xgb_model.predict_proba(sel_feat)[0][1])
    return prob


def quantum_predict(smiles: str, show_progress=False):
    # Extract & scale orthogonal features
    phys_raw = get_orthogonal_features(smiles).reshape(1, -1)
    phys_scaled = np.nan_to_num(scaler.transform(phys_raw))[0]

    # Compute K_new_m (1 x m) via simulated fidelities
    K_new_m = np.zeros((1, m))
    iterable = range(m)
    if show_progress:
        st.write(f"Simulating {m} quantum circuits on 20 qubits...")
        progress_bar = st.progress(0)
        iterable = list(range(m))
    for j in iterable:
        K_new_m[0, j] = compute_single_fidelity(phys_scaled, landmarks_scaled[j])
        if show_progress:
            progress_bar.progress((j + 1) / m)

    # Nystrom reconstruction
    K_new_train = K_new_m @ K_mm_inv @ K_nm.T

    # Cosine normalization consistent with training
    K_new_self = np.sum((K_new_m @ K_mm_inv) * K_new_m, axis=1)
    diag_new = np.sqrt(np.maximum(K_new_self, 1e-12))
    K_new_train = K_new_train / np.outer(diag_new, diag_train)
    K_new_train = np.clip(K_new_train, 0, 1)

    prob = float(svm_model.predict_proba(K_new_train)[0][1])
    return prob


def ensemble_prob_from_components(
    xgb_prob, q_prob, w_xgb=0.55, w_q=0.45, alert_threshold=0.60
):
    ensemble_avg = w_xgb * xgb_prob + w_q * q_prob
    either_flags = xgb_prob > alert_threshold or q_prob > alert_threshold
    ensemble_prob = (
        max(ensemble_avg, max(xgb_prob, q_prob) * 0.85)
        if either_flags
        else ensemble_avg
    )
    return ensemble_prob


# ================================================================
# Simple rule-based baseline (fast heuristics) for quick comparison
# ================================================================
def baseline_rule_score(smiles: str) -> float:
    """
    Returns a 0..1 heuristic toxicity score based on:
    - nitro group presence
    - high logP
    - many aromatic rings
    - many heavy atoms
    This is NOT a model; it's a quick baseline to help spot big disagreements.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    score = 0.0
    # nitro group substructure
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[N+](=O)[O-]")):
        score += 0.35
    # many aromatic rings
    try:
        narom = Descriptors.NumAromaticRings(mol)
        if narom >= 2:
            score += min(0.25, 0.08 * narom)
    except Exception:
        pass
    # heavy atoms
    hat = Descriptors.HeavyAtomCount(mol)
    if hat > 30:
        score += 0.15
    # high logP
    try:
        logp = Descriptors.MolLogP(mol)
        if logp > 3.5:
            score += min(0.25, 0.05 * (logp - 3.5))
    except Exception:
        pass
    return float(np.clip(score, 0.0, 1.0))


# ================================================================
# FRONTEND INTERFACE (keeps your sidebar examples)
# ================================================================
st.sidebar.header("Molecule Input & Validation")
st.sidebar.markdown("Try these SMILES:")
st.sidebar.code("CC(=O)OC1=CC=CC=C1C(=O)O  # Aspirin (Safe)")
st.sidebar.code("C1=CC=C2C(=C1)C=CC3=CC=CC=C32  # Phenanthrene (Toxic)")

smiles_input = st.sidebar.text_input(
    "Enter custom SMILES string:", "CC(=O)OC1=CC=CC=C1C(=O)O"
)

# Upload CSV for batch validation
st.sidebar.markdown("---")
st.sidebar.markdown("Upload ground-truth CSV for batch validation:")
st.sidebar.markdown(
    "CSV must contain columns: `smiles`, `experimental` (0-1 or binary 0/1)."
)
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Single-run: Run Hybrid Analysis (keeps your original UI but uses wrappers)
if st.sidebar.button("Run Hybrid Analysis"):
    col1, col2 = st.columns(2)

    # --- CLASSICAL ROUTER ---
    with col1:
        st.subheader("💻 Classical XGBoost Router")
        st.caption(
            "Using Morgan r2+r3 + MACCS Keys + RDKit FP + PhysChem Descriptors (V2)"
        )

        start_time = time.time()
        xgb_prob = classical_predict(smiles_input)
        xgb_time = time.time() - start_time

        st.metric(label="Toxicity Probability", value=f"{xgb_prob:.2%}")
        st.write(f"⏱️ **Inference Time:** {xgb_time:.4f} seconds")
        st.progress(float(xgb_prob))

    # --- QUANTUM ORACLE ---
    with col2:
        st.subheader("⚛️ 20-Qubit Quantum Oracle")
        st.caption("Using Nystrom Approximation on Orthogonal Descriptors")

        start_time = time.time()
        svm_prob = quantum_predict(smiles_input, show_progress=True)
        q_time = time.time() - start_time

        st.metric(label="Toxicity Probability", value=f"{svm_prob:.2%}")
        st.write(f"⏱️ **Inference Time:** {q_time:.2f} seconds")

    # ================================================================
    # HYBRID ENSEMBLE (Conservative Max-Alert Policy)
    # ================================================================
    st.divider()
    st.subheader("🧬 Hybrid Ensemble Verdict (with validation helpers)")
    st.caption(
        "Combines Topological (XGBoost) and Quantum Phase-Space (QSVM) signals "
        "using a conservative max-alert drug safety policy"
    )

    W_XGB, W_QML = 0.55, 0.45
    ensemble_prob = ensemble_prob_from_components(xgb_prob, svm_prob, W_XGB, W_QML)

    ecol1, ecol2, ecol3 = st.columns(3)
    with ecol1:
        st.metric("XGBoost (Topology)", f"{xgb_prob:.2%}")
    with ecol2:
        st.metric("Quantum (Physics)", f"{svm_prob:.2%}")
    with ecol3:
        delta_str = ""
        # In Streamlit metric delta expects a numeric; keep message in caption instead
        st.metric("Ensemble Verdict", f"{ensemble_prob:.2%}")

    st.progress(float(np.clip(ensemble_prob, 0, 1)))

    # small baseline heuristic & downloadable per-sample JSON
    baseline_score = baseline_rule_score(smiles_input)
    st.write("Baseline heuristic toxicity score:", f"{baseline_score:.2%}")

    result_detail = {
        "smiles": smiles_input,
        "xgb_prob": xgb_prob,
        "quantum_prob": svm_prob,
        "ensemble_prob": ensemble_prob,
        "baseline_rule_score": baseline_score,
        "timings": {"xgb": xgb_time, "quantum": q_time},
    }
    st.download_button(
        "Download per-molecule JSON",
        json.dumps(result_detail, indent=2),
        file_name="prediction_detail.json",
    )

    if ensemble_prob > 0.5:
        st.error(f"⚠️ HIGH TOXICITY RISK — Confidence: {ensemble_prob:.1%}")
    else:
        st.success(f"✅ LOW TOXICITY RISK — Confidence: {1 - ensemble_prob:.1%} safe.")

# ================================================================
# Batch Validation (if CSV uploaded)
# ================================================================
if uploaded is not None:
    st.header("📊 Batch Validation")
    gt_df = pd.read_csv(uploaded)
    if "smiles" not in gt_df.columns or "experimental" not in gt_df.columns:
        st.error("CSV must contain 'smiles' and 'experimental' columns.")
    else:
        # Run predictions for each row (could be slow for quantum part)
        preds = []
        pbar = st.progress(0)
        total = len(gt_df)
        t0 = time.time()
        for i, row in gt_df.reset_index(drop=True).iterrows():
            s = row["smiles"]
            x_p = classical_predict(s)
            q_p = quantum_predict(
                s, show_progress=False
            )  # batch will be serial - consider small batch sizes
            e_p = ensemble_prob_from_components(x_p, q_p, 0.55, 0.45)
            b_p = baseline_rule_score(s)
            preds.append(
                {
                    "smiles": s,
                    "experimental": float(row["experimental"]),
                    "xgb": x_p,
                    "quantum": q_p,
                    "ensemble": e_p,
                    "baseline": b_p,
                }
            )
            pbar.progress((i + 1) / total)
        t_elapsed = time.time() - t0

        preds_df = pd.DataFrame(preds)
        merged = preds_df.copy()

        st.write(f"Completed {total} predictions in {t_elapsed:.1f}s")

        # Metrics: Brier score for probabilities
        brier_xgb = brier_score_loss(merged["experimental"], merged["xgb"])
        brier_q = brier_score_loss(merged["experimental"], merged["quantum"])
        brier_ens = brier_score_loss(merged["experimental"], merged["ensemble"])
        brier_base = brier_score_loss(merged["experimental"], merged["baseline"])

        st.subheader("Metrics")
        st.write(
            f"Brier score — XGBoost: {brier_xgb:.4f}, Quantum: {brier_q:.4f}, Ensemble: {brier_ens:.4f}, Baseline: {brier_base:.4f}"
        )

        # If experimental appears binary, compute ROC AUC
        try:
            if set(merged["experimental"].unique()).issubset({0, 1}):
                auc_xgb = roc_auc_score(merged["experimental"], merged["xgb"])
                auc_q = roc_auc_score(merged["experimental"], merged["quantum"])
                auc_ens = roc_auc_score(merged["experimental"], merged["ensemble"])
                st.write(
                    f"ROC AUC — XGBoost: {auc_xgb:.4f}, Quantum: {auc_q:.4f}, Ensemble: {auc_ens:.4f}"
                )
        except Exception as e:
            st.info(
                "ROC AUC not computed: experimental column not binary or calculation failed."
            )

        st.subheader("Calibration (reliability) curves")
        from sklearn.calibration import calibration_curve

        fig, ax = plt.subplots(figsize=(6, 4))
        for label, col, style in [
            ("XGBoost", "xgb", "-"),
            ("Quantum", "quantum", "--"),
            ("Ensemble", "ensemble", ":"),
            ("Baseline", "baseline", "-."),
        ]:
            prob_true, prob_pred = calibration_curve(
                merged["experimental"], merged[col], n_bins=10, strategy="uniform"
            )
            ax.plot(prob_pred, prob_true, marker="o", linestyle=style, label=label)
        ax.plot([0, 1], [0, 1], "k:", label="ideal")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction positive (observed)")
        ax.legend()
        ax.set_title("Reliability Diagram")
        st.pyplot(fig)

        st.subheader("Predicted vs Experimental (scatter)")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.scatter(
            merged["experimental"], merged["ensemble"], alpha=0.7, label="Ensemble"
        )
        ax2.scatter(
            merged["experimental"],
            merged["xgb"],
            alpha=0.5,
            label="XGBoost",
            marker="x",
        )
        ax2.scatter(
            merged["experimental"],
            merged["quantum"],
            alpha=0.5,
            label="Quantum",
            marker="^",
        )
        ax2.set_xlabel("Experimental (ground truth)")
        ax2.set_ylabel("Predicted probability")
        ax2.legend()
        st.pyplot(fig2)

        # Show top disagreements
        merged["abs_diff_x_q"] = np.abs(merged["xgb"] - merged["quantum"])
        top_disagree = merged.sort_values("abs_diff_x_q", ascending=False).head(10)
        st.subheader("Top 10 model disagreements (|XGB - Quantum|)")
        st.dataframe(
            top_disagree[
                ["smiles", "experimental", "xgb", "quantum", "ensemble", "baseline"]
            ]
        )

        # Allow download of full report
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
            "predictions": merged.to_dict(orient="records"),
        }
        st.download_button(
            "Download validation report (JSON)",
            json.dumps(report),
            file_name="validation_report.json",
        )

# ================================================================
# Hardware certificate area remains (unchanged)
# ================================================================
st.divider()
st.subheader("🏆 IBM Quantum Hardware Verification")
st.info(
    """
**Verified Execution on Physical Quantum Matter**
* **Hardware:** `ibm_fez` (156-Qubit IBM Heron r2 Processor)
* **Job ID:** `d6deb9954hss73b9lc40`
* **Hardware Fidelity:** 98.2% Diagonal Self-Similarity
* **Status:** Verified. Physical-to-Phase mapping successfully compiled and executed with tunable-coupler crosstalk mitigation.
"""
)
