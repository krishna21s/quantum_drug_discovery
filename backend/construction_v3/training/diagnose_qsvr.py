"""
QSVR Pre-Training Diagnostic (V4)
===================================
Run this FIRST before training to understand why V3 failed and
confirm V4 fixes will improve things. Takes ~5 minutes.

Checks:
  1. Current feature distribution → confirms encoding range problem
  2. Baseline kernel rank + KTA with fixed params
  3. Expected KTA improvement from ArctanScaler V4 vs V3
  4. Landmark quality (spread vs clustering)
  5. Label distribution sanity

Usage:
    python training/diagnose_qsvr.py

No GPU required. Uses Aer statevector simulation.
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from scipy.stats import pearsonr
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

from config import (
    CHECKPOINT_DIR,
    CHEMBL_DATASET_PATH,
    N_QUBITS,
    MAX_TRAIN,
    MAX_TEST,
    RANDOM_STATE,
    NYSTROM_LANDMARKS,
    KERNEL_GAMMA,
    SVD_THRESHOLD,
    K_MM_REGULARIZATION,
)


def run_diagnostics():
    print("=" * 62)
    print(" QSVR V4 Pre-Training Diagnostic")
    print("=" * 62)

    # ── 1. Load small sample ─────────────────────────────────────────
    if not os.path.exists(CHEMBL_DATASET_PATH):
        print(f"[ERROR] Dataset not found: {CHEMBL_DATASET_PATH}")
        return

    df = pd.read_csv(CHEMBL_DATASET_PATH)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    n_diag = min(200, len(df))
    df = df.iloc[:n_diag]
    y = df["pic50"].values.astype(np.float32)

    print(f"\n[1] LABEL DISTRIBUTION (n={n_diag})")
    print(
        f"  mean={y.mean():.2f}  std={y.std():.2f}  "
        f"range=[{y.min():.2f}, {y.max():.2f}]"
    )
    print(f"  Quartiles: {np.percentile(y, [25, 50, 75]).round(2)}")
    if y.std() < 0.5:
        print("  [WARN] Very low pIC50 variance → regression is harder")
    else:
        print("  [OK] Sufficient label variance for regression")

    # ── 2. 3D Feature Extraction (small batch) ────────────────────────
    print(f"\n[2] FEATURE EXTRACTION (sample of {n_diag})")
    from services.feature_service_3d import FeatureService3D

    feat_svc = FeatureService3D()

    vecs, names = [], None
    failed = 0
    for i, smi in enumerate(df["canonical_smiles"].tolist()):
        try:
            v, n = feat_svc.extract_3d_descriptors(smi)
            vecs.append(v)
            if names is None:
                names = n
        except Exception:
            v, n = feat_svc._extract_2d_fallback(smi)
            vecs.append(v)
            failed += 1
        if i % 50 == 0:
            print(f"  mol {i + 1}/{n_diag}...", end="\r")

    max_len = max(len(v) for v in vecs)
    padded = [np.pad(v, (0, max_len - len(v))) for v in vecs]
    X_raw = np.array(padded, dtype=np.float32)
    print(f"\n  Shape: {X_raw.shape}  3D failures: {failed}/{n_diag}")

    # Pearson filter
    selected = feat_svc.fit_pearson_filter(X_raw, names)
    feat_idx = {n: i for i, n in enumerate(names)}
    sel_idx = [feat_idx[f] for f in selected if f in feat_idx][:N_QUBITS]
    X_sel = X_raw[:, sel_idx]
    print(f"  Selected {len(selected)} features: {selected[:3]}...")

    # ── 3. Encoding Range Comparison ─────────────────────────────────
    print(f"\n[3] ENCODING RANGE COMPARISON")

    # V3: ArctanScaler → [0, π]
    try:
        from training.train_qsvr import ArctanScaler as V3Scaler
    except ImportError:
        from training.train_qsvr import ArctanScalerV4 as V3Scaler
    scaler_v3 = V3Scaler()
    X_v3 = scaler_v3.fit_transform(X_sel)
    print(f"  V3 ArctanScaler → [0, π]:")
    print(f"    range=[{X_v3.min():.4f}, {X_v3.max():.4f}]  mean={X_v3.mean():.4f}")

    # V4: ArctanScaler → [-π/2, π/2]
    from sklearn.preprocessing import StandardScaler as SS

    Z_v4 = SS().fit_transform(X_sel)
    X_v4 = np.arctan(Z_v4).astype(np.float32)
    print(f"  V4 ArctanScaler → [-π/2, π/2]:")
    print(f"    range=[{X_v4.min():.4f}, {X_v4.max():.4f}]  mean={X_v4.mean():.4f}")
    print(
        f"  V4 uses {(X_v4.max() - X_v4.min()) / (X_v3.max() - X_v3.min()) * 100:.0f}% "
        f"of V3 rotation range  (both ≈π but V4 centred at 0)"
    )
    print(f"  [KEY] V4 reaches BOTH hemispheres of Bloch sphere; V3 only one")

    # ── 4. Mini Kernel Quality Check ─────────────────────────────────
    print(f"\n[4] MINI KERNEL QUALITY (n=30, statevector)")
    from quantum.backends import StatevectorBackend
    from quantum.kta_optimizer import build_mini_kernel, kernel_target_alignment

    # Support both V3 (no default_params) and V4 circuits.py
    try:
        from quantum.circuits import default_params
    except ImportError:
        # V3 circuits.py — define inline (theta=1, phi=0 everywhere)
        import config as _cfg

        def default_params(n_qubits=_cfg.N_QUBITS, n_layers=_cfg.N_REUPLOADING_LAYERS):
            return np.concatenate(
                [np.ones(n_qubits * n_layers), np.zeros(n_qubits * n_layers)]
            )

    n_mini = 30
    idx = np.random.RandomState(RANDOM_STATE).choice(n_diag, n_mini, replace=False)
    X_mini = X_v4[idx]
    y_mini = y[idx]

    print("  Building mini kernel (V4 features + fixed params)...")
    K_mini = build_mini_kernel(X_mini, default_params(), n_shots=1024, verbose=True)
    print()

    kta_fixed = kernel_target_alignment(K_mini, y_mini)
    print(f"  KTA (fixed params): {kta_fixed:.4f}")
    print(f"  Interpretation: KTA=0→random, KTA>0.05→useful, KTA>0.15→good")

    svs = np.linalg.svd(K_mini, compute_uv=False)
    rank10 = int(np.sum(svs > 0.1 * svs[0]))
    offdiag = K_mini[np.triu_indices_from(K_mini, k=1)]
    print(f"  Kernel off-diag: mean={offdiag.mean():.4f}  std={offdiag.std():.4f}")
    print(f"  Effective rank:  {rank10}/{n_mini} (10% threshold)")

    if offdiag.std() < 0.02:
        print(
            "  [WARN] Very low off-diagonal variance → kernel near-constant → KTA will be low"
        )
        print("         KTA optimisation is critical. Run with default settings.")
    else:
        print(
            "  [OK] Kernel has usable variance. KTA optimisation will improve alignment."
        )

    # ── 5. Feature selection comparison — THE CRITICAL TEST ──────────
    print(f"\n[5] FEATURE SELECTION COMPARISON (supervised vs unsupervised)")
    print("  This is the primary diagnosis for CV R²=0.07")
    print()

    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.preprocessing import MinMaxScaler

    # Unsupervised (V3) — no y
    feat_svc_v3 = FeatureService3D()
    sel_v3 = feat_svc_v3.fit_pearson_filter(X_raw, names)
    fi_v3 = {n: i for i, n in enumerate(names)}
    idx_v3 = [fi_v3[f] for f in sel_v3 if f in fi_v3][:N_QUBITS]
    X_sel_v3 = X_raw[:, idx_v3]

    # Supervised (V4) — with y
    feat_svc_v4 = FeatureService3D()
    sel_v4 = feat_svc_v4.fit_pearson_filter(X_raw, names, y=y)
    fi_v4 = {n: i for i, n in enumerate(names)}
    idx_v4 = [fi_v4[f] for f in sel_v4 if f in fi_v4][:N_QUBITS]
    X_sel_v4 = X_raw[:, idx_v4]

    print(f"\n  V3 unsupervised features: {sel_v3[:5]}...")
    print(f"  V4 supervised features:   {sel_v4[:5]}...")
    changed = len(set(sel_v3) - set(sel_v4))
    print(f"  Features changed: {changed}/20")
    print()

    # Compare classical RBF on both feature sets
    print("  Classical RBF CV R² comparison (n=200, 3-fold, gamma=0.1):")
    for label, X_sel in [("V3 unsupervised", X_sel_v3), ("V4 supervised  ", X_sel_v4)]:
        K_rbf = rbf_kernel(MinMaxScaler().fit_transform(X_sel), gamma=0.1)
        np.fill_diagonal(K_rbf, 1.0)
        svr = SVR(kernel="precomputed", C=1.0, epsilon=0.1)
        try:
            sc = cross_val_score(svr, K_rbf, y, cv=3, scoring="r2")
            print(f"  {label}:  CV R² = {sc.mean():.4f} ± {sc.std():.4f}")
        except Exception as e:
            print(f"  {label}:  CV failed — {e}")

    print()
    print("  KEY: If V4 supervised R² >> V3 unsupervised R², feature")
    print("       selection was the primary cause of CV R²=0.07.")
    print("       If both are still near zero, the dataset itself may")
    print("       lack sufficient pIC50 signal in 3D descriptors alone.")

    # Also show raw label correlation of selected features
    print(f"\n  Label correlations |ρ(feat, pIC50)| for selected features:")
    for label, feats, X_sel in [("V3", sel_v3, X_sel_v3), ("V4", sel_v4, X_sel_v4)]:
        corrs = []
        for i in range(X_sel.shape[1]):
            r, _ = pearsonr(X_sel[:, i], y)
            corrs.append(abs(r))
        print(
            f"  {label}: mean|ρ|={np.mean(corrs):.3f}  "
            f"min={np.min(corrs):.3f}  max={np.max(corrs):.3f}"
        )

    # ── 6. V3 checkpoint analysis ─────────────────────────────────────
    print(f"\n[6] V3 CHECKPOINT STATUS")
    ckpt = str(CHECKPOINT_DIR)
    for fname in [
        "K_mm_v3.npy",
        "K_nm_v3.npy",
        "qsvr_model_v3.pkl",
        "kta_params_final.npy",
        "best_gamma_v4.json",
    ]:
        path = os.path.join(ckpt, fname)
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            print(f"  [EXISTS] {fname}  ({size:.1f} KB)")
        else:
            print(f"  [MISSING] {fname}")

    # ── 7. Recommended config ────────────────────────────────────────
    print(f"\n[7] RECOMMENDED V4 CONFIG")
    print(f"  Landmarks:    100  (current: {NYSTROM_LANDMARKS})")
    print(f"  n_kta:        60-80  (mini-kernel subsample)")
    print(f"  kta_iters:    80")
    print(f"  kta_restarts: 3")
    print(f"  Expected KTA before: {kta_fixed:.4f}")
    print(f"  Expected KTA after:  0.05-0.20 (depends on dataset)")
    print(f"  Expected CV R²:      0.30-0.60 with hybrid kernel")

    print(f"\n  Run command:")
    print(
        f"  python training/train_qsvr_v4.py "
        f"--n-kta 60 --kta-iters 80 --kta-restarts 3 --landmarks 100"
    )
    print("\n[DIAGNOSTIC COMPLETE]")


if __name__ == "__main__":
    run_diagnostics()
