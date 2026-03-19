"""
Train QSVR — Phase B: 20-Qubit QSVR with Data Reuploading Circuit
===================================================================
Phase B upgrades:
  1. Arctan feature normalization (better Bloch sphere coverage)
  2. Batch fidelity execution (4-8x faster K_nm via single sim.run)
  3. Reduced landmarks: 50 (was 100) → 2x faster
  4. Single clean test evaluation loop (removed redundant first pass)
  5. RBF-Q kernel transform in reconstruction

Usage:
    python training/train_qsvr.py

Outputs (saved to checkpoints/):
    - qsvr_model_v3.pkl
    - qsvr_scaler_v3.pkl (or StandardScaler for arctan)
    - qsvr_landmarks_scaled_v3.npy
    - qsvr_selected_features_v3.json
    - qsvr_K_mm_inv_v3.npy
    - qsvr_diag_train_v3.npy
    - training_report_qsvr.txt
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

from config import (
    CHECKPOINT_DIR, CHEMBL_DATASET_PATH, N_QUBITS,
    MAX_TRAIN, MAX_TEST, RANDOM_STATE, NYSTROM_LANDMARKS,
    PIC50_MIN, PIC50_MAX, KERNEL_GAMMA, SVD_THRESHOLD, K_MM_REGULARIZATION
)


# ================================================================
# ARCTAN NORMALIZATION (Phase B: replaces MinMaxScaler)
# ================================================================

class ArctanScaler:
    """
    Arctan normalization: X -> arctan(z-scored X) * 2/pi, mapped to [0, pi].

    Advantages over MinMaxScaler([0,pi]):
      - Robust to outliers (arctan saturates)
      - Near-uniform distribution → maximal Bloch sphere coverage
      - Molecules with similar standard values map to similar angles
        but the distribution is spread across the full [0, pi] range
    """

    def __init__(self):
        self.std_scaler = StandardScaler()

    def fit_transform(self, X):
        Z = self.std_scaler.fit_transform(X)
        # arctan maps R → (-pi/2, pi/2), scale to (0, pi)
        return (np.arctan(Z) / np.pi + 0.5) * np.pi

    def transform(self, X):
        Z = self.std_scaler.transform(X)
        return (np.arctan(Z) / np.pi + 0.5) * np.pi


# ================================================================
# BATCH K_nm COMPUTATION (Phase B: 4-8x faster)
# ================================================================

def compute_K_nm_batch(X_train, landmarks, backend, checkpoint_dir, fname="K_nm_v3.npy"):
    """
    Compute N×m training-to-landmark kernel matrix using BATCH fidelity calls.

    Phase B speedup: for each row i, submits ALL m landmark circuits in one
    sim.run([c1,...,cm]) call instead of m separate calls.
    Reduction: N*m Python round-trips → N Python round-trips.

    Also resumes from partial checkpoint if available.
    """
    N, m = len(X_train), len(landmarks)
    path = os.path.join(checkpoint_dir, fname)

    # Load checkpoint if exists
    if os.path.exists(path):
        K = np.load(path)
        if K.shape == (N, m):
            n_done = int(np.sum(~np.any(np.isnan(K), axis=1)))
            if n_done == N:
                print(f"  [CACHED] K_nm fully loaded ({N}x{m})")
                return K
            else:
                print(f"  [RESUME] K_nm: {n_done}/{N} rows done, resuming...")
        else:
            K = np.full((N, m), np.nan)
    else:
        K = np.full((N, m), np.nan)

    t0 = time.time()
    for i in range(N):
        if not np.any(np.isnan(K[i])):   # already done
            continue

        # BATCH: all m fidelities for row i in one sim.run() call
        K[i] = backend.fidelity_batch(X_train[i], landmarks)

        if i % 10 == 0:
            np.save(path, K)   # checkpoint
            elapsed = time.time() - t0
            rows_done_since_start = i + 1
            rate = rows_done_since_start / max(elapsed, 1)
            eta_min = (N - i - 1) / rate / 60
            print(f"  K_nm row {i+1}/{N}  ETA: {eta_min:.1f} min", end="\r")

    np.save(path, K)
    print(f"\n  [DONE] K_nm ({N}x{m}) complete in {(time.time()-t0)/60:.1f} min")
    return K


# ================================================================
# BATCH K_mm COMPUTATION
# ================================================================

def compute_K_mm_batch(landmarks, backend, checkpoint_dir, fname="K_mm_v3.npy"):
    """Symmetric m×m landmark kernel using batch fidelity per row."""
    m    = len(landmarks)
    path = os.path.join(checkpoint_dir, fname)

    if os.path.exists(path):
        K = np.load(path)
        if K.shape == (m, m) and not np.any(np.isnan(K)):
            print(f"  [CACHED] K_mm loaded ({m}x{m})")
            return K

    K = np.full((m, m), np.nan)
    np.fill_diagonal(K, 1.0)

    for i in range(m):
        # Only compute upper triangle and mirror
        upper_landmarks = landmarks[i+1:]
        if len(upper_landmarks) == 0:
            continue
        fids = backend.fidelity_batch(landmarks[i], upper_landmarks)
        K[i, i+1:] = fids
        K[i+1:, i] = fids

        if i % 10 == 0:
            np.save(path, K)
            print(f"  K_mm row {i+1}/{m}", end="\r")

    np.save(path, K)
    print(f"\n  [DONE] K_mm ({m}x{m}) complete")
    return K


# ================================================================
# MAIN TRAINING PIPELINE
# ================================================================

def main():
    print("=" * 60)
    print(" QSVR Training: Phase B — 20-Qubit Data Reuploading")
    print("=" * 60)

    t_start = time.time()

    # ── STEP 1: Load ChEMBL dataset ──────────────────────────────────
    if not os.path.exists(CHEMBL_DATASET_PATH):
        print(f"\n[ERROR] Dataset not found at {CHEMBL_DATASET_PATH}")
        print("  Run train_xgb_regressor.py first.")
        sys.exit(1)

    df = pd.read_csv(CHEMBL_DATASET_PATH)
    print(f"\n[DATA] Loaded {len(df)} molecules.")
    print(f"  pIC50 range: {df['pic50'].min():.2f} - {df['pic50'].max():.2f}")

    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    n_total = min(len(df), MAX_TRAIN + MAX_TEST)
    df       = df.iloc[:n_total]
    n_train  = min(MAX_TRAIN, int(0.83 * n_total))
    df_train = df.iloc[:n_train]
    df_test  = df.iloc[n_train:]
    print(f"  Train: {len(df_train)}, Test: {len(df_test)}")

    # ── STEP 2: 3D Feature Extraction ────────────────────────────────
    from services.feature_service_3d import FeatureService3D
    feat_svc = FeatureService3D()

    print("\n[3D FEATURES] Extracting conformers and descriptors...")

    def extract_dataset(smiles_list):
        all_vecs, all_names = [], None
        failed = 0
        for i, smi in enumerate(smiles_list):
            if i % 50 == 0:
                print(f"  Molecule {i+1}/{len(smiles_list)}...", end="\r")
            try:
                vec, names = feat_svc.extract_3d_descriptors(smi)
                all_vecs.append(vec)
                if all_names is None:
                    all_names = names
            except Exception:
                vec, names = feat_svc._extract_2d_fallback(smi)
                all_vecs.append(vec)
                failed += 1
        max_len = max(len(v) for v in all_vecs)
        padded  = [np.pad(v, (0, max_len - len(v))) for v in all_vecs]
        X = np.array(padded, dtype=np.float32)
        print(f"\n  Done! Failed 3D: {failed}/{len(smiles_list)}  Shape: {X.shape}")
        return X, all_names

    X_train_raw, feat_names = extract_dataset(df_train["canonical_smiles"].tolist())
    X_test_raw, _           = extract_dataset(df_test["canonical_smiles"].tolist())
    y_train = df_train["pic50"].values.astype(np.float32)
    y_test  = df_test["pic50"].values.astype(np.float32)

    # ── STEP 3: Pearson filter → 20 orthogonal features ──────────────
    print(f"\n[PEARSON] Selecting {N_QUBITS} features from {X_train_raw.shape[1]}...")
    selected_features = feat_svc.fit_pearson_filter(X_train_raw, feat_names)
    print(f"  Selected: {selected_features[:5]}...")

    feat_idx = {n: i for i, n in enumerate(feat_names)}
    sel_idx  = [feat_idx[f] for f in selected_features if f in feat_idx]
    while len(sel_idx) < N_QUBITS:
        sel_idx.append(sel_idx[-1])
    sel_idx = sel_idx[:N_QUBITS]

    X_train_3d = X_train_raw[:, sel_idx]
    X_test_3d  = X_test_raw[:, sel_idx]

    # ── STEP 4: Phase B — Arctan Normalization ────────────────────────
    print(f"\n[SCALER] Phase B: Arctan normalization to [0, pi]")
    scaler = ArctanScaler()
    X_train_scaled = scaler.fit_transform(X_train_3d).astype(np.float32)
    X_test_scaled  = scaler.transform(X_test_3d).astype(np.float32)
    print(f"  Scaled range: [{X_train_scaled.min():.4f}, {X_train_scaled.max():.4f}]")
    print(f"  Mean: {X_train_scaled.mean():.4f}  Std: {X_train_scaled.std():.4f}")

    # ── STEP 5: Nystrom Kernel (Batch Mode) ───────────────────────────
    print(f"\n[NYSTROM] Computing quantum kernel (BATCH mode, {NYSTROM_LANDMARKS} landmarks)...")
    print(f"  Phase B: fidelity_batch() — all {NYSTROM_LANDMARKS} circuits per row in one sim.run()")

    from services.nystrom_engine import NystromEngine
    from quantum.backends import StatevectorBackend

    nystrom    = NystromEngine(checkpoint_dir=str(CHECKPOINT_DIR))
    backend_sv = StatevectorBackend()

    # Landmark selection (cached for reproducibility)
    landmarks_scaled, _ = nystrom.select_landmarks(
        X_train_scaled, m=NYSTROM_LANDMARKS, method="kmeans"
    )
    print(f"  Landmarks: {len(landmarks_scaled)}")

    # K_mm — batch mode
    print(f"\n  Computing K_mm ({NYSTROM_LANDMARKS}x{NYSTROM_LANDMARKS})...")
    K_mm = compute_K_mm_batch(
        landmarks_scaled, backend_sv, str(CHECKPOINT_DIR)
    )
    nystrom.K_mm = K_mm

    # K_nm — batch mode (the big one)
    print(f"\n  Computing K_nm ({len(X_train_scaled)}x{NYSTROM_LANDMARKS})...")
    print(f"  NOTE: {NYSTROM_LANDMARKS} circuits per row → ~{len(X_train_scaled)} sim.run() calls")
    K_nm = compute_K_nm_batch(
        X_train_scaled, landmarks_scaled, backend_sv, str(CHECKPOINT_DIR)
    )
    nystrom.K_nm = K_nm

    # ── STEP 6: Kernel Reconstruction with RBF-Q transform ───────────
    print(f"\n[KERNEL] Reconstructing with RBF-Q (gamma={KERNEL_GAMMA})...")
    K_train, K_mm_inv, diag_train = nystrom.reconstruct_kernel(
        K_mm=K_mm, K_nm=K_nm,
        kernel_gamma=KERNEL_GAMMA,
        svd_threshold=SVD_THRESHOLD,
        regularization=K_MM_REGULARIZATION
    )

    # ── STEP 7: SVR Hyperparameter Search ───────────────────────────
    print(f"\n[SVR] Grid search over C and epsilon...")
    param_grid = {
        "C":       [0.1, 1.0, 10.0, 50.0, 100.0],
        "epsilon": [0.01, 0.05, 0.1, 0.2],
    }
    grid = GridSearchCV(
        SVR(kernel="precomputed"), param_grid,
        cv=5, scoring="r2", n_jobs=-1, verbose=0
    )
    grid.fit(K_train, y_train)
    best_C   = grid.best_params_["C"]
    best_eps = grid.best_params_["epsilon"]
    print(f"  Best CV R2: {grid.best_score_:.4f}  C={best_C}  eps={best_eps}")

    # ── STEP 8: Fit final SVR ────────────────────────────────────────
    print(f"\n[SVR] Fitting final model...")
    svr = SVR(kernel="precomputed", C=best_C, epsilon=best_eps)
    svr.fit(K_train, y_train)

    # ── STEP 9: Test Evaluation (single clean loop) ──────────────────
    print(f"\n[TEST] Building test kernel ({len(X_test_scaled)}x{len(X_train_scaled)})...")
    K_test_full = np.zeros((len(X_test_scaled), len(X_train_scaled)))

    for i, x_new in enumerate(X_test_scaled):
        if i % 10 == 0:
            print(f"  Test row {i+1}/{len(X_test_scaled)}...", end="\r")

        # Batch: all 50 landmark fidelities in one sim.run()
        K_row = backend_sv.fidelity_batch(x_new, landmarks_scaled).reshape(1, -1)

        # Apply RBF-Q transform (must match training)
        K_row_t = NystromEngine.apply_rbf_transform(K_row, KERNEL_GAMMA)

        # Nystrom reconstruction
        K_new_train = K_row_t @ K_mm_inv @ (NystromEngine.apply_rbf_transform(K_nm, KERNEL_GAMMA)).T
        K_new_self  = np.sum((K_row_t @ K_mm_inv) * K_row_t, axis=1)
        diag_new    = np.sqrt(np.maximum(K_new_self, 1e-12))
        K_new_train = K_new_train / np.outer(diag_new, diag_train)
        K_new_train = np.clip(K_new_train, 0, 1)
        K_test_full[i] = K_new_train[0]

    y_pred = svr.predict(K_test_full)

    r2        = r2_score(y_test, y_pred)
    rmse      = np.sqrt(mean_squared_error(y_test, y_pred))
    pearson_r, _ = pearsonr(y_test, y_pred)

    print(f"\n[RESULTS] -----------------------------------------")
    print(f"  Test R2:      {r2:.4f}   (target > 0.65)")
    print(f"  Pearson r:    {pearson_r:.4f}")
    print(f"  Test RMSE:    {rmse:.4f}  (target < 1.0)")
    print(f"  Best SVR C:   {best_C}   eps: {best_eps}")

    # ── STEP 10: Save Checkpoints ───────────────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    with open(CHECKPOINT_DIR / "qsvr_model_v3.pkl", "wb") as f:
        pickle.dump(svr, f)

    scaler_payload = {
        "type":       "arctan",
        "std_mean":   scaler.std_scaler.mean_.tolist(),
        "std_scale":  scaler.std_scaler.scale_.tolist(),
    }
    with open(CHECKPOINT_DIR / "qsvr_scaler_v3.pkl", "wb") as f:
        pickle.dump(scaler, f)

    np.save(CHECKPOINT_DIR / "qsvr_landmarks_scaled_v3.npy", landmarks_scaled)
    np.save(CHECKPOINT_DIR / "qsvr_K_mm_inv_v3.npy", K_mm_inv)
    np.save(CHECKPOINT_DIR / "qsvr_diag_train_v3.npy", diag_train)

    with open(CHECKPOINT_DIR / "qsvr_selected_features_v3.json", "w") as f:
        json.dump(selected_features, f, indent=2)

    elapsed = (time.time() - t_start) / 60
    report = (
        f"QSVR V3 Phase B Training Report\n"
        f"Target: EGFR Lung Cancer (20-qubit Data Reuploading + CZ)\n"
        f"{'-'*40}\n"
        f"Circuit: 2-layer data reuploading + CZ ring\n"
        f"Scaler: ArctanScaler\n"
        f"Landmarks: {NYSTROM_LANDMARKS}\n"
        f"KERNEL_GAMMA: {KERNEL_GAMMA}\n"
        f"SVD_THRESHOLD: {SVD_THRESHOLD}\n"
        f"K_MM_REG: {K_MM_REGULARIZATION}\n"
        f"SVR C={best_C}, eps={best_eps}\n"
        f"{'-'*40}\n"
        f"Test R2:      {r2:.4f}\n"
        f"Pearson r:    {pearson_r:.4f}\n"
        f"Test RMSE:    {rmse:.4f}\n"
        f"{'-'*40}\n"
        f"Selected features: {json.dumps(selected_features, indent=2)}\n"
        f"Elapsed: {elapsed:.1f} min\n"
    )
    with open(CHECKPOINT_DIR / "training_report_qsvr.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n[SAVED] All checkpoints saved to {CHECKPOINT_DIR}")
    print(f"[TIME]  Total elapsed: {elapsed:.1f} min")
    print("\nQSVR Phase B training complete!")


if __name__ == "__main__":
    main()
