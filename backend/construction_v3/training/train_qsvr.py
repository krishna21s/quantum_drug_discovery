"""
Train QSVR — 20-Qubit Quantum Support Vector Regressor for pIC50
=================================================================
Trains the quantum kernel SVR using the 3D conformer-based 20-feature
vectors (one per qubit). Runs Nystrom kernel approximation + SVR.

Uses data from EGFR ChEMBL (downloaded by train_xgb_regressor.py first).

Usage:
    python training/train_qsvr.py

Outputs (saved to ../checkpoints/):
    - qsvr_model_v3.pkl         (fitted SVR with precomputed kernel)
    - qsvr_scaler_v3.pkl        (MinMaxScaler for 3D features)
    - qsvr_landmarks_scaled_v3.npy
    - qsvr_selected_features_v3.json
    - training_report_qsvr.txt
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

from config import (
    CHECKPOINT_DIR, CHEMBL_DATASET_PATH, N_QUBITS,
    MAX_TRAIN, MAX_TEST, RANDOM_STATE, NYSTROM_LANDMARKS,
    PIC50_MIN, PIC50_MAX
)


def main():
    print("=" * 60)
    print(" QSVR Training: 20-Qubit Binding Affinity Regression (V3)")
    print("=" * 60)

    # ----------------------------------------------------------------
    # STEP 1: Load ChEMBL dataset (must run train_xgb_regressor.py first)
    # ----------------------------------------------------------------
    if not os.path.exists(CHEMBL_DATASET_PATH):
        print(f"\n[ERROR] Dataset not found at {CHEMBL_DATASET_PATH}")
        print("  Run: python training/train_xgb_regressor.py first to download it.")
        sys.exit(1)

    df = pd.read_csv(CHEMBL_DATASET_PATH)
    print(f"\n[DATA] Loaded {len(df)} molecules from cache.")
    print(f"  pIC50 range: {df['pic50'].min():.2f} – {df['pic50'].max():.2f}")

    # Sample for QSVR (smaller set due to kernel computation cost)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    n_total = min(len(df), MAX_TRAIN + MAX_TEST)
    df       = df.iloc[:n_total]
    n_train  = min(MAX_TRAIN, int(0.83 * n_total))
    df_train = df.iloc[:n_train]
    df_test  = df.iloc[n_train:]

    print(f"  Train: {len(df_train)}, Test: {len(df_test)}")

    # ----------------------------------------------------------------
    # STEP 2: 3D Feature Extraction
    # ----------------------------------------------------------------
    from services.feature_service_3d import FeatureService3D
    feat_svc = FeatureService3D()

    print("\n[3D FEATURES] Generating 3D conformers + extracting descriptors...")
    print("  This may take several minutes (ETKDG v3 + MMFF94 per molecule)...")

    def extract_for_dataset(smiles_list):
        all_vecs, all_names = [], None
        failed = 0
        for i, smi in enumerate(smiles_list):
            if i % 50 == 0:
                print(f"    Processing molecule {i+1}/{len(smiles_list)}...", end="\r")
            try:
                vec, names = feat_svc.extract_3d_descriptors(smi)
                all_vecs.append(vec)
                if all_names is None:
                    all_names = names
            except Exception:
                # try fallback
                vec, names = feat_svc._extract_2d_fallback(smi)
                all_vecs.append(vec)
                failed += 1

        # Pad to same length
        max_len = max(len(v) for v in all_vecs)
        padded  = [np.pad(v, (0, max_len - len(v))) for v in all_vecs]
        X = np.array(padded, dtype=np.float32)
        print(f"\n    Done! Failed 3D: {failed}/{len(smiles_list)}  Shape: {X.shape}")
        return X, all_names

    X_train_raw, feat_names = extract_for_dataset(df_train["canonical_smiles"].tolist())
    X_test_raw, _           = extract_for_dataset(df_test["canonical_smiles"].tolist())
    y_train = df_train["pic50"].values.astype(np.float32)
    y_test  = df_test["pic50"].values.astype(np.float32)

    # ----------------------------------------------------------------
    # STEP 3: Pearson orthogonality filter → 20 features for 20 qubits
    # ----------------------------------------------------------------
    print(f"\n[PEARSON] Selecting {N_QUBITS} orthogonal features from {X_train_raw.shape[1]}...")
    selected_features = feat_svc.fit_pearson_filter(X_train_raw, feat_names)

    # Extract the 20-dim vectors
    feat_name_idx = {n: i for i, n in enumerate(feat_names)}
    sel_idx       = [feat_name_idx[f] for f in selected_features if f in feat_name_idx]
    # Pad if necessary
    while len(sel_idx) < N_QUBITS:
        sel_idx.append(sel_idx[-1])
    sel_idx = sel_idx[:N_QUBITS]

    X_train_3d = X_train_raw[:, sel_idx]
    X_test_3d  = X_test_raw[:, sel_idx]
    print(f"  Selected features (first 5): {selected_features[:5]}")

    # ----------------------------------------------------------------
    # STEP 4: Scale to [0, π] for RY qubit encoding
    # ----------------------------------------------------------------
    scaler = MinMaxScaler(feature_range=(0.0, 3.14159))
    X_train_scaled = scaler.fit_transform(X_train_3d).astype(np.float32)
    X_test_scaled  = scaler.transform(X_test_3d).astype(np.float32)
    print(f"\n[SCALER] Features scaled to [0, π] for qubit RY angles.")

    # ----------------------------------------------------------------
    # STEP 5: Nystrom Kernel Computation
    # ----------------------------------------------------------------
    print(f"\n[NYSTROM] Computing quantum kernel matrix...")
    print("  ⚠️  This is the compute-intensive step (~10–60 min for 500 molecules).")
    print("      Kernel rows are checkpointed every 10 rows for resumability.\n")

    from services.nystrom_engine import NystromEngine
    from quantum.backends import StatevectorBackend

    nystrom   = NystromEngine(checkpoint_dir=str(CHECKPOINT_DIR))
    backend_sv = StatevectorBackend()

    # Select landmarks from training set
    landmarks_scaled, _ = nystrom.select_landmarks(
        X_train_scaled, m=NYSTROM_LANDMARKS, method="kmeans"
    )
    print(f"  Selected {len(landmarks_scaled)} landmarks via k-means.")

    # Compute K_mm (landmark × landmark)
    print(f"\n  Computing K_mm ({NYSTROM_LANDMARKS}×{NYSTROM_LANDMARKS})...")
    K_mm = nystrom.compute_K_mm(landmarks_scaled, backend_sv)

    # Compute K_nm (train × landmark)
    print(f"\n  Computing K_nm ({len(X_train_scaled)}×{NYSTROM_LANDMARKS})...")
    K_nm = nystrom.compute_K_nm(X_train_scaled, landmarks_scaled, backend_sv)

    # Reconstruct full kernel
    K_train, K_mm_inv, diag_train = nystrom.reconstruct_kernel()

    # ----------------------------------------------------------------
    # STEP 6: SVR Training on Quantum Kernel
    # ----------------------------------------------------------------
    print(f"\n[SVR] Fitting SVR(kernel='precomputed') on {K_train.shape[0]}×{K_train.shape[0]} kernel...")
    svr = SVR(kernel="precomputed", C=10.0, epsilon=0.1, gamma="scale")
    svr.fit(K_train, y_train)
    print("  SVR fitted!")

    # ----------------------------------------------------------------
    # STEP 7: Test Set Evaluation
    # ----------------------------------------------------------------
    print("\n[EVALUATION] Computing test kernel rows...")
    K_test_rows = np.zeros((len(X_test_scaled), NYSTROM_LANDMARKS))
    for i, x_new in enumerate(X_test_scaled):
        if i % 10 == 0:
            print(f"  Test row {i+1}/{len(X_test_scaled)}...", end="\r")
        K_row           = nystrom.compute_single_kernel_row(x_new, landmarks_scaled, backend_sv)
        K_test_rows[i]  = nystrom.predict_pic50_from_kernel_row.__func__(
            nystrom, K_row, K_mm_inv, K_nm, diag_train, svr
        )

    # Actually build the full test kernel for SVR prediction
    # (precomputed kernel needs N_test × N_train)
    print("\n  Building full test kernel...")
    K_test_full = np.zeros((len(X_test_scaled), len(X_train_scaled)))
    for i, x_new in enumerate(X_test_scaled):
        K_row             = nystrom.compute_single_kernel_row(x_new, landmarks_scaled, backend_sv)
        K_new_train       = K_row @ K_mm_inv @ K_nm.T
        K_new_self        = np.sum((K_row @ K_mm_inv) * K_row, axis=1)
        diag_new          = np.sqrt(np.maximum(K_new_self, 1e-12))
        K_new_train       = K_new_train / np.outer(diag_new, diag_train)
        K_new_train       = np.clip(K_new_train, 0, 1)
        K_test_full[i]    = K_new_train[0]

    y_pred = svr.predict(K_test_full)

    r2         = r2_score(y_test, y_pred)
    rmse       = np.sqrt(mean_squared_error(y_test, y_pred))
    pearson_r, _ = pearsonr(y_test, y_pred)

    print(f"\n[RESULTS] ─────────────────────────────────")
    print(f"  Test R²:    {r2:.4f}   (target > 0.65)")
    print(f"  Pearson r:  {pearson_r:.4f}")
    print(f"  RMSE:       {rmse:.4f}  (target < 1.0)")

    # ----------------------------------------------------------------
    # STEP 8: Save Checkpoints
    # ----------------------------------------------------------------
    svr_path       = CHECKPOINT_DIR / "qsvr_model_v3.pkl"
    scaler_path    = CHECKPOINT_DIR / "qsvr_scaler_v3.pkl"
    landmarks_path = CHECKPOINT_DIR / "qsvr_landmarks_scaled_v3.npy"
    feats_path     = CHECKPOINT_DIR / "qsvr_selected_features_v3.json"

    with open(svr_path, "wb") as f:
        pickle.dump(svr, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    np.save(landmarks_path, landmarks_scaled)
    with open(feats_path, "w") as f:
        json.dump(selected_features, f, indent=2)

    # Save K_mm_inv and diag_train for inference
    np.save(CHECKPOINT_DIR / "qsvr_K_mm_inv_v3.npy", K_mm_inv)
    np.save(CHECKPOINT_DIR / "qsvr_diag_train_v3.npy", diag_train)

    # Training report
    report = (
        f"QSVR V3 Training Report\n"
        f"Target: EGFR Lung Cancer (20-qubit Quantum SVR)\n"
        f"{'─'*40}\n"
        f"Train molecules: {len(df_train)}\n"
        f"Test molecules:  {len(df_test)}\n"
        f"Qubits:          {N_QUBITS}\n"
        f"Nystrom landmarks: {NYSTROM_LANDMARKS}\n"
        f"{'─'*40}\n"
        f"Test R²:         {r2:.4f}\n"
        f"Pearson r:       {pearson_r:.4f}\n"
        f"RMSE:            {rmse:.4f}\n"
        f"{'─'*40}\n"
        f"Selected features: {json.dumps(selected_features, indent=2)}\n"
    )
    with open(CHECKPOINT_DIR / "training_report_qsvr.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n[SAVED] All QSVR checkpoints saved to {CHECKPOINT_DIR}")
    print("\n✅ QSVR training complete!")


if __name__ == "__main__":
    main()
