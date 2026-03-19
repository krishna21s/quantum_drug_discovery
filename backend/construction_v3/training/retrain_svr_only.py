"""
Retrain SVR Only — Phase A Fix (No Quantum Circuits Needed)
=============================================================
Loads the K_mm and K_nm checkpoint files from the Kaggle training run,
applies the RBF-Q kernel transformation, and refits the SVR model.

This script runs in ~30 seconds on CPU — no quantum simulation needed.

Usage:
    python training/retrain_svr_only.py

Inputs (from checkpoints_v3_final/):
    - K_mm_v3.npy, K_nm_v3.npy        (raw fidelity kernels)
    - egfr_chembl_ic50.csv             (pIC50 labels)
    - qsvr_selected_features_v3.json   (feature names)
    - qsvr_scaler_v3.pkl               (feature scaler)
    - qsvr_landmarks_scaled_v3.npy     (landmarks)

Outputs (saved to checkpoints/):
    - qsvr_model_v3.pkl                (retrained SVR)
    - qsvr_K_mm_inv_v3.npy             (new K_mm_inv)
    - qsvr_diag_train_v3.npy           (new diag_train)
    - qsvr_K_nm_transformed_v3.npy     (transformed K_nm for inference)
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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

from config import (
    CHECKPOINT_DIR, CHECKPOINTS_FINAL_DIR, CHEMBL_DATASET_PATH,
    MAX_TRAIN, MAX_TEST, RANDOM_STATE, NYSTROM_LANDMARKS,
    KERNEL_GAMMA, SVD_THRESHOLD, K_MM_REGULARIZATION,
    N_QUBITS, PIC50_MIN, PIC50_MAX
)


def main():
    print("=" * 60)
    print(" Phase A: Retrain SVR with RBF-Q Kernel Transform")
    print(" (No quantum circuits — uses existing checkpoints)")
    print("=" * 60)

    t0 = time.time()

    # ----------------------------------------------------------------
    # STEP 1: Load kernel matrices from Kaggle checkpoints
    # ----------------------------------------------------------------
    src = CHECKPOINTS_FINAL_DIR
    print(f"\n[LOAD] Source checkpoint dir: {src}")

    K_mm_raw = np.load(src / "K_mm_v3.npy")
    K_nm_raw = np.load(src / "K_nm_v3.npy")
    landmarks = np.load(src / "qsvr_landmarks_scaled_v3.npy")
    with open(src / "qsvr_selected_features_v3.json") as f:
        selected_features = json.load(f)
    with open(src / "qsvr_scaler_v3.pkl", "rb") as f:
        scaler = pickle.load(f)

    print(f"  K_mm: {K_mm_raw.shape}")
    print(f"  K_nm: {K_nm_raw.shape}")
    print(f"  Landmarks: {landmarks.shape}")
    print(f"  Selected features: {len(selected_features)}")

    # Analyze raw fidelity distribution
    K_mm_offdiag = K_mm_raw[np.triu_indices_from(K_mm_raw, k=1)]
    print(f"\n[ANALYSIS] Raw fidelity K_mm (off-diagonal):")
    print(f"  min={K_mm_offdiag.min():.4f}  max={K_mm_offdiag.max():.4f}")
    print(f"  mean={K_mm_offdiag.mean():.4f}  std={K_mm_offdiag.std():.4f}")

    K_nm_flat = K_nm_raw.flatten()
    print(f"  Raw K_nm: min={K_nm_flat.min():.4f}  max={K_nm_flat.max():.4f}  mean={K_nm_flat.mean():.4f}")

    raw_svs = np.linalg.svd(K_mm_raw, compute_uv=False)
    raw_rank = int(np.sum(raw_svs > 0.1 * raw_svs[0]))
    print(f"  Raw K_mm effective rank (10% cutoff): {raw_rank}/100")

    # ----------------------------------------------------------------
    # STEP 2: Apply RBF-Q Transform
    # ----------------------------------------------------------------
    print(f"\n[TRANSFORM] Applying RBF-Q: exp(-{KERNEL_GAMMA} * (1 - K))")

    from services.nystrom_engine import NystromEngine
    nystrom = NystromEngine(checkpoint_dir=str(CHECKPOINT_DIR))

    K_mm_t = nystrom.apply_rbf_transform(K_mm_raw, KERNEL_GAMMA)
    K_nm_t = nystrom.apply_rbf_transform(K_nm_raw, KERNEL_GAMMA)

    # Analyze transformed kernel
    K_mm_t_offdiag = K_mm_t[np.triu_indices_from(K_mm_t, k=1)]
    print(f"  Transformed K_mm (off-diagonal):")
    print(f"    min={K_mm_t_offdiag.min():.4f}  max={K_mm_t_offdiag.max():.4f}")
    print(f"    mean={K_mm_t_offdiag.mean():.4f}  std={K_mm_t_offdiag.std():.4f}")

    trans_svs = np.linalg.svd(K_mm_t, compute_uv=False)
    trans_rank = int(np.sum(trans_svs > 0.1 * trans_svs[0]))
    print(f"  Transformed K_mm effective rank: {raw_rank} -> {trans_rank}")

    # ----------------------------------------------------------------
    # STEP 3: Reconstruct Kernel (with Tikhonov + lower SVD threshold)
    # ----------------------------------------------------------------
    print(f"\n[RECONSTRUCT] Nystrom reconstruction with:")
    print(f"  SVD threshold: {SVD_THRESHOLD}")
    print(f"  Tikhonov lambda: {K_MM_REGULARIZATION}")

    # Set raw kernels on nystrom engine so reconstruct can use transformed
    nystrom.K_mm = K_mm_raw
    nystrom.K_nm = K_nm_raw

    K_train, K_mm_inv, diag_train = nystrom.reconstruct_kernel(
        K_mm=K_mm_raw, K_nm=K_nm_raw,
        kernel_gamma=KERNEL_GAMMA,
        svd_threshold=SVD_THRESHOLD,
        regularization=K_MM_REGULARIZATION
    )

    print(f"  K_train shape: {K_train.shape}")
    print(f"  K_train range: [{K_train.min():.4f}, {K_train.max():.4f}]")
    print(f"  K_train mean off-diag: {K_train[np.triu_indices_from(K_train,k=1)].mean():.4f}")

    # ----------------------------------------------------------------
    # STEP 4: Load pIC50 labels
    # ----------------------------------------------------------------
    print("\n[DATA] Loading pIC50 labels...")
    df = pd.read_csv(src / "egfr_chembl_ic50.csv")
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    n_total = min(len(df), MAX_TRAIN + MAX_TEST)
    df = df.iloc[:n_total]
    n_train = min(MAX_TRAIN, int(0.83 * n_total))
    df_train = df.iloc[:n_train]
    df_test = df.iloc[n_train:]

    y_train = df_train["pic50"].values.astype(np.float32)
    y_test = df_test["pic50"].values.astype(np.float32)
    print(f"  Train: {len(y_train)}, Test: {len(y_test)}")

    # Verify shapes match
    assert K_nm_raw.shape[0] == len(y_train), (
        f"K_nm rows ({K_nm_raw.shape[0]}) != train samples ({len(y_train)})"
    )

    # ----------------------------------------------------------------
    # STEP 5: Grid search over SVR hyperparameters
    # ----------------------------------------------------------------
    print("\n[SVR] Grid search over C and epsilon...")

    param_grid = {
        "C":       [0.1, 1.0, 10.0, 50.0, 100.0],
        "epsilon": [0.01, 0.05, 0.1, 0.2, 0.5],
    }

    svr_base = SVR(kernel="precomputed")
    grid = GridSearchCV(
        svr_base, param_grid,
        cv=5, scoring="r2", n_jobs=-1, verbose=0
    )
    grid.fit(K_train, y_train)

    best_C   = grid.best_params_["C"]
    best_eps = grid.best_params_["epsilon"]
    best_cv_r2 = grid.best_score_
    print(f"  Best CV R2: {best_cv_r2:.4f}")
    print(f"  Best C={best_C}, epsilon={best_eps}")

    # ----------------------------------------------------------------
    # STEP 6: Fit final SVR
    # ----------------------------------------------------------------
    print("\n[SVR] Fitting final SVR with best params...")
    svr = SVR(kernel="precomputed", C=best_C, epsilon=best_eps)
    svr.fit(K_train, y_train)

    # Training set performance
    y_train_pred = svr.predict(K_train)
    train_r2   = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    print(f"  Train R2:  {train_r2:.4f}")
    print(f"  Train RMSE: {train_rmse:.4f}")

    # ----------------------------------------------------------------
    # STEP 7: Test Set Evaluation
    # ----------------------------------------------------------------
    print("\n[TEST] Building test kernel from existing K_nm transform logic...")

    # For test evaluation, we need to compute K_test via 3D features
    # But in Phase A we don't have the test quantum kernel rows,
    # so we use a train-validation split instead
    from sklearn.model_selection import cross_val_predict

    y_cv_pred = cross_val_predict(
        SVR(kernel="precomputed", C=best_C, epsilon=best_eps),
        K_train, y_train, cv=5
    )
    cv_r2   = r2_score(y_train, y_cv_pred)
    cv_rmse = np.sqrt(mean_squared_error(y_train, y_cv_pred))
    cv_pearson, _ = pearsonr(y_train, y_cv_pred)

    print(f"\n[RESULTS] -----------------------------------------")
    print(f"  Cross-Val R2:       {cv_r2:.4f}   (target > 0.65)")
    print(f"  Cross-Val Pearson:  {cv_pearson:.4f}")
    print(f"  Cross-Val RMSE:     {cv_rmse:.4f}  (target < 1.0)")
    print(f"  Train R2:           {train_r2:.4f}")

    # ----------------------------------------------------------------
    # STEP 8: Save Checkpoints
    # ----------------------------------------------------------------
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Save SVR model
    with open(CHECKPOINT_DIR / "qsvr_model_v3.pkl", "wb") as f:
        pickle.dump(svr, f)

    # Save transformed kernel artifacts for inference
    np.save(CHECKPOINT_DIR / "qsvr_K_mm_inv_v3.npy", K_mm_inv)
    np.save(CHECKPOINT_DIR / "qsvr_diag_train_v3.npy", diag_train)

    # Save K_nm transformed for inference
    np.save(CHECKPOINT_DIR / "K_nm_v3.npy", K_nm_t)

    # Copy over scaler, landmarks, features, raw K_mm
    np.save(CHECKPOINT_DIR / "qsvr_landmarks_scaled_v3.npy", landmarks)
    np.save(CHECKPOINT_DIR / "K_mm_v3.npy", K_mm_raw)
    with open(CHECKPOINT_DIR / "qsvr_scaler_v3.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(CHECKPOINT_DIR / "qsvr_selected_features_v3.json", "w") as f:
        json.dump(selected_features, f, indent=2)

    # Training report
    elapsed = time.time() - t0
    report = (
        f"QSVR V3 Phase A Retrain Report\n"
        f"Target: EGFR Lung Cancer (20-qubit Quantum SVR)\n"
        f"{'-'*40}\n"
        f"Method: RBF-Q kernel transform (no retraining of quantum circuits)\n"
        f"KERNEL_GAMMA: {KERNEL_GAMMA}\n"
        f"SVD_THRESHOLD: {SVD_THRESHOLD}\n"
        f"K_MM_REGULARIZATION: {K_MM_REGULARIZATION}\n"
        f"{'-'*40}\n"
        f"Raw K_mm effective rank: {raw_rank}/100\n"
        f"Transformed K_mm rank:   {trans_rank}/100\n"
        f"SVR C={best_C}, epsilon={best_eps}\n"
        f"{'-'*40}\n"
        f"Cross-Val R2:       {cv_r2:.4f}\n"
        f"Cross-Val Pearson:  {cv_pearson:.4f}\n"
        f"Cross-Val RMSE:     {cv_rmse:.4f}\n"
        f"Train R2:           {train_r2:.4f}\n"
        f"{'-'*40}\n"
        f"Elapsed: {elapsed:.1f}s\n"
    )
    with open(CHECKPOINT_DIR / "training_report_qsvr.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n[SAVED] All QSVR checkpoints saved to {CHECKPOINT_DIR}")
    print(f"[TIME] Total elapsed: {elapsed:.1f}s")
    print("\nDone! Phase A retrain complete!")


if __name__ == "__main__":
    main()
