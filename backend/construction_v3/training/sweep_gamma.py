"""
Quick gamma sweep using cached K_mm/K_nm — tests if raw fidelity has useful signal.
Runs in ~30 seconds total (no quantum simulation needed).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

from config import (
    CHECKPOINT_DIR, CHEMBL_DATASET_PATH,
    MAX_TRAIN, MAX_TEST, RANDOM_STATE, SVD_THRESHOLD, K_MM_REGULARIZATION
)
from services.nystrom_engine import NystromEngine

# Load cached kernels
K_mm_raw = np.load(CHECKPOINT_DIR / "K_mm_v3.npy")
K_nm_raw = np.load(CHECKPOINT_DIR / "K_nm_v3.npy")

# Load labels (same split as training)
df = pd.read_csv(CHEMBL_DATASET_PATH)
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
n_total = min(len(df), MAX_TRAIN + MAX_TEST)
df = df.iloc[:n_total]
n_train = min(MAX_TRAIN, int(0.83 * n_total))
y_train = df.iloc[:n_train]["pic50"].values.astype(np.float32)

print(f"K_mm: {K_mm_raw.shape}, K_nm: {K_nm_raw.shape}")
print(f"y_train: {len(y_train)}, range: [{y_train.min():.1f}, {y_train.max():.1f}]")

# Raw fidelity diagnostics
offdiag = K_mm_raw[np.triu_indices_from(K_mm_raw, k=1)]
print(f"\nRaw K_mm off-diag: mean={offdiag.mean():.4f} std={offdiag.std():.4f} max={offdiag.max():.4f}")
print(f"Raw K_nm: mean={K_nm_raw.mean():.4f}  max={K_nm_raw.max():.4f}")
print(f"Raw K_nm nonzero fraction: {(K_nm_raw > 0.001).mean():.4f}")

# Sweep gamma values
gammas = [0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
nystrom = NystromEngine(checkpoint_dir=str(CHECKPOINT_DIR))

print(f"\n{'gamma':>8} | {'CV R2':>8} | {'Train R2':>8} | K_train mean | K_train std | K_train max")
print("-" * 85)

best_gamma = 0
best_cv_r2 = -999

for gamma in gammas:
    # Transform kernels
    if gamma > 0:
        K_mm_t = NystromEngine.apply_rbf_transform(K_mm_raw, gamma)
        K_nm_t = NystromEngine.apply_rbf_transform(K_nm_raw, gamma)
    else:
        K_mm_t = K_mm_raw.copy()
        K_nm_t = K_nm_raw.copy()
    
    # Regularize K_mm
    m = len(K_mm_t)
    K_mm_reg = K_mm_t + K_MM_REGULARIZATION * np.eye(m)
    
    # SVD pseudoinverse
    U, s, Vt = np.linalg.svd(K_mm_reg, full_matrices=False)
    threshold = SVD_THRESHOLD * s[0]
    s_inv = np.where(s > threshold, 1.0 / s, 0.0)
    K_mm_inv = Vt.T @ np.diag(s_inv) @ U.T
    
    # Nystrom reconstruction
    K_train = K_nm_t @ K_mm_inv @ K_nm_t.T
    np.fill_diagonal(K_train, 1.0)
    K_train = (K_train + K_train.T) / 2.0
    
    # PSD projection
    eigvals, eigvecs = np.linalg.eigh(K_train)
    eigvals = np.maximum(eigvals, 0)
    K_train = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Cosine normalization
    diag = np.sqrt(np.maximum(np.diag(K_train), 1e-12))
    K_train = K_train / np.outer(diag, diag)
    K_train = np.clip(K_train, 0, 1)
    np.fill_diagonal(K_train, 1.0)
    
    # Quick SVR + CV
    try:
        svr = SVR(kernel="precomputed", C=10.0, epsilon=0.1)
        y_cv = cross_val_predict(svr, K_train, y_train, cv=5)
        cv_r2 = r2_score(y_train, y_cv)
        
        svr.fit(K_train, y_train)
        train_r2 = r2_score(y_train, svr.predict(K_train))
        
        offdiag_train = K_train[np.triu_indices_from(K_train, k=1)]
        
        print(f"{gamma:8.2f} | {cv_r2:8.4f} | {train_r2:8.4f} | {offdiag_train.mean():12.6f} | {offdiag_train.std():11.6f} | {offdiag_train.max():11.4f}")
        
        if cv_r2 > best_cv_r2:
            best_cv_r2 = cv_r2
            best_gamma = gamma
    except Exception as e:
        print(f"{gamma:8.2f} | ERROR: {e}")

print(f"\nBest gamma: {best_gamma}  Best CV R2: {best_cv_r2:.4f}")
print("\nConclusion: If ALL gammas give CV R2 < 0, the 20-qubit kernel lacks")
print("discriminative power. Solution: reduce qubits to 6-8 for useful fidelities.")
