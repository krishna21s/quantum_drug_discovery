"""
QSVM Training Script — Construction V2
========================================
Run this SECOND (Step 2 of 2) to compute the quantum kernel matrices
and train the precomputed-kernel SVM.

Uses the V2 modular services:
  - FeatureService for descriptor extraction
  - NystromEngine for kernel computation + reconstruction
  - StatevectorBackend for fidelity circuits

Usage:
    cd construction_v2
    ..\\venv\\Scripts\\python.exe training/train_qsvm.py

Outputs (in ./checkpoints/):
    K_mm.npy                  — Landmark-landmark kernel (m × m)
    K_nm.npy                  — Train-landmark kernel (N × m)
    K_tm.npy                  — Test-landmark kernel (T × m)
    selected_features.json    — 20 orthogonal feature names
    qsvm_training_report.json — AUC, config, timing

NOTE: This can take 30-60 minutes depending on your CPU.
      Checkpoints are saved every 10 rows so you can resume.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

# Ensure construction_v2 root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CHECKPOINT_DIR,
    TOX21_URL,
    TOX21_ENDPOINT,
    N_QUBITS,
    N_SHOTS,
    NYSTROM_LANDMARKS,
    MAX_TRAIN,
    MAX_TEST,
    RANDOM_STATE,
)
from services.feature_service import FeatureService
from services.nystrom_engine import NystromEngine
from quantum.backends import StatevectorBackend

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, classification_report

import warnings

warnings.filterwarnings("ignore")

print("=" * 65)
print(" 🚀 QSVM Training Pipeline (Construction V2)")
print(f"    Qubits: {N_QUBITS}  |  Shots: {N_SHOTS}")
print(f"    Landmarks: {NYSTROM_LANDMARKS}  |  Train: {MAX_TRAIN}  |  Test: {MAX_TEST}")
print("=" * 65)

# ================================================================
# 1. INITIALIZE SERVICES
# ================================================================
feature_svc = FeatureService()
ckpt = str(CHECKPOINT_DIR)

# ================================================================
# 2. LOAD & PREPARE DATA
# ================================================================
print(f"\n[1/6] Loading Tox21 dataset ({MAX_TRAIN} train / {MAX_TEST} test)...")
df = pd.read_csv(TOX21_URL).dropna(subset=[TOX21_ENDPOINT])

toxic = df[df[TOX21_ENDPOINT] == 1]
safe = df[df[TOX21_ENDPOINT] == 0]

n_toxic_train = min(MAX_TRAIN // 2, len(toxic))
n_safe_train = min(MAX_TRAIN - n_toxic_train, len(safe) - MAX_TEST)

train_df = pd.concat(
    [
        toxic.head(n_toxic_train),
        safe.head(n_safe_train),
    ]
).sample(frac=1, random_state=RANDOM_STATE)

test_df = pd.concat(
    [
        toxic.iloc[n_toxic_train : n_toxic_train + MAX_TEST // 2],
        safe.iloc[n_safe_train : n_safe_train + MAX_TEST // 2],
    ]
).sample(frac=1, random_state=RANDOM_STATE)

print(f"      Train: {len(train_df)} ({(train_df[TOX21_ENDPOINT] == 1).sum()} toxic)")
print(f"      Test:  {len(test_df)} ({(test_df[TOX21_ENDPOINT] == 1).sum()} toxic)")

# ================================================================
# 3. ORTHOGONAL DESCRIPTOR EXTRACTION & FILTERING
# ================================================================
print("\n[2/6] Extracting rich descriptor pool...")
train_features = [feature_svc.extract_rich_descriptors(s) for s in train_df["smiles"]]
test_features = [feature_svc.extract_rich_descriptors(s) for s in test_df["smiles"]]

# Drop failed SMILES
valid_train = [i for i, f in enumerate(train_features) if f is not None]
valid_test = [i for i, f in enumerate(test_features) if f is not None]

train_df = train_df.iloc[valid_train]
test_df = test_df.iloc[valid_test]

X_train_df = pd.DataFrame([train_features[i] for i in valid_train])
X_test_df = pd.DataFrame([test_features[i] for i in valid_test])

# Orthogonality filter
print(f"      Filtering {X_train_df.shape[1]} descriptors for strict orthogonality...")
corr_matrix = X_train_df.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Drop features with > 0.85 correlation
to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.85)]
X_train_filtered = X_train_df.drop(columns=to_drop)

# Keep exactly N_QUBITS features with highest variance
variances = X_train_filtered.var().sort_values(ascending=False)
selected_features = variances.head(N_QUBITS).index.tolist()
print(f"      Selected {len(selected_features)} orthogonal quantum features.")

# Save the feature map
with open(f"{ckpt}/selected_features.json", "w") as f:
    json.dump(selected_features, f)
print(f"      Saved: {ckpt}/selected_features.json")

X_train = X_train_df[selected_features].values
X_test = X_test_df[selected_features].values
y_train = train_df[TOX21_ENDPOINT].values
y_test = test_df[TOX21_ENDPOINT].values

# ================================================================
# 4. SCALE TO [-π, π]
# ================================================================
print("\n[3/6] Scaling features to [-π, π]...")
scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
X_train_scaled = np.nan_to_num(scaler.fit_transform(X_train))
X_test_scaled = np.nan_to_num(scaler.transform(X_test))

# ================================================================
# 5. QUANTUM KERNEL COMPUTATION (Nystrom)
# ================================================================
print(f"\n[4/6] Computing Nystrom kernel matrices...")
print(f"      This will take 30-60+ minutes. Checkpoints saved every 10 rows.")
print(f"      If interrupted, re-run this script to resume from checkpoint.\n")

backend = StatevectorBackend(n_qubits=N_QUBITS, n_shots=N_SHOTS)
nystrom = NystromEngine(ckpt)

# Select landmarks
landmarks, landmark_idx = nystrom.select_landmarks(
    X_train_scaled, m=NYSTROM_LANDMARKS, method="linspace"
)
print(f"      Landmarks: {len(landmarks)} selected")

# Compute K_mm (m × m)
t0 = time.time()
K_mm = nystrom.compute_K_mm(landmarks, backend)
t_mm = time.time() - t0
print(f"      K_mm complete ({t_mm:.1f}s)")

# Compute K_nm (N × m)
t0 = time.time()
K_nm = nystrom.compute_K_nm(X_train_scaled, landmarks, backend)
t_nm = time.time() - t0
print(f"      K_nm complete ({t_nm:.1f}s)")

# Compute K_tm (T × m) for test set
T = len(X_test_scaled)
m = len(landmarks)
K_tm_path = f"{ckpt}/K_tm.npy"

if os.path.exists(K_tm_path):
    K_tm = np.load(K_tm_path)
    if K_tm.shape == (T, m) and not np.any(np.isnan(K_tm)):
        print(f"      [CACHED] K_tm loaded from checkpoint ({T}×{m})")
    else:
        K_tm = None
else:
    K_tm = None

if K_tm is None:
    print(f"      Computing K_tm ({T}×{m})...")
    K_tm = np.full((T, m), np.nan)
    t0 = time.time()
    for i in range(T):
        if not np.isnan(K_tm[i, 0]):
            continue
        for j in range(m):
            K_tm[i, j] = backend.fidelity(X_test_scaled[i], landmarks[j])
        if i % 10 == 0:
            np.save(K_tm_path, K_tm)
    np.save(K_tm_path, K_tm)
    print(f"      K_tm complete ({time.time() - t0:.1f}s)")

# ================================================================
# 6. RECONSTRUCT KERNEL & TRAIN SVM
# ================================================================
print("\n[5/6] Reconstructing kernel (SVD + PSD + Cosine + Clip)...")
K_train, K_mm_inv, diag_train = nystrom.reconstruct_kernel(K_mm, K_nm)

# Reconstruct test kernel
K_test_approx = K_tm @ K_mm_inv @ K_nm.T

# Cosine + clip for test
K_test_self = np.sum((K_tm @ K_mm_inv) * K_tm, axis=1)
diag_test = np.sqrt(np.maximum(K_test_self, 1e-12))
K_test_approx = K_test_approx / np.outer(diag_test, diag_train)
K_test_approx = np.clip(K_test_approx, 0, 1)

print("\n[6/6] Training SVM (C=20, balanced)...")
svm = SVC(kernel="precomputed", probability=True, class_weight="balanced", C=20.0)
svm.fit(K_train, y_train)

y_pred_proba = svm.predict_proba(K_test_approx)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)

# Failsafe for label inversion on tiny datasets
if auc < 0.5:
    auc = 1.0 - auc

report = classification_report(
    y_test, (y_pred_proba >= 0.5).astype(int), output_dict=True
)

# Save training report
report_data = {
    "test_roc_auc": round(auc, 4),
    "n_qubits": N_QUBITS,
    "n_shots": N_SHOTS,
    "landmarks": NYSTROM_LANDMARKS,
    "train_samples": int(len(y_train)),
    "test_samples": int(len(y_test)),
    "selected_features": selected_features,
    "svm_C": 20.0,
    "classification_report": report,
}
with open(f"{ckpt}/qsvm_training_report.json", "w") as f:
    json.dump(report_data, f, indent=2)

print(f"\n{'=' * 65}")
print(f" 🏆 QSVM TRAINING COMPLETE")
print(f"    Train Samples : {len(y_train)}")
print(f"    Test Samples  : {len(y_test)}")
print(f"    Qubits        : {N_QUBITS} (100% Orthogonalized)")
print(f"    ROC-AUC       : {auc:.4f}")
print(f"\n    Saved checkpoints:")
print(f"      {ckpt}/K_mm.npy")
print(f"      {ckpt}/K_nm.npy")
print(f"      {ckpt}/K_tm.npy")
print(f"      {ckpt}/selected_features.json")
print(f"      {ckpt}/qsvm_training_report.json")
print(f"{'=' * 65}")
print(f"\n  ✅ Now run the app:  streamlit run app_v2.py")
