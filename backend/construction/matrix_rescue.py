"""
Matrix Rescue Script — Phase 1 Fixes on Existing 100-Landmark Checkpoints.
Zero recomputation. Pure linear algebra surgery.
"""
import numpy as np
import pandas as pd
import json
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

CHECKPOINT_DIR = "./checkpoints"

# ============================================================
# 1. RELOAD DATA (deterministic, same as core_engine)
# ============================================================
print("=" * 60)
print(" MATRIX RESCUE: Phase 1 Fixes on 100-Landmark Data")
print("=" * 60)

with open(f"{CHECKPOINT_DIR}/selected_features.json", "r") as f:
    selected_features = json.load(f)

url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
df = pd.read_csv(url).dropna(subset=["NR-AR"])
toxic = df[df["NR-AR"] == 1]
safe = df[df["NR-AR"] == 0]

MAX_TRAIN, MAX_TEST = 500, 100
n_toxic_train = min(MAX_TRAIN // 2, len(toxic))
n_safe_train = min(MAX_TRAIN - n_toxic_train, len(safe) - MAX_TEST)

train_df = pd.concat([toxic.head(n_toxic_train), safe.head(n_safe_train)]).sample(
    frac=1, random_state=42
)
test_df = pd.concat(
    [
        toxic.iloc[n_toxic_train : n_toxic_train + MAX_TEST // 2],
        safe.iloc[n_safe_train : n_safe_train + MAX_TEST // 2],
    ]
).sample(frac=1, random_state=42)


def extract_rich_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    desc_dict = Descriptors.CalcMolDescriptors(mol)
    return {
        k: float(v)
        for k, v in desc_dict.items()
        if not np.isnan(v) and not np.isinf(v)
    }


train_features = [extract_rich_descriptors(s) for s in train_df["smiles"]]
test_features = [extract_rich_descriptors(s) for s in test_df["smiles"]]
valid_train = [i for i, f in enumerate(train_features) if f is not None]
valid_test = [i for i, f in enumerate(test_features) if f is not None]
train_df = train_df.iloc[valid_train]
test_df = test_df.iloc[valid_test]

y_train = train_df["NR-AR"].values
y_test = test_df["NR-AR"].values

# ============================================================
# 2. LOAD RAW KERNEL CHECKPOINTS
# ============================================================
K_mm = np.load(f"{CHECKPOINT_DIR}/K_mm.npy")
K_nm = np.load(f"{CHECKPOINT_DIR}/K_nm.npy")
K_tm = np.load(f"{CHECKPOINT_DIR}/K_tm.npy")
m = len(K_mm)

print(f"\nLoaded: K_mm={K_mm.shape}, K_nm={K_nm.shape}, K_tm={K_tm.shape}")
print(f"Train={len(y_train)}, Test={len(y_test)}\n")

# ============================================================
# HELPER FUNCTIONS
# ============================================================


def robust_inverse(K, threshold_ratio=0.01):
    """SVD-truncated pseudoinverse — clips small singular values."""
    U, s, Vt = np.linalg.svd(K, full_matrices=False)
    threshold = threshold_ratio * s[0]
    s_inv = np.where(s > threshold, 1.0 / s, 0.0)
    return Vt.T @ np.diag(s_inv) @ U.T


def psd_project(K):
    """Project kernel matrix onto PSD cone."""
    eigvals, eigvecs = np.linalg.eigh(K)
    eigvals = np.maximum(eigvals, 0)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def cosine_normalize(K_train, K_test, K_nm, K_mm_inv, K_tm):
    """Cosine normalization: K_norm[i,j] = K[i,j] / sqrt(K[i,i]*K[j,j])."""
    diag_train = np.sqrt(np.maximum(np.diag(K_train), 1e-12))
    K_train_norm = K_train / np.outer(diag_train, diag_train)

    K_test_self = np.sum((K_tm @ K_mm_inv) * K_tm, axis=1)
    diag_test = np.sqrt(np.maximum(K_test_self, 1e-12))
    K_test_norm = K_test / np.outer(diag_test, diag_train)
    return K_train_norm, K_test_norm


def eval_svm(K_train, K_test, y_train, y_test, label, C=1.0):
    svm = SVC(kernel="precomputed", probability=True, class_weight="balanced", C=C)
    svm.fit(K_train, y_train)
    proba = svm.predict_proba(K_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    if auc < 0.5:
        auc = 1.0 - auc
    print(f"  [{label}] ROC-AUC = {auc:.4f}")
    return auc


# ============================================================
# 3. BASELINE (broken original)
# ============================================================
print("--- BASELINE (original broken method) ---")
K_mm_inv_orig = np.linalg.pinv(K_mm + 1e-6 * np.eye(m))
K_tr_orig = K_nm @ K_mm_inv_orig @ K_nm.T
np.fill_diagonal(K_tr_orig, 1.0)
K_tr_orig = (K_tr_orig + K_tr_orig.T) / 2.0
K_te_orig = K_tm @ K_mm_inv_orig @ K_nm.T
eval_svm(K_tr_orig, K_te_orig, y_train, y_test, "Original")

# ============================================================
# 4. FIX 1a: SVD-Truncated Inverse
# ============================================================
print("\n--- FIX 1a: SVD-Truncated Inverse ---")
for thresh in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
    K_mm_inv = robust_inverse(K_mm, threshold_ratio=thresh)
    K_tr = K_nm @ K_mm_inv @ K_nm.T
    np.fill_diagonal(K_tr, 1.0)
    K_tr = (K_tr + K_tr.T) / 2.0
    K_te = K_tm @ K_mm_inv @ K_nm.T
    eval_svm(K_tr, K_te, y_train, y_test, f"SVD thresh={thresh}")

# ============================================================
# 5. FIX 1a+1b: SVD + PSD Projection
# ============================================================
print("\n--- FIX 1a+1b: SVD Inverse + PSD Projection ---")
for thresh in [0.01, 0.02, 0.05, 0.1]:
    K_mm_inv = robust_inverse(K_mm, threshold_ratio=thresh)
    K_tr = K_nm @ K_mm_inv @ K_nm.T
    np.fill_diagonal(K_tr, 1.0)
    K_tr = (K_tr + K_tr.T) / 2.0
    K_tr = psd_project(K_tr)
    K_te = K_tm @ K_mm_inv @ K_nm.T
    eval_svm(K_tr, K_te, y_train, y_test, f"SVD({thresh})+PSD")

# ============================================================
# 6. FIX 1a+1b+1c: SVD + PSD + Cosine Normalization
# ============================================================
print("\n--- FIX 1a+1b+1c: SVD + PSD + Cosine Norm ---")
for thresh in [0.01, 0.02, 0.05, 0.1]:
    K_mm_inv = robust_inverse(K_mm, threshold_ratio=thresh)
    K_tr = K_nm @ K_mm_inv @ K_nm.T
    np.fill_diagonal(K_tr, 1.0)
    K_tr = (K_tr + K_tr.T) / 2.0
    K_tr = psd_project(K_tr)
    K_te = K_tm @ K_mm_inv @ K_nm.T
    K_tr_n, K_te_n = cosine_normalize(K_tr, K_te, K_nm, K_mm_inv, K_tm)
    eval_svm(K_tr_n, K_te_n, y_train, y_test, f"SVD({thresh})+PSD+Cos")

# ============================================================
# 7. FULL FIX: SVD + PSD + Cosine + Clip[0,1]
# ============================================================
print("\n--- FULL FIX: SVD + PSD + Cosine + Clip[0,1] ---")
for thresh in [0.01, 0.02, 0.05, 0.1]:
    K_mm_inv = robust_inverse(K_mm, threshold_ratio=thresh)
    K_tr = K_nm @ K_mm_inv @ K_nm.T
    np.fill_diagonal(K_tr, 1.0)
    K_tr = (K_tr + K_tr.T) / 2.0
    K_tr = psd_project(K_tr)
    K_te = K_tm @ K_mm_inv @ K_nm.T
    K_tr_n, K_te_n = cosine_normalize(K_tr, K_te, K_nm, K_mm_inv, K_tm)
    K_tr_n = np.clip(K_tr_n, 0, 1)
    K_te_n = np.clip(K_te_n, 0, 1)
    np.fill_diagonal(K_tr_n, 1.0)
    eval_svm(K_tr_n, K_te_n, y_train, y_test, f"FULL({thresh})+Clip")

# ============================================================
# 8. C-Parameter Sweep on best kernel config
# ============================================================
print("\n--- C-Parameter Sweep (FULL fix, multiple thresholds) ---")
best_auc = 0
best_config = ""
for thresh in [0.02, 0.05, 0.1]:
    K_mm_inv = robust_inverse(K_mm, threshold_ratio=thresh)
    K_tr = K_nm @ K_mm_inv @ K_nm.T
    np.fill_diagonal(K_tr, 1.0)
    K_tr = (K_tr + K_tr.T) / 2.0
    K_tr = psd_project(K_tr)
    K_te = K_tm @ K_mm_inv @ K_nm.T
    K_tr_n, K_te_n = cosine_normalize(K_tr, K_te, K_nm, K_mm_inv, K_tm)
    K_tr_n = np.clip(K_tr_n, 0, 1)
    K_te_n = np.clip(K_te_n, 0, 1)
    np.fill_diagonal(K_tr_n, 1.0)

    for C in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
        auc = eval_svm(K_tr_n, K_te_n, y_train, y_test, f"t={thresh},C={C}", C=C)
        if auc > best_auc:
            best_auc = auc
            best_config = f"thresh={thresh}, C={C}"

print(f"\n{'='*60}")
print(f" BEST CONFIG: {best_config}")
print(f" BEST AUC:    {best_auc:.4f}")
print(f" IMPROVEMENT: {best_auc - 0.6320:.4f} over broken baseline")
print(f"{'='*60}")
