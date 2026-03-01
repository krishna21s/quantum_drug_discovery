"""
XGBoost V2 Training Script — Quantum Drug Discovery Platform
============================================================
Run this script ONCE to produce a calibrated, high-accuracy XGBoost
model that replaces the naive in-app trained model.

Improvements over V1:
  1. Multi-fingerprint feature ensemble  (Morgan r2 + r3 + MACCS + RDKit + PhysChem)
  2. Full Tox21 NR-AR dataset            (~6700 molecules, real class imbalance)
  3. scale_pos_weight for imbalance      (no forced 50/50 split)
  4. Optuna Bayesian hyperparameter search (50 trials, ROC-AUC objective)
  5. CalibratedClassifierCV (isotonic)   (true probabilities, not raw scores)
  6. Checkpoint persistence              (app.py loads model, never retrains)

Usage:
    python train_xgb_v2.py

Outputs (in ./checkpoints/):
    xgb_model_v2.pkl          — calibrated XGBoost pipeline
    xgb_feature_names.json    — ordered feature name list
    xgb_training_report.json  — AUC, precision, recall, thresholds
"""

import os
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, average_precision_score
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ================================================================
# CONFIGURATION
# ================================================================
OPTUNA_TRIALS   = 60     # Bayesian search trials
CV_FOLDS        = 5      # Stratified cross-validation folds
RANDOM_STATE    = 42
MIN_VARIANCE    = 0.01   # Variance threshold for feature pruning
# scale_pos_weight: cap at 8 — high enough to handle imbalance,
# low enough to avoid probability collapse after calibration
SPW_MAX         = 8.0

print("=" * 65)
print(" 🚀 XGBoost V2 Training Pipeline")
print(f"    Optuna trials: {OPTUNA_TRIALS}  |  CV folds: {CV_FOLDS}")
print("=" * 65)


# ================================================================
# 1. MULTI-FINGERPRINT FEATURE EXTRACTION
# ================================================================
PHYSCHEM_DESCS = [
    "MolWt", "MolLogP", "TPSA", "NumRotatableBonds",
    "NumHAcceptors", "NumHDonors", "NumAromaticRings",
    "RingCount", "FractionCSP3", "HeavyAtomCount",
    "NumAliphaticRings", "NumSaturatedRings",
    "BalabanJ", "BertzCT", "Chi0",
]

def extract_features(smiles: str) -> np.ndarray | None:
    """
    Returns a flat feature vector combining:
      - Morgan r=2 (1024-bit)
      - Morgan r=3 (1024-bit)
      - MACCS Keys (167-bit)   ← structural toxicophore alerts
      - RDKit Topological (2048-bit)
      - Physicochemical descriptors (15 values)
    Total raw: 4,278 features.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Morgan fingerprints
    fp_m2 = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024), dtype=np.float32)
    fp_m3 = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024), dtype=np.float32)

    # MACCS Keys (167 expert-curated structural keys)
    fp_maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)

    # RDKit topological fingerprint
    fp_rdk = np.array(Chem.RDKFingerprint(mol, fpSize=2048), dtype=np.float32)

    # Physicochemical descriptors
    desc_dict = Descriptors.CalcMolDescriptors(mol)
    phys = np.array(
        [float(desc_dict.get(d, 0.0)) for d in PHYSCHEM_DESCS], dtype=np.float32
    )
    phys = np.nan_to_num(phys, nan=0.0, posinf=0.0, neginf=0.0)

    return np.concatenate([fp_m2, fp_m3, fp_maccs, fp_rdk, phys])


def build_feature_names() -> list[str]:
    names  = [f"Morgan2_{i}" for i in range(1024)]
    names += [f"Morgan3_{i}" for i in range(1024)]
    names += [f"MACCS_{i}"   for i in range(167)]
    names += [f"RDKit_{i}"   for i in range(2048)]
    names += PHYSCHEM_DESCS
    return names


# ================================================================
# 2. LOAD FULL TOX21 NR-AR DATASET
# ================================================================
print("\n[1/5] Loading full Tox21 NR-AR dataset...")
url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
df  = pd.read_csv(url).dropna(subset=["NR-AR"])

print(f"      Total molecules with NR-AR labels: {len(df)}")
toxic_count = (df["NR-AR"] == 1).sum()
safe_count  = (df["NR-AR"] == 0).sum()
print(f"      Toxic (NR-AR=1): {toxic_count}  |  Safe (NR-AR=0): {safe_count}")
scale_pos_weight = safe_count / toxic_count
print(f"      scale_pos_weight = {scale_pos_weight:.2f}  (handles real imbalance)")

# Stratified train/test split (80/20)
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(
    df, test_size=0.20, random_state=RANDOM_STATE, stratify=df["NR-AR"]
)
print(f"      Train: {len(train_df)}  |  Test: {len(test_df)}")


# ================================================================
# 3. FEATURE EXTRACTION
# ================================================================
print("\n[2/5] Extracting multi-fingerprint features...")
t0 = time.time()

def safe_extract(smiles, default_size=4278):
    feat = extract_features(smiles)
    return feat if feat is not None else np.zeros(default_size, dtype=np.float32)

X_train_raw = np.array([safe_extract(s) for s in train_df["smiles"]])
y_train      = train_df["NR-AR"].values.astype(int)

X_test_raw   = np.array([safe_extract(s) for s in test_df["smiles"]])
y_test        = test_df["NR-AR"].values.astype(int)

print(f"      Feature matrix shape: {X_train_raw.shape}  ({time.time()-t0:.1f}s)")


# ================================================================
# 4. FEATURE SELECTION (Variance filter)
# ================================================================
print(f"\n[3/5] Pruning near-zero-variance features (threshold={MIN_VARIANCE})...")
var_selector = VarianceThreshold(threshold=MIN_VARIANCE)
X_train_sel  = var_selector.fit_transform(X_train_raw)
X_test_sel   = var_selector.transform(X_test_raw)

n_kept = X_train_sel.shape[1]
print(f"      Features retained: {n_kept} / {X_train_raw.shape[1]}")

# Save feature names after selection
all_names      = build_feature_names()
selected_names = [n for n, keep in zip(all_names, var_selector.get_support()) if keep]


# ================================================================
# 5. OPTUNA HYPERPARAMETER SEARCH
# ================================================================
print(f"\n[4/5] Optuna Bayesian search ({OPTUNA_TRIALS} trials, {CV_FOLDS}-fold CV)...")

def objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 1200, step=100),
        "max_depth":         trial.suggest_int("max_depth", 3, 7),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.2, 0.7),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 12),
        "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        # Cap scale_pos_weight: too high collapses probabilities toward base rate
        "scale_pos_weight":  trial.suggest_float("scale_pos_weight", 2.0, SPW_MAX),
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    clf = XGBClassifier(**params)
    cv  = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    # average_precision (PR-AUC) is far more informative than ROC-AUC under heavy imbalance
    scores = cross_val_score(clf, X_train_sel, y_train, cv=cv,
                             scoring="average_precision", n_jobs=1)
    return scores.mean()

study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

best_params = study.best_params
best_params.update({
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "tree_method": "hist",
})
print(f"\n      Best CV PR-AUC: {study.best_value:.4f}")
print(f"      Best params: {best_params}")


# ================================================================
# 6. TRAIN FINAL MODEL + SIGMOID CALIBRATION
# ================================================================
print("\n[5/5] Training final model with Platt scaling calibration (sigmoid)...")
base_xgb = XGBClassifier(**best_params)

# Use holdout calibration to avoid fold-level probability collapse.
# sigmoid (Platt scaling) is robust under class imbalance;
# isotonic overfits badly when toxic samples are <5% of each fold.
from sklearn.model_selection import train_test_split as _tts
X_fit, X_cal, y_fit, y_cal = _tts(
    X_train_sel, y_train, test_size=0.20,
    random_state=RANDOM_STATE, stratify=y_train
)
base_xgb.fit(X_fit, y_fit)
calibrated_xgb = CalibratedClassifierCV(base_xgb, method="sigmoid", cv="prefit")
calibrated_xgb.fit(X_cal, y_cal)

# Evaluate on held-out test set
y_pred_proba = calibrated_xgb.predict_proba(X_test_sel)[:, 1]
y_pred       = (y_pred_proba >= 0.5).astype(int)
test_auc     = roc_auc_score(y_test, y_pred_proba)
test_pr_auc  = average_precision_score(y_test, y_pred_proba)
report       = classification_report(y_test, y_pred, output_dict=True)

print(f"\n      Test ROC-AUC  : {test_auc:.4f}")
print(f"      Test PR-AUC   : {test_pr_auc:.4f}")
print(f"      Test Precision (toxic): {report['1']['precision']:.3f}")
print(f"      Test Recall    (toxic): {report['1']['recall']:.3f}")
print(f"      Test F1        (toxic): {report['1']['f1-score']:.3f}")

# ── Validation on reference molecules ──────────────────────────
REFERENCE_MOLS = {
    "Aspirin (Safe)":       ("CC(=O)OC1=CC=CC=C1C(=O)O", 0),
    "Phenanthrene (Toxic)": ("C1=CC=C2C(=C1)C=CC3=CC=CC=C32", 1),
    "Ibuprofen (Safe)":     ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", 0),
    "Bisphenol A (Toxic)":  ("CC(c1ccc(O)cc1)(c1ccc(O)cc1)C", 1),
    "Paracetamol (Safe)":   ("CC(=O)Nc1ccc(O)cc1", 0),
}
print("\n  ── Reference Molecule Validation ──")
for name, (smiles, true_label) in REFERENCE_MOLS.items():
    feat = safe_extract(smiles)
    feat_sel = var_selector.transform(feat.reshape(1, -1))
    prob = calibrated_xgb.predict_proba(feat_sel)[0][1]
    status = "✓" if (prob > 0.5) == bool(true_label) else "✗"
    print(f"  {status}  {name:<28} → {prob:.1%}  (true={'Toxic' if true_label else 'Safe '})")


# ================================================================
# 7. SAVE CHECKPOINTS
# ================================================================
print("\n[SAVE] Writing checkpoints...")

# Save the calibrated model
with open(f"{CHECKPOINT_DIR}/xgb_model_v2.pkl", "wb") as f:
    pickle.dump(calibrated_xgb, f)

# Save the variance selector
with open(f"{CHECKPOINT_DIR}/xgb_var_selector.pkl", "wb") as f:
    pickle.dump(var_selector, f)

# Save metadata
report_data = {
    "test_roc_auc":          round(test_auc, 4),
    "test_pr_auc":           round(test_pr_auc, 4),
    "best_cv_pr_auc":        round(study.best_value, 4),
    "best_params":           best_params,
    "calibration_method":    "sigmoid (Platt, prefit holdout)",
    "features_total":        int(X_train_raw.shape[1]),
    "features_after_filter": int(n_kept),
    "train_samples":         int(len(train_df)),
    "test_samples":          int(len(test_df)),
    "scale_pos_weight_base": float(scale_pos_weight),
    "spw_cap":               SPW_MAX,
    "toxic_train":           int((y_train == 1).sum()),
    "safe_train":            int((y_train == 0).sum()),
    "classification_report": report,
    "fingerprint_types":     ["Morgan_r2_1024", "Morgan_r3_1024", "MACCS_167", "RDKit_2048", "PhysChem_15"],
}
with open(f"{CHECKPOINT_DIR}/xgb_training_report.json", "w") as f:
    json.dump(report_data, f, indent=2)

print(f"\n  Saved: {CHECKPOINT_DIR}/xgb_model_v2.pkl")
print(f"  Saved: {CHECKPOINT_DIR}/xgb_var_selector.pkl")
print(f"  Saved: {CHECKPOINT_DIR}/xgb_training_report.json")

print("\n" + "=" * 65)
print(f" ✅ XGBoost V2 training complete!  Test AUC: {test_auc:.4f}  PR-AUC: {test_pr_auc:.4f}")
print("    Run app.py — it will now load from checkpoint.")
print("=" * 65)
