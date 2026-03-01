"""
XGBoost V2 Training Script — Construction V2
==============================================
Run this FIRST (Step 1 of 2) to produce a calibrated, high-accuracy
XGBoost model using the V2 FeatureService.

Usage:
    cd construction_v2
    ..\\venv\\Scripts\\python.exe training/train_xgb_v2.py

Outputs (in ./checkpoints/):
    xgb_model_v2.pkl          — calibrated XGBoost pipeline
    xgb_var_selector.pkl      — variance threshold selector
    xgb_training_report.json  — AUC, precision, recall, config
"""

import os
import sys
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd

# Ensure construction_v2 root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CHECKPOINT_DIR,
    TOX21_URL,
    TOX21_ENDPOINT,
    RANDOM_STATE,
    OPTUNA_TRIALS,
    CV_FOLDS,
    MIN_VARIANCE,
    SPW_MAX,
    PHYSCHEM_DESCS,
    MULTI_FP_DIM,
    REFERENCE_MOLECULES,
)
from services.feature_service import FeatureService

from sklearn.feature_selection import VarianceThreshold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    average_precision_score,
)
from xgboost import XGBClassifier
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

print("=" * 65)
print(" 🚀 XGBoost V2 Training Pipeline (Construction V2)")
print(f"    Optuna trials: {OPTUNA_TRIALS}  |  CV folds: {CV_FOLDS}")
print("=" * 65)

# ================================================================
# 1. INITIALIZE FEATURE SERVICE
# ================================================================
feature_svc = FeatureService()

# ================================================================
# 2. LOAD FULL TOX21 NR-AR DATASET
# ================================================================
print("\n[1/5] Loading full Tox21 NR-AR dataset...")
df = pd.read_csv(TOX21_URL).dropna(subset=[TOX21_ENDPOINT])

print(f"      Total molecules with {TOX21_ENDPOINT} labels: {len(df)}")
toxic_count = (df[TOX21_ENDPOINT] == 1).sum()
safe_count = (df[TOX21_ENDPOINT] == 0).sum()
print(f"      Toxic: {toxic_count}  |  Safe: {safe_count}")
scale_pos_weight = safe_count / toxic_count
print(f"      scale_pos_weight = {scale_pos_weight:.2f}")

# Stratified train/test split (80/20)
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df[TOX21_ENDPOINT],
)
print(f"      Train: {len(train_df)}  |  Test: {len(test_df)}")

# ================================================================
# 3. FEATURE EXTRACTION (using V2 FeatureService)
# ================================================================
print("\n[2/5] Extracting multi-fingerprint features via FeatureService...")
t0 = time.time()

X_train_raw = np.array(
    [feature_svc.extract_multi_fingerprint(s) for s in train_df["smiles"]]
)
y_train = train_df[TOX21_ENDPOINT].values.astype(int)

X_test_raw = np.array(
    [feature_svc.extract_multi_fingerprint(s) for s in test_df["smiles"]]
)
y_test = test_df[TOX21_ENDPOINT].values.astype(int)

print(f"      Feature matrix shape: {X_train_raw.shape}  ({time.time() - t0:.1f}s)")

# ================================================================
# 4. FEATURE SELECTION (Variance filter)
# ================================================================
print(f"\n[3/5] Pruning near-zero-variance features (threshold={MIN_VARIANCE})...")
var_selector = VarianceThreshold(threshold=MIN_VARIANCE)
X_train_sel = var_selector.fit_transform(X_train_raw)
X_test_sel = var_selector.transform(X_test_raw)

n_kept = X_train_sel.shape[1]
print(f"      Features retained: {n_kept} / {X_train_raw.shape[1]}")

# ================================================================
# 5. OPTUNA HYPERPARAMETER SEARCH
# ================================================================
print(f"\n[4/5] Optuna Bayesian search ({OPTUNA_TRIALS} trials, {CV_FOLDS}-fold CV)...")


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.7),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 2.0, SPW_MAX),
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    clf = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        clf,
        X_train_sel,
        y_train,
        cv=cv,
        scoring="average_precision",
        n_jobs=1,
    )
    return scores.mean()


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
)
study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

best_params = study.best_params
best_params.update(
    {
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist",
    }
)
print(f"\n      Best CV PR-AUC: {study.best_value:.4f}")
print(f"      Best params: {best_params}")

# ================================================================
# 6. TRAIN FINAL MODEL + SIGMOID CALIBRATION
# ================================================================
print("\n[5/5] Training final model with Platt scaling calibration...")
base_xgb = XGBClassifier(**best_params)

X_fit, X_cal, y_fit, y_cal = train_test_split(
    X_train_sel,
    y_train,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_train,
)
base_xgb.fit(X_fit, y_fit)
calibrated_xgb = CalibratedClassifierCV(base_xgb, method="sigmoid", cv="prefit")
calibrated_xgb.fit(X_cal, y_cal)

# Evaluate
y_pred_proba = calibrated_xgb.predict_proba(X_test_sel)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)
test_auc = roc_auc_score(y_test, y_pred_proba)
test_pr_auc = average_precision_score(y_test, y_pred_proba)
report = classification_report(y_test, y_pred, output_dict=True)

print(f"\n      Test ROC-AUC  : {test_auc:.4f}")
print(f"      Test PR-AUC   : {test_pr_auc:.4f}")
print(f"      Test Precision (toxic): {report['1']['precision']:.3f}")
print(f"      Test Recall    (toxic): {report['1']['recall']:.3f}")
print(f"      Test F1        (toxic): {report['1']['f1-score']:.3f}")

# Reference molecule validation
print("\n  ── Reference Molecule Validation ──")
for name, (smiles, true_label) in REFERENCE_MOLECULES.items():
    feat = feature_svc.extract_multi_fingerprint(smiles)
    feat_sel = var_selector.transform(feat.reshape(1, -1))
    prob = calibrated_xgb.predict_proba(feat_sel)[0][1]
    status = "✓" if (prob > 0.5) == bool(true_label) else "✗"
    print(
        f"  {status}  {name:<28} → {prob:.1%}  (true={'Toxic' if true_label else 'Safe '})"
    )

# ================================================================
# 7. SAVE CHECKPOINTS
# ================================================================
print("\n[SAVE] Writing checkpoints...")
ckpt = str(CHECKPOINT_DIR)

with open(f"{ckpt}/xgb_model_v2.pkl", "wb") as f:
    pickle.dump(calibrated_xgb, f)

with open(f"{ckpt}/xgb_var_selector.pkl", "wb") as f:
    pickle.dump(var_selector, f)

report_data = {
    "test_roc_auc": round(test_auc, 4),
    "test_pr_auc": round(test_pr_auc, 4),
    "best_cv_pr_auc": round(study.best_value, 4),
    "best_params": best_params,
    "calibration_method": "sigmoid (Platt, prefit holdout)",
    "features_total": int(X_train_raw.shape[1]),
    "features_after_filter": int(n_kept),
    "train_samples": int(len(train_df)),
    "test_samples": int(len(test_df)),
    "classification_report": report,
    "fingerprint_types": [
        "Morgan_r2_1024",
        "Morgan_r3_1024",
        "MACCS_167",
        "RDKit_2048",
        "PhysChem_15",
    ],
}
with open(f"{ckpt}/xgb_training_report.json", "w") as f:
    json.dump(report_data, f, indent=2)

print(f"\n  Saved: {ckpt}/xgb_model_v2.pkl")
print(f"  Saved: {ckpt}/xgb_var_selector.pkl")
print(f"  Saved: {ckpt}/xgb_training_report.json")

print("\n" + "=" * 65)
print(
    f" ✅ XGBoost V2 training complete!  Test AUC: {test_auc:.4f}  PR-AUC: {test_pr_auc:.4f}"
)
print("=" * 65)
