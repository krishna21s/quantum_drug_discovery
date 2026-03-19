"""
Train XGBRegressor for pIC50 (Binding Affinity) Prediction
============================================================
Downloads EGFR ChEMBL IC50 data, converts to pIC50, extracts 2D
multi-fingerprint features, and trains an XGBRegressor with Optuna
hyperparameter optimization.

Usage:
    python training/train_xgb_regressor.py

Outputs (saved to ../checkpoints/):
    - xgb_regressor_v3.pkl
    - xgb_var_selector_v3.pkl
    - selected_features_v3.json  (feature names)
    - training_report_xgb.txt
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle
import numpy as np
import pandas as pd
import requests
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

from config import (
    CHECKPOINT_DIR, CHEMBL_TARGET_ID, CHEMBL_DATASET_PATH,
    MAX_TRAIN, MAX_TEST, RANDOM_STATE, OPTUNA_TRIALS, CV_FOLDS,
    MIN_VARIANCE, PIC50_MIN, PIC50_MAX, PHYSCHEM_DESCS
)

import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ================================================================
# STEP 1: DATA DOWNLOAD & pIC50 CONVERSION
# ================================================================

def download_egfr_chembl(n_pages: int = 20, page_size: int = 500) -> pd.DataFrame:
    """
    Download EGFR IC50 data from ChEMBL REST API and convert to pIC50.
    Caches to CHEMBL_DATASET_PATH.
    """
    if os.path.exists(CHEMBL_DATASET_PATH):
        print(f"[DATA] Loading cached dataset from {CHEMBL_DATASET_PATH}")
        return pd.read_csv(CHEMBL_DATASET_PATH)

    print(f"[DATA] Downloading EGFR IC50 data from ChEMBL...")
    all_records = []
    offset      = 0

    for page in range(n_pages):
        url = (
            f"https://www.ebi.ac.uk/chembl/api/data/activity.json"
            f"?target_chembl_id={CHEMBL_TARGET_ID}"
            f"&standard_type=IC50"
            f"&standard_relation=%3D"
            f"&standard_units=nM"
            f"&pchembl_value__isnull=false"
            f"&limit={page_size}&offset={offset}"
        )
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data     = r.json()
            records  = data.get("activities", [])
            if not records:
                break
            all_records.extend(records)
            offset += page_size
            total   = data.get("page_meta", {}).get("total_count", "?")
            print(f"  Page {page+1}: {len(all_records)}/{total} records")
            if len(all_records) >= int(data.get("page_meta", {}).get("total_count", 0)):
                break
        except Exception as e:
            print(f"  [WARNING] Page {page+1} failed: {e}")
            break

    if not all_records:
        raise RuntimeError(
            "No ChEMBL data downloaded. Check internet connection."
        )

    df = pd.DataFrame(all_records)

    # Keep essential columns
    keep_cols = [
        "canonical_smiles", "pchembl_value", "standard_value",
        "molecule_chembl_id", "activity_id"
    ]
    # Flatten nested molecule dict if present
    if "molecule_structures" in df.columns:
        df["canonical_smiles"] = df["molecule_structures"].apply(
            lambda x: x.get("canonical_smiles") if isinstance(x, dict) else None
        )

    available = [c for c in keep_cols if c in df.columns]
    df = df[available].copy()

    # pIC50 calculation: pIC50 = -log10(IC50 in mol/L) = pChEMBL value
    df = df.rename(columns={"pchembl_value": "pic50"})
    df["pic50"] = pd.to_numeric(df["pic50"], errors="coerce")
    df          = df.dropna(subset=["canonical_smiles", "pic50"])
    df          = df[(df["pic50"] >= PIC50_MIN) & (df["pic50"] <= PIC50_MAX)]
    df          = df.drop_duplicates(subset=["canonical_smiles"])
    df          = df.reset_index(drop=True)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    df.to_csv(CHEMBL_DATASET_PATH, index=False)
    print(f"[DATA] Saved {len(df)} records to {CHEMBL_DATASET_PATH}")

    return df


# ================================================================
# STEP 2: FEATURE EXTRACTION
# ================================================================

def extract_features(smiles_list: list) -> np.ndarray:
    """
    Extract 2D multi-fingerprint + PhysChem features for all molecules.
    Morgan r2(1024) + Morgan r3(1024) + MACCS(167) + RDKit(2048) + PhysChem(10)
    = 4273 total features.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint
    from rdkit.ML.Descriptors import MoleculeDescriptors

    calc   = MoleculeDescriptors.MolecularDescriptorCalculator(PHYSCHEM_DESCS)
    rows   = []
    failed = 0

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(np.zeros(4273))
            failed += 1
            continue

        try:
            fp_m2   = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
            fp_m3   = list(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024))
            fp_maccs = list(MACCSkeys.GenMACCSKeys(mol))
            fp_rdk  = list(RDKFingerprint(mol, maxPath=5, fpSize=2048))
            phys    = [float(v) if np.isfinite(float(v)) else 0.0
                       for v in calc.CalcDescriptors(mol)]
            row = fp_m2 + fp_m3 + fp_maccs + fp_rdk + phys
        except Exception:
            row = [0.0] * 4273

        rows.append(row)

    print(f"  Feature extraction: {failed}/{len(smiles_list)} molecules failed (set to zeros)")
    return np.array(rows, dtype=np.float32)


# ================================================================
# STEP 3: OPTUNA HYPERPARAMETER OPTIMIZATION
# ================================================================

def optimize_xgb(X_train, y_train):
    """Optuna-based XGBRegressor hyperparameter search (CV R²)."""

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 300, 2000),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 20),
            "random_state":      RANDOM_STATE,
            "tree_method":       "hist",
            "device":            "cpu",
            "eval_metric":       "rmse",
        }
        model = xgb.XGBRegressor(**params)
        kf    = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        r2s   = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1)
        return r2s.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

    best_params = study.best_params
    best_params.update({"random_state": RANDOM_STATE, "tree_method": "hist", "device": "cpu"})
    print(f"\n[OPTUNA] Best R² = {study.best_value:.4f}")
    print(f"[OPTUNA] Best params: {best_params}")
    return best_params


# ================================================================
# STEP 4: MAIN TRAINING PIPELINE
# ================================================================

def main():
    print("=" * 60)
    print(" XGBRegressor Training: EGFR pIC50 Prediction (V3)")
    print("=" * 60)

    # Download data
    df = download_egfr_chembl()
    print(f"\n[DATA] Dataset: {len(df)} molecules, pIC50 range: "
          f"{df['pic50'].min():.2f} – {df['pic50'].max():.2f}")

    # Sample for training limits
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    n_total = min(len(df), MAX_TRAIN + MAX_TEST)
    df       = df.iloc[:n_total]
    n_train  = min(MAX_TRAIN, int(0.83 * n_total))
    df_train = df.iloc[:n_train]
    df_test  = df.iloc[n_train:]

    print(f"[DATA] Train: {len(df_train)}, Test: {len(df_test)}")

    # Feature extraction
    print("\n[FEATURES] Extracting 2D multi-fingerprint features...")
    X_train_raw = extract_features(df_train["canonical_smiles"].tolist())
    X_test_raw  = extract_features(df_test["canonical_smiles"].tolist())
    y_train     = df_train["pic50"].values.astype(np.float32)
    y_test      = df_test["pic50"].values.astype(np.float32)

    # Variance filter
    selector = VarianceThreshold(threshold=MIN_VARIANCE)
    X_train  = selector.fit_transform(X_train_raw)
    X_test   = selector.transform(X_test_raw)
    print(f"[FEATURES] After variance filter: {X_train.shape[1]} features")

    # Hyperparameter optimization
    print(f"\n[OPTUNA] Running {OPTUNA_TRIALS} trials...")
    best_params = optimize_xgb(X_train, y_train)

    # Final model training
    print("\n[TRAINING] Fitting final XGBRegressor...")
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Evaluation
    y_pred     = model.predict(X_test)
    r2         = r2_score(y_test, y_pred)
    rmse       = np.sqrt(mean_squared_error(y_test, y_pred))
    pearson_r, _ = pearsonr(y_test, y_pred)

    print(f"\n[RESULTS] -----------------------------------------")
    print(f"  Test R2:      {r2:.4f}   (target > 0.65)")
    print(f"  Pearson r:    {pearson_r:.4f}")
    print(f"  Test RMSE:    {rmse:.4f}  (target < 1.0)")

    # Save checkpoints
    xgb_path = CHECKPOINT_DIR / "xgb_regressor_v3.pkl"
    sel_path  = CHECKPOINT_DIR / "xgb_var_selector_v3.pkl"

    with open(xgb_path, "wb") as f:
        pickle.dump(model, f)
    with open(sel_path, "wb") as f:
        pickle.dump(selector, f)

    # Save training report
    report = (
        f"XGBRegressor V3 Training Report\n"
        f"Target: EGFR Lung Cancer (ChEMBL {CHEMBL_TARGET_ID})\n"
        f"{'─'*40}\n"
        f"Train molecules: {len(df_train)}\n"
        f"Test molecules:  {len(df_test)}\n"
        f"Features used:   {X_train.shape[1]}\n"
        f"{'─'*40}\n"
        f"Test R2:         {r2:.4f}\n"
        f"Pearson r:       {pearson_r:.4f}\n"
        f"RMSE:            {rmse:.4f}\n"
        f"{'─'*40}\n"
        f"Best params: {json.dumps(best_params, indent=2)}\n"
    )
    report_path = CHECKPOINT_DIR / "training_report_xgb.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n[SAVED] xgb_regressor_v3.pkl → {xgb_path}")
    print(f"[SAVED] xgb_var_selector_v3.pkl → {sel_path}")
    print(f"[SAVED] Training report → {report_path}")
    print("\n✅ XGBRegressor training complete!")


if __name__ == "__main__":
    main()
