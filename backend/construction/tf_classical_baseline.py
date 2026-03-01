import os
import warnings
import numpy as np
import pandas as pd
import deepchem as dc
import tensorflow as tf
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem import AllChem

# Suppress annoying warnings for cleaner terminal output
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # Disables RDKit's internal C++ warnings

print(f"Using TensorFlow version: {tf.__version__}")


def build_clean_tox21_dataset():
    """
    Downloads raw Tox21 CSV, physically verifies every molecule,
    and purges corrupted data to prevent NumPy crashes.
    """
    print("1. Downloading raw Tox21 CSV...")
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
    df = pd.read_csv(url)

    tasks = [
        "NR-AR",
        "NR-AR-LBD",
        "NR-AhR",
        "NR-Aromatase",
        "NR-ER",
        "NR-ER-LBD",
        "NR-PPAR-gamma",
        "SR-ARE",
        "SR-ATAD5",
        "SR-HSE",
        "SR-MMP",
        "SR-p53",
    ]

    print("2. Scrubbing corrupted molecules (Validating via RDKit)...")
    valid_smiles = []
    valid_indices = []

    for idx, smiles in enumerate(df["smiles"]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_smiles.append(smiles)
            valid_indices.append(idx)

    print(f"   Kept {len(valid_indices)} valid molecules out of {len(df)}.")
    clean_df = df.iloc[valid_indices].copy()

    # Extract Labels (y) and Weights (w, for missing data)
    y = clean_df[tasks].values
    w = np.where(pd.isna(clean_df[tasks]), 0.0, 1.0)  # 0.0 masks out NaNs
    y = np.nan_to_num(
        y, nan=0.0
    )  # Replace NaNs with 0 safely since w handles the masking

    return clean_df, valid_smiles, y, w, tasks


# ==========================================
# 1. BASELINE A: XGBoost on Tabular Data
# ==========================================
def run_xgboost_baseline(valid_smiles, y, w, tasks):
    print("\n--- Starting Baseline A: XGBoost on Morgan Fingerprints ---")

    print("Generating Morgan Fingerprints...")
    fps = [
        AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=1024)
        for s in valid_smiles
    ]
    X_tabular = np.array(fps)

    # Wrap in DeepChem dataset and apply rigorous Scaffold Split
    dataset = dc.data.NumpyDataset(X=X_tabular, y=y, w=w, ids=valid_smiles)
    splitter = dc.splits.ScaffoldSplitter()
    train_data, valid_data, test_data = splitter.train_valid_test_split(
        dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1
    )

    task_idx = 0  # Testing the first assay: NR-AR
    task_name = tasks[task_idx]
    print(f"Training XGBoost on endpoint: {task_name}...")

    # Extract single task and drop masked (missing) values
    X_train = train_data.X[train_data.w[:, task_idx] > 0]
    y_train = train_data.y[:, task_idx][train_data.w[:, task_idx] > 0]

    X_valid = valid_data.X[valid_data.w[:, task_idx] > 0]
    y_valid = valid_data.y[:, task_idx][valid_data.w[:, task_idx] > 0]

    # Calculate dynamic imbalance weight
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_weight = num_neg / num_pos if num_pos > 0 else 1.0

    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_weight,
        eval_metric="logloss",
    )
    xgb_model.fit(X_train, y_train)

    valid_preds = xgb_model.predict_proba(X_valid)[:, 1]
    auc_score = roc_auc_score(y_valid, valid_preds)
    print(f"✅ XGBoost ROC-AUC on Validation Set ({task_name}): {auc_score:.4f}")


# ==========================================
# 2. BASELINE B: TensorFlow Graph Neural Network
# ==========================================
def run_tf_gnn_baseline(valid_smiles, y, w, tasks):
    print("\n--- Starting Baseline B: TensorFlow GraphConvModel ---")

    print("Generating Molecular Graphs for GNN...")
    # Because we scrubbed the bad SMILES, this featurizer will NOT crash
    featurizer = dc.feat.ConvMolFeaturizer()
    X_graph = featurizer.featurize(valid_smiles)

    dataset = dc.data.NumpyDataset(X=X_graph, y=y, w=w, ids=valid_smiles)
    splitter = dc.splits.ScaffoldSplitter()
    train_data, valid_data, test_data = splitter.train_valid_test_split(
        dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1
    )

    model = dc.models.GraphConvModel(
        n_tasks=len(tasks), mode="classification", dropout=0.2, batch_size=50, batch_normalize=False # <--- ADD THIS LINE TO FIX THE KERAS 3 CRASH
    )

    print("Training TensorFlow GNN (this may take 1-2 minutes)...")
    model.fit(train_data, nb_epoch=10)

    metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)
    valid_score = model.evaluate(valid_data, [metric])

    print(
        f"✅ TF GNN Mean ROC-AUC on Validation Set: {valid_score['mean-roc_auc_score']:.4f}"
    )


if __name__ == "__main__":
    # 1. Build the clean dataset once
    clean_df, valid_smiles, y, w, tasks = build_clean_tox21_dataset()

    # 2. Run both baselines on the perfectly clean data
    run_xgboost_baseline(valid_smiles, y, w, tasks)
    run_tf_gnn_baseline(valid_smiles, y, w, tasks)
