"""
Quantum Drug Discovery Platform — V2 Global Configuration
==========================================================
All constants, paths, feature flags, and latency budgets in one place.
Every module imports from here — no magic numbers in service code.
"""

import os
from pathlib import Path

# ================================================================
# PATHS
# ================================================================
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ================================================================
# DATA SOURCE
# ================================================================
TOX21_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
TOX21_ENDPOINT = "NR-AR"  # Target column

# ================================================================
# QUANTUM PARAMETERS
# ================================================================
N_QUBITS = 20
N_SHOTS = 1024
NYSTROM_LANDMARKS = 100
MAX_TRAIN = 500
MAX_TEST = 100
RANDOM_STATE = 42

# ================================================================
# CLASSICAL MODEL
# ================================================================
PHYSCHEM_DESCS = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumRotatableBonds",
    "NumHAcceptors",
    "NumHDonors",
    "NumAromaticRings",
    "RingCount",
    "FractionCSP3",
    "HeavyAtomCount",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "BalabanJ",
    "BertzCT",
    "Chi0",
]
MULTI_FP_DIM = (
    4278  # Morgan r2(1024) + r3(1024) + MACCS(167) + RDKit(2048) + PhysChem(15)
)

# XGBoost training
OPTUNA_TRIALS = 60
CV_FOLDS = 5
MIN_VARIANCE = 0.01
SPW_MAX = 8.0

# GNN embedding
GNN_EMBEDDING_DIM = 128
GNN_HIDDEN_DIM = 128
GNN_NUM_LAYERS = 3
GNN_PROJECTION_DIM = 20  # projected to match N_QUBITS

# ================================================================
# ENSEMBLE WEIGHTS
# ================================================================
W_XGB = 0.55
W_QML = 0.45
ALERT_THRESHOLD = 0.60

# ================================================================
# FEATURE FLAGS
# ================================================================
ENABLE_GNN = True  # GNN model trained and checkpoint available
ENABLE_SHOT_MODE = True  # Allow shot-based final evaluation
ENABLE_HARDWARE_CHECK = False  # Real hardware integration

# ================================================================
# LATENCY BUDGETS (for monitoring / SLA enforcement)
# ================================================================
XGB_LATENCY_TARGET_MS = 50
GNN_LATENCY_TARGET_MS = 100
STATEVECTOR_LATENCY_TARGET_MS = 500
SHOT_LATENCY_TARGET_S = 120
INTERACTIVE_SLA_S = 3.0  # Max acceptable latency for fast path

# ================================================================
# CACHE CONFIGURATION
# ================================================================
EMBEDDING_CACHE_TTL_HOURS = 168  # 7 days
KERNEL_ROW_CACHE_PERSIST = True  # Persist kernel rows indefinitely

# ================================================================
# PARALLEL WORKERS
# ================================================================
N_KERNEL_WORKERS = max(1, os.cpu_count() - 1) if os.cpu_count() else 4
CHECKPOINT_EVERY_N_ROWS = 10

# ================================================================
# REFERENCE MOLECULES (for validation & regression tests)
# ================================================================
REFERENCE_MOLECULES = {
    "Aspirin (Safe)": ("CC(=O)OC1=CC=CC=C1C(=O)O", 0),
    "Phenanthrene (Toxic)": ("C1=CC=C2C(=C1)C=CC3=CC=CC=C32", 1),
    "Ibuprofen (Safe)": ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", 0),
    "Bisphenol A (Toxic)": ("CC(c1ccc(O)cc1)(c1ccc(O)cc1)C", 1),
    "Paracetamol (Safe)": ("CC(=O)Nc1ccc(O)cc1", 0),
}
