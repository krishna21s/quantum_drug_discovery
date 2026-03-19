"""
Quantum Drug Discovery Platform — V3 Global Configuration
==========================================================
Target Binding Affinity Regression (Option A)
Target: EGFR — Epidermal Growth Factor Receptor (Lung Cancer)

Shift: Classification (Safe/Toxic) → Regression (continuous pIC50)
All constants, paths, and thresholds in one place.
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
# CHEMBL TARGET CONFIG — EGFR (Lung Cancer)
# ================================================================
CHEMBL_TARGET_ID   = "CHEMBL203"           # EGFR ChEMBL target ID
CHEMBL_TARGET_NAME = "EGFR"
DISEASE_AREA       = "Lung Cancer"
CHEMBL_DATASET_URL = (
    "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    "?target_chembl_id=CHEMBL203&standard_type=IC50"
    "&standard_relation=%3D&standard_units=nM"
    "&pchembl_value__isnull=false&limit=1000"
)
# Locally cached dataset path after download
CHEMBL_DATASET_PATH = CHECKPOINT_DIR / "egfr_chembl_ic50.csv"

# ================================================================
# pIC50 REGRESSION SCALE
# ================================================================
PIC50_MIN = 4.0     # Lower bound for training filter (weak binders)
PIC50_MAX = 11.0    # Upper bound for training filter (stronger than possible)
PIC50_HIGH_THRESHOLD = 7.0   # pIC50 ≥ 7 → strong binder (nM potency)
PIC50_MED_THRESHOLD  = 5.0   # pIC50 ≥ 5 → moderate binder

# ================================================================
# QUANTUM PARAMETERS
# ================================================================
N_QUBITS          = 20
N_SHOTS           = 1024
NYSTROM_LANDMARKS = 100
MAX_TRAIN         = 600
MAX_TEST          = 120
RANDOM_STATE      = 42

# ================================================================
# 3D FEATURE ENGINEERING
# ================================================================
N_3D_FEATURES     = 20   # Must match N_QUBITS
CONFORMER_ATTEMPTS = 3   # Number of ETKDG conformer generation attempts
PEARSON_THRESHOLD  = 0.85  # |ρ| < 0.85 for orthogonality filter

# 3D Descriptor groups selected from
# WHIM (geometry / symmetry) and 3D-MoRSE (scattering)
WHIM_DESCRIPTORS = [
    "WHIM_L1", "WHIM_L2", "WHIM_L3",
    "WHIM_P1", "WHIM_P2",
    "WHIM_G1", "WHIM_G2", "WHIM_G3",
    "WHIM_E1", "WHIM_E2", "WHIM_E3",
]
MORSE_DESCRIPTORS = [
    "Mor01", "Mor02", "Mor03", "Mor04",
    "Mor05", "Mor06", "Mor07", "Mor08",
    "Mor09", "Mor10",
]

# 2D PhysChem fallback (used inside XGBoost only)
PHYSCHEM_DESCS = [
    "MolWt", "MolLogP", "TPSA", "NumRotatableBonds",
    "NumHAcceptors", "NumHDonors", "NumAromaticRings",
    "RingCount", "FractionCSP3", "HeavyAtomCount",
]

# ================================================================
# CLASSICAL MODEL (XGBRegressor)
# ================================================================
OPTUNA_TRIALS = 60
CV_FOLDS      = 5
MIN_VARIANCE  = 0.01

# ================================================================
# ENSEMBLE WEIGHTS (Regression blend)
# ================================================================
W_XGB = 0.50   # XGBRegressor (topological baseline)
W_QML = 0.50   # QSVR (quantum physical depth)

# ================================================================
# FEATURE FLAGS
# ================================================================
ENABLE_SHOT_MODE       = True
ENABLE_HARDWARE_CHECK  = False

# ================================================================
# LATENCY BUDGETS
# ================================================================
XGB_LATENCY_TARGET_MS        = 50
STATEVECTOR_LATENCY_TARGET_MS = 500
SHOT_LATENCY_TARGET_S        = 120
INTERACTIVE_SLA_S            = 3.0

# ================================================================
# PARALLEL WORKERS
# ================================================================
N_KERNEL_WORKERS       = max(1, os.cpu_count() - 1) if os.cpu_count() else 4
CHECKPOINT_EVERY_N_ROWS = 10

# ================================================================
# REFERENCE MOLECULES FOR VALIDATION (known EGFR binders)
# ================================================================
REFERENCE_MOLECULES = {
    "Erlotinib (Strong binder, pIC50≈9.2)":   ("COCCOC1=C(OCCO)C=C2C(=CC1)NCNC3=CC=CC(=C3)C#C", 9.2),
    "Gefitinib (Strong binder, pIC50≈8.7)":   ("COC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)F)OCC4=CC=NO4", 8.7),
    "Lapatinib (Moderate binder, pIC50≈8.1)": ("CS(=O)(=O)CCN1CCN(CC1)C2=CC=C(C=C2)NC3=NC=CC(=N3)C4=CC=C(Cl)C=C4.OC(=O)C=C", 8.1),
    "Aspirin (Non-binder, pIC50≈4.0)":        ("CC(=O)OC1=CC=CC=C1C(=O)O", 4.0),
    "Ibuprofen (Non-binder, pIC50≈4.0)":      ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", 4.0),
}
