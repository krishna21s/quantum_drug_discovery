"""
Quantum Drug Discovery Platform — V4 De Novo Drug Generation Configuration
============================================================================
All constants, paths, and hyperparameters for the SMILES RNN generative
module. No logic — only configuration.

Target: EGFR — Epidermal Growth Factor Receptor (Lung Cancer)
Generator: REINVENT-style CharRNN with pocket conditioning + RL fine-tuning
"""

from pathlib import Path

# ================================================================
# PATHS
# ================================================================
BASE_DIR = Path(__file__).resolve().parent
V3_DIR = BASE_DIR.parent / "construction_v3"
V3_CHECKPOINT_DIR = V3_DIR / "checkpoints"
V4_CHECKPOINT_DIR = BASE_DIR / "checkpoints"
V4_CHECKPOINT_DIR.mkdir(exist_ok=True)

ZINC_DATA_PATH = BASE_DIR / "data" / "zinc250k_clean.csv"
EGFR_PDB_ID = "1M17"
EGFR_PHI_PATH = V4_CHECKPOINT_DIR / "egfr_phi.npy"

# ================================================================
# SMILES VOCABULARY
# ================================================================
# Fixed character set covering all ZINC250k SMILES.
# Multi-character tokens (Cl, Br) handled by regex tokeniser.
# Never inferred from data — ensures model portability.
SMILES_CHARS = [
    "C", "N", "O", "S", "F", "P", "B", "I",
    "c", "n", "o", "s",              # aromatic lowercase
    "(", ")", "[", "]",
    "=", "#", "@", "+", "-",
    ".", "/", "\\",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
]
MULTI_CHAR_TOKENS = ["Cl", "Br"]     # tokenised as single units
SPECIAL_TOKENS = ["<SOS>", "<EOS>", "<PAD>"]

# Vocab size = single chars + multi-char tokens + special tokens
VOCAB_SIZE = len(SMILES_CHARS) + len(MULTI_CHAR_TOKENS) + len(SPECIAL_TOKENS)

# ================================================================
# RNN ARCHITECTURE
# ================================================================
EMBED_DIM = 128
HIDDEN_DIM = 512
N_LAYERS = 3
DROPOUT = 0.1
MAX_SMILES_LEN = 120  # covers 99.9% of ZINC250k

# ================================================================
# PRE-TRAINING (Phase 1 — Kaggle GPU)
# ================================================================
PRETRAIN_EPOCHS = 30
PRETRAIN_BATCH = 128
PRETRAIN_LR = 1e-3
PRETRAIN_PATIENCE = 3  # early stopping patience

# ================================================================
# RL FINE-TUNING (Phase 2 — local)
# ================================================================
RL_EPISODES = 500
RL_BATCH_SIZE = 32
RL_LR = 1e-4
RL_BASELINE_DECAY = 0.95
RL_KL_WEIGHT = 0.1          # KL regularisation coefficient
RL_TEMPERATURE = 1.2        # sampling temperature during RL
RL_GRAD_CLIP = 1.0          # gradient clipping max norm
RL_LOG_EVERY = 10            # log metrics every N episodes
RL_PLATEAU_PATIENCE = 50    # early stopping if reward plateaus

# ================================================================
# REWARD FUNCTION WEIGHTS
# ================================================================
# R = α·norm(pIC50) − β·tox_proxy + γ·QED − δ·SA_norm − diversity_penalty
ALPHA = 1.0   # pIC50 weight (primary objective)
BETA = 0.5    # toxicity / Lipinski penalty weight
GAMMA = 0.3   # QED (drug-likeness) weight
DELTA = 0.2   # synthetic accessibility weight

# Diversity enforcement
DIVERSITY_RADIUS = 0.7    # Tanimoto threshold for diversity penalty
DIVERSITY_PENALTY = 0.3   # penalty magnitude if too similar

# ================================================================
# POCKET CONDITIONING (Phase 2)
# ================================================================
PHI_DIM = 7   # SASA, volume, HBD, HBA, charge, aromatic_frac, depth

# EGFR binding site parameters (PDB 1M17, chain A)
EGFR_POCKET_RESIDUES_START = 718
EGFR_POCKET_RESIDUES_END = 835
EGFR_POCKET_CHAIN = "A"
POCKET_RADIUS_A = 6.0  # Angstroms radius for pocket residue selection

# ================================================================
# QUANTUM FINAL EVALUATION (Phase 2)
# ================================================================
QUANTUM_TOP_K = 50         # candidates to pass to quantum oracle
QUANTUM_N_JOBS = 4         # parallel workers for quantum scoring
QUANTUM_DIVERSITY_MAX = 0.6  # max Tanimoto for final candidate set
SA_SCORE_CUTOFF = 6.0      # SA ≤ 6 for synthesisability filter

# ================================================================
# API SERVER
# ================================================================
API_HOST = "0.0.0.0"
API_PORT = 8001
WS_PUSH_INTERVAL_S = 2.0   # WebSocket update frequency

# ================================================================
# RANDOM STATE
# ================================================================
RANDOM_STATE = 42
