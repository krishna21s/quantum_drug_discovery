# V4 De Novo Drug Generation — Complete Implementation Plan

> **Status:** Ready for implementation  
> **Prerequisite:** V3 QSVR + XGBoost training complete (or in progress — does not block V4)  
> **Estimated total runtime:** Pretraining 4–8h GPU (20–48h CPU), RL fine-tuning 30–60 min

---

## Overview

V4 adds a **generative AI module** to the quantum drug discovery platform. It produces novel EGFR drug candidate SMILES strings conditioned on the EGFR binding pocket geometry, using a pre-trained SMILES RNN fine-tuned with quantum-oracle-guided reinforcement learning.

### Core design decisions (and why)

| Decision                  | Chosen approach                      | Rejected approach        | Reason                                                                                      |
| ------------------------- | ------------------------------------ | ------------------------ | ------------------------------------------------------------------------------------------- |
| Generator backbone        | SMILES RNN (REINVENT-style)          | Graph VAE + MLP decoder  | VAE plan capped at 9 atoms — EGFR drugs need 30–50. RNN has no atom-count limit.            |
| RL oracle during training | XGBoost pIC50 (ms/call)              | QSVR quantum oracle      | 16,000 RL calls × 5–15s = 22–67h blocked. XGBoost runs the same 16,000 calls in ~4 seconds. |
| Quantum oracle usage      | Final eval of top-50 candidates only | Every RL step            | Preserves quantum advantage at the decision point without blocking RL training.             |
| Pocket conditioning       | 7D static PDB features               | 8D including RMSF        | RMSF requires MD trajectory. All 7 chosen features computable from a static PDB file.       |
| Diversity enforcement     | Tanimoto penalty in reward           | None (plan omitted this) | RL without diversity penalty always collapses to one repeated molecule (mode collapse).     |

---

## Architecture

```
ZINC250k SMILES
      │
      ▼
┌─────────────────────────────┐
│  Phase 1: RNN Pre-training  │   ~4–8h GPU / ~20h CPU
│  CharRNN on 250k molecules  │
│  Validity target: ≥ 95%     │
└─────────────┬───────────────┘
              │  vae_pretrained.pt
              ▼
┌─────────────────────────────┐
│  Phase 2: Pocket encoding   │   ~5 min (one-time per PDB)
│  7D φ from PDB 1M17         │
│  SASA, volume, HBD, HBA,    │
│  charge, aromatic%, depth   │
└─────────────┬───────────────┘
              │  egfr_phi.npy
              ▼
┌─────────────────────────────┐
│  Phase 3: RL Fine-tuning    │   ~30–60 min
│  REINFORCE on XGB oracle    │
│  + Tanimoto diversity       │
│  500 episodes × 32 mols     │
└─────────────┬───────────────┘
              │  policy_egfr_rl.pt
              ▼
┌─────────────────────────────┐
│  Phase 4: Quantum eval      │   ~10–20 min
│  Top-50 candidates →        │
│  V3 QSVR quantum kernel     │
│  Full pIC50 scoring         │
└─────────────┬───────────────┘
              │  final_candidates.json
              ▼
         API + Frontend
```

---

## Directory Structure

```
backend/
└── construction_v4/
    ├── config_v4.py
    ├── app_v4.py
    ├── data/
    │   ├── zinc_downloader.py
    │   ├── smiles_dataset.py
    │   └── sa_scorer.py               ← copied from rdkit/Contrib/SA_Score/
    ├── models/
    │   ├── char_rnn.py
    │   └── conditioned_rnn.py
    ├── oracle/
    │   ├── quantum_oracle.py
    │   ├── xgb_oracle.py
    │   ├── admet_scorer.py
    │   └── reward_function.py
    ├── training/
    │   ├── pretrain_rnn.py
    │   ├── pocket_conditioner.py
    │   └── rl_finetune.py
    └── checkpoints/                   ← auto-created

frontend/src/
├── pages/Generator.tsx
└── components/
    ├── GeneratorControls.tsx
    ├── MoleculeCard.tsx
    └── GenerationProgress.tsx
```

---

## Component Specifications

---

### `config_v4.py`

All V4 constants in one place. No logic — only configuration.

```python
# Paths
V3_CHECKPOINT_DIR  = "../construction_v3/checkpoints"
V4_CHECKPOINT_DIR  = "./checkpoints"
ZINC_DATA_PATH     = "./data/zinc250k_clean.csv"
EGFR_PDB_ID        = "1M17"
EGFR_PHI_PATH      = "./checkpoints/egfr_phi.npy"

# RNN architecture
VOCAB_SIZE         = 40         # unique characters in SMILES vocabulary
EMBED_DIM          = 128
HIDDEN_DIM         = 512
N_LAYERS           = 3
DROPOUT            = 0.1
MAX_SMILES_LEN     = 120        # covers 99.9% of ZINC250k

# Training
PRETRAIN_EPOCHS    = 30
PRETRAIN_BATCH     = 128
PRETRAIN_LR        = 1e-3

# RL
RL_EPISODES        = 500
RL_BATCH_SIZE      = 32
RL_LR              = 1e-4
RL_BASELINE_DECAY  = 0.95
DIVERSITY_RADIUS   = 0.7        # Tanimoto threshold for diversity penalty
DIVERSITY_PENALTY  = 0.3        # penalty magnitude subtracted from reward

# Reward weights (all normalised to [0,1] internally)
ALPHA = 1.0    # pIC50 weight
BETA  = 0.5    # toxicity/SA weight
GAMMA = 0.3    # QED weight
DELTA = 0.2    # synthetic accessibility weight

# Pocket conditioning
PHI_DIM = 7    # SASA, volume, HBD, HBA, charge, aromatic_frac, depth

# Quantum final eval
QUANTUM_TOP_K = 50   # number of RL-generated candidates to pass to quantum oracle
```

---

### `data/zinc_downloader.py`

Downloads ZINC250k from a stable, versioned source. Validates SMILES with RDKit. Caches locally.

**Source:** Uses the `datasets` library (`zpn/zinc250k` on HuggingFace) — stable, versioned, no raw GitHub URL fragility.

**Fallback:** If HuggingFace unavailable, falls back to direct ZINC download with retry logic.

**Key behaviour:**

- Validates every SMILES with `Chem.MolFromSmiles()` — drops ~1,200 invalid entries
- Canonicalises all SMILES
- Filters to drug-like range: MW 150–600, at least 5 heavy atoms
- Saves `zinc250k_clean.csv` with columns: `smiles`, `mw`, `logp`
- Prints row count and sample on completion

**Expected output:** ~247,000 rows in ~2 minutes.

---

### `data/smiles_dataset.py`

Tokenises SMILES strings into character-level sequences for the RNN.

**Vocabulary construction:**

- 38 characters covering all ZINC250k SMILES: `C N O S F Cl Br I P B ( ) [ ] = # @ + - . / \ 1 2 3 4 5 6 7 8 9 0` plus `<SOS>` and `<EOS>` tokens
- Fixed vocabulary (never inferred from data) so the RNN is portable

**`SMILESDataset(Dataset)`:**

- Loads `zinc250k_clean.csv`
- Encodes each SMILES as: `[SOS] + char_tokens + [EOS] + [PAD to MAX_SMILES_LEN]`
- Returns `(input_ids, target_ids, lengths)` where `target_ids` is `input_ids` shifted left by one (next-character prediction)
- `collate_fn` uses `pack_padded_sequence` for efficient variable-length batching

**`decode_smiles(token_ids, vocab) -> str`:**

- Converts integer token sequence back to SMILES string
- Stops at first EOS token
- Returns empty string if no EOS found (truncated — treat as invalid)

---

### `data/sa_scorer.py`

Exact copy of `rdkit/Contrib/SA_Score/sascorer.py` — included verbatim in the repo.

**Why include it:** SA Score is not in the standard RDKit Python install. It lives in the Contrib directory which is not always present in virtualenv installs. Including it explicitly prevents import failures.

**Source:** `https://github.com/rdkit/rdkit/blob/master/Contrib/SA_Score/sascorer.py`

**Do not modify.** Reference as `from data.sa_scorer import calculateScore`.

---

### `models/char_rnn.py`

Character-level LSTM for SMILES generation.

**Architecture:**

```
Input:  (batch, seq_len) integer token ids
        │
        ▼
Embedding(VOCAB_SIZE, EMBED_DIM)
        │
        ▼
LSTM(EMBED_DIM, HIDDEN_DIM, num_layers=N_LAYERS, dropout=DROPOUT, batch_first=True)
        │
        ▼
Linear(HIDDEN_DIM, VOCAB_SIZE)   ← logits over vocabulary at each position
        │
        ▼
Output: (batch, seq_len, VOCAB_SIZE) logits
```

**`CharRNN` class — public API:**

```python
class CharRNN(nn.Module):
    def forward(self, x, hidden=None) -> (logits, hidden)
    def init_hidden(self, batch_size) -> hidden_state
    def sample(self, n, temperature=1.0, device='cpu') -> List[str]
        # Autoregressively samples n SMILES strings.
        # Starts with SOS token, samples until EOS or MAX_SMILES_LEN.
        # Returns list of decoded SMILES strings (may include invalid ones).
    def sample_with_logprobs(self, n, temperature=1.0) -> (smiles_list, log_prob_tensor)
        # Same as sample() but also returns log P(smiles) for REINFORCE gradient.
    def save(self, path: str)
    @classmethod
    def load(cls, path: str) -> 'CharRNN'
```

**`temperature` parameter:**

- `temperature=1.0`: sample from learned distribution
- `temperature < 1.0`: more conservative, higher validity, less diversity
- `temperature > 1.0`: more exploratory, lower validity, higher novelty
- RL fine-tuning uses `temperature=1.2` to maintain exploration

---

### `models/conditioned_rnn.py`

Extends `CharRNN` with EGFR pocket conditioning.

**Conditioning mechanism — concat-to-hidden:**

The 7D pocket vector φ is projected to `HIDDEN_DIM` and added to the initial LSTM hidden state before generation begins. This is simpler and more stable than injecting φ at every step.

```
φ (7,) → Linear(PHI_DIM, HIDDEN_DIM) → tanh → h_0 for all layers
```

**`ConditionedRNN` class — public API:**

```python
class ConditionedRNN(CharRNN):
    def __init__(self, phi_dim=PHI_DIM, **kwargs)
    def forward(self, x, phi, hidden=None) -> (logits, hidden)
    def sample_conditioned(self, phi, n, temperature=1.0) -> List[str]
    def sample_conditioned_with_logprobs(self, phi, n, temperature=1.0) -> (smiles_list, log_probs)
```

**Important:** During pre-training, train `CharRNN` (unconditioned) on ZINC250k. Fine-tune `ConditionedRNN` (load pre-trained weights, add the φ projection layer) during RL. This two-stage approach is more stable than training the conditioned model from scratch.

**φ at inference:** For EGFR, φ is always the pre-computed `egfr_phi.npy` vector. For other targets, the user supplies a PDB ID and `pocket_conditioner.py` computes φ on-the-fly.

---

### `oracle/xgb_oracle.py`

Wraps the V3 XGBoost regressor as a fast RL training oracle.

**Key requirement:** Must be stateless between calls — the same object is called ~16,000 times during RL. Load the model once in `__init__`, cache it, never reload.

```python
class XGBOracle:
    def __init__(self, checkpoint_dir=V3_CHECKPOINT_DIR):
        # Loads xgb_regressor_v3.pkl and xgb_var_selector_v3.pkl
        # Raises FileNotFoundError with clear message if not found

    def score(self, smiles: str) -> float:
        # Returns predicted pIC50 (float in [2.0, 12.0])
        # Returns 2.0 (floor) for invalid SMILES — no exception raised
        # Internally: extract_xgb_features → selector.transform → model.predict

    def score_batch(self, smiles_list: List[str]) -> np.ndarray:
        # Vectorised batch scoring — ~50x faster than looping score()
        # Returns (n,) array of pIC50 values
        # Uses numpy batching on the XGB model internally
```

**Speed expectation:** `score_batch(32 smiles)` should complete in < 50ms. If slower, check that selector is not re-fitting on each call.

---

### `oracle/quantum_oracle.py`

Wraps the V3 QSVR as a final-evaluation oracle. Called only on the top-50 candidates after RL converges — not during RL training.

**Loading:** Loads all V3 checkpoints: `qsvr_model_v4.pkl`, `qsvr_scaler_v4.pkl`, `qsvr_landmarks_scaled_v4.npy`, `qsvr_K_mm_inv_v4.npy`, `qsvr_diag_train_v4.npy`, `qsvr_K_nm_transformed_v4.npy`, `kta_params_final.npy`, `hybrid_kernel_params_v4.pkl`.

**Falls back to V3 checkpoints** (`_v3` suffix) if V4 checkpoints not yet available — useful if QSVR V4 training is still running when V4 generative module is ready.

```python
class QuantumOracle:
    def __init__(self, checkpoint_dir=V3_CHECKPOINT_DIR):
        # Loads V4 checkpoints, falls back to V3 if missing
        # Prints which version was loaded

    def score(self, smiles: str) -> dict:
        # Returns:
        # {
        #   "pic50": float,
        #   "mode": "quantum_hybrid",
        #   "latency_s": float,
        #   "error": None | str,   # set if SMILES invalid or oracle failed
        # }

    def score_batch(self, smiles_list: List[str], n_jobs=4) -> List[dict]:
        # Scores top-K candidates in parallel using multiprocessing.Pool
        # n_jobs=4 recommended — more causes Aer simulator contention
```

---

### `oracle/admet_scorer.py`

Pure RDKit ADMET scoring. No network calls. Runs in < 1ms per molecule.

```python
class ADMETScorer:
    def score(self, smiles: str) -> dict:
        # Returns:
        # {
        #   "qed": float,           # 0–1, rdkit.Chem.QED.qed()
        #   "sa_score": float,      # 1–10, sascorer.calculateScore()
        #   "mw": float,            # molecular weight
        #   "logp": float,          # Crippen LogP
        #   "hbd": int,             # H-bond donors (Lipinski)
        #   "hba": int,             # H-bond acceptors (Lipinski)
        #   "lipinski_pass": bool,  # MW<500, LogP<5, HBD≤5, HBA≤10
        #   "tpsa": float,          # topological polar surface area
        #   "rotatable_bonds": int,
        #   "error": None | str,
        # }
        # Returns all None values (error set) for invalid SMILES — never raises.
```

**SA score normalisation for reward:** `sa_score_norm = (sa_score - 1) / 9.0` → maps [1,10] to [0,1] where 0 = easiest to synthesise.

---

### `oracle/reward_function.py`

Computes the scalar reward for REINFORCE.

**Formula:**

```
R = α·norm(pIC50) − β·tox_proxy + γ·QED − δ·SA_norm − diversity_penalty

where:
  norm(pIC50) = clip((pIC50 - 2) / 10, 0, 1)   maps [2,12] → [0,1]
  tox_proxy   = 1 - lipinski_pass                 0 if passes, 1 if fails
  SA_norm     = (sa_score - 1) / 9               maps [1,10] → [0,1]
  diversity_penalty = DIVERSITY_PENALTY if Tanimoto(smiles, batch) > DIVERSITY_RADIUS else 0
```

**Validity:** Returns `-1.0` immediately for any invalid SMILES (failed `Chem.MolFromSmiles`). This is the hardest possible penalty — the model learns to never generate invalid SMILES very quickly.

**Diversity penalty implementation:**

```python
def _tanimoto_penalty(smiles: str, batch_smiles: List[str],
                      radius: float = DIVERSITY_RADIUS,
                      penalty: float = DIVERSITY_PENALTY) -> float:
    # Computes Morgan FP for smiles
    # Computes Tanimoto similarity against all batch_smiles
    # Returns penalty if max similarity > radius, else 0.0
    # Returns 0.0 (no penalty) if batch is empty
```

**`compute_reward` signature:**

```python
def compute_reward(
    smiles: str,
    pic50: float | None,          # from XGBOracle.score()
    admet: dict | None,           # from ADMETScorer.score()
    batch_smiles: List[str],      # current episode batch for diversity
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
    delta: float = DELTA,
) -> float
```

---

### `training/pocket_conditioner.py`

Extracts the 7D pocket vector φ from a PDB file. Runs once per target protein; result is cached as `.npy`.

**The 7 features and how each is computed:**

| #   | Feature            | Method                                            | Notes                                                                               |
| --- | ------------------ | ------------------------------------------------- | ----------------------------------------------------------------------------------- |
| 1   | SASA (Å²)          | FreeSASA via RDKit or `Bio.PDB` + SASA            | Solvent-accessible surface area of pocket residues only. Proxy for pocket openness. |
| 2   | Pocket volume (Å³) | Bounding box of pocket atoms × packing factor 0.7 | Rough volume. Not exact but consistent.                                             |
| 3   | H-bond donors      | Count N–H and O–H in pocket residues              | Standard RDKit HBD count on pocket fragment                                         |
| 4   | H-bond acceptors   | Count N and O with lone pairs in pocket           | Standard RDKit HBA count on pocket fragment                                         |
| 5   | Net charge         | Sum of formal charges on pocket residues          | From PDB ATOM records + RDKit                                                       |
| 6   | Aromatic fraction  | Aromatic residues (F,Y,W,H) / total residues      | Encodes hydrophobic stacking propensity                                             |
| 7   | Pocket depth (Å)   | Max distance from pocket centroid to surface      | Estimates burial depth of the binding site                                          |

**Pocket residue identification:** Uses a 6Å radius sphere around the known EGFR ATP binding site centroid (pre-computed for PDB 1M17: residues 718–835 of chain A). For novel PDB inputs without known binding sites, uses ConCavity-style maximum curvature detection as a fallback.

**`PocketConditioner` class:**

```python
class PocketConditioner:
    def compute_phi(self, pdb_path: str) -> np.ndarray:
        # Returns (7,) float32 array — the pocket condition vector
        # Normalises each feature to [0,1] using pre-computed EGFR reference ranges
        # Caches result to {pdb_id}_phi.npy

    def load_or_compute(self, pdb_id: str, pdb_path: str = None) -> np.ndarray:
        # Loads cached phi if available, else downloads PDB and computes
        # For PDB 1M17: returns pre-committed egfr_phi.npy (skip network call)
```

**Pre-committed checkpoint:** `checkpoints/egfr_phi.npy` — the pre-computed 7D φ vector for EGFR PDB 1M17 is committed to the repo so generation works immediately without computing from PDB.

---

### `training/pretrain_rnn.py`

Pre-training loop for the SMILES RNN on ZINC250k.

**Training loop:**

```
For each epoch (30 total):
    For each batch of 128 SMILES:
        1. Encode: SMILES → token ids (with SOS/EOS/PAD)
        2. Forward: CharRNN(input_ids) → logits
        3. Loss: CrossEntropyLoss(logits, target_ids, ignore_index=PAD_ID)
        4. Backward + Adam step

    Every 5 epochs:
        Sample 512 molecules at temperature=1.0
        Compute: validity %, uniqueness %, novelty % (vs ZINC training set)
        Log metrics + save checkpoint
```

**Stopping criterion:** Stop if validation loss does not improve for 3 consecutive epochs (patience=3). In practice, 30 epochs is sufficient for ≥ 95% validity.

**Validity definition:** `Chem.MolFromSmiles(smi) is not None`

**Uniqueness definition:** Fraction of sampled molecules that are canonical-SMILES unique within the sample

**Novelty definition:** Fraction not present in ZINC250k training set (exact canonical SMILES match)

**Checkpoints saved:**

- `checkpoints/rnn_epoch_{n}.pt` every 5 epochs
- `checkpoints/rnn_pretrained.pt` — best validation loss

**Expected metrics at epoch 30:**

| Metric        | Target |
| ------------- | ------ |
| Validity      | ≥ 95%  |
| Uniqueness    | ≥ 98%  |
| Novelty       | ≥ 99%  |
| Training loss | ≤ 0.35 |

---

### `training/rl_finetune.py`

REINFORCE policy gradient fine-tuning conditioned on EGFR pocket.

**Full algorithm:**

```
Load:
    policy ← ConditionedRNN (pre-trained weights + φ projection layer)
    prior  ← CharRNN (pre-trained, frozen — used for KL regularisation)
    xgb    ← XGBOracle
    admet  ← ADMETScorer
    phi    ← load egfr_phi.npy

baseline = 0.0  (running mean reward)

For episode in range(RL_EPISODES):

    # 1. Sample batch
    smiles_batch, log_probs = policy.sample_conditioned_with_logprobs(
        phi, n=RL_BATCH_SIZE, temperature=1.2
    )

    # 2. Score batch (fast — XGB oracle)
    pic50_batch  = xgb.score_batch(smiles_batch)
    admet_batch  = [admet.score(s) for s in smiles_batch]

    # 3. Compute rewards with diversity penalty
    rewards = [
        compute_reward(s, p, a, smiles_batch)
        for s, p, a in zip(smiles_batch, pic50_batch, admet_batch)
    ]
    rewards = torch.tensor(rewards, dtype=torch.float32)

    # 4. KL regularisation vs prior (prevents catastrophic forgetting)
    #    Penalises if policy drifts too far from ZINC chemistry distribution
    prior_log_probs = prior.log_prob_batch(smiles_batch)
    kl_penalty = (log_probs - prior_log_probs).mean()

    # 5. REINFORCE gradient with baseline
    baseline = BASELINE_DECAY * baseline + (1 - BASELINE_DECAY) * rewards.mean()
    advantages = rewards - baseline
    loss = -(advantages * log_probs).mean() + 0.1 * kl_penalty

    # 6. Update
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()

    # 7. Logging every 10 episodes
    if episode % 10 == 0:
        mean_reward = rewards.mean().item()
        mean_pic50  = np.mean([p for p in pic50_batch if p > 2.0])
        validity    = sum(1 for s in smiles_batch if Chem.MolFromSmiles(s)) / len(smiles_batch)
        print(f"  ep={episode:4d}  reward={mean_reward:.3f}  "
              f"pIC50={mean_pic50:.2f}  valid={validity:.0%}")

    # 8. Early stopping: plateau detection
    if plateau_detected(reward_history, patience=50):
        print(f"  Early stop at episode {episode}")
        break

# Save
policy.save("checkpoints/policy_egfr_rl.pt")

# Phase 4: Quantum eval of top-K
top_smiles = select_top_k(all_generated, k=QUANTUM_TOP_K, by="xgb_pic50")
quantum_oracle = QuantumOracle()
final_results  = quantum_oracle.score_batch(top_smiles)
save_json("checkpoints/final_candidates.json", final_results)
```

**KL regularisation:** The `prior` (frozen pre-trained RNN) acts as a chemical prior. Without this, the policy drifts into invalid chemistry as it over-optimises the reward. The KL term keeps generated molecules in the drug-like chemical space learned from ZINC250k. KL weight 0.1 is a recommended starting point; reduce to 0.05 if the policy is too conservative.

**Mode collapse prevention — two mechanisms:**

1. Tanimoto diversity penalty in reward (per-batch, in `reward_function.py`)
2. KL regularisation vs prior (keeps population diverse at the distribution level)

**Expected RL progression:**

| Episode | Mean reward | Mean pIC50 | Validity |
| ------- | ----------- | ---------- | -------- |
| 0       | ~0.15       | ~5.0       | ~95%     |
| 100     | ~0.35       | ~6.0       | ~93%     |
| 300     | ~0.55       | ~6.8       | ~91%     |
| 500     | ~0.65       | ~7.2       | ~90%     |

---

### `oracle/quantum_oracle.py` — Phase 4 flow

After RL converges, the top-50 molecules by XGB pIC50 are passed to the quantum oracle for final scoring. This is where the quantum kernel's unique similarity structure contributes its advantage.

**Selection criteria for top-50:**

1. Valid SMILES
2. Lipinski pass
3. SA score ≤ 6 (synthesisable)
4. Ranked by XGB pIC50 descending
5. Filtered for Tanimoto diversity (max 0.6 pairwise) — ensure structural variety in final set

**Output `final_candidates.json` schema:**

```json
{
  "generated_at": "ISO timestamp",
  "target": "EGFR (PDB 1M17)",
  "n_rl_episodes": 423,
  "candidates": [
    {
      "rank": 1,
      "smiles": "...",
      "xgb_pic50": 7.82,
      "quantum_pic50": 8.14,
      "qed": 0.71,
      "sa_score": 3.2,
      "mw": 412.5,
      "lipinski_pass": true,
      "tpsa": 88.2,
      "is_novel": true
    }
  ]
}
```

---

### `app_v4.py`

FastAPI server exposing V4 generation as async endpoints.

**Endpoints:**

```
POST /api/v4/generate
    Body: {
        "pdb_id": "1M17",
        "n_candidates": 10,        # returned in results
        "alpha": 1.0,
        "beta": 0.5,
        "gamma": 0.3,
        "delta": 0.2,
        "use_rl": true,            # false = pure RNN sampling (fast, no RL)
        "temperature": 1.2
    }
    Returns: { "job_id": "uuid4" }
    Note: Generation runs in background via asyncio + concurrent.futures.
          Returns immediately; poll /status for progress.

GET /api/v4/status/{job_id}
    Returns: {
        "status": "running" | "complete" | "failed",
        "phase": "pretrain" | "rl" | "quantum_eval" | "done",
        "episode": 342,
        "total_episodes": 500,
        "current_reward": 0.58,
        "current_mean_pic50": 6.9,
        "validity_pct": 91.2,
        "elapsed_s": 1842
    }

GET /api/v4/results/{job_id}
    Returns: { "candidates": [...] }   ← final_candidates.json contents
    Returns 404 if job still running.

GET /api/v4/pocket_phi/{pdb_id}
    Returns: {
        "phi": [0.72, 0.45, 0.31, 0.68, 0.12, 0.55, 0.38],
        "feature_names": ["sasa_norm", "volume_norm", "hbd_norm",
                          "hba_norm", "charge_norm", "aromatic_frac", "depth_norm"],
        "pocket_residues": ["ALA719", "LYS721", ...]
    }

WS  /api/v4/stream/{job_id}
    WebSocket: pushes JSON status updates every 2s during generation.
    Closes when status = "complete" or "failed".
```

**Job management:** Uses an in-memory dict `{job_id: JobState}` + `asyncio.Queue`. For production, replace with Redis. Each job runs in a `ProcessPoolExecutor` worker to avoid blocking the event loop during RL.

---

## Frontend Components

---

### `src/pages/Generator.tsx`

Three-panel layout at `/generator`. All panels are visible simultaneously — no tabs.

**Left panel — Target & controls (280px fixed width):**

- PDB ID text input with "Load" button → calls `GET /api/v4/pocket_phi/{pdb_id}`
- φ vector visualisation: horizontal bar chart (7 bars, one per pocket feature)
- Reward weight sliders with live formula preview:
  ```
  R = [α=1.0]·pIC50 − [β=0.5]·SA + [γ=0.3]·QED − [δ=0.2]·Tox
  ```
- Temperature slider (0.8 → 1.5)
- N candidates selector (5, 10, 25, 50)
- "Generate" button → `POST /api/v4/generate`

**Centre panel — Progress (flex grow):**

- Episode number + phase label
- `GenerationProgress` chart (reward curve)
- Live stats: current mean pIC50, validity %, elapsed time
- Phase progress steps: Pre-train → RL → Quantum eval → Done

**Right panel — Results (320px fixed width):**

- Scrollable list of `MoleculeCard` components
- Sorted by quantum pIC50 descending (XGB pIC50 if quantum not yet run)
- Empty state with message until first results arrive

**State management:** Uses React Query for polling `/api/v4/status/{job_id}` every 2s during generation. WebSocket subscription for live updates.

---

### `src/components/GeneratorControls.tsx`

Reward weight sliders. Each slider uses `<input type="range" step="0.1" min="0" max="2">`.

Live formula update: re-renders the formula string as sliders move. No API call on slider change — weights are only sent on "Generate" click.

**Pocket φ visualisation:** `recharts BarChart` with 7 bars. X-axis: feature names shortened to abbreviations. Y-axis: 0–1 (normalised). Colour: teal fill.

**Disabled states:** "Generate" button is disabled while a job is running. Sliders and PDB input are disabled too — prevents re-submission during active generation.

---

### `src/components/MoleculeCard.tsx`

Card per generated candidate. Uses existing design system colours.

**Contents:**

- SMILES string (monospace, truncated with expand-on-click)
- pIC50 badge: green ≥ 7.0, amber 5.5–7.0, red < 5.5 (matches V3 colour scheme)
- QED progress bar (0–1)
- SA score with label: "Easy" (≤ 3), "Moderate" (3–6), "Hard" (> 6)
- MW, LogP, Lipinski pass/fail chips
- "Send to ADMET" button → navigates to `/admet?smiles=<encoded>`
- "View 3D" button → navigates to `/visualization?smiles=<encoded>`
- "Copy SMILES" icon button

**Hover state:** Subtle border highlight. Card does not expand on hover — interaction is via explicit buttons only.

---

### `src/components/GenerationProgress.tsx`

Live reward curve using `recharts LineChart`.

**Lines plotted:**

- Reward per episode (primary — teal)
- Mean pIC50 of batch (secondary — purple, right Y axis)
- Baseline running mean (dashed gray)

**Axes:** X = episode number, Y = reward [−1, 1], Y2 = pIC50 [2, 12]

**Updates:** Receives new data points via WebSocket or polling. Appends to local state array — never re-fetches full history. Chart animates new points smoothly with `isAnimationActive={false}` on existing points (only new points animate).

**Plateau indicator:** Dashed horizontal line appears when early stopping triggers.

---

### Modified existing files

**`src/components/AppSidebar.tsx`:**

```typescript
{ path: "/generator", label: "Drug Generator", icon: Sparkles, color: "hsl(280 80% 65%)" }
```

Add between "ADMET Analysis" and "Simulation Studio" in the nav order.

**`src/pages/Index.tsx`:**

- Add "Drug Generator" to quick actions panel (icon: Sparkles)
- Add "Generated Candidates" stat card (value from `/api/v4/results` or 0 if no jobs run)

**`src/App.tsx`:**

```typescript
import Generator from './pages/Generator';
<Route path="/generator" element={<Generator />} />
```

**`src/pages/Simulation.tsx`:**
Add "Send to Generator" button in the protein prep panel footer. On click: navigate to `/generator` with router state `{ pdbId: currentPdbId }`. The Generator page reads this from `useLocation().state` and pre-fills the PDB input.

---

## Implementation Order

Build and validate each step before proceeding to the next. Do not skip ahead.

```
Step 1 (30 min)   config_v4.py + directory structure
Step 2 (45 min)   data/zinc_downloader.py → verify zinc250k_clean.csv
Step 3 (30 min)   data/sa_scorer.py (copy from rdkit), data/smiles_dataset.py
Step 4 (1h)       models/char_rnn.py → unit test sample() produces valid strings
Step 5 (4–8h GPU) training/pretrain_rnn.py → run to completion, verify validity ≥ 95%
Step 6 (30 min)   oracle/xgb_oracle.py → sanity test (aspirin: pIC50 ~ 4–6)
Step 7 (30 min)   oracle/admet_scorer.py → sanity test (aspirin: QED, SA, Lipinski)
Step 8 (30 min)   oracle/reward_function.py → unit tests (valid + invalid SMILES)
Step 9 (1h)       training/pocket_conditioner.py + verify egfr_phi.npy loads
Step 10 (1h)      models/conditioned_rnn.py → verify conditioning changes output
Step 11 (30–60m)  training/rl_finetune.py → run 50 episodes, verify reward trending up
Step 12 (30 min)  oracle/quantum_oracle.py → test loads V3/V4 checkpoints
Step 13 (1h)      app_v4.py → test all endpoints with curl
Step 14 (2h)      Frontend: Generator.tsx + GeneratorControls.tsx
Step 15 (1h)      Frontend: MoleculeCard.tsx + GenerationProgress.tsx
Step 16 (30 min)  Frontend: AppSidebar.tsx + App.tsx + Index.tsx + Simulation.tsx
Step 17 (1h)      End-to-end test: PDB input → generate → quantum eval → display
```

---

## Verification Plan

### Backend unit tests

**1. Downloader:**

```bash
python data/zinc_downloader.py
# ✓ zinc250k_clean.csv exists with ≥ 245,000 rows
# ✓ head -3 shows valid SMILES column
python -c "import pandas as pd; df=pd.read_csv('data/zinc250k_clean.csv'); print(len(df), df.columns.tolist())"
```

**2. SMILES tokenisation round-trip:**

```python
from data.smiles_dataset import SMILESDataset, VOCAB
ds = SMILESDataset("data/zinc250k_clean.csv")
smi = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
tokens = ds.encode(smi)
decoded = ds.decode(tokens)
assert decoded == smi, f"Round-trip failed: {decoded}"
print("Tokenisation PASSED")
```

**3. RNN samples valid SMILES:**

```python
from models.char_rnn import CharRNN
rnn = CharRNN.load("checkpoints/rnn_pretrained.pt")
samples = rnn.sample(100, temperature=1.0)
from rdkit import Chem
valid = [s for s in samples if Chem.MolFromSmiles(s)]
print(f"Validity: {len(valid)/100:.0%}")  # expect ≥ 95%
```

**4. XGB oracle sanity:**

```python
from oracle.xgb_oracle import XGBOracle
oracle = XGBOracle()
pic50 = oracle.score("CCO")  # ethanol — expect low activity, ~3–5
assert 2.0 <= pic50 <= 12.0, f"pIC50 out of range: {pic50}"
print(f"Ethanol pIC50: {pic50:.2f}  PASSED")

pic50_invalid = oracle.score("INVALID_SMILES")
assert pic50_invalid == 2.0, "Invalid SMILES should return floor value 2.0"
print("Invalid SMILES handling PASSED")
```

**5. ADMET scorer:**

```python
from oracle.admet_scorer import ADMETScorer
scorer = ADMETScorer()
res = scorer.score("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
assert 0 <= res["qed"] <= 1
assert 1 <= res["sa_score"] <= 10
assert res["lipinski_pass"] == True   # aspirin passes Lipinski
assert res["mw"] < 200                # aspirin MW = 180
print(f"ADMET PASSED: {res}")
```

**6. Reward function:**

```python
from oracle.reward_function import compute_reward

# Valid high-reward molecule
r = compute_reward("CC(=O)Oc1ccccc1C(=O)O",
                   pic50=7.5, admet={"qed":0.8,"sa_score":2.0,"lipinski_pass":True},
                   batch_smiles=[])
assert -1.0 < r < 1.5, f"Reward out of range: {r}"

# Invalid SMILES → penalty
r_inv = compute_reward("INVALID", pic50=None, admet=None, batch_smiles=[])
assert r_inv == -1.0, "Invalid SMILES must return -1.0"

# Diversity penalty triggered
r_dup = compute_reward("CCO", pic50=5.0,
                        admet={"qed":0.5,"sa_score":3.0,"lipinski_pass":True},
                        batch_smiles=["CCO", "CCO", "CCO"])  # identical mols
assert r_dup < r  # must be penalised
print("Reward function PASSED")
```

**7. Pocket conditioner:**

```python
from training.pocket_conditioner import PocketConditioner
phi = PocketConditioner().load_or_compute("1M17")
assert phi.shape == (7,), f"phi shape wrong: {phi.shape}"
assert all(0 <= v <= 1 for v in phi), f"phi not normalised: {phi}"
print(f"Pocket phi PASSED: {phi.round(3)}")
```

**8. Quantum oracle loads (both V3 and V4):**

```python
from oracle.quantum_oracle import QuantumOracle
oracle = QuantumOracle()
result = oracle.score("CCO")
assert result["error"] is None or result["pic50"] is not None
print(f"Quantum oracle PASSED: {result}")
```

**9. RL distribution shift (post fine-tuning):**

```bash
python training/rl_finetune.py --eval-distribution \
    --checkpoint checkpoints/policy_egfr_rl.pt --n-samples 512
# Expected output:
#   Pre-RL mean pIC50:  ~5.0
#   Post-RL mean pIC50: ≥ 6.5
#   Post-RL validity:   ≥ 88%
#   Post-RL uniqueness: ≥ 85%
```

**10. End-to-end API test:**

```bash
uvicorn app_v4:app --port 8001 &

# Submit job
JOB=$(curl -s -X POST http://localhost:8001/api/v4/generate \
  -H "Content-Type: application/json" \
  -d '{"pdb_id":"1M17","n_candidates":10,"use_rl":true}' | python -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "Job: $JOB"

# Poll until done
while true; do
  STATUS=$(curl -s http://localhost:8001/api/v4/status/$JOB | python -c "import sys,json; d=json.load(sys.stdin); print(d['status'])")
  echo "Status: $STATUS"
  [ "$STATUS" = "complete" ] && break
  sleep 10
done

# Fetch results
curl -s http://localhost:8001/api/v4/results/$JOB | python -m json.tool | head -60
```

### Frontend verification

```bash
cd frontend && npm run test   # existing suite must still pass

# Manual checks:
# 1. /generator loads with 3-panel layout
# 2. PDB ID "1M17" → loads φ bar chart
# 3. Sliders update formula display in real-time
# 4. "Generate" click → job submitted, progress chart updates
# 5. Molecule cards appear with correct colour-coded pIC50 badges
# 6. "Send to ADMET" → navigates to /admet with SMILES in URL
# 7. "View 3D" → navigates to /visualization with SMILES in URL
# 8. Simulation page "Send to Generator" → pre-fills Generator PDB input
```

---

## Dependencies

Add to `requirements.txt` in `construction_v4/`:

```
torch>=2.0.0
torch-geometric>=2.3.0          # for future graph work; optional for V4 RNN
rdkit>=2023.3.1
requests>=2.28.0
fastapi>=0.100.0
uvicorn[standard]>=0.22.0
datasets>=2.14.0                # for ZINC250k download
biopython>=1.81                 # for PDB parsing in pocket_conditioner
MDAnalysis>=2.6.0               # optional — for SASA; can use pure RDKit fallback
scipy>=1.10.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=1.7.0
```

**Note:** `MDAnalysis` is optional. If not installed, `pocket_conditioner.py` falls back to a pure RDKit SASA implementation using `rdkit.Chem.rdFreeSASA`. The fallback is slightly less accurate but produces consistent φ vectors.

---

## Known Limitations and Future Work

**Atom count:** SMILES RNN has no structural atom-count constraint — it can generate valid SMILES of any length. However, very large molecules (> 60 heavy atoms) become increasingly invalid as generation length grows. The `MAX_SMILES_LEN=120` cap effectively limits practical molecule size to ~40–50 heavy atoms, which covers all known EGFR inhibitors.

**Pocket conditioning fidelity:** The 7D φ vector is a coarse approximation of a complex 3D binding site. Conditioning improves pIC50 distribution but does not guarantee binding-mode complementarity. For higher-fidelity conditioning, future work should explore docking-based scoring (AutoDock-GPU) as an additional reward signal.

**Multi-target generation:** The current design assumes a single target (EGFR). Extending to multi-target generation requires either separate models per target or a conditional model trained on multiple pocket φ vectors. The architecture supports this — `ConditionedRNN` accepts any φ vector.

**Synthesis validation:** SA score is a proxy. Real synthesisability requires retrosynthetic analysis (e.g. ASKCOS or IBM RXN). This is out of scope for V4 but could be added as an optional reward component.

**Quantum advantage:** The quantum oracle is used as a final evaluator, not a training signal. True quantum-enhanced generative models (e.g. quantum VAE, quantum GAN) are not yet computationally feasible on classical simulators at drug-relevant molecule sizes. This is the correct V4 boundary — preserve quantum involvement at the evaluation stage where it is already validated.
