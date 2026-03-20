"""
Train QSVR — V4 (Production): Full KTA + Hybrid Kernel Pipeline
================================================================
Complete rewrite of train_qsvr.py with all V4 fixes applied:

  FIX 1 — Feature encoding range:
      ArctanScaler → [-π/2, π/2]  (was [0, π], now uses full Bloch sphere)

  FIX 2 — Quantum Kernel Alignment (QKA):
      Trainable circuit params (theta, phi) optimised via L-BFGS-B
      to maximise centred Kernel Target Alignment (KTA) with pIC50.
      Cached: re-runs KTA only when --force-kta flag is set.

  FIX 3 — Hybrid kernel blend:
      K_hybrid = α·K_quantum + (1-α)·K_rbf
      α and rbf_gamma selected via 5-fold CV on training kernel.

  FIX 4 — Gamma sweep:
      KERNEL_GAMMA swept over [1, 5, 10, 20] before final run.
      Best gamma selected by CV R² on training kernel.

  FIX 5 — Stale checkpoint management:
      Detects when any upstream param changes and invalidates
      K_mm/K_nm so they are recomputed with correct params.

  FIX 6 — Increased landmarks:
      Default NYSTROM_LANDMARKS=100 (was 50).

Pipeline order:
    1. Load data
    2. Extract 3D features → Pearson filter → 20 features
    3. ArctanScaler → [-π/2, π/2]
    4. KTA optimisation (or load cached)
    5. Gamma sweep (or load cached)
    6. Compute K_mm, K_nm with best (params, gamma)
    7. Reconstruct K_train with Nystrom + RBF-Q + PSD + normalise
    8. HybridKernelBuilder: blend K_quantum with K_rbf, CV-search α
    9. Wide SVR grid search on K_hybrid
    10. Evaluate on test set
    11. Save all checkpoints

Usage:
    python training/train_qsvr_v4.py
    python training/train_qsvr_v4.py --force-kta      # re-run KTA even if cached
    python training/train_qsvr_v4.py --skip-kta       # skip KTA, use fixed params
    python training/train_qsvr_v4.py --skip-gamma-sweep  # skip gamma sweep

Outputs (saved to checkpoints/):
    kta_params_final.npy          - Optimised circuit params
    qsvr_scaler_v4.pkl            - ArctanScaler ([-π/2, π/2])
    qsvr_model_v4.pkl             - Fitted SVR(kernel='precomputed')
    qsvr_landmarks_scaled_v4.npy  - Landmark feature vectors
    qsvr_K_mm_inv_v4.npy          - Pseudoinverse of K_mm
    qsvr_diag_train_v4.npy        - Cosine normalisation factors
    qsvr_K_nm_transformed_v4.npy  - Transformed K_nm (for inference)
    hybrid_kernel_params_v4.pkl   - Blend weights (alpha, rbf_gamma)
    hybrid_X_train_scaled_v4.npy  - Training features for inference
    qsvr_selected_features_v4.json
    training_report_qsvr_v4.txt

Expected outcome:
    Baseline (V3):   CV R² ≈ 0.07
    After FIX 1:     CV R² ≈ 0.10-0.15
    After FIX 1+2:   CV R² ≈ 0.20-0.40
    After FIX 1+2+3: CV R² ≈ 0.40-0.60
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import pickle
import time
import hashlib
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

from config import (
    CHECKPOINT_DIR,
    CHEMBL_DATASET_PATH,
    N_QUBITS,
    N_REUPLOADING_LAYERS,
    MAX_TRAIN,
    MAX_TEST,
    RANDOM_STATE,
    NYSTROM_LANDMARKS,
    PIC50_MIN,
    PIC50_MAX,
    SVD_THRESHOLD,
    K_MM_REGULARIZATION,
)


# ================================================================
# CLI FLAGS
# ================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Train QSVR V4")
    parser.add_argument(
        "--force-kta",
        action="store_true",
        help="Re-run KTA optimisation even if cached params exist",
    )
    parser.add_argument(
        "--skip-kta",
        action="store_true",
        help="Skip KTA optimisation entirely (use theta=1, phi=0)",
    )
    parser.add_argument(
        "--skip-gamma-sweep",
        action="store_true",
        help="Skip gamma sweep, use KERNEL_GAMMA from config",
    )
    parser.add_argument(
        "--n-kta",
        type=int,
        default=60,
        help="Number of training samples for KTA subsample (default: 60)",
    )
    parser.add_argument(
        "--kta-iters",
        type=int,
        default=80,
        help="L-BFGS-B iterations per KTA restart (default: 80)",
    )
    parser.add_argument(
        "--kta-restarts",
        type=int,
        default=3,
        help="Number of random restarts for KTA (default: 3)",
    )
    parser.add_argument(
        "--landmarks",
        type=int,
        default=100,
        help="Number of Nystrom landmarks (default: 100)",
    )
    return parser.parse_args()


# ================================================================
# ARCTAN SCALER — [-π/2, π/2]  (FIX 1)
# ================================================================


class ArctanScalerV4:
    """
    Maps features to [-π/2, π/2] via z-score then arctan.

    Why [-π/2, π/2] instead of [0, π]:
        RY(θ) is maximally expressive across [-π, π].
        V3 used [0, π] (half the sphere). V4 centres on 0 for
        full expressivity — both hemispheres of the Bloch sphere.

    arctan(z) ∈ (-π/2, π/2) for z ∈ (-∞, ∞) — naturally in range.
    Outlier robust: extreme z-scores saturate gracefully.
    """

    def __init__(self):
        self.std_scaler = StandardScaler()

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        Z = self.std_scaler.fit_transform(X)
        return np.arctan(Z).astype(np.float32)  # ∈ (-π/2, π/2)

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = self.std_scaler.transform(X)
        return np.arctan(Z).astype(np.float32)


# ================================================================
# CHECKPOINT UTILITIES
# ================================================================


def param_hash(n_landmarks, kernel_gamma, kta_params_path):
    """Hash of all params that affect K_mm and K_nm."""
    h = hashlib.md5()
    h.update(str(n_landmarks).encode())
    h.update(str(kernel_gamma).encode())
    if kta_params_path and os.path.exists(kta_params_path):
        with open(kta_params_path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


def should_recompute(checkpoint_dir, tag, n_landmarks, kernel_gamma, kta_params_path):
    """Return True if K_mm/K_nm checkpoints are stale or missing."""
    hash_path = os.path.join(checkpoint_dir, f"kernel_hash_{tag}.txt")
    if not os.path.exists(hash_path):
        return True
    with open(hash_path) as f:
        stored = f.read().strip()
    current = param_hash(n_landmarks, kernel_gamma, kta_params_path)
    return stored != current


def save_hash(checkpoint_dir, tag, n_landmarks, kernel_gamma, kta_params_path):
    hash_path = os.path.join(checkpoint_dir, f"kernel_hash_{tag}.txt")
    with open(hash_path, "w") as f:
        f.write(param_hash(n_landmarks, kernel_gamma, kta_params_path))


def invalidate_kernel_checkpoints(checkpoint_dir, tag):
    """Delete K_mm/K_nm files so they are recomputed from scratch."""
    for fname in [
        f"K_mm_{tag}.npy",
        f"K_nm_{tag}.npy",
        f"qsvr_landmarks_scaled_{tag}.npy",
    ]:
        path = os.path.join(checkpoint_dir, fname)
        if os.path.exists(path):
            os.remove(path)
            print(f"  [INVALIDATED] Deleted stale checkpoint: {fname}")


# ================================================================
# KERNEL DIAGNOSTICS
# ================================================================


def log_kernel_stats(name, K, is_square=True):
    print(
        f"  [{name}] shape={K.shape}  "
        f"range=[{K.min():.4f}, {K.max():.4f}]  "
        f"mean={K.mean():.4f}  std={K.std():.4f}"
    )
    if is_square and K.shape[0] == K.shape[1]:
        offdiag = K[np.triu_indices_from(K, k=1)]
        print(
            f"  [{name}] off-diag: mean={offdiag.mean():.4f}  "
            f"std={offdiag.std():.4f}  "
            f"min={offdiag.min():.4f}  max={offdiag.max():.4f}"
        )
        svs = np.linalg.svd(K, compute_uv=False)
        rank10 = int(np.sum(svs > 0.1 * svs[0]))
        rank01 = int(np.sum(svs > 0.01 * svs[0]))
        print(
            f"  [{name}] effective rank: {rank10} (10% threshold), "
            f"{rank01} (1% threshold) of {len(svs)}"
        )


# ================================================================
# BATCH KERNEL COMPUTATION
# ================================================================


def compute_K_mm_batch(landmarks, backend, checkpoint_dir, tag="v4"):
    m = len(landmarks)
    path = os.path.join(checkpoint_dir, f"K_mm_{tag}.npy")

    if os.path.exists(path):
        K = np.load(path)
        if K.shape == (m, m) and not np.any(np.isnan(K)):
            print(f"  [CACHED] K_mm loaded ({m}×{m})")
            return K

    K = np.full((m, m), np.nan)
    np.fill_diagonal(K, 1.0)

    for i in range(m):
        upper = landmarks[i + 1 :]
        if len(upper) == 0:
            continue
        fids = backend.fidelity_batch(landmarks[i], upper)
        K[i, i + 1 :] = fids
        K[i + 1 :, i] = fids
        if i % 10 == 0:
            np.save(path, K)
            print(f"  K_mm row {i + 1}/{m}", end="\r")

    np.save(path, K)
    print(f"\n  [DONE] K_mm ({m}×{m})")
    return K


def compute_K_nm_batch(X_train, landmarks, backend, checkpoint_dir, tag="v4"):
    N, m = len(X_train), len(landmarks)
    path = os.path.join(checkpoint_dir, f"K_nm_{tag}.npy")

    if os.path.exists(path):
        K = np.load(path)
        if K.shape == (N, m):
            n_done = int(np.sum(~np.any(np.isnan(K), axis=1)))
            if n_done == N:
                print(f"  [CACHED] K_nm fully loaded ({N}×{m})")
                return K
            print(f"  [RESUME] K_nm {n_done}/{N} rows done, resuming...")
        else:
            K = np.full((N, m), np.nan)
    else:
        K = np.full((N, m), np.nan)

    t0 = time.time()
    for i in range(N):
        if not np.any(np.isnan(K[i])):
            continue
        K[i] = backend.fidelity_batch(X_train[i], landmarks)
        if i % 10 == 0:
            np.save(path, K)
            rate = (i + 1) / max(time.time() - t0, 1)
            eta_min = (N - i - 1) / rate / 60
            print(f"  K_nm row {i + 1}/{N}  ETA {eta_min:.1f} min", end="\r")

    np.save(path, K)
    print(f"\n  [DONE] K_nm ({N}×{m}) in {(time.time() - t0) / 60:.1f} min")
    return K


# ================================================================
# GAMMA SWEEP  (FIX 4)
# ================================================================


def gamma_sweep(K_mm, K_nm, y_train, svr_C=1.0, svr_eps=0.1, cv=5):
    """
    Find best KERNEL_GAMMA via 5-fold CV on the reconstructed training kernel.

    Returns best_gamma (float).
    """
    from services.nystrom_engine import NystromEngine

    gammas = [1.0, 5.0, 10.0, 20.0]
    best_gamma = gammas[0]
    best_score = -np.inf

    print("\n[GAMMA SWEEP]")
    for g in gammas:
        try:
            K_t, _, _ = NystromEngine().reconstruct_kernel(
                K_mm=K_mm,
                K_nm=K_nm,
                kernel_gamma=g,
                svd_threshold=SVD_THRESHOLD,
                regularization=K_MM_REGULARIZATION,
            )
            svr = SVR(kernel="precomputed", C=svr_C, epsilon=svr_eps)
            scores = cross_val_score(svr, K_t, y_train, cv=cv, scoring="r2")
            mean_r2 = float(scores.mean())
        except Exception as e:
            print(f"  gamma={g}  ERROR: {e}")
            mean_r2 = -999.0

        print(f"  gamma={g:5.1f}  CV R2={mean_r2:.4f}")
        if mean_r2 > best_score:
            best_score = mean_r2
            best_gamma = g

    print(f"  Best gamma: {best_gamma}  (CV R2={best_score:.4f})")
    return best_gamma


# ================================================================
# MAIN PIPELINE
# ================================================================


def main():
    args = parse_args()

    print("=" * 64)
    print(" QSVR V4 — KTA + Hybrid Kernel Production Training")
    print("=" * 64)

    t_start = time.time()
    ckpt = str(CHECKPOINT_DIR)
    os.makedirs(ckpt, exist_ok=True)

    TAG = "v4"  # checkpoint file suffix

    # ── STEP 1: Load Dataset ─────────────────────────────────────────
    if not os.path.exists(CHEMBL_DATASET_PATH):
        print(f"[ERROR] Dataset not found: {CHEMBL_DATASET_PATH}")
        sys.exit(1)

    df = pd.read_csv(CHEMBL_DATASET_PATH)
    print(
        f"\n[DATA] {len(df)} molecules  "
        f"pIC50 ∈ [{df['pic50'].min():.2f}, {df['pic50'].max():.2f}]"
    )

    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    n_total = min(len(df), MAX_TRAIN + MAX_TEST)
    df = df.iloc[:n_total]
    n_train = min(MAX_TRAIN, int(0.83 * n_total))
    df_train = df.iloc[:n_train]
    df_test = df.iloc[n_train:]
    print(f"  Train: {len(df_train)}  Test: {len(df_test)}")
    print(
        f"  y_train: mean={df_train['pic50'].mean():.2f}  std={df_train['pic50'].std():.2f}"
    )

    # ── STEP 2: 3D Feature Extraction ────────────────────────────────
    from services.feature_service_3d import FeatureService3D

    feat_svc = FeatureService3D()

    print("\n[3D FEATURES] Extracting...")

    def extract(smiles_list):
        vecs, names = [], None
        failed = 0
        for i, smi in enumerate(smiles_list):
            if i % 50 == 0:
                print(f"  mol {i + 1}/{len(smiles_list)}...", end="\r")
            try:
                v, n = feat_svc.extract_3d_descriptors(smi)
                vecs.append(v)
                if names is None:
                    names = n
            except Exception:
                v, n = feat_svc._extract_2d_fallback(smi)
                vecs.append(v)
                failed += 1
        max_len = max(len(v) for v in vecs)
        padded = [np.pad(v, (0, max_len - len(v))) for v in vecs]
        X = np.array(padded, dtype=np.float32)
        print(
            f"\n  Done: {len(smiles_list)} mols  failures: {failed}  shape: {X.shape}"
        )
        return X, names

    X_train_raw, feat_names = extract(df_train["canonical_smiles"].tolist())
    X_test_raw, _ = extract(df_test["canonical_smiles"].tolist())
    y_train = df_train["pic50"].values.astype(np.float32)
    y_test = df_test["pic50"].values.astype(np.float32)

    # ── STEP 3: Supervised feature selection → N_QUBITS features ─────
    print(f"\n[FEATURE SELECTION] Supervised filter (sorted by |ρ(feat, y_pIC50)|)...")
    print(f"  This is the V4 fix — V3 used variance-sorted (unsupervised) selection")
    print(f"  which picked orthogonal but label-uncorrelated features.")
    selected_features = feat_svc.fit_pearson_filter(X_train_raw, feat_names, y=y_train)

    feat_idx = {n: i for i, n in enumerate(feat_names)}
    sel_idx = [feat_idx[f] for f in selected_features if f in feat_idx]
    while len(sel_idx) < N_QUBITS:
        sel_idx.append(sel_idx[-1])
    sel_idx = sel_idx[:N_QUBITS]

    X_train_3d = X_train_raw[:, sel_idx]
    X_test_3d = X_test_raw[:, sel_idx]

    # ── STEP 4: ArctanScaler → [-π/2, π/2]  (FIX 1) ─────────────────
    print(f"\n[SCALER] ArctanScaler V4 → [-π/2, π/2]")
    scaler = ArctanScalerV4()
    X_train_scaled = scaler.fit_transform(X_train_3d)
    X_test_scaled = scaler.transform(X_test_3d)
    print(f"  Scaled range: [{X_train_scaled.min():.4f}, {X_train_scaled.max():.4f}]")
    print(f"  Expected ≈ (-1.57, 1.57)  — full Bloch sphere coverage")

    # ── STEP 5: KTA Optimisation  (FIX 2) ────────────────────────────
    from quantum.kta_optimizer import KTAOptimizer
    from quantum.circuits import default_params

    kta_path = os.path.join(ckpt, "kta_params_final.npy")

    if args.skip_kta:
        print("\n[KTA] Skipping — using fixed params (theta=1, phi=0)")
        circuit_params = default_params()

    elif not args.force_kta and os.path.exists(kta_path):
        print(f"\n[KTA] Loading cached params from {kta_path}")
        circuit_params = np.load(kta_path)
        print(f"  Loaded params shape: {circuit_params.shape}")

    else:
        print(
            f"\n[KTA] Running optimisation "
            f"(n_kta={args.n_kta}, iters={args.kta_iters}, restarts={args.kta_restarts})"
        )
        print("  This is the most important step — aligns quantum geometry to pIC50.")
        optimizer = KTAOptimizer(
            n_kta=args.n_kta,
            n_shots=1024,
            max_iter=args.kta_iters,
            n_restarts=args.kta_restarts,
            checkpoint_dir=ckpt,
            verbose=True,
        )
        circuit_params, _ = optimizer.fit(X_train_scaled, y_train)
        optimizer.report_param_shift(circuit_params)
        print(f"\n  Best KTA: {optimizer.best_kta_:.4f}")

    # ── STEP 6: Backend with KTA params ──────────────────────────────
    from quantum.backends import StatevectorBackend
    from services.nystrom_engine import NystromEngine

    backend_sv = StatevectorBackend(params=circuit_params)
    nystrom = NystromEngine(checkpoint_dir=ckpt)

    # ── STEP 7: Landmark Selection ───────────────────────────────────
    n_landmarks = args.landmarks

    # Check if stale before selecting landmarks
    if should_recompute(ckpt, TAG, n_landmarks, 0, kta_path):
        print(f"\n[LANDMARKS] Param change detected — invalidating kernel checkpoints")
        invalidate_kernel_checkpoints(ckpt, TAG)

    landmarks_path = os.path.join(ckpt, f"qsvr_landmarks_scaled_{TAG}.npy")
    if os.path.exists(landmarks_path):
        landmarks_scaled = np.load(landmarks_path)
        if landmarks_scaled.shape[0] == n_landmarks:
            print(f"\n[LANDMARKS] Loaded cached ({n_landmarks} points)")
        else:
            print(f"\n[LANDMARKS] Cached shape mismatch — reselecting")
            landmarks_scaled = None
    else:
        landmarks_scaled = None

    if landmarks_scaled is None:
        from sklearn.cluster import KMeans
        from scipy.spatial.distance import cdist

        print(f"\n[LANDMARKS] K-means selection ({n_landmarks} points)...")
        km = KMeans(n_clusters=n_landmarks, random_state=RANDOM_STATE, n_init=3)
        km.fit(X_train_scaled)
        dists = cdist(km.cluster_centers_, X_train_scaled)
        indices = np.argmin(dists, axis=1)
        indices = np.unique(indices)
        if len(indices) < n_landmarks:
            remaining = sorted(set(range(len(X_train_scaled))) - set(indices))
            extra = np.linspace(
                0, len(remaining) - 1, n_landmarks - len(indices), dtype=int
            )
            indices = np.concatenate([indices, [remaining[e] for e in extra]])
        indices = indices[:n_landmarks].astype(int)
        landmarks_scaled = X_train_scaled[indices]
        np.save(landmarks_path, landmarks_scaled)
        print(f"  Saved {n_landmarks} landmarks.")

    # ── STEP 8: Compute K_mm, K_nm ───────────────────────────────────
    print(f"\n[KERNEL] Computing K_mm ({n_landmarks}×{n_landmarks})...")
    K_mm = compute_K_mm_batch(landmarks_scaled, backend_sv, ckpt, TAG)
    nystrom.K_mm = K_mm
    log_kernel_stats("K_mm_raw", K_mm)

    print(f"\n[KERNEL] Computing K_nm ({len(X_train_scaled)}×{n_landmarks})...")
    K_nm = compute_K_nm_batch(X_train_scaled, landmarks_scaled, backend_sv, ckpt, TAG)
    nystrom.K_nm = K_nm
    log_kernel_stats("K_nm_raw", K_nm, is_square=False)

    # ── STEP 9: Gamma Sweep  (FIX 4) ────────────────────────────────
    gamma_cache_path = os.path.join(ckpt, "best_gamma_v4.json")

    if args.skip_gamma_sweep and os.path.exists(gamma_cache_path):
        with open(gamma_cache_path) as f:
            best_gamma = json.load(f)["best_gamma"]
        print(f"\n[GAMMA] Using cached best gamma: {best_gamma}")
    else:
        best_gamma = gamma_sweep(K_mm, K_nm, y_train)
        with open(gamma_cache_path, "w") as f:
            json.dump({"best_gamma": best_gamma}, f)

    # ── STEP 10: Reconstruct K_train ─────────────────────────────────
    print(f"\n[KERNEL] Reconstructing with gamma={best_gamma}...")
    K_train, K_mm_inv, diag_train = nystrom.reconstruct_kernel(
        K_mm=K_mm,
        K_nm=K_nm,
        kernel_gamma=best_gamma,
        svd_threshold=SVD_THRESHOLD,
        regularization=K_MM_REGULARIZATION,
    )
    log_kernel_stats("K_train", K_train)

    # Pre-transform K_nm for inference consistency
    K_nm_transformed = NystromEngine.apply_rbf_transform(K_nm, best_gamma)

    # Save kernel hash for future stale detection
    save_hash(ckpt, TAG, n_landmarks, best_gamma, kta_path)

    # ── STEP 11: Hybrid Kernel  (FIX 3) ─────────────────────────────
    print(f"\n[HYBRID] Building quantum + classical RBF blend...")
    print("  Note: alpha=0.0 → pure classical, alpha=1.0 → pure quantum")

    from services.hybrid_kernel import HybridKernelBuilder

    # Quick SVR C search first so hybrid CV uses a reasonable C
    quick_grid = GridSearchCV(
        SVR(kernel="precomputed"),
        {"C": [0.1, 1.0, 10.0], "epsilon": [0.05, 0.1]},
        cv=3,
        scoring="r2",
        n_jobs=-1,
    )
    quick_grid.fit(K_train, y_train)
    quick_C = quick_grid.best_params_["C"]
    quick_eps = quick_grid.best_params_["epsilon"]
    print(f"  Quick C={quick_C}  eps={quick_eps}")

    hybrid_builder = HybridKernelBuilder()
    K_hybrid, best_alpha, best_rbf_gamma = hybrid_builder.fit(
        K_train, X_train_scaled, y_train, C=quick_C, epsilon=quick_eps, cv=5
    )
    hybrid_builder.save(ckpt)

    log_kernel_stats("K_hybrid", K_hybrid)

    # ── STEP 12: Wide SVR Grid Search ────────────────────────────────
    print(f"\n[SVR] Wide grid search on K_hybrid...")
    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0],
        "epsilon": [0.001, 0.01, 0.05, 0.1, 0.2, 0.5],
    }
    grid = GridSearchCV(
        SVR(kernel="precomputed"), param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=0
    )
    grid.fit(K_hybrid, y_train)
    best_C = grid.best_params_["C"]
    best_eps = grid.best_params_["epsilon"]
    print(f"  Best CV R2: {grid.best_score_:.4f}  C={best_C}  eps={best_eps}")

    cv_results = pd.DataFrame(grid.cv_results_)
    for _, row in (
        cv_results.sort_values("mean_test_score", ascending=False).head(5).iterrows()
    ):
        print(
            f"    C={row['param_C']:<7} eps={row['param_epsilon']:<6}  "
            f"R2={row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}"
        )

    # ── STEP 13: Final SVR ────────────────────────────────────────────
    svr = SVR(kernel="precomputed", C=best_C, epsilon=best_eps)
    svr.fit(K_hybrid, y_train)

    y_train_pred = svr.predict(K_hybrid)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    # CV predictions on hybrid kernel
    y_cv_pred = cross_val_predict(
        SVR(kernel="precomputed", C=best_C, epsilon=best_eps), K_hybrid, y_train, cv=5
    )
    cv_r2 = r2_score(y_train, y_cv_pred)
    cv_rmse = np.sqrt(mean_squared_error(y_train, y_cv_pred))
    cv_pearson, _ = pearsonr(y_train, y_cv_pred)

    # ── STEP 14: Test Evaluation ────────────────────────────────────
    print(
        f"\n[TEST] Building test kernel ({len(X_test_scaled)}×{len(X_train_scaled)})..."
    )

    K_test_full = np.zeros((len(X_test_scaled), len(X_train_scaled)))

    for i, x_new in enumerate(X_test_scaled):
        if i % 10 == 0:
            print(f"  row {i + 1}/{len(X_test_scaled)}...", end="\r")

        # Quantum kernel row (Nystrom)
        K_row_raw = backend_sv.fidelity_batch(x_new, landmarks_scaled).reshape(1, -1)
        K_row_t = NystromEngine.apply_rbf_transform(K_row_raw, best_gamma)

        K_new_train = K_row_t @ K_mm_inv @ K_nm_transformed.T
        K_new_self = np.sum((K_row_t @ K_mm_inv) * K_row_t, axis=1)
        diag_new = np.sqrt(np.maximum(K_new_self, 1e-12))
        K_new_train = K_new_train / np.outer(diag_new, diag_train)
        K_new_train = np.clip(K_new_train, 0, 1)  # quantum row (1, N_train)

        # Hybrid blend
        K_hybrid_row = hybrid_builder.predict_row(K_new_train, x_new, X_train_scaled)
        K_test_full[i] = K_hybrid_row[0]

    print()
    y_pred = svr.predict(K_test_full)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pearson_r, _ = pearsonr(y_test, y_pred)

    print(f"\n{'=' * 60}")
    print(f"  [RESULTS] QSVR V4")
    print(f"{'=' * 60}")
    print(f"  Test R2:        {r2:.4f}   (target > 0.55)")
    print(f"  Test Pearson:   {pearson_r:.4f}")
    print(f"  Test RMSE:      {rmse:.4f}   (target < 0.90)")
    print(f"  ---")
    print(f"  CV R2:          {cv_r2:.4f}")
    print(f"  CV Pearson:     {cv_pearson:.4f}")
    print(f"  CV RMSE:        {cv_rmse:.4f}")
    print(f"  ---")
    print(f"  Train R2:       {train_r2:.4f}")
    print(f"  Train RMSE:     {train_rmse:.4f}")
    print(f"  ---")
    print(f"  Best SVR C:     {best_C}   eps: {best_eps}")
    print(f"  Best gamma:     {best_gamma}")
    print(f"  Best alpha:     {best_alpha}  (quantum weight)")
    print(f"  Best rbf_gamma: {best_rbf_gamma}")
    print(f"{'=' * 60}")

    # ── STEP 15: Save All Checkpoints ───────────────────────────────
    with open(os.path.join(ckpt, "qsvr_model_v4.pkl"), "wb") as f:
        pickle.dump(svr, f)
    with open(os.path.join(ckpt, "qsvr_scaler_v4.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    np.save(os.path.join(ckpt, "qsvr_landmarks_scaled_v4.npy"), landmarks_scaled)
    np.save(os.path.join(ckpt, "qsvr_K_mm_inv_v4.npy"), K_mm_inv)
    np.save(os.path.join(ckpt, "qsvr_diag_train_v4.npy"), diag_train)
    np.save(os.path.join(ckpt, "qsvr_K_nm_transformed_v4.npy"), K_nm_transformed)
    np.save(os.path.join(ckpt, "qsvr_X_train_scaled_v4.npy"), X_train_scaled)

    with open(os.path.join(ckpt, "qsvr_selected_features_v4.json"), "w") as f:
        json.dump(selected_features, f, indent=2)

    elapsed = (time.time() - t_start) / 60
    report = (
        f"QSVR V4 Training Report\n"
        f"{'=' * 50}\n"
        f"Fixes applied:\n"
        f"  FIX 1: ArctanScaler → [-π/2, π/2]\n"
        f"  FIX 2: KTA circuit param optimisation\n"
        f"  FIX 3: Hybrid quantum+RBF kernel  alpha={best_alpha}\n"
        f"  FIX 4: Gamma sweep → gamma={best_gamma}\n"
        f"  FIX 5: Stale checkpoint detection\n"
        f"  FIX 6: {n_landmarks} landmarks (was 50)\n"
        f"{'=' * 50}\n"
        f"KTA params: {kta_path}\n"
        f"Landmarks:  {n_landmarks}\n"
        f"Gamma:      {best_gamma}\n"
        f"Alpha:      {best_alpha}\n"
        f"rbf_gamma:  {best_rbf_gamma}\n"
        f"SVR C:      {best_C}   eps: {best_eps}\n"
        f"{'=' * 50}\n"
        f"Test R2:        {r2:.4f}\n"
        f"Test Pearson:   {pearson_r:.4f}\n"
        f"Test RMSE:      {rmse:.4f}\n"
        f"{'=' * 50}\n"
        f"CV R2:          {cv_r2:.4f}\n"
        f"CV Pearson:     {cv_pearson:.4f}\n"
        f"CV RMSE:        {cv_rmse:.4f}\n"
        f"{'=' * 50}\n"
        f"Train R2:       {train_r2:.4f}\n"
        f"Train RMSE:     {train_rmse:.4f}\n"
        f"{'=' * 50}\n"
        f"Selected features:\n{json.dumps(selected_features, indent=2)}\n"
        f"Elapsed: {elapsed:.1f} min\n"
    )
    with open(
        os.path.join(ckpt, "training_report_qsvr_v4.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(report)

    print(f"\n[SAVED] All checkpoints in {ckpt}")
    print(f"[TIME]  Total elapsed: {elapsed:.1f} min")


if __name__ == "__main__":
    main()
