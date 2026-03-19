"""
Nystrom Engine — Kernel Approximation for QSVR
================================================
Adapted from V2 NystromEngine for Support Vector Regression (SVR).

Key change from V2:
  - predict_from_kernel_row() now calls svr_model.predict() instead of
    svc_model.predict_proba(), returning a continuous pIC50 value.
  - SVD truncation and PSD projection remain identical for numerical stability.
"""

import os
import numpy as np
from sklearn.cluster import KMeans

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NYSTROM_LANDMARKS, RANDOM_STATE, CHECKPOINT_DIR


class NystromEngine:
    """
    Nystrom low-rank kernel approximation for regression (SVR).

    Stores K_mm, K_nm, and the inverse/diagonal for prediction.
    Supports training-time full matrix computation and inference-time
    single-row prediction returning a continuous pIC50 value.
    """

    def __init__(self, checkpoint_dir=None):
        self.checkpoint_dir = checkpoint_dir or str(CHECKPOINT_DIR)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.K_mm = None
        self.K_nm = None
        self.K_mm_inv = None
        self.diag_train = None
        self.landmarks = None
        self.landmark_indices = None

    # ------------------------------------------------------------------
    # LANDMARK SELECTION
    # ------------------------------------------------------------------

    def select_landmarks(self, X_train, m=NYSTROM_LANDMARKS, method="kmeans"):
        """Select m landmark points from the training data.
        If landmarks are already cached to disk, reload them for consistency."""
        m = min(m, len(X_train))
        landmark_path = os.path.join(self.checkpoint_dir, "qsvr_landmarks_scaled_v3.npy")

        # ── Load cached landmarks (critical: same landmarks = valid cached K_nm) ──
        if os.path.exists(landmark_path):
            cached = np.load(landmark_path)
            if cached.shape == (m, X_train.shape[1]):
                print(f"  [CACHED] Landmarks loaded from checkpoint ({m} points)")
                self.landmarks = cached
                self.landmark_indices = None
                return self.landmarks, None

        # ── Fresh selection ──────────────────────────────────────────────────────
        if method == "kmeans":
            kmeans = KMeans(n_clusters=m, random_state=RANDOM_STATE, n_init=3)
            kmeans.fit(X_train)
            from scipy.spatial.distance import cdist
            dists   = cdist(kmeans.cluster_centers_, X_train)
            indices = np.argmin(dists, axis=1)
            indices = np.unique(indices)
            if len(indices) < m:
                all_idx   = set(range(len(X_train)))
                remaining = sorted(all_idx - set(indices))
                extra     = np.linspace(0, len(remaining) - 1, m - len(indices), dtype=int)
                indices   = np.concatenate([indices, [remaining[e] for e in extra]])
            indices = indices[:m].astype(int)

        elif method == "linspace":
            indices = np.linspace(0, len(X_train) - 1, m, dtype=int)

        elif method == "random":
            rng     = np.random.RandomState(RANDOM_STATE)
            indices = rng.choice(len(X_train), size=m, replace=False)
            indices.sort()
        else:
            raise ValueError(f"Unknown landmark selection method: {method}")

        self.landmark_indices = indices
        self.landmarks        = X_train[indices]

        # Save so the next run reloads the SAME landmarks
        np.save(landmark_path, self.landmarks)
        print(f"  [SAVED] Landmarks saved to checkpoint ({m} points)")
        return self.landmarks, indices

    # ------------------------------------------------------------------
    # KERNEL MATRIX COMPUTATION
    # ------------------------------------------------------------------

    def compute_K_mm(self, landmarks, backend):
        """Compute m×m landmark-landmark kernel matrix (symmetric)."""
        m    = len(landmarks)
        path = os.path.join(self.checkpoint_dir, "K_mm_v3.npy")

        if os.path.exists(path):
            K = np.load(path)
            if K.shape == (m, m) and not np.any(np.isnan(K)):
                print(f"  [CACHED] K_mm loaded from checkpoint ({m}×{m})")
                self.K_mm = K
                return K

        K = np.full((m, m), np.nan)
        for i in range(m):
            for j in range(m):
                if j < i:
                    K[i, j] = K[j, i]
                elif i == j:
                    K[i, j] = 1.0
                else:
                    K[i, j] = backend.fidelity(landmarks[i], landmarks[j])
            if i % 10 == 0:
                np.save(path, K)

        np.save(path, K)
        self.K_mm = K
        return K

    def compute_K_nm(self, X_train, landmarks, backend):
        """Compute N×m training-to-landmark kernel matrix with row-level resumption."""
        N, m = len(X_train), len(landmarks)
        path = os.path.join(self.checkpoint_dir, "K_nm_v3.npy")

        # ── Load checkpoint and check if fully done ──────────────────────────────
        if os.path.exists(path):
            K = np.load(path)
            if K.shape == (N, m):
                n_done = int(np.sum(~np.any(np.isnan(K), axis=1)))  # rows with no NaN
                if n_done == N:
                    print(f"  [CACHED] K_nm fully loaded from checkpoint ({N}x{m})")
                    self.K_nm = K
                    return K
                else:
                    print(f"  [RESUME] K_nm checkpoint found: {n_done}/{N} rows done. Resuming...")
            else:
                print(f"  [RESET] K_nm checkpoint shape mismatch. Restarting.")
                K = np.full((N, m), np.nan)
        else:
            K = np.full((N, m), np.nan)

        # ── Compute / resume row by row ──────────────────────────────────────────
        for i in range(N):
            if not np.any(np.isnan(K[i])):  # already done
                continue
            for j in range(m):
                K[i, j] = backend.fidelity(X_train[i], landmarks[j])
            if i % 10 == 0:
                np.save(path, K)
                n_done = int(np.sum(~np.any(np.isnan(K), axis=1)))
                print(f"  K_nm row {i+1}/{N}  ({n_done} rows done)", end="\r")

        np.save(path, K)  # final save — all rows complete, no NaNs remain
        print(f"\n  [DONE] K_nm ({N}x{m}) fully computed and saved.")
        self.K_nm = K
        return K

    # ------------------------------------------------------------------
    # ROBUST KERNEL RECONSTRUCTION (unchanged from V2)
    # ------------------------------------------------------------------

    def reconstruct_kernel(self, K_mm=None, K_nm=None, svd_threshold=0.10):
        """
        Full Nystrom kernel reconstruction with robust fixes.

        Pipeline:
          1. SVD-truncated pseudoinverse of K_mm
          2. Nystrom: K_train ≈ K_nm · K_mm_inv · K_nm^T
          3. PSD projection (clip negative eigenvalues to 0)
          4. Cosine normalization
          5. Clip to [0, 1]

        Returns:
            K_train: (N, N) reconstructed training kernel
            K_mm_inv: pseudoinverse of K_mm
            diag_train: diagonal normalization factors
        """
        K_mm = K_mm if K_mm is not None else self.K_mm
        K_nm = K_nm if K_nm is not None else self.K_nm

        if K_mm is None or K_nm is None:
            raise ValueError("K_mm and K_nm must be computed or provided.")

        m = len(K_mm)

        # Step 1: SVD-truncated pseudoinverse
        U, s, Vt = np.linalg.svd(K_mm, full_matrices=False)
        threshold = svd_threshold * s[0]
        s_inv     = np.where(s > threshold, 1.0 / s, 0.0)
        K_mm_inv  = Vt.T @ np.diag(s_inv) @ U.T
        kept      = int(np.sum(s > threshold))
        print(f"  SVD: Kept {kept}/{m} singular values (threshold={threshold:.4f})")

        # Step 2: Nystrom reconstruction
        K_train = K_nm @ K_mm_inv @ K_nm.T
        np.fill_diagonal(K_train, 1.0)
        K_train = (K_train + K_train.T) / 2.0

        # Step 3: PSD projection
        eigvals, eigvecs = np.linalg.eigh(K_train)
        neg_count = int(np.sum(eigvals < 0))
        eigvals   = np.maximum(eigvals, 0)
        K_train   = eigvecs @ np.diag(eigvals) @ eigvecs.T
        print(f"  PSD: Projected {neg_count} negative eigenvalues to zero")

        # Step 4: Cosine normalization
        diag_train = np.sqrt(np.maximum(np.diag(K_train), 1e-12))
        K_train    = K_train / np.outer(diag_train, diag_train)

        # Step 5: Clip to valid fidelity range
        K_train = np.clip(K_train, 0, 1)
        np.fill_diagonal(K_train, 1.0)
        print(f"  Cosine normalization + clip applied. Kernel ready.")

        self.K_mm_inv   = K_mm_inv
        self.diag_train = diag_train

        return K_train, K_mm_inv, diag_train

    # ------------------------------------------------------------------
    # SINGLE-ROW PREDICTION — REGRESSION (key change from V2)
    # ------------------------------------------------------------------

    def compute_single_kernel_row(self, x_new, landmarks, backend):
        """Compute kernel row for a single new molecule against landmarks."""
        m      = len(landmarks)
        K_new_m = np.zeros((1, m))
        for j in range(m):
            K_new_m[0, j] = backend.fidelity(x_new, landmarks[j])
        return K_new_m

    def predict_pic50_from_kernel_row(
        self, K_new_m, K_mm_inv=None, K_nm=None, diag_train=None, svr_model=None
    ) -> float:
        """
        Reconstruct full kernel prediction and return continuous pIC50.

        Key difference from V2:
            V2: svc_model.predict_proba(K_row) → toxicity probability
            V3: svr_model.predict(K_row)        → continuous pIC50 value

        Args:
            K_new_m:   (1, m) kernel row against landmarks
            K_mm_inv:  Pseudoinverse of K_mm
            K_nm:      (N, m) training-to-landmark matrix
            diag_train: Diagonal normalization factors
            svr_model:  Fitted SVR(kernel='precomputed')

        Returns:
            float: Predicted pIC50 value
        """
        K_mm_inv   = K_mm_inv   if K_mm_inv   is not None else self.K_mm_inv
        K_nm       = K_nm       if K_nm       is not None else self.K_nm
        diag_train = diag_train if diag_train is not None else self.diag_train

        # Nystrom reconstruction for the new point
        K_new_train = K_new_m @ K_mm_inv @ K_nm.T

        # Cosine normalization
        K_new_self  = np.sum((K_new_m @ K_mm_inv) * K_new_m, axis=1)
        diag_new    = np.sqrt(np.maximum(K_new_self, 1e-12))
        K_new_train = K_new_train / np.outer(diag_new, diag_train)
        K_new_train = np.clip(K_new_train, 0, 1)

        # SVR prediction (returns continuous value)
        pic50 = float(svr_model.predict(K_new_train)[0])
        return float(np.clip(pic50, 2.0, 12.0))

    # ------------------------------------------------------------------
    # LOAD / SAVE
    # ------------------------------------------------------------------

    def load_checkpoints(self):
        """Load K_mm, K_nm and reconstruct K_mm_inv + diag_train."""
        K_mm_path = os.path.join(self.checkpoint_dir, "K_mm_v3.npy")
        K_nm_path = os.path.join(self.checkpoint_dir, "K_nm_v3.npy")

        if os.path.exists(K_mm_path) and os.path.exists(K_nm_path):
            self.K_mm = np.load(K_mm_path)
            self.K_nm = np.load(K_nm_path)
            print(f"  Loaded K_mm ({self.K_mm.shape}) and K_nm ({self.K_nm.shape})")
            return True
        return False
