"""
Nystrom Engine — Improved Nystrom Kernel Approximation
======================================================
Refactored from V1 core_engine_shot.py::compute_nystrom_stateful().

Improvements over V1:
  - k-means landmark selection (instead of linspace)
  - Configurable landmark count (adaptive m)
  - Modular reconstruction: SVD + PSD + Cosine + Clip
  - Clean checkpoint management
  - Incremental single-row prediction for live inference
"""

import os
import numpy as np
from sklearn.cluster import KMeans

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NYSTROM_LANDMARKS, RANDOM_STATE, CHECKPOINT_DIR


class NystromEngine:
    """
    Nystrom low-rank kernel approximation with robust reconstruction.

    Stores K_mm, K_nm, and the inverse/diagonal needed for prediction.
    Supports both training-time full matrix computation and inference-time
    single-row prediction.
    """

    def __init__(self, checkpoint_dir=None):
        self.checkpoint_dir = checkpoint_dir or str(CHECKPOINT_DIR)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Loaded / computed state
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
        """
        Select m landmark points from the training data.

        Args:
            X_train: (N, d) training features
            m: Number of landmarks
            method: 'kmeans' (default, diversity), 'linspace' (V1 compat), 'random'

        Returns:
            X_landmarks: (m, d) landmark feature vectors
            landmark_indices: indices into X_train (approximate for kmeans)
        """
        m = min(m, len(X_train))

        if method == "kmeans":
            kmeans = KMeans(n_clusters=m, random_state=RANDOM_STATE, n_init=3)
            kmeans.fit(X_train)
            # Map each cluster center to its nearest training point
            from scipy.spatial.distance import cdist

            dists = cdist(kmeans.cluster_centers_, X_train)
            indices = np.argmin(dists, axis=1)
            # Ensure unique indices
            indices = np.unique(indices)
            if len(indices) < m:
                # Fill remaining with linspace
                all_idx = set(range(len(X_train)))
                remaining = sorted(all_idx - set(indices))
                extra = np.linspace(0, len(remaining) - 1, m - len(indices), dtype=int)
                indices = np.concatenate([indices, [remaining[e] for e in extra]])
            indices = indices[:m].astype(int)

        elif method == "linspace":
            # V1 compatible: evenly spaced
            indices = np.linspace(0, len(X_train) - 1, m, dtype=int)

        elif method == "random":
            rng = np.random.RandomState(RANDOM_STATE)
            indices = rng.choice(len(X_train), size=m, replace=False)
            indices.sort()

        else:
            raise ValueError(f"Unknown method: {method}")

        self.landmark_indices = indices
        self.landmarks = X_train[indices]
        return self.landmarks, indices

    # ------------------------------------------------------------------
    # KERNEL MATRIX COMPUTATION (uses backend directly, not parallel)
    # ------------------------------------------------------------------

    def compute_K_mm(self, landmarks, backend):
        """
        Compute m×m landmark-landmark kernel matrix.
        Symmetric: only computes upper triangle + mirrors.
        """
        m = len(landmarks)
        path = os.path.join(self.checkpoint_dir, "K_mm.npy")

        if os.path.exists(path):
            K = np.load(path)
            if K.shape == (m, m) and not np.any(np.isnan(K)):
                print(f"  [CACHED] K_mm loaded from checkpoint ({m}×{m})")
                self.K_mm = K
                return K

        K = np.full((m, m), np.nan)
        for i in range(m):
            if not np.isnan(K[i, 0]) and i > 0:
                continue
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
        """Compute N×m training-to-landmark kernel matrix."""
        N, m = len(X_train), len(landmarks)
        path = os.path.join(self.checkpoint_dir, "K_nm.npy")

        if os.path.exists(path):
            K = np.load(path)
            if K.shape == (N, m) and not np.any(np.isnan(K)):
                print(f"  [CACHED] K_nm loaded from checkpoint ({N}×{m})")
                self.K_nm = K
                return K

        K = np.full((N, m), np.nan)
        for i in range(N):
            if not np.isnan(K[i, 0]):
                continue
            for j in range(m):
                K[i, j] = backend.fidelity(X_train[i], landmarks[j])
            if i % 10 == 0:
                np.save(path, K)

        np.save(path, K)
        self.K_nm = K
        return K

    # ------------------------------------------------------------------
    # ROBUST KERNEL RECONSTRUCTION
    # ------------------------------------------------------------------

    def reconstruct_kernel(self, K_mm=None, K_nm=None, svd_threshold=0.10):
        """
        Full Nystrom kernel reconstruction with robust fixes.

        Pipeline:
          1. SVD-truncated pseudoinverse of K_mm
          2. Nystrom approximation: K_train ≈ K_nm · K_mm_inv · K_nm^T
          3. PSD projection (clip negative eigenvalues)
          4. Cosine normalization (bounds to [-1, 1])
          5. Clip to [0, 1] (valid fidelity range)

        Returns:
            K_train: (N, N) reconstructed training kernel
            K_mm_inv: pseudoinverse of K_mm
            diag_train: diagonal normalization factors
        """
        K_mm = K_mm if K_mm is not None else self.K_mm
        K_nm = K_nm if K_nm is not None else self.K_nm

        if K_mm is None or K_nm is None:
            raise ValueError("K_mm and K_nm must be computed or provided")

        m = len(K_mm)

        # Step 1: SVD-truncated pseudoinverse
        U, s, Vt = np.linalg.svd(K_mm, full_matrices=False)
        threshold = svd_threshold * s[0]
        s_inv = np.where(s > threshold, 1.0 / s, 0.0)
        K_mm_inv = Vt.T @ np.diag(s_inv) @ U.T
        kept = int(np.sum(s > threshold))
        print(f"  SVD: Kept {kept}/{m} singular values (threshold={threshold:.4f})")

        # Step 2: Nystrom reconstruction
        K_train = K_nm @ K_mm_inv @ K_nm.T
        np.fill_diagonal(K_train, 1.0)
        K_train = (K_train + K_train.T) / 2.0

        # Step 3: PSD projection
        eigvals, eigvecs = np.linalg.eigh(K_train)
        neg_count = int(np.sum(eigvals < 0))
        eigvals = np.maximum(eigvals, 0)
        K_train = eigvecs @ np.diag(eigvals) @ eigvecs.T
        print(f"  PSD: Projected {neg_count} negative eigenvalues to zero")

        # Step 4: Cosine normalization
        diag_train = np.sqrt(np.maximum(np.diag(K_train), 1e-12))
        K_train = K_train / np.outer(diag_train, diag_train)

        # Step 5: Clip to valid fidelity range
        K_train = np.clip(K_train, 0, 1)
        np.fill_diagonal(K_train, 1.0)
        print(f"  Cosine normalization + clip applied. Kernel ready.")

        self.K_mm_inv = K_mm_inv
        self.diag_train = diag_train

        return K_train, K_mm_inv, diag_train

    # ------------------------------------------------------------------
    # SINGLE-ROW PREDICTION (for live inference)
    # ------------------------------------------------------------------

    def compute_single_kernel_row(self, x_new, landmarks, backend):
        """
        Compute kernel row for a single new molecule against landmarks.

        Args:
            x_new: (d,) scaled feature vector for the new molecule
            landmarks: (m, d) landmark vectors
            backend: StatevectorBackend or ShotBackend

        Returns:
            K_new_m: (1, m) kernel row
        """
        m = len(landmarks)
        K_new_m = np.zeros((1, m))
        for j in range(m):
            K_new_m[0, j] = backend.fidelity(x_new, landmarks[j])
        return K_new_m

    def predict_from_kernel_row(
        self, K_new_m, K_mm_inv=None, K_nm=None, diag_train=None, svm_model=None
    ):
        """
        Reconstruct full kernel prediction from a single kernel row.

        Args:
            K_new_m: (1, m) kernel row against landmarks
            K_mm_inv: Pseudoinverse of K_mm
            K_nm: (N, m) training-to-landmark matrix
            diag_train: Diagonal normalization factors
            svm_model: Fitted SVC(kernel='precomputed')

        Returns:
            float: Predicted toxicity probability
        """
        K_mm_inv = K_mm_inv if K_mm_inv is not None else self.K_mm_inv
        K_nm = K_nm if K_nm is not None else self.K_nm
        diag_train = diag_train if diag_train is not None else self.diag_train

        # Nystrom reconstruction for new point
        K_new_train = K_new_m @ K_mm_inv @ K_nm.T

        # Cosine normalization
        K_new_self = np.sum((K_new_m @ K_mm_inv) * K_new_m, axis=1)
        diag_new = np.sqrt(np.maximum(K_new_self, 1e-12))
        K_new_train = K_new_train / np.outer(diag_new, diag_train)
        K_new_train = np.clip(K_new_train, 0, 1)

        # SVM prediction
        prob = float(svm_model.predict_proba(K_new_train)[0][1])
        return prob

    # ------------------------------------------------------------------
    # LOAD / SAVE
    # ------------------------------------------------------------------

    def load_checkpoints(self):
        """Load K_mm, K_nm and reconstruct K_mm_inv + diag_train."""
        K_mm_path = os.path.join(self.checkpoint_dir, "K_mm.npy")
        K_nm_path = os.path.join(self.checkpoint_dir, "K_nm.npy")

        if os.path.exists(K_mm_path) and os.path.exists(K_nm_path):
            self.K_mm = np.load(K_mm_path)
            self.K_nm = np.load(K_nm_path)
            print(f"  Loaded K_mm ({self.K_mm.shape}) and K_nm ({self.K_nm.shape})")
            return True
        return False
