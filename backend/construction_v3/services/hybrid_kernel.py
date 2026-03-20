"""
Hybrid Kernel Module (V4)
==========================
Blends the quantum Nystrom kernel with a classical RBF kernel computed
on the same 3D descriptors.

Motivation:
    The classical RBF on 3D descriptors is likely already positively
    correlated with pIC50 (the features were Pearson-selected FOR this).
    Blending gives the SVR a useful gradient signal immediately, while
    the quantum component contributes its unique similarity structure.

    K_hybrid = α · K_quantum + (1-α) · K_rbf

    Optimal α is found via cross-validation on the training kernel.
    In practice α=0.2–0.5 typically wins for regression on small datasets
    because the quantum kernel is still noisy.

This module also provides test-time reconstruction of the hybrid kernel
row, mirroring train-time construction exactly.
"""

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR


class HybridKernelBuilder:
    """
    Build and optimise a hybrid quantum-classical kernel.

    Usage (training):
        builder = HybridKernelBuilder(alpha=None)   # None → CV search
        K_hybrid, best_alpha, best_gamma_rbf = builder.fit(
            K_quantum_train,
            X_train_scaled,
            y_train
        )
        builder.save(checkpoint_dir)

    Usage (inference):
        K_row_hybrid = builder.predict_row(
            K_quantum_row,     # (1, N_train)
            x_new_scaled,      # (n_features,)
            X_train_scaled     # (N_train, n_features)
        )
    """

    def __init__(self, alpha=None, rbf_gamma=None):
        """
        Args:
            alpha:      Blend weight for quantum kernel (0=pure classical, 1=pure quantum).
                        None → cross-validated search over [0.0, 0.1, 0.2, ..., 1.0].
            rbf_gamma:  RBF gamma for classical kernel.
                        None → cross-validated search over [0.01, 0.05, 0.1, 0.5, 1.0, 2.0].
        """
        self.alpha     = alpha
        self.rbf_gamma = rbf_gamma

        # Set after fit()
        self.best_alpha_     = None
        self.best_rbf_gamma_ = None
        self.X_train_scaled_ = None  # stored for inference

    def fit(self, K_quantum, X_scaled, y, C=1.0, epsilon=0.1, cv=5):
        """
        Find best alpha and rbf_gamma via CV, then return blended kernel.

        Args:
            K_quantum:  (N, N) quantum training kernel (precomputed, normalised)
            X_scaled:   (N, n_features) scaled features
            y:          (N,) pIC50 labels
            C, epsilon: SVR hyperparams (use the ones from your grid search)
            cv:         Cross-validation folds

        Returns:
            K_hybrid:       (N, N) blended kernel
            best_alpha:     float
            best_rbf_gamma: float
        """
        self.X_train_scaled_ = X_scaled.copy()

        alpha_grid     = [self.alpha]    if self.alpha     is not None else [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        rbf_gamma_grid = [self.rbf_gamma] if self.rbf_gamma is not None else [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]

        best_score      = -np.inf
        best_alpha      = 0.3
        best_rbf_gamma  = 0.1

        print(f"  [HYBRID] Searching alpha ∈ {alpha_grid} × gamma_rbf ∈ {rbf_gamma_grid}")

        for rbf_g in rbf_gamma_grid:
            K_rbf = rbf_kernel(X_scaled, gamma=rbf_g)

            for alpha in alpha_grid:
                K_blend = self._blend(K_quantum, K_rbf, alpha)
                svr     = SVR(kernel="precomputed", C=C, epsilon=epsilon)

                try:
                    scores = cross_val_score(svr, K_blend, y, cv=cv, scoring="r2")
                    mean_r2 = float(scores.mean())
                except Exception:
                    mean_r2 = -999.0

                print(f"    alpha={alpha:.1f}  gamma_rbf={rbf_g}  CV_R2={mean_r2:.4f}")

                if mean_r2 > best_score:
                    best_score     = mean_r2
                    best_alpha     = alpha
                    best_rbf_gamma = rbf_g

        print(f"  [HYBRID] Best: alpha={best_alpha}  gamma_rbf={best_rbf_gamma}  CV_R2={best_score:.4f}")

        self.best_alpha_     = best_alpha
        self.best_rbf_gamma_ = best_rbf_gamma

        K_rbf_best = rbf_kernel(X_scaled, gamma=best_rbf_gamma)
        K_hybrid   = self._blend(K_quantum, K_rbf_best, best_alpha)

        return K_hybrid, best_alpha, best_rbf_gamma

    @staticmethod
    def _blend(K_quantum, K_rbf, alpha):
        """Blend two kernel matrices, clip to [0, 1], set diagonal=1."""
        K = alpha * K_quantum + (1.0 - alpha) * K_rbf
        K = np.clip(K, 0, 1)
        np.fill_diagonal(K, 1.0)
        return K

    def predict_row(self, K_quantum_row, x_new_scaled, X_train_scaled):
        """
        Build a single hybrid kernel row for inference.

        Args:
            K_quantum_row:  (1, N_train) quantum kernel row (Nystrom reconstructed)
            x_new_scaled:   (n_features,) new molecule scaled features
            X_train_scaled: (N_train, n_features) training scaled features

        Returns:
            (1, N_train) blended kernel row
        """
        K_rbf_row = rbf_kernel(
            x_new_scaled.reshape(1, -1),
            X_train_scaled,
            gamma=self.best_rbf_gamma_
        )
        K_row = (  self.best_alpha_ * K_quantum_row
                 + (1.0 - self.best_alpha_) * K_rbf_row )
        K_row = np.clip(K_row, 0, 1)
        return K_row

    def save(self, checkpoint_dir: str):
        """Persist blend params and training features for inference."""
        import os, pickle
        os.makedirs(checkpoint_dir, exist_ok=True)
        payload = {
            "best_alpha":     self.best_alpha_,
            "best_rbf_gamma": self.best_rbf_gamma_,
        }
        with open(os.path.join(checkpoint_dir, "hybrid_kernel_params_v4.pkl"), "wb") as f:
            pickle.dump(payload, f)
        np.save(os.path.join(checkpoint_dir, "hybrid_X_train_scaled_v4.npy"),
                self.X_train_scaled_)
        print(f"  [HYBRID] Saved blend params to {checkpoint_dir}")

    @classmethod
    def load(cls, checkpoint_dir: str):
        """Reload a saved HybridKernelBuilder."""
        import os, pickle
        param_path = os.path.join(checkpoint_dir, "hybrid_kernel_params_v4.pkl")
        feat_path  = os.path.join(checkpoint_dir, "hybrid_X_train_scaled_v4.npy")
        with open(param_path, "rb") as f:
            payload = pickle.load(f)
        obj                  = cls(alpha=payload["best_alpha"],
                                    rbf_gamma=payload["best_rbf_gamma"])
        obj.best_alpha_      = payload["best_alpha"]
        obj.best_rbf_gamma_  = payload["best_rbf_gamma"]
        obj.X_train_scaled_  = np.load(feat_path)
        return obj
