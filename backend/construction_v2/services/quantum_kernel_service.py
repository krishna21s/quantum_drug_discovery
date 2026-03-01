"""
Quantum Kernel Service — Two-Mode Kernel Prediction
=====================================================
Orchestrates quantum kernel computation in two modes:
  1. Statevector (fast, deterministic): for screening
  2. Shot-based (noisy, with CI): for final evaluation

Combines backends, Nystrom engine, and error mitigation.
"""

import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_QUBITS, N_SHOTS


class QuantumKernelService:
    """
    Two-mode quantum kernel prediction service.

    Modes:
      - 'statevector': Fast deterministic (for screening, cached)
      - 'shot': Hardware-realistic with measurement counts (for final eval)
    """

    def __init__(
        self,
        backend_sv,
        backend_shot,
        nystrom_engine,
        svm_model,
        scaler,
        landmarks_scaled,
        feature_service=None,
        selected_features=None,
    ):
        """
        Args:
            backend_sv: StatevectorBackend instance
            backend_shot: ShotBackend instance
            nystrom_engine: NystromEngine with loaded K_mm_inv, K_nm, diag_train
            svm_model: Fitted SVC(kernel='precomputed')
            scaler: MinMaxScaler fitted on training data
            landmarks_scaled: (m, d) scaled landmark vectors
            feature_service: FeatureService for descriptor extraction
            selected_features: List of orthogonal feature names
        """
        self.backend_sv = backend_sv
        self.backend_shot = backend_shot
        self.nystrom = nystrom_engine
        self.svm_model = svm_model
        self.scaler = scaler
        self.landmarks_scaled = landmarks_scaled
        self.feature_svc = feature_service
        self.selected_features = selected_features

        # Cache for kernel rows (canonical_smiles → kernel_row)
        self._kernel_cache = {}

    def predict(self, smiles, mode="statevector", progress_callback=None):
        """
        Predict toxicity probability using quantum kernel.

        Args:
            smiles: SMILES string
            mode: 'statevector' (fast) or 'shot' (noisy + counts)
            progress_callback: Callable(step, total) for progress reporting

        Returns:
            dict: {
                'probability': float,
                'mode': str,
                'latency_s': float,
                'kernel_row': np.ndarray (if available)
            }
        """
        t0 = time.time()

        # Get scaled features
        phys_scaled = self._get_scaled_features(smiles)

        # Select backend
        backend = self.backend_sv if mode == "statevector" else self.backend_shot

        # Check cache (only for statevector mode)
        cache_key = None
        if mode == "statevector" and self.feature_svc:
            cache_key = self.feature_svc.canonical_smiles(smiles)
            if cache_key and cache_key in self._kernel_cache:
                K_new_m = self._kernel_cache[cache_key]
                prob = self.nystrom.predict_from_kernel_row(
                    K_new_m, svm_model=self.svm_model
                )
                return {
                    "probability": prob,
                    "mode": mode,
                    "latency_s": time.time() - t0,
                    "cached": True,
                }

        # Compute kernel row
        m = len(self.landmarks_scaled)
        K_new_m = np.zeros((1, m))
        for j in range(m):
            K_new_m[0, j] = backend.fidelity(phys_scaled, self.landmarks_scaled[j])
            if progress_callback:
                progress_callback(j + 1, m)

        # Cache the kernel row
        if cache_key:
            self._kernel_cache[cache_key] = K_new_m

        # Predict
        prob = self.nystrom.predict_from_kernel_row(K_new_m, svm_model=self.svm_model)

        return {
            "probability": prob,
            "mode": mode,
            "latency_s": time.time() - t0,
            "cached": False,
            "kernel_row": K_new_m,
        }

    def predict_with_ci(self, smiles, n_bootstrap=10, progress_callback=None):
        """
        Shot-based prediction with bootstrap confidence interval.

        Runs n_bootstrap independent shot-based kernel computations
        and returns mean ± CI.

        Args:
            smiles: SMILES string
            n_bootstrap: Number of bootstrap repeats
            progress_callback: Callable(step, total)

        Returns:
            dict: {
                'probability': float (mean),
                'std': float,
                'ci_lower': float (2.5th percentile),
                'ci_upper': float (97.5th percentile),
                'raw_probs': list[float],
                'mode': 'shot_bootstrap',
                'latency_s': float,
            }
        """
        t0 = time.time()
        phys_scaled = self._get_scaled_features(smiles)
        m = len(self.landmarks_scaled)

        probs = []
        total_steps = n_bootstrap * m

        for rep in range(n_bootstrap):
            K_new_m = np.zeros((1, m))
            for j in range(m):
                K_new_m[0, j] = self.backend_shot.fidelity(
                    phys_scaled, self.landmarks_scaled[j]
                )
                if progress_callback:
                    step = rep * m + j + 1
                    progress_callback(step, total_steps)

            prob = self.nystrom.predict_from_kernel_row(
                K_new_m, svm_model=self.svm_model
            )
            probs.append(prob)

        probs_arr = np.array(probs)
        return {
            "probability": float(np.mean(probs_arr)),
            "std": float(np.std(probs_arr)),
            "ci_lower": float(np.percentile(probs_arr, 2.5)),
            "ci_upper": float(np.percentile(probs_arr, 97.5)),
            "raw_probs": probs,
            "mode": "shot_bootstrap",
            "latency_s": time.time() - t0,
            "n_bootstrap": n_bootstrap,
        }

    def _get_scaled_features(self, smiles):
        """Extract and scale orthogonal features for a SMILES string."""
        if self.feature_svc:
            raw = self.feature_svc.extract_orthogonal_descriptors(
                smiles, self.selected_features
            )
        else:
            # Fallback: assume descriptors are passed externally
            raise ValueError("FeatureService required for SMILES-based prediction")

        scaled = np.nan_to_num(self.scaler.transform(raw.reshape(1, -1)))[0]
        return scaled

    def clear_cache(self):
        """Clear kernel row cache."""
        self._kernel_cache.clear()

    @property
    def cache_size(self):
        return len(self._kernel_cache)
