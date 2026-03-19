"""
Quantum Kernel Service — Two-Mode QSVR pIC50 Prediction
=========================================================
Orchestrates quantum kernel computation in two modes for regression:
  1. Statevector (fast, deterministic): for screening
  2. Shot-based (noisy, with CI): for final evaluation

Key change from V2:
  Predicts continuous pIC50 value (regression) instead of toxicity probability.
"""

import time
import numpy as np
from typing import Optional, Callable

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_QUBITS, N_SHOTS


class QuantumKernelService:
    """
    Two-mode quantum kernel pIC50 prediction service (QSVR).

    Modes:
      - 'statevector': Fast deterministic (for screening)
      - 'shot':        Hardware-realistic with measurement counts (for final eval)
    """

    def __init__(
        self,
        backend_sv,
        backend_shot,
        nystrom_engine,
        svr_model,
        scaler,
        landmarks_scaled,
        feature_service=None,
        selected_features=None,
    ):
        """
        Args:
            backend_sv:        StatevectorBackend instance
            backend_shot:      ShotBackend instance
            nystrom_engine:    NystromEngine with loaded K_mm_inv, K_nm, diag_train
            svr_model:         Fitted SVR(kernel='precomputed') — regression model
            scaler:            MinMaxScaler fitted on training data
            landmarks_scaled:  (m, d) scaled landmark vectors
            feature_service:   FeatureService3D for descriptor extraction
            selected_features: List of 20 orthogonal feature names
        """
        self.backend_sv       = backend_sv
        self.backend_shot     = backend_shot
        self.nystrom          = nystrom_engine
        self.svr_model        = svr_model
        self.scaler           = scaler
        self.landmarks_scaled = landmarks_scaled
        self.feature_svc      = feature_service
        self.selected_features = selected_features

        # Cache: canonical_smiles → kernel row
        self._kernel_cache = {}

    def predict_pic50(self, smiles: str, mode: str = "statevector",
                      progress_callback: Optional[Callable] = None) -> dict:
        """
        Predict pIC50 using the quantum kernel SVR.

        Args:
            smiles:            SMILES string
            mode:              'statevector' (fast) or 'shot' (noisy + counts)
            progress_callback: Callable(step, total) for progress reporting

        Returns:
            dict: {
                'pic50': float,
                'mode': str,
                'latency_s': float,
                'cached': bool,
                'kernel_row': np.ndarray (if not cached)
            }
        """
        t0 = time.time()

        # Get scaled 3D features
        phys_scaled = self._get_scaled_features(smiles)
        backend     = self.backend_sv if mode == "statevector" else self.backend_shot

        # Cache lookup (statevector only)
        cache_key = None
        if mode == "statevector" and self.feature_svc:
            cache_key = self.feature_svc.canonical_smiles(smiles)
            if cache_key and cache_key in self._kernel_cache:
                K_new_m = self._kernel_cache[cache_key]
                pic50   = self.nystrom.predict_pic50_from_kernel_row(
                    K_new_m, svr_model=self.svr_model
                )
                return {
                    "pic50":     pic50,
                    "mode":      mode,
                    "latency_s": time.time() - t0,
                    "cached":    True,
                }

        # Compute kernel row
        m       = len(self.landmarks_scaled)
        K_new_m = np.zeros((1, m))
        for j in range(m):
            K_new_m[0, j] = backend.fidelity(phys_scaled, self.landmarks_scaled[j])
            if progress_callback:
                progress_callback(j + 1, m)

        if cache_key:
            self._kernel_cache[cache_key] = K_new_m

        pic50 = self.nystrom.predict_pic50_from_kernel_row(K_new_m, svr_model=self.svr_model)

        return {
            "pic50":      pic50,
            "mode":       mode,
            "latency_s":  time.time() - t0,
            "cached":     False,
            "kernel_row": K_new_m,
        }

    def predict_with_ci(self, smiles: str, n_bootstrap: int = 10,
                        progress_callback: Optional[Callable] = None) -> dict:
        """
        Shot-based prediction with bootstrap confidence interval.

        Returns:
            dict: {
                'pic50': float (mean),
                'std': float,
                'ci_lower': float (2.5th pct),
                'ci_upper': float (97.5th pct),
                'raw_pic50s': list[float],
                'mode': 'shot_bootstrap',
                'latency_s': float,
            }
        """
        t0           = time.time()
        phys_scaled  = self._get_scaled_features(smiles)
        m            = len(self.landmarks_scaled)
        pic50s       = []
        total_steps  = n_bootstrap * m

        for rep in range(n_bootstrap):
            K_new_m = np.zeros((1, m))
            for j in range(m):
                K_new_m[0, j] = self.backend_shot.fidelity(phys_scaled, self.landmarks_scaled[j])
                if progress_callback:
                    progress_callback(rep * m + j + 1, total_steps)

            pic50 = self.nystrom.predict_pic50_from_kernel_row(K_new_m, svr_model=self.svr_model)
            pic50s.append(pic50)

        arr = np.array(pic50s)
        return {
            "pic50":       float(np.mean(arr)),
            "std":         float(np.std(arr)),
            "ci_lower":    float(np.percentile(arr, 2.5)),
            "ci_upper":    float(np.percentile(arr, 97.5)),
            "raw_pic50s":  pic50s,
            "mode":        "shot_bootstrap",
            "latency_s":   time.time() - t0,
            "n_bootstrap": n_bootstrap,
        }

    def _get_scaled_features(self, smiles: str) -> np.ndarray:
        """Extract and scale 3D orthogonal features for a SMILES string."""
        if self.feature_svc is None:
            raise ValueError("FeatureService3D required for SMILES-based prediction.")
        raw    = self.feature_svc.extract_orthogonal_descriptors(smiles, self.selected_features)
        scaled = np.nan_to_num(self.scaler.transform(raw.reshape(1, -1)))[0]
        return scaled

    def clear_cache(self):
        self._kernel_cache.clear()

    @property
    def cache_size(self) -> int:
        return len(self._kernel_cache)
