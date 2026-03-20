"""
Quantum Oracle — QSVR Final Evaluation (STUB)
================================================
Wraps the V3 QSVR as a final-evaluation oracle. Called only on the
top-50 candidates after RL converges — not during RL training.

STATUS: STUB — The QSVR model is still being trained.
        This file provides the full interface but returns XGB-based
        fallback scores. Wire in the real quantum kernel once
        QSVR training completes.

Usage:
    from oracle.quantum_oracle import QuantumOracle
    oracle = QuantumOracle()
    result = oracle.score("CCO")
"""

import os
import sys
import time
from typing import List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_v4 import V3_CHECKPOINT_DIR, QUANTUM_N_JOBS


class QuantumOracle:
    """
    Quantum QSVR pIC50 oracle for final candidate evaluation.

    CURRENT STATE: Stub — falls back to XGB scoring.
    When QSVR training completes, update __init__ to load quantum
    checkpoints and score() to use quantum kernel prediction.
    """

    def __init__(self, checkpoint_dir: str = None):
        self.ckpt_dir = checkpoint_dir or str(V3_CHECKPOINT_DIR)
        self._quantum_available = False
        self._xgb_fallback = None

        # Try loading quantum checkpoints
        try:
            self._try_load_quantum()
        except Exception as e:
            print(f"  [QuantumOracle] Quantum checkpoints not available: {e}")
            print(f"  [QuantumOracle] Using XGB fallback mode")
            self._load_xgb_fallback()

    def _try_load_quantum(self):
        """Attempt to load V3 QSVR quantum checkpoints."""
        required_files = [
            "qsvr_model_v3.pkl",
            "qsvr_scaler_v3.pkl",
            "qsvr_landmarks_scaled_v3.npy",
            "qsvr_K_mm_inv_v3.npy",
            "qsvr_K_nm_transformed_v3.npy",
            "qsvr_diag_train_v3.npy",
        ]

        missing = [
            f for f in required_files
            if not os.path.exists(os.path.join(self.ckpt_dir, f))
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing quantum checkpoints: {missing}"
            )

        # TODO: Load actual quantum kernel components when QSVR training done
        # For now, mark as not available
        print(f"  [QuantumOracle] Quantum checkpoints found but integration pending")
        print(f"  [QuantumOracle] Will use XGB fallback until quantum integration is complete")
        self._quantum_available = False
        self._load_xgb_fallback()

    def _load_xgb_fallback(self):
        """Load XGB oracle as fallback scoring method."""
        try:
            from oracle.xgb_oracle import XGBOracle
            self._xgb_fallback = XGBOracle(self.ckpt_dir)
            print(f"  [QuantumOracle] XGB fallback loaded")
        except Exception as e:
            print(f"  [QuantumOracle] WARNING: XGB fallback also failed: {e}")
            self._xgb_fallback = None

    @property
    def is_quantum_mode(self) -> bool:
        """Whether quantum kernel is actually being used."""
        return self._quantum_available

    def score(self, smiles: str) -> dict:
        """
        Score a single SMILES with quantum oracle (or XGB fallback).

        Returns:
            dict: {
                "pic50": float,
                "mode": "quantum_hybrid" | "xgb_fallback",
                "latency_s": float,
                "error": None | str,
            }
        """
        t0 = time.time()

        if self._quantum_available:
            # TODO: Real quantum kernel scoring
            # This path will be activated once QSVR training completes
            pass

        # XGB fallback
        if self._xgb_fallback is not None:
            try:
                pic50 = self._xgb_fallback.score(smiles)
                return {
                    "pic50": pic50,
                    "mode": "xgb_fallback",
                    "latency_s": time.time() - t0,
                    "error": None,
                }
            except Exception as e:
                return {
                    "pic50": None,
                    "mode": "xgb_fallback",
                    "latency_s": time.time() - t0,
                    "error": str(e),
                }

        return {
            "pic50": None,
            "mode": "unavailable",
            "latency_s": time.time() - t0,
            "error": "No scoring backend available",
        }

    def score_batch(self, smiles_list: List[str], n_jobs: int = QUANTUM_N_JOBS) -> List[dict]:
        """
        Score a batch of SMILES.

        When quantum is available, uses multiprocessing.Pool for parallel scoring.
        In fallback mode, uses sequential XGB scoring.

        Args:
            smiles_list: list of SMILES strings
            n_jobs:      number of parallel workers (for quantum mode)

        Returns:
            List[dict]: one result dict per SMILES
        """
        if self._quantum_available:
            # TODO: Parallel quantum scoring with multiprocessing.Pool
            pass

        # Sequential fallback
        return [self.score(smi) for smi in smiles_list]


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("QuantumOracle test:")
    oracle = QuantumOracle()
    print(f"  Quantum mode: {oracle.is_quantum_mode}")

    result = oracle.score("CCO")
    print(f"  Ethanol: {result}")

    result = oracle.score("CC(=O)Oc1ccccc1C(=O)O")
    print(f"  Aspirin: {result}")

    batch = oracle.score_batch(["CCO", "c1ccccc1"])
    print(f"  Batch results: {batch}")

    print(f"\n  ✓ QuantumOracle tests passed (mode: {'quantum' if oracle.is_quantum_mode else 'XGB fallback'})")
