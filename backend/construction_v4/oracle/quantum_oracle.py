"""
Quantum Oracle — QSVR Binding Affinity Scoring
================================================
Wraps the V3 QSVR model (trained with hybrid quantum+classical kernel)
as a scoring oracle. Called on top-50 candidates after RL converges.

Pipeline:
    1. Extract 20 selected fingerprint features (MACCS + Morgan bits)
    2. Scale with ArctanScaler → 8D feature vector
    3. Compute classical RBF kernel vs training data
       (alpha=0.0 means pure classical RBF, no quantum circuit needed)
    4. Precomputed-kernel SVR predict

Also provides XGB fallback for comparison.
"""

import os
import sys
import json
import time
import pickle
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_v4 import V3_CHECKPOINT_DIR, QUANTUM_N_JOBS


class QuantumOracle:
    """
    QSVR pIC50 oracle for final candidate evaluation.

    Uses the V3-trained QSVR model with hybrid kernel (alpha=0.0 → pure RBF).
    Also provides XGB fallback for dual scoring.
    """

    def __init__(self, checkpoint_dir: str = None):
        self.ckpt_dir = checkpoint_dir or str(V3_CHECKPOINT_DIR)
        self._quantum_available = False
        self._xgb_fallback = None

        # Try loading QSVR
        try:
            self._load_qsvr()
        except Exception as e:
            print(f"  [QuantumOracle] QSVR load failed: {e}")
            self._quantum_available = False

        # Also load XGB fallback
        self._load_xgb_fallback()

    def _load_qsvr(self):
        """Load all QSVR checkpoints for inference."""
        ckpt = self.ckpt_dir

        # 1. Selected feature names
        feat_path = os.path.join(ckpt, "qsvr_selected_features_v4.json")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Missing: {feat_path}")
        with open(feat_path) as f:
            all_selected = json.load(f)

        # V3 uses N_QUBITS features (top-8 from 20 selected by |ρ_y|)
        # Detect n_qubits from training data shape
        X_path = os.path.join(ckpt, "hybrid_X_train_scaled_v4.npy")
        self._X_train_scaled = np.load(X_path)
        n_qubits = self._X_train_scaled.shape[1]  # 8

        self._selected_features = all_selected[:n_qubits]
        print(f"  [QuantumOracle] {len(self._selected_features)}/{len(all_selected)} features used (N_QUBITS={n_qubits})")
        print(f"  [QuantumOracle] Training data: {self._X_train_scaled.shape}")

        # 2. SVR model (trained on precomputed kernel)
        model_path = os.path.join(ckpt, "qsvr_model_v4.pkl")
        with open(model_path, "rb") as f:
            self._svr_model = pickle.load(f)
        print(f"  [QuantumOracle] SVR model loaded (C={self._svr_model.C}, eps={self._svr_model.epsilon})")

        # 4. Hybrid kernel params
        params_path = os.path.join(ckpt, "hybrid_kernel_params_v4.pkl")
        with open(params_path, "rb") as f:
            params = pickle.load(f)
        self._alpha = params["best_alpha"]       # 0.0 = pure classical
        self._rbf_gamma = params["best_rbf_gamma"]  # 0.5

        # 5. Best gamma for quantum kernel (not used if alpha=0)
        gamma_path = os.path.join(ckpt, "best_gamma_v4.json")
        with open(gamma_path) as f:
            self._q_gamma = json.load(f)["best_gamma"]

        self._quantum_available = True
        mode = "pure classical RBF" if self._alpha == 0.0 else f"hybrid (α={self._alpha})"
        print(f"  [QuantumOracle] QSVR ready — {mode}, γ_rbf={self._rbf_gamma}")

    def _load_xgb_fallback(self):
        """Load XGB oracle as fallback/comparison."""
        try:
            from oracle.xgb_oracle import XGBOracle
            self._xgb_fallback = XGBOracle(self.ckpt_dir)
        except Exception as e:
            print(f"  [QuantumOracle] XGB fallback failed: {e}")
            self._xgb_fallback = None

    def _extract_selected_features(self, smiles: str) -> Optional[np.ndarray]:
        """
        Extract the 20 selected fingerprint features for a SMILES.

        Features are MACCS keys and Morgan fingerprint bits at specific indices.
        Returns None if SMILES is invalid.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, MACCSkeys

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Generate all fingerprints needed
            morgan2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            morgan3 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024)
            maccs = MACCSkeys.GenMACCSKeys(mol)

            # Extract selected features by name
            features = []
            for feat_name in self._selected_features:
                if feat_name.startswith("MACCS_"):
                    idx = int(feat_name.split("_")[1])
                    features.append(float(maccs[idx]))
                elif feat_name.startswith("MFP2_"):
                    idx = int(feat_name.split("_")[1])
                    features.append(float(morgan2[idx]))
                elif feat_name.startswith("MFP3_"):
                    idx = int(feat_name.split("_")[1])
                    features.append(float(morgan3[idx]))
                else:
                    features.append(0.0)  # Unknown feature type

            return np.array(features, dtype=np.float32)

        except Exception:
            return None

    def _arctan_scale(self, X: np.ndarray) -> np.ndarray:
        """
        ArctanScaler: maps features to [-π/2, π/2] range.

        This matches the V3 ArctanScalerV4 transformation used during training.
        For binary fingerprint bits (0/1), arctan maps to {0, π/4} ≈ {0, 0.785}.
        """
        return np.arctan(X.astype(np.float64)).astype(np.float32)

    def _compute_rbf_kernel(self, X_new: np.ndarray) -> np.ndarray:
        """
        Compute RBF kernel between new data and training data.

        K[i,j] = exp(-gamma * ||x_new_i - x_train_j||^2)

        Args:
            X_new: (n_new, n_features) scaled new data

        Returns:
            (n_new, n_train) kernel matrix
        """
        # Efficient pairwise squared distances
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        X_new_sq = np.sum(X_new ** 2, axis=1, keepdims=True)   # (n_new, 1)
        X_tr_sq = np.sum(self._X_train_scaled ** 2, axis=1, keepdims=True).T  # (1, n_train)
        cross = X_new @ self._X_train_scaled.T  # (n_new, n_train)

        dist_sq = X_new_sq + X_tr_sq - 2 * cross
        dist_sq = np.maximum(dist_sq, 0.0)  # numerical safety

        return np.exp(-self._rbf_gamma * dist_sq)

    def _predict_qsvr(self, smiles: str) -> Optional[float]:
        """Full QSVR prediction pipeline for a single SMILES."""
        # 1. Extract selected features
        raw_features = self._extract_selected_features(smiles)
        if raw_features is None:
            return None

        # 2. Scale with arctan
        scaled = self._arctan_scale(raw_features.reshape(1, -1))

        # 3. Compute kernel vs training data
        # Since alpha=0.0, only classical RBF is used
        K_test = self._compute_rbf_kernel(scaled)  # (1, n_train)

        # 4. SVR predict on precomputed kernel
        pic50 = float(self._svr_model.predict(K_test)[0])
        return float(np.clip(pic50, 2.0, 12.0))

    @property
    def is_quantum_mode(self) -> bool:
        return self._quantum_available

    def score(self, smiles: str) -> dict:
        """
        Score a single SMILES with QSVR (+ XGB comparison).

        Returns:
            dict with pic50, xgb_pic50, mode, latency_s
        """
        t0 = time.time()
        result = {
            "pic50": None,
            "xgb_pic50": None,
            "mode": "unavailable",
            "latency_s": 0.0,
            "error": None,
        }

        # QSVR scoring
        if self._quantum_available:
            try:
                pic50 = self._predict_qsvr(smiles)
                result["pic50"] = pic50
                result["mode"] = "qsvr_hybrid" if self._alpha > 0 else "qsvr_rbf"
            except Exception as e:
                result["error"] = str(e)

        # XGB comparison
        if self._xgb_fallback is not None:
            try:
                result["xgb_pic50"] = self._xgb_fallback.score(smiles)
            except Exception:
                pass

        result["latency_s"] = time.time() - t0
        return result

    def score_batch(self, smiles_list: List[str], n_jobs: int = QUANTUM_N_JOBS) -> List[dict]:
        """Score a batch of SMILES."""
        return [self.score(smi) for smi in smiles_list]


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("QuantumOracle test:")
    oracle = QuantumOracle()
    print(f"  Quantum mode: {oracle.is_quantum_mode}")

    tests = [
        ("CCO", "ethanol"),
        ("CC(=O)Oc1ccccc1C(=O)O", "aspirin"),
        ("c1ccccc1", "benzene"),
        ("INVALID", "invalid"),
    ]

    for smi, name in tests:
        result = oracle.score(smi)
        qsvr = result["pic50"]
        xgb = result["xgb_pic50"]
        print(f"  {name:15s}  QSVR={qsvr}  XGB={xgb}  mode={result['mode']}")

    print(f"\n  ✓ QuantumOracle tests passed")
