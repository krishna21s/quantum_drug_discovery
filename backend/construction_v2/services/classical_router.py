"""
Classical Router — Multi-Model Classical Prediction
=====================================================
Manages XGBoost (legacy, always available) and GNN classifier (when enabled)
with calibrated stacking for ensemble predictions.
"""

import pickle
import time
import numpy as np
from pathlib import Path

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHECKPOINT_DIR, ENABLE_GNN


class ClassicalRouter:
    """
    Multi-model classical prediction with graceful degradation.

    Models:
      - XGBoost: Always available, <50ms, loaded from checkpoint
      - GNN: Optional (ENABLE_GNN flag), ~100ms GPU / ~1s CPU
      - Stacking: Calibrated combination when both available
    """

    def __init__(
        self,
        feature_service,
        xgb_model=None,
        xgb_selector=None,
        gnn_model=None,
        calibrators=None,
    ):
        """
        Args:
            feature_service: FeatureService instance for descriptor extraction
            xgb_model: Calibrated XGBoost model (loaded from checkpoint)
            xgb_selector: VarianceThreshold selector for XGBoost
            gnn_model: Optional GNN classifier
            calibrators: Optional dict of {model_name: calibrator}
        """
        self.feature_svc = feature_service
        self.xgb_model = xgb_model
        self.xgb_selector = xgb_selector
        self.gnn_model = gnn_model
        self.calibrators = calibrators or {}

    @classmethod
    def from_checkpoints(cls, feature_service, checkpoint_dir=None):
        """
        Load XGBoost model and selector from checkpoint files.

        Args:
            feature_service: FeatureService instance
            checkpoint_dir: Directory containing xgb_model_v2.pkl and xgb_var_selector.pkl
        """
        ckpt_dir = Path(checkpoint_dir or CHECKPOINT_DIR)

        xgb_path = ckpt_dir / "xgb_model_v2.pkl"
        sel_path = ckpt_dir / "xgb_var_selector.pkl"

        if not xgb_path.exists() or not sel_path.exists():
            raise FileNotFoundError(
                f"XGBoost checkpoints not found in {ckpt_dir}. "
                "Run: python training/train_xgb_v2.py"
            )

        with open(xgb_path, "rb") as f:
            xgb_model = pickle.load(f)
        with open(sel_path, "rb") as f:
            xgb_selector = pickle.load(f)

        return cls(
            feature_service=feature_service,
            xgb_model=xgb_model,
            xgb_selector=xgb_selector,
        )

    def predict_xgb(self, smiles):
        """
        XGBoost toxicity probability.
        Target latency: ≤50ms.

        Args:
            smiles: SMILES string

        Returns:
            dict: {'probability': float, 'model': 'xgboost', 'latency_ms': float}
        """
        t0 = time.time()

        raw_feat = self.feature_svc.extract_multi_fingerprint(smiles).reshape(1, -1)
        sel_feat = self.xgb_selector.transform(raw_feat)
        prob = float(self.xgb_model.predict_proba(sel_feat)[0][1])

        return {
            "probability": prob,
            "model": "xgboost",
            "latency_ms": (time.time() - t0) * 1000,
        }

    def predict_gnn(self, smiles):
        """
        GNN classifier probability (when enabled).
        Target latency: ≤100ms (GPU), ≤1s (CPU).

        Args:
            smiles: SMILES string

        Returns:
            dict: {'probability': float, 'model': 'gnn', 'latency_ms': float}
            or None if GNN not available
        """
        if not ENABLE_GNN or self.gnn_model is None:
            return None

        t0 = time.time()
        # Placeholder for GNN prediction
        # When GNN is trained, this will use EmbeddingService
        prob = 0.5  # Will be replaced with actual GNN inference
        return {
            "probability": prob,
            "model": "gnn",
            "latency_ms": (time.time() - t0) * 1000,
        }

    def predict_stacked(self, smiles):
        """
        Calibrated stacking of all available classical models.

        When GNN is available: weighted average of calibrated XGB + GNN.
        Otherwise: falls back to XGBoost alone.

        Args:
            smiles: SMILES string

        Returns:
            dict: {
                'probability': float,
                'model': 'stacked' | 'xgboost',
                'components': dict,
                'latency_ms': float,
            }
        """
        t0 = time.time()

        xgb_result = self.predict_xgb(smiles)
        components = {"xgboost": xgb_result["probability"]}

        gnn_result = self.predict_gnn(smiles)
        if gnn_result is not None:
            components["gnn"] = gnn_result["probability"]
            # Simple calibrated average (can be upgraded to trained stacker later)
            prob = 0.6 * xgb_result["probability"] + 0.4 * gnn_result["probability"]
            model_name = "stacked"
        else:
            prob = xgb_result["probability"]
            model_name = "xgboost"

        # Apply calibrator if available
        if model_name in self.calibrators:
            prob = float(
                self.calibrators[model_name].predict_proba(np.array([[prob]]))[0][1]
            )

        return {
            "probability": prob,
            "model": model_name,
            "components": components,
            "latency_ms": (time.time() - t0) * 1000,
        }

    @property
    def available_models(self):
        """List of currently available classical models."""
        models = []
        if self.xgb_model is not None:
            models.append("xgboost")
        if self.gnn_model is not None and ENABLE_GNN:
            models.append("gnn")
        return models
