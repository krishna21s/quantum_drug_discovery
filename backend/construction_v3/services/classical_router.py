"""
Classical Router — XGBRegressor for pIC50 Prediction
======================================================
Manages the XGBoost regression model for continuous pIC50 prediction.
Uses 2D fingerprints + PhysChem descriptors as the topological baseline.
"""

import pickle
import time
import numpy as np
from pathlib import Path
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHECKPOINT_DIR


class ClassicalRouter:
    """
    XGBRegressor-based pIC50 prediction service (topological baseline).
    
    Uses multi-fingerprint features (Morgan + RDKit + MACCS + PhysChem).
    Checkpointed after Optuna optimization in train_xgb_regressor.py.
    """

    def __init__(self, feature_service, xgb_model=None, xgb_selector=None):
        """
        Args:
            feature_service: FeatureService3D instance
            xgb_model: Fitted XGBRegressor loaded from checkpoint
            xgb_selector: VarianceThreshold selector fitted on training data
        """
        self.feature_svc = feature_service
        self.xgb_model   = xgb_model
        self.xgb_selector = xgb_selector

    @classmethod
    def from_checkpoints(cls, feature_service, checkpoint_dir=None):
        """Load XGBRegressor and selector from checkpoint files."""
        ckpt_dir = Path(checkpoint_dir or CHECKPOINT_DIR)

        xgb_path = ckpt_dir / "xgb_regressor_v3.pkl"
        sel_path  = ckpt_dir / "xgb_var_selector_v3.pkl"

        if not xgb_path.exists() or not sel_path.exists():
            raise FileNotFoundError(
                f"XGBRegressor checkpoints not found in {ckpt_dir}.\n"
                "Run: python training/train_xgb_regressor.py"
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

    def predict_pic50(self, smiles: str) -> dict:
        """
        Predict pIC50 via XGBRegressor.

        Args:
            smiles: SMILES string

        Returns:
            dict: {
                'pic50': float,
                'model': 'xgb_regressor',
                'latency_ms': float,
            }
        """
        t0 = time.time()

        raw_feat = self._extract_xgb_features(smiles)
        sel_feat  = self.xgb_selector.transform(raw_feat.reshape(1, -1))
        pic50     = float(self.xgb_model.predict(sel_feat)[0])
        # Clamp to physically meaningful range
        pic50 = float(np.clip(pic50, 2.0, 12.0))

        return {
            "pic50":      pic50,
            "model":      "xgb_regressor",
            "latency_ms": (time.time() - t0) * 1000,
        }

    def _extract_xgb_features(self, smiles: str) -> np.ndarray:
        """Extract 2D multi-fingerprint + PhysChem features for XGBoost."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint, Descriptors
            from rdkit.ML.Descriptors import MoleculeDescriptors
            from config import PHYSCHEM_DESCS

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(4278, dtype=np.float32)

            fp_morgan2 = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
            fp_morgan3 = list(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024))
            fp_maccs   = list(MACCSkeys.GenMACCSKeys(mol))
            fp_rdkit   = list(RDKFingerprint(mol, maxPath=5, fpSize=2048))

            calc = MoleculeDescriptors.MolecularDescriptorCalculator(PHYSCHEM_DESCS)
            phys = [v if np.isfinite(v) else 0.0 for v in calc.CalcDescriptors(mol)]

            features = (
                fp_morgan2 + fp_morgan3 + fp_maccs + fp_rdkit + phys
            )
            return np.array(features, dtype=np.float32)

        except Exception:
            return np.zeros(4278, dtype=np.float32)

    @property
    def is_ready(self) -> bool:
        return self.xgb_model is not None and self.xgb_selector is not None
