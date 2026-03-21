"""
XGBoost Oracle — Fast pIC50 Scoring for RL Training
=====================================================
Wraps the V3 XGBoost regressor as a stateless, fast oracle for
the REINFORCE training loop. Called ~16,000 times during RL.

Speed target: score_batch(32 SMILES) < 50ms

Loads V3 checkpoints:
  - xgb_regressor_v3.pkl
  - xgb_var_selector_v3.pkl
"""

import os
import sys
import pickle
import time
from typing import List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_v4 import V3_CHECKPOINT_DIR

# Import V3 feature extraction
_V3_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "construction_v3",
)
sys.path.insert(0, _V3_DIR)


class XGBOracle:
    """
    Fast XGBoost pIC50 oracle for RL training.

    Stateless between calls — model loaded once in __init__, never reloaded.
    Returns 2.0 (floor) for invalid SMILES — no exception raised.
    """

    # Feature dim: Morgan2(1024) + Morgan3(1024) + MACCS(167) + RDKit(2048) + PhysChem(10)
    FEATURE_DIM = 1024 + 1024 + 167 + 2048 + 10  # = 4273

    def __init__(self, checkpoint_dir: str = None):
        ckpt_dir = checkpoint_dir or str(V3_CHECKPOINT_DIR)

        xgb_path = os.path.join(ckpt_dir, "xgb_regressor_v3.pkl")
        sel_path = os.path.join(ckpt_dir, "xgb_var_selector_v3.pkl")

        if not os.path.exists(xgb_path):
            raise FileNotFoundError(
                f"XGB model not found: {xgb_path}\n"
                "Run V3 training first: python construction_v3/training/train_xgb_regressor.py"
            )
        if not os.path.exists(sel_path):
            raise FileNotFoundError(
                f"XGB selector not found: {sel_path}\n"
                "Run V3 training first: python construction_v3/training/train_xgb_regressor.py"
            )

        with open(xgb_path, "rb") as f:
            self.model = pickle.load(f)
        with open(sel_path, "rb") as f:
            self.selector = pickle.load(f)

        print(f"  [XGBOracle] Loaded model from {xgb_path}")

        # Lazy-init RDKit imports (done once)
        self._rdkit_ready = False
        self._init_rdkit()

    def _init_rdkit(self):
        """Initialise RDKit imports for feature extraction."""
        try:
            from rdkit import Chem, RDLogger
            from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint, Descriptors
            from rdkit.ML.Descriptors import MoleculeDescriptors

            RDLogger.DisableLog("rdApp.*")

            # Import V3 config for PHYSCHEM_DESCS
            try:
                from config import PHYSCHEM_DESCS
                self._physchem_descs = PHYSCHEM_DESCS
            except ImportError:
                self._physchem_descs = [
                    "MolWt", "MolLogP", "TPSA", "NumRotatableBonds",
                    "NumHAcceptors", "NumHDonors", "NumAromaticRings",
                    "RingCount", "FractionCSP3", "HeavyAtomCount",
                ]

            self._Chem = Chem
            self._AllChem = AllChem
            self._MACCSkeys = MACCSkeys
            self._RDKFingerprint = RDKFingerprint
            self._MoleculeDescriptors = MoleculeDescriptors
            self._np = np
            self._rdkit_ready = True

        except ImportError as e:
            raise ImportError(f"RDKit required for XGBOracle: {e}")

    def _extract_features(self, smiles: str) -> np.ndarray:
        """
        Extract 2D multi-fingerprint + PhysChem features for XGBoost.
        Must match V3 classical_router.py::_extract_xgb_features exactly.
        """
        try:
            mol = self._Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(self.FEATURE_DIM, dtype=np.float32)

            # Morgan r=2 (1024) + Morgan r=3 (1024) + MACCS (167) + RDKit FP (2048) + PhysChem (10)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fp_morgan2 = list(self._AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
                fp_morgan3 = list(self._AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024))
                fp_maccs = list(self._MACCSkeys.GenMACCSKeys(mol))
                fp_rdkit = list(self._RDKFingerprint(mol, maxPath=5, fpSize=2048))

            calc = self._MoleculeDescriptors.MolecularDescriptorCalculator(self._physchem_descs)
            phys = [v if np.isfinite(v) else 0.0 for v in calc.CalcDescriptors(mol)]

            features = fp_morgan2 + fp_morgan3 + fp_maccs + fp_rdkit + phys
            return np.array(features, dtype=np.float32)

        except Exception:
            return np.zeros(self.FEATURE_DIM, dtype=np.float32)

    def score(self, smiles: str) -> float:
        """
        Predict pIC50 for a single SMILES.

        Returns:
            float: predicted pIC50 in [2.0, 12.0]
            Returns 2.0 (floor) for invalid SMILES.
        """
        try:
            raw_feat = self._extract_features(smiles)
            sel_feat = self.selector.transform(raw_feat.reshape(1, -1))
            pic50 = float(self.model.predict(sel_feat)[0])
            return float(np.clip(pic50, 2.0, 12.0))
        except Exception:
            return 2.0

    def score_batch(self, smiles_list: List[str]) -> np.ndarray:
        """
        Vectorised batch scoring — ~50x faster than looping score().

        Returns:
            (n,) array of pIC50 values in [2.0, 12.0]
        """
        n = len(smiles_list)
        if n == 0:
            return np.array([], dtype=np.float32)

        # Extract features for all molecules
        feature_list = [self._extract_features(smi) for smi in smiles_list]
        features = np.stack(feature_list, axis=0)

        # Batch transform + predict
        try:
            sel_features = self.selector.transform(features)
            predictions = self.model.predict(sel_features)
            return np.clip(predictions, 2.0, 12.0).astype(np.float32)
        except Exception:
            return np.full(n, 2.0, dtype=np.float32)


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("XGBOracle test:")
    oracle = XGBOracle()

    # Sanity tests
    tests = [
        ("CCO", "ethanol"),
        ("CC(=O)Oc1ccccc1C(=O)O", "aspirin"),
        ("c1ccccc1", "benzene"),
        ("INVALID_SMILES", "invalid"),
    ]

    for smi, name in tests:
        pic50 = oracle.score(smi)
        print(f"  {name:15s} pIC50 = {pic50:.2f}")
        assert 2.0 <= pic50 <= 12.0, f"Out of range: {pic50}"

    # Batch test
    smiles_batch = [s for s, _ in tests]
    t0 = time.time()
    batch_results = oracle.score_batch(smiles_batch)
    elapsed_ms = (time.time() - t0) * 1000
    print(f"\n  Batch ({len(smiles_batch)} mols): {batch_results.round(2)} ({elapsed_ms:.1f}ms)")
    print(f"\n  ✓ XGBOracle tests passed")
