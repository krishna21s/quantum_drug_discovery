"""
Feature Service — Unified Molecular Feature Extraction
=======================================================
Consolidates all descriptor/fingerprint extraction from V1 into one
service with canonical SMILES normalization and in-memory caching.

Source lineage:
  - extract_rich_descriptors() ← core_engine_shot.py
  - extract_xgb_features()    ← app_with_validation.py
  - extract_features()        ← train_xgb_v2.py
  - get_orthogonal_features() ← app_with_validation.py (inner fn)
"""

import json
import numpy as np
from functools import lru_cache
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem, MACCSkeys

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PHYSCHEM_DESCS, MULTI_FP_DIM, CHECKPOINT_DIR

RDLogger.DisableLog("rdApp.*")


class FeatureService:
    """
    Single source of truth for all molecular feature extraction.

    Provides:
      1. Multi-fingerprint vector (4278-d) for XGBoost
      2. Rich descriptor pool for orthogonal filtering
      3. Orthogonal descriptors (20-d) for quantum kernel
      4. Canonical SMILES normalization
    """

    def __init__(self, selected_features_path=None):
        """
        Args:
            selected_features_path: Path to JSON listing the 20 selected
                orthogonal feature names (output of core_engine_shot.py).
                If None, tries CHECKPOINT_DIR/selected_features.json.
        """
        self._cache = {}  # canonical_smiles → {multi_fp, ortho_desc}
        self._selected_features = None

        if selected_features_path is None:
            selected_features_path = CHECKPOINT_DIR / "selected_features.json"

        if Path(selected_features_path).exists():
            with open(selected_features_path, "r") as f:
                self._selected_features = json.load(f)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def canonical_smiles(self, smiles: str) -> str | None:
        """Return RDKit canonical SMILES, or None if invalid."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)

    def extract_multi_fingerprint(self, smiles: str) -> np.ndarray:
        """
        Morgan r2 (1024) + Morgan r3 (1024) + MACCS (167) +
        RDKit Topo (2048) + PhysChem (15) = 4278-d flat vector.

        Used by XGBoost / classical router.
        Returns zeros if SMILES is invalid.
        """
        canon = self.canonical_smiles(smiles)
        if canon and canon in self._cache and "multi_fp" in self._cache[canon]:
            return self._cache[canon]["multi_fp"]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(MULTI_FP_DIM, dtype=np.float32)

        fp_m2 = np.array(
            AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024),
            dtype=np.float32,
        )
        fp_m3 = np.array(
            AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024),
            dtype=np.float32,
        )
        fp_maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
        fp_rdk = np.array(Chem.RDKFingerprint(mol, fpSize=2048), dtype=np.float32)

        desc = Descriptors.CalcMolDescriptors(mol)
        phys = np.array(
            [float(desc.get(d, 0.0)) for d in PHYSCHEM_DESCS], dtype=np.float32
        )
        phys = np.nan_to_num(phys, nan=0.0, posinf=0.0, neginf=0.0)

        result = np.concatenate([fp_m2, fp_m3, fp_maccs, fp_rdk, phys])

        # Cache
        if canon:
            self._cache.setdefault(canon, {})["multi_fp"] = result

        return result

    def extract_rich_descriptors(self, smiles: str) -> dict | None:
        """
        Compute ALL safe RDKit descriptors (typically ~200).
        Used for orthogonal filtering during training.
        Returns None if SMILES is invalid.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        desc_dict = Descriptors.CalcMolDescriptors(mol)
        safe_descs = {
            k: float(v)
            for k, v in desc_dict.items()
            if not np.isnan(v) and not np.isinf(v)
        }
        return safe_descs

    def extract_orthogonal_descriptors(
        self, smiles: str, selected_features: list[str] | None = None
    ) -> np.ndarray:
        """
        Extract exactly the 20 orthogonal descriptors used by the quantum kernel.
        Falls back to instance-level selected_features if not provided.
        Returns zeros if SMILES is invalid.
        """
        features = selected_features or self._selected_features
        if features is None:
            raise ValueError(
                "No selected features available. Provide selected_features or "
                "ensure checkpoints/selected_features.json exists."
            )

        canon = self.canonical_smiles(smiles)
        n_feats = len(features)

        if canon and canon in self._cache and "ortho" in self._cache[canon]:
            return self._cache[canon]["ortho"]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_feats, dtype=np.float64)

        desc_dict = Descriptors.CalcMolDescriptors(mol)
        result = np.array(
            [float(desc_dict.get(f, 0.0)) for f in features], dtype=np.float64
        )

        # Cache
        if canon:
            self._cache.setdefault(canon, {})["ortho"] = result

        return result

    def build_feature_names(self) -> list[str]:
        """Return ordered feature name list for the 4278-d vector."""
        names = [f"Morgan2_{i}" for i in range(1024)]
        names += [f"Morgan3_{i}" for i in range(1024)]
        names += [f"MACCS_{i}" for i in range(167)]
        names += [f"RDKit_{i}" for i in range(2048)]
        names += PHYSCHEM_DESCS
        return names

    def baseline_rule_score(self, smiles: str) -> float:
        """
        Simple heuristic toxicity score (0..1) based on substructural alerts.
        NOT a model — used as a quick baseline for spotting big disagreements.
        Preserved from V1 app_with_validation.py.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0

        score = 0.0

        # Nitro group substructure
        if mol.HasSubstructMatch(Chem.MolFromSmarts("[N+](=O)[O-]")):
            score += 0.35

        # Many aromatic rings
        try:
            narom = Descriptors.NumAromaticRings(mol)
            if narom >= 2:
                score += min(0.25, 0.08 * narom)
        except Exception:
            pass

        # Heavy atoms
        hat = Descriptors.HeavyAtomCount(mol)
        if hat > 30:
            score += 0.15

        # High logP
        try:
            logp = Descriptors.MolLogP(mol)
            if logp > 3.5:
                score += min(0.25, 0.05 * (logp - 3.5))
        except Exception:
            pass

        return float(np.clip(score, 0.0, 1.0))

    def clear_cache(self):
        """Clear in-memory feature cache."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        return len(self._cache)
