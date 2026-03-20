"""
Feature Service 3D — RDKit 3D Conformer + WHIM/3D-MoRSE Descriptors
=====================================================================
Generates 3D molecular geometry using ETKDG v3 + MMFF94 optimization,
then extracts quantum-ready 3D descriptors for the 20-qubit QSVR.

Pipeline:
  1. Parse SMILES → RDKit Mol (with Hs)
  2. Generate 3D conformer using ETKDG v3
  3. Optimize geometry with MMFF94 force field
  4. Extract WHIM (symmetry/size) + 3D-MoRSE (scattering) descriptors
  5. Apply Pearson filter (|ρ| < 0.85) → select 20 orthogonal features

The resulting 20-dimensional vector maps 1-to-1 to the 20 qubits.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    N_3D_FEATURES, CONFORMER_ATTEMPTS, PEARSON_THRESHOLD,
    PHYSCHEM_DESCS, CHECKPOINT_DIR, RANDOM_STATE
)

# Silence RDKit warnings
warnings.filterwarnings("ignore")

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
    from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
    from rdkit.ML.Descriptors import MoleculeDescriptors
    RDKIT_AVAILABLE = True
except ImportError as e:
    RDKIT_AVAILABLE = False
    print(f"[WARNING] RDKit import failed: {e}. Feature service disabled.")

try:
    from mordred import Calculator, descriptors as mordred_descs
    MORDRED_AVAILABLE = True
except ImportError:
    MORDRED_AVAILABLE = False
    print("[WARNING] mordred not installed. 3D-MoRSE descriptors will use RDKit AUTOCORR fallback.")


class FeatureService3D:
    """
    3D-aware molecular feature extraction service.

    Produces exactly N_3D_FEATURES (20) orthogonal features for the
    20-qubit quantum kernel. Falls back to 2D descriptors if 3D
    conformer generation fails.
    """

    def __init__(self):
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for FeatureService3D.")

        self._desc_calculator = None
        self._selected_features: Optional[List[str]] = None
        self._pearson_mask: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def smiles_to_mol3d(self, smiles: str) -> Optional[object]:
        """
        Parse SMILES → 3D-optimized RDKit molecule.

        Steps:
          1. Parse SMILES
          2. Add explicit Hs
          3. Embed 3D coords via ETKDG v3
          4. Optimize with MMFF94

        Returns:
            mol3d (RDKit Mol with 3D conformer) or None if failed
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mol = Chem.AddHs(mol)

        # Try ETKDG v3 multiple times
        params = AllChem.ETKDGv3()
        params.randomSeed = RANDOM_STATE
        params.enforceChirality = True
        params.useSmallRingTorsions = True
        params.useMacrocycleTorsions = True

        for attempt in range(CONFORMER_ATTEMPTS):
            params.randomSeed = RANDOM_STATE + attempt
            result = AllChem.EmbedMolecule(mol, params)
            if result == 0:
                break
        else:
            # Fallback: ETKDG without version constraints
            params_simple = AllChem.EmbedParameters()
            params_simple.randomSeed = RANDOM_STATE
            result = AllChem.EmbedMolecule(mol, params_simple)
            if result != 0:
                return None  # Cannot generate 3D

        # MMFF94 geometry optimization
        try:
            MMFFOptimizeMolecule(mol, maxIters=2000, mmffVariant="MMFF94")
        except Exception:
            pass  # Use unoptimized conformer if MMFF fails

        return mol

    def extract_3d_descriptors(self, smiles: str) -> Tuple[np.ndarray, List[str]]:
        """
        Extract all available 3D descriptors for a SMILES string.

        Returns:
            (feature_vector, feature_names)
        """
        mol = self.smiles_to_mol3d(smiles)
        if mol is None:
            # Fall back to 2D-only descriptors
            return self._extract_2d_fallback(smiles)

        features = {}

        # ---- WHIM Descriptors (3D shape, symmetry, and size) ----
        try:
            whim = rdMolDescriptors.CalcWHIM(mol)
            whim_names = [f"WHIM_{i+1}" for i in range(len(whim))]
            for name, val in zip(whim_names, whim):
                features[name] = val if np.isfinite(val) else 0.0
        except Exception:
            pass

        # ---- 3D Autocorrelation (AUTOCORR3D) — builtin RDKit ----
        try:
            autocorr3d = rdMolDescriptors.CalcAUTOCORR3D(mol)
            for i, val in enumerate(autocorr3d):
                features[f"AUTOCORR3D_{i+1}"] = val if np.isfinite(val) else 0.0
        except Exception:
            pass

        # ---- 3D-MoRSE via mordred (if available) ----
        if MORDRED_AVAILABLE:
            try:
                calc = Calculator(mordred_descs, ignore_3D=False)
                result = calc(mol)
                for key, val in result.items():
                    name = str(key)
                    if "Mor" in name or "RDF" in name:
                        try:
                            features[name] = float(val) if np.isfinite(float(val)) else 0.0
                        except (TypeError, ValueError):
                            features[name] = 0.0
            except Exception:
                pass

        # ---- 2D PhysChem Fallback (always included) ----
        phys2d = self._extract_physchem_2d(mol)
        features.update(phys2d)

        if not features:
            return self._extract_2d_fallback(smiles)

        names = list(features.keys())
        values = np.array([features[n] for n in names], dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        return values, names

    def extract_orthogonal_descriptors(
        self,
        smiles: str,
        selected_features: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Extract exactly N_3D_FEATURES orthogonal 3D descriptors.

        When called after fit_pearson_filter(), uses stored selected_features.
        Otherwise selects first N_3D_FEATURES features as fallback.

        Returns:
            (N_3D_FEATURES,) numpy array
        """
        values, names = self.extract_3d_descriptors(smiles)

        feats = selected_features or self._selected_features
        if feats is not None:
            name_to_idx = {n: i for i, n in enumerate(names)}
            out = np.array(
                [values[name_to_idx[f]] if f in name_to_idx else 0.0 for f in feats],
                dtype=np.float32
            )
        else:
            # Pad or truncate to N_3D_FEATURES
            if len(values) >= N_3D_FEATURES:
                out = values[:N_3D_FEATURES]
            else:
                out = np.pad(values, (0, N_3D_FEATURES - len(values)))

        return out.astype(np.float32)

    # ------------------------------------------------------------------
    # PEARSON FILTER — called once during dataset preparation
    # ------------------------------------------------------------------

    def fit_pearson_filter(self, X: np.ndarray, feature_names: List[str]) -> List[str]:
        """
        Select N_3D_FEATURES orthogonal features using Pearson correlation.

        Strategy:
          1. Remove zero-variance columns
          2. Greedily select features where |ρ| < PEARSON_THRESHOLD
             with all already-selected features

        Args:
            X: (N_samples, N_raw_features) matrix
            feature_names: list of feature name strings

        Returns:
            selected_features: list of N_3D_FEATURES feature names
        """
        n_samples, n_raw = X.shape
        print(f"  Pearson filter: input shape {X.shape}")

        # Step 1: Remove zero-variance features
        std = np.std(X, axis=0)
        valid_mask = std > 1e-6
        X_valid  = X[:, valid_mask]
        names_valid = [n for n, m in zip(feature_names, valid_mask) if m]
        print(f"  After variance filter: {X_valid.shape[1]}/{n_raw} features remain")

        if X_valid.shape[1] == 0:
            raise ValueError("No valid features after variance filtering.")

        # Step 1b: Sort candidates by variance (descending) so the most
        # informative features are considered first in greedy selection
        variances = np.var(X_valid, axis=0)
        variance_order = np.argsort(-variances)  # highest variance first
        X_sorted = X_valid[:, variance_order]
        names_sorted = [names_valid[i] for i in variance_order]

        # Normalize for correlation computation
        X_norm = (X_sorted - X_sorted.mean(axis=0)) / (X_sorted.std(axis=0) + 1e-8)

        # Step 2: Greedy Pearson filter (variance-prioritized)
        selected_idx = [0]  # Start with highest-variance feature
        for i in range(1, len(names_sorted)):
            col = X_norm[:, i]
            too_correlated = False
            for sel_i in selected_idx:
                rho = float(np.corrcoef(col, X_norm[:, sel_i])[0, 1])
                if abs(rho) >= PEARSON_THRESHOLD:
                    too_correlated = True
                    break
            if not too_correlated:
                selected_idx.append(i)
            if len(selected_idx) >= N_3D_FEATURES:
                break

        # If we have fewer than N_3D_FEATURES, pad with remaining features
        if len(selected_idx) < N_3D_FEATURES:
            remaining = [i for i in range(len(names_sorted)) if i not in set(selected_idx)]
            need = N_3D_FEATURES - len(selected_idx)
            selected_idx.extend(remaining[:need])

        selected_idx = selected_idx[:N_3D_FEATURES]
        self._selected_features = [names_sorted[i] for i in selected_idx]
        print(f"  Pearson filter selected {len(self._selected_features)} features: "
              f"{self._selected_features[:5]}...")
        print(f"  Feature types: {len(set(n.split('_')[0] for n in self._selected_features))} distinct groups")

        return self._selected_features

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _extract_physchem_2d(self, mol) -> dict:
        """Extract 2D PhysChem descriptors from RDKit mol."""
        results = {}
        try:
            mol_noh = Chem.RemoveHs(mol)
            calc = MoleculeDescriptors.MolecularDescriptorCalculator(PHYSCHEM_DESCS)
            vals = calc.CalcDescriptors(mol_noh)
            for name, val in zip(PHYSCHEM_DESCS, vals):
                results[f"2D_{name}"] = float(val) if np.isfinite(float(val)) else 0.0
        except Exception:
            pass
        return results

    def _extract_2d_fallback(self, smiles: str) -> Tuple[np.ndarray, List[str]]:
        """Pure 2D fallback when 3D conformer fails."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(N_3D_FEATURES, dtype=np.float32), [f"zero_{i}" for i in range(N_3D_FEATURES)]

        phys = self._extract_physchem_2d(mol)
        # Also add fingerprint bits as extra features
        try:
            fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=64))
            for i, bit in enumerate(fp):
                phys[f"MFP_{i}"] = float(bit)
        except Exception:
            pass

        names = list(phys.keys())
        values = np.array([phys[n] for n in names], dtype=np.float32)
        values = np.nan_to_num(values)
        return values, names

    def canonical_smiles(self, smiles: str) -> Optional[str]:
        """Return canonical SMILES for caching, or None if invalid."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)

    @property
    def selected_features(self) -> Optional[List[str]]:
        return self._selected_features
