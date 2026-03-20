"""
Reward Function — Multi-objective Reward for REINFORCE
========================================================
Computes a scalar reward combining pIC50, drug-likeness (QED),
synthetic accessibility (SA), Lipinski compliance, and a Tanimoto
diversity penalty to prevent mode collapse.

Formula:
    R = α·norm(pIC50) − β·tox_proxy + γ·QED − δ·SA_norm − diversity_penalty

Invalid SMILES → returns -1.0 (hardest penalty).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_v4 import (
    ALPHA, BETA, GAMMA, DELTA,
    DIVERSITY_RADIUS, DIVERSITY_PENALTY,
)

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def _tanimoto_penalty(
    smiles: str,
    batch_smiles: list,
    radius: float = DIVERSITY_RADIUS,
    penalty: float = DIVERSITY_PENALTY,
) -> float:
    """
    Compute diversity penalty based on Tanimoto similarity.

    If the molecule is too similar (Tanimoto > radius) to any molecule
    in the current batch, apply a penalty. This prevents mode collapse
    where RL converges to generating the same molecule repeatedly.

    Args:
        smiles:        SMILES of current molecule
        batch_smiles:  list of other SMILES in the batch
        radius:        Tanimoto threshold (default 0.7)
        penalty:       penalty magnitude (default 0.3)

    Returns:
        float: 0.0 (diverse) or penalty value (too similar)
    """
    if not batch_smiles or not RDKIT_AVAILABLE:
        return 0.0

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0

    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    except Exception:
        return 0.0

    max_similarity = 0.0
    for other_smi in batch_smiles:
        if other_smi == smiles:
            continue
        other_mol = Chem.MolFromSmiles(other_smi)
        if other_mol is None:
            continue
        try:
            other_fp = AllChem.GetMorganFingerprintAsBitVect(other_mol, 2, nBits=2048)
            sim = DataStructs.TanimotoSimilarity(fp, other_fp)
            max_similarity = max(max_similarity, sim)
        except Exception:
            continue

    return penalty if max_similarity > radius else 0.0


def compute_reward(
    smiles: str,
    pic50: float = None,
    admet: dict = None,
    batch_smiles: list = None,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
    delta: float = DELTA,
) -> float:
    """
    Compute scalar reward for REINFORCE.

    Args:
        smiles:        generated SMILES string
        pic50:         predicted pIC50 from XGBOracle (or None for invalid)
        admet:         dict from ADMETScorer (or None for invalid)
        batch_smiles:  list of other SMILES in the batch for diversity
        alpha - delta: reward weights

    Returns:
        float: reward value. -1.0 for invalid SMILES, otherwise ~[-0.5, 1.5]
    """
    # Hard penalty for invalid SMILES
    if not RDKIT_AVAILABLE:
        return 0.0

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -1.0

    # If oracles returned None/error, use floor values
    if pic50 is None:
        return -1.0
    if admet is None:
        return -1.0
    if admet.get("error") is not None:
        return -1.0

    # Normalise pIC50: maps [2, 12] → [0, 1]
    pic50_norm = float(np.clip((pic50 - 2.0) / 10.0, 0.0, 1.0))

    # Toxicity proxy: 0 if passes Lipinski, 1 if fails
    lipinski_pass = admet.get("lipinski_pass", False)
    tox_proxy = 0.0 if lipinski_pass else 1.0

    # QED: already in [0, 1]
    qed = float(admet.get("qed", 0.0) or 0.0)

    # SA normalisation: maps [1, 10] → [0, 1] where 0 = easiest
    sa_score = float(admet.get("sa_score", 5.0) or 5.0)
    sa_norm = (sa_score - 1.0) / 9.0

    # Core reward
    reward = alpha * pic50_norm - beta * tox_proxy + gamma * qed - delta * sa_norm

    # Diversity penalty
    if batch_smiles:
        div_penalty = _tanimoto_penalty(smiles, batch_smiles)
        reward -= div_penalty

    return float(reward)


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Reward function test:")

    # Valid high-reward molecule
    r1 = compute_reward(
        "CC(=O)Oc1ccccc1C(=O)O",
        pic50=7.5,
        admet={"qed": 0.8, "sa_score": 2.0, "lipinski_pass": True, "error": None},
        batch_smiles=[],
    )
    print(f"  Aspirin (pIC50=7.5):    R = {r1:.3f}")
    assert -1.0 < r1 < 1.5, f"Reward out of range: {r1}"

    # Invalid SMILES
    r2 = compute_reward("INVALID", pic50=None, admet=None, batch_smiles=[])
    print(f"  Invalid SMILES:         R = {r2:.3f}")
    assert r2 == -1.0, f"Should be -1.0: {r2}"

    # Low activity
    r3 = compute_reward(
        "CCO",
        pic50=3.0,
        admet={"qed": 0.3, "sa_score": 1.5, "lipinski_pass": True, "error": None},
        batch_smiles=[],
    )
    print(f"  Ethanol (pIC50=3.0):    R = {r3:.3f}")

    # Diversity penalty
    r4 = compute_reward(
        "CCO",
        pic50=5.0,
        admet={"qed": 0.5, "sa_score": 3.0, "lipinski_pass": True, "error": None},
        batch_smiles=["CCO", "CCO", "CCO"],
    )
    r5 = compute_reward(
        "CCO",
        pic50=5.0,
        admet={"qed": 0.5, "sa_score": 3.0, "lipinski_pass": True, "error": None},
        batch_smiles=[],
    )
    print(f"  With dupes in batch:    R = {r4:.3f}")
    print(f"  Without dupes:          R = {r5:.3f}")
    assert r4 < r5, "Diversity penalty should reduce reward"

    print(f"\n  ✓ Reward function tests passed")
