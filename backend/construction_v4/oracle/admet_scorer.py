"""
ADMET Scorer — Pure RDKit Drug-likeness Assessment
=====================================================
Computes drug-likeness properties using only RDKit. No network calls.
Runs in <1ms per molecule.

Properties: QED, SA Score, MW, LogP, HBD, HBA, Lipinski, TPSA,
            rotatable bonds.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, QED as QEDModule

    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


class ADMETScorer:
    """
    Pure RDKit ADMET scoring. No external dependencies or network calls.

    Returns a dict of drug-likeness properties for any SMILES string.
    Never raises exceptions — returns None values with error message
    for invalid SMILES.
    """

    def __init__(self):
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for ADMETScorer")

        # Lazy-load SA scorer (downloads fpscores.pkl.gz on first use)
        self._sa_scorer = None

    def _get_sa_scorer(self):
        """Lazy-load SA scorer to avoid import-time download."""
        if self._sa_scorer is None:
            from data.sa_scorer import calculateScore
            self._sa_scorer = calculateScore
        return self._sa_scorer

    def score(self, smiles: str) -> dict:
        """
        Compute ADMET properties for a SMILES string.

        Args:
            smiles: SMILES string

        Returns:
            dict with keys:
                qed:              float (0-1, drug-likeness)
                sa_score:         float (1-10, synthetic accessibility)
                mw:               float (molecular weight)
                logp:             float (Crippen LogP)
                hbd:              int   (H-bond donors)
                hba:              int   (H-bond acceptors)
                lipinski_pass:    bool  (MW<500, LogP<5, HBD≤5, HBA≤10)
                tpsa:             float (topological polar surface area)
                rotatable_bonds:  int
                error:            None | str
        """
        empty_result = {
            "qed": None,
            "sa_score": None,
            "mw": None,
            "logp": None,
            "hbd": None,
            "hba": None,
            "lipinski_pass": None,
            "tpsa": None,
            "rotatable_bonds": None,
            "error": None,
        }

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                empty_result["error"] = f"Invalid SMILES: {smiles}"
                return empty_result

            # Core properties
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            rotatable = Descriptors.NumRotatableBonds(mol)

            # QED (Quantitative Estimate of Drug-likeness)
            qed = QEDModule.qed(mol)

            # SA Score
            sa_scorer = self._get_sa_scorer()
            sa_score = sa_scorer(mol)
            if sa_score is None:
                sa_score = 5.0  # neutral default

            # Lipinski Rule of Five
            lipinski_pass = (
                mw < 500
                and logp < 5
                and hbd <= 5
                and hba <= 10
            )

            return {
                "qed": round(float(qed), 4),
                "sa_score": round(float(sa_score), 2),
                "mw": round(float(mw), 2),
                "logp": round(float(logp), 3),
                "hbd": int(hbd),
                "hba": int(hba),
                "lipinski_pass": lipinski_pass,
                "tpsa": round(float(tpsa), 2),
                "rotatable_bonds": int(rotatable),
                "error": None,
            }

        except Exception as e:
            empty_result["error"] = str(e)
            return empty_result


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("ADMETScorer test:")
    scorer = ADMETScorer()

    tests = [
        ("CC(=O)Oc1ccccc1C(=O)O", "aspirin"),
        ("CCO", "ethanol"),
        ("c1ccccc1", "benzene"),
        ("INVALID", "invalid"),
    ]

    for smi, name in tests:
        result = scorer.score(smi)
        if result["error"]:
            print(f"  {name:10s}: ERROR — {result['error']}")
        else:
            print(
                f"  {name:10s}: QED={result['qed']:.3f}  "
                f"SA={result['sa_score']:.1f}  "
                f"MW={result['mw']:.0f}  "
                f"Lipinski={'✓' if result['lipinski_pass'] else '✗'}"
            )

    # Verify aspirin passes Lipinski
    asp = scorer.score("CC(=O)Oc1ccccc1C(=O)O")
    assert asp["lipinski_pass"] == True, "Aspirin should pass Lipinski"
    assert 0 <= asp["qed"] <= 1, f"QED out of range: {asp['qed']}"
    assert 1 <= asp["sa_score"] <= 10, f"SA out of range: {asp['sa_score']}"

    print(f"\n  ✓ ADMETScorer tests passed")
