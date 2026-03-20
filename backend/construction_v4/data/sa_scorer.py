"""
Synthetic Accessibility (SA) Score — RDKit Contrib
====================================================
Verbatim copy of rdkit/Contrib/SA_Score/sascorer.py
Source: https://github.com/rdkit/rdkit/blob/master/Contrib/SA_Score/sascorer.py

Included explicitly because SA Score is not in the standard RDKit pip install.
The fpscores.pkl.gz data file is downloaded automatically on first use.

DO NOT MODIFY this file. Reference as:
    from data.sa_scorer import calculateScore
"""

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

import math
import pickle

import os.path as op

_fscores = None
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)


def readFragmentScores(name="fpscores.pkl.gz"):
    import gzip
    global _fscores

    # generate the full path filename:
    if name == "fpscores.pkl.gz":
        name = op.join(op.dirname(__file__), name)

    # Auto-download if not present
    if not op.exists(name):
        _download_fpscores(name)

    data = pickle.load(gzip.open(name))
    outDict = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict


def _download_fpscores(dest_path):
    """Download fpscores.pkl.gz from RDKit GitHub if not present."""
    import urllib.request
    url = (
        "https://raw.githubusercontent.com/rdkit/rdkit/master/"
        "Contrib/SA_Score/fpscores.pkl.gz"
    )
    print(f"  [SA Scorer] Downloading fpscores.pkl.gz from RDKit GitHub...")
    os.makedirs(op.dirname(dest_path), exist_ok=True) if op.dirname(dest_path) else None
    urllib.request.urlretrieve(url, dest_path)
    print(f"  [SA Scorer] Saved to {dest_path}")


import os  # needed for _download_fpscores


def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro


def calculateScore(m):
    """
    Calculate the Synthetic Accessibility score for an RDKit molecule.

    Args:
        m: RDKit Mol object

    Returns:
        float: SA score in range [1, 10] where 1 = easy to synthesise
    """
    if not m.GetNumAtoms():
        return None

    if _fscores is None:
        readFragmentScores()

    # fragment score
    sfp = mfpgen.GetSparseCountFingerprint(m)

    score1 = 0.
    nf = 0
    nze = sfp.GetNonzeroElements()
    for id, count in nze.items():
        nf += count
        score1 += _fscores.get(id, -4) * count

    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    score3 = 0.
    numBits = len(nze)
    if nAtoms > numBits:
        score3 = math.log(float(nAtoms) / numBits) * .5

    sascore = score1 + score2 + score3

    # transform "raw" value into scale between 1 and 10
    min_score = -4.0
    max_score = 2.5
    sascore = 11. - (sascore - min_score + 1) / (max_score - min_score) * 9.

    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0

    return sascore


if __name__ == "__main__":
    # Quick test
    smi = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
    mol = Chem.MolFromSmiles(smi)
    score = calculateScore(mol)
    print(f"SA Score for aspirin: {score:.2f} (expected ~2.5, easy to synthesise)")
