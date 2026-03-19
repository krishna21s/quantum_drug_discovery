"""
Quantum Circuits — Phase B: Data Reuploading + CZ Entanglement (V3)
====================================================================
Upgraded from simple single-layer HEA to data reuploading PQC
(Pérez-Salinas 2020) with CZ ring entanglement.

Key improvements over V2/V3-Phase-A circuit:
  1. CZ ring entanglement (instead of linear CX) — better long-range
     correlations between non-adjacent qubits
  2. Two-layer data reuploading — feature vector encoded twice with
     a π/4 offset, doubling the expressivity of the encoding
  3. Fidelity estimation remains: |<ψ(x1)|ψ(x2)>|² as before
"""

from qiskit import QuantumCircuit
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_QUBITS

import numpy as np


def _cz_ring(qc: QuantumCircuit, n_qubits: int):
    """Apply a ring of CZ gates: q0-q1, q1-q2, ..., q(n-1)-q0."""
    for i in range(n_qubits - 1):
        qc.cz(i, i + 1)
    qc.cz(n_qubits - 1, 0)   # close the ring


def build_hea_circuit(x1, x2, n_qubits=N_QUBITS, measure=True):
    """
    Build a Data-Reuploading fidelity circuit: U(x1)† · U(x2).

    U(x) architecture (two-layer data reuploading with CZ rings):
      Layer 1: RY(xᵢ)   for all qubits
               CZ ring
      Layer 2: RY(xᵢ + π/4) for all qubits     ← reuploaded with offset
               CZ ring

    Fidelity = P(|00...0⟩) after U†(x1) · U(x2)

    The data reuploading (encoding features multiple times) is the key
    technique from Pérez-Salinas et al. (2020) that creates richer,
    more distinguishable quantum states for nearby molecules.

    Args:
        x1, x2:   20-dim scaled feature vectors (values in [0, π])
        n_qubits: Number of qubits (default: 20)
        measure:  If True, add measurement gates

    Returns:
        QuantumCircuit
    """
    if measure:
        qc = QuantumCircuit(n_qubits, n_qubits)
    else:
        qc = QuantumCircuit(n_qubits)

    x2 = list(x2)
    x1 = list(x1)
    OFFSET = np.pi / 4   # reuploading offset

    # ── Forward encoding U(x2) ────────────────────────────────────────
    # Layer 1: RY(x) + CZ ring
    for i in range(n_qubits):
        qc.ry(float(x2[i]), i)
    _cz_ring(qc, n_qubits)

    # Layer 2: RY(x + π/4) + CZ ring
    for i in range(n_qubits):
        qc.ry(float(x2[i]) + OFFSET, i)
    _cz_ring(qc, n_qubits)

    # ── Adjoint U†(x1): reverse circuit of U(x1) ─────────────────────
    # Reverse Layer 2: inverse CZ ring + inverse RY(x + π/4)
    _cz_ring(qc, n_qubits)          # CZ† = CZ (self-inverse)
    for i in range(n_qubits):
        qc.ry(-(float(x1[i]) + OFFSET), i)

    # Reverse Layer 1: inverse CZ ring + inverse RY(x)
    _cz_ring(qc, n_qubits)
    for i in range(n_qubits):
        qc.ry(-float(x1[i]), i)

    if measure:
        qc.measure(range(n_qubits), range(n_qubits))

    return qc
