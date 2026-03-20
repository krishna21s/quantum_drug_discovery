"""
Quantum Circuits — 8-Qubit Multi-Layer Data Reuploading (Production)
=====================================================================
Compact 8-qubit architecture that encodes 20 features via 3 reuploading
layers. This avoids the 20-qubit exponential concentration problem where
fidelities collapse to ~0 in a 2²⁰-dimensional Hilbert space.

Architecture:
  Layer 1: RY(x[0..7])           + CZ ring   (features 0-7)
  Layer 2: RY(x[8..15] + π/4)   + CZ ring   (features 8-15)
  Layer 3: RY(x[16..19] + π/2)  + CZ ring   (features 16-19, q4-q7 get π/2)

Fidelity circuit: U†(x1) · U(x2), measured as P(|00000000⟩)

Key advantages:
  - 2⁸ = 256 dim Hilbert space → fidelities in useful 0.1-0.9 range
  - All 20 features encoded (no information loss)
  - CZ ring entanglement captures correlations between features
  - Multiple reuploading layers create richer quantum states
    (Pérez-Salinas 2020, Schuld & Killoran 2019)
"""

from qiskit import QuantumCircuit
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_QUBITS, N_REUPLOADING_LAYERS

import numpy as np


def _cz_ring(qc: QuantumCircuit, n_qubits: int):
    """Apply a ring of CZ gates: q0-q1, q1-q2, ..., q(n-1)-q0."""
    for i in range(n_qubits - 1):
        qc.cz(i, i + 1)
    if n_qubits > 2:
        qc.cz(n_qubits - 1, 0)   # close the ring


def _encode_layer(qc, x, n_qubits, layer_idx, n_features):
    """
    Encode one layer of features into the circuit.

    Each layer uses a different subset of the feature vector and
    a different angular offset for expressivity:
      Layer 0: features[0:n_qubits],     offset = 0
      Layer 1: features[n_qubits:2*n],   offset = π/4
      Layer 2: features[2*n:3*n],        offset = π/2
      ...

    If fewer features remain than qubits, remaining qubits get
    the offset only (no data-dependent rotation).
    """
    offset = layer_idx * np.pi / 4   # π/4 spacing between layers
    start_idx = layer_idx * n_qubits

    for i in range(n_qubits):
        feat_idx = start_idx + i
        if feat_idx < n_features:
            qc.ry(float(x[feat_idx]) + offset, i)
        else:
            # Pad qubit with offset-only rotation (still contributes to entanglement)
            qc.ry(offset, i)

    _cz_ring(qc, n_qubits)


def build_reuploading_circuit(x1, x2, n_qubits=N_QUBITS,
                               n_layers=N_REUPLOADING_LAYERS,
                               n_features=None, measure=True):
    """
    Build a multi-layer data reuploading fidelity circuit: U†(x1) · U(x2).

    U(x) encodes ALL features across multiple layers, with each layer
    encoding n_qubits features and a CZ ring for entanglement.

    Fidelity = P(|00...0⟩) after U†(x1) · U(x2)

    Args:
        x1, x2:     Feature vectors (up to n_qubits * n_layers features)
        n_qubits:   Number of qubits (default: 8)
        n_layers:   Number of reuploading layers (default: 3)
        n_features: Number of features to encode (default: len(x2))
        measure:    If True, add measurement gates

    Returns:
        QuantumCircuit
    """
    if n_features is None:
        n_features = len(x2)

    if measure:
        qc = QuantumCircuit(n_qubits, n_qubits)
    else:
        qc = QuantumCircuit(n_qubits)

    x2 = list(x2)
    x1 = list(x1)

    # ── Forward encoding U(x2) ────────────────────────────────────────
    for layer in range(n_layers):
        _encode_layer(qc, x2, n_qubits, layer, n_features)

    # ── Adjoint U†(x1): reverse all layers in reverse order ──────────
    for layer in range(n_layers - 1, -1, -1):
        offset = layer * np.pi / 4
        start_idx = layer * n_qubits

        # Inverse CZ ring (CZ is self-inverse)
        _cz_ring(qc, n_qubits)

        # Inverse RY rotations
        for i in range(n_qubits):
            feat_idx = start_idx + i
            if feat_idx < n_features:
                qc.ry(-(float(x1[feat_idx]) + offset), i)
            else:
                qc.ry(-offset, i)

    if measure:
        qc.measure(range(n_qubits), range(n_qubits))

    return qc


# ── Legacy compatibility wrapper ──────────────────────────────────────
def build_hea_circuit(x1, x2, n_qubits=N_QUBITS, measure=True):
    """Legacy wrapper: routes to the new reuploading circuit."""
    return build_reuploading_circuit(x1, x2, n_qubits=n_qubits, measure=measure)
