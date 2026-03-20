"""
Quantum Circuits — 8-Qubit Multi-Layer Data Reuploading (V4)
============================================================
V4 adds TRAINABLE parameters to every RY gate for Quantum Kernel
Alignment (QKA). Each qubit in each layer gets:
    - theta_i  : learnable scale   (init=1.0)
    - phi_i    : learnable bias    (init=0.0)

So the rotation becomes: RY(theta_i * x[feat] + phi_i + layer_offset)

This allows gradient-free optimisation (L-BFGS-B on KTA) to find
the parameter set that makes the quantum kernel best correlated
with pIC50 labels — the root fix for CV R²=0.07.

Backward compatible: calling with params=None gives the old
fixed behaviour (theta=1, phi=0 everywhere).

Architecture unchanged:
    Layer 0: features[0..7]   offset=0
    Layer 1: features[8..15]  offset=π/4
    Layer 2: features[16..19] offset=π/2
    CZ ring after each layer.
"""

import numpy as np
from qiskit import QuantumCircuit
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_QUBITS, N_REUPLOADING_LAYERS


# ── Total trainable params = n_qubits * n_layers * 2 (scale + bias) ──
N_CIRCUIT_PARAMS = N_QUBITS * N_REUPLOADING_LAYERS * 2   # 8*3*2 = 48


def unpack_params(params, n_qubits=N_QUBITS, n_layers=N_REUPLOADING_LAYERS):
    """
    Unpack flat param vector into (theta, phi) arrays.

    params: (2 * n_qubits * n_layers,) or None
    Returns:
        theta: (n_layers, n_qubits)  scale params, init 1.0
        phi:   (n_layers, n_qubits)  bias  params, init 0.0
    """
    n = n_qubits * n_layers
    if params is None:
        return np.ones((n_layers, n_qubits)), np.zeros((n_layers, n_qubits))
    theta = np.array(params[:n]).reshape(n_layers, n_qubits)
    phi   = np.array(params[n:]).reshape(n_layers, n_qubits)
    return theta, phi


def default_params(n_qubits=N_QUBITS, n_layers=N_REUPLOADING_LAYERS):
    """Return flat default param vector (theta=1, phi=0)."""
    theta = np.ones(n_qubits * n_layers)
    phi   = np.zeros(n_qubits * n_layers)
    return np.concatenate([theta, phi])


def _cz_ring(qc: QuantumCircuit, n_qubits: int):
    """Ring of CZ gates: q0-q1, q1-q2, ..., q(n-1)-q0."""
    for i in range(n_qubits - 1):
        qc.cz(i, i + 1)
    if n_qubits > 2:
        qc.cz(n_qubits - 1, 0)


def _encode_layer(qc, x, n_qubits, layer_idx, n_features,
                  theta_row, phi_row):
    """
    Encode one layer with trainable scale/bias.

    RY angle = theta_i * x[feat] + phi_i + layer_offset
    """
    offset = layer_idx * np.pi / 4
    start  = layer_idx * n_qubits

    for i in range(n_qubits):
        feat_idx = start + i
        sc = float(theta_row[i])
        bi = float(phi_row[i])
        if feat_idx < n_features:
            angle = sc * float(x[feat_idx]) + bi + offset
        else:
            angle = bi + offset
        qc.ry(angle, i)

    _cz_ring(qc, n_qubits)


def build_reuploading_circuit(x1, x2,
                               n_qubits=N_QUBITS,
                               n_layers=N_REUPLOADING_LAYERS,
                               n_features=None,
                               measure=True,
                               params=None):
    """
    Build fidelity circuit U†(x1)·U(x2) with optional trainable params.

    Fidelity = P(|00...0⟩) after measurement.

    Args:
        x1, x2:    Feature vectors (up to n_qubits * n_layers features)
        n_qubits:  Number of qubits
        n_layers:  Reuploading layers
        n_features: Features to encode (default: len(x2))
        measure:   Add measurement
        params:    Flat (2*n_qubits*n_layers,) param vector or None
                   [theta_flat | phi_flat]. None = fixed (theta=1, phi=0).

    Returns:
        QuantumCircuit
    """
    if n_features is None:
        n_features = len(x2)

    theta, phi = unpack_params(params, n_qubits, n_layers)

    qc = QuantumCircuit(n_qubits, n_qubits) if measure else QuantumCircuit(n_qubits)

    x2 = list(x2)
    x1 = list(x1)

    # ── Forward encoding U(x2) ────────────────────────────────────────
    for layer in range(n_layers):
        _encode_layer(qc, x2, n_qubits, layer, n_features,
                      theta[layer], phi[layer])

    # ── Adjoint U†(x1): reverse layers in reverse order ──────────────
    for layer in range(n_layers - 1, -1, -1):
        offset = layer * np.pi / 4
        start  = layer * n_qubits
        sc_row = theta[layer]
        bi_row = phi[layer]

        _cz_ring(qc, n_qubits)   # CZ is self-inverse

        for i in range(n_qubits):
            feat_idx = start + i
            sc = float(sc_row[i])
            bi = float(bi_row[i])
            if feat_idx < n_features:
                angle = -(sc * float(x1[feat_idx]) + bi + offset)
            else:
                angle = -(bi + offset)
            qc.ry(angle, i)

    if measure:
        qc.measure(range(n_qubits), range(n_qubits))

    return qc


# ── Legacy compatibility wrappers ─────────────────────────────────────

def build_hea_circuit(x1, x2, n_qubits=N_QUBITS, measure=True, params=None):
    """Legacy wrapper → reuploading circuit."""
    return build_reuploading_circuit(x1, x2, n_qubits=n_qubits,
                                     measure=measure, params=params)
