"""
Quantum Circuits — HEA Fidelity Circuit Builder
================================================
Extracted from V1 core_engine_shot.py::build_hea_circuit().

Provides the Hardware-Efficient Ansatz (HEA) fidelity circuit
used by both statevector and shot-based backends.
"""

from qiskit import QuantumCircuit
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_QUBITS


def build_hea_circuit(x1, x2, n_qubits=N_QUBITS, measure=True):
    """
    Build a Hardware-Efficient Ansatz fidelity circuit: U(x1)† · U(x2).

    The circuit computes the overlap (fidelity) between two feature vectors
    encoded as rotation angles. Measurement of all-zeros state gives fidelity.

    Architecture:
      1. Forward embedding U(x1): RY rotations
      2. Alternating even/odd CX entanglement (IBM coupling-map friendly)
      3. Adjoint U†(x2): reverse CX + inverse RY rotations

    Args:
        x1: First feature vector (n_qubits-dimensional)
        x2: Second feature vector (n_qubits-dimensional)
        n_qubits: Number of qubits (default: N_QUBITS from config)
        measure: If True, add measurement gates (required for shot-based)

    Returns:
        QuantumCircuit with the fidelity estimation circuit
    """
    if measure:
        qc = QuantumCircuit(n_qubits, n_qubits)
    else:
        qc = QuantumCircuit(n_qubits)

    # Forward embedding U(x1)
    for i in range(n_qubits):
        qc.ry(float(x1[i]), i)

    # Alternating Even/Odd Entanglement (better for IBM coupling maps)
    for i in range(0, n_qubits - 1, 2):
        qc.cx(i, i + 1)
    for i in range(1, n_qubits - 1, 2):
        qc.cx(i, i + 1)

    # Adjoint embedding U†(x2)
    for i in range(1, n_qubits - 1, 2)[::-1]:
        qc.cx(i, i + 1)
    for i in range(0, n_qubits - 1, 2)[::-1]:
        qc.cx(i, i + 1)
    for i in range(n_qubits):
        qc.ry(-float(x2[i]), i)

    if measure:
        qc.measure(range(n_qubits), range(n_qubits))

    return qc
