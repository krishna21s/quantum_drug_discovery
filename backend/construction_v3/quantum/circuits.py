"""
Quantum Circuits — HEA Fidelity Circuit Builder (V3)
======================================================
Identical to V2 circuits — the HEA fidelity circuit is mode-agnostic.
Encodes 20-dim 3D feature vectors into the 20-qubit Hilbert space.
"""

from qiskit import QuantumCircuit
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_QUBITS


def build_hea_circuit(x1, x2, n_qubits=N_QUBITS, measure=True):
    """
    Build a Hardware-Efficient Ansatz fidelity circuit: U(x1)† · U(x2).

    Encodes 3D molecular features as RY rotation angles in the 20-qubit
    Hilbert space. The |0...0⟩ measurement probability gives the fidelity
    (quantum similarity) between two molecules.

    Architecture:
      1. Forward embedding U(x1): RY rotations per qubit
      2. Alternating even/odd CX entanglement (IBM coupling-map friendly)
      3. Adjoint U†(x2): reverse CX + inverse RY rotations

    Args:
        x1, x2: 20-dimensional scaled feature vectors (one per qubit)
        n_qubits: Number of qubits (default: 20)
        measure:  If True, add measurement gates

    Returns:
        QuantumCircuit with fidelity estimation circuit
    """
    if measure:
        qc = QuantumCircuit(n_qubits, n_qubits)
    else:
        qc = QuantumCircuit(n_qubits)

    # Forward embedding U(x1)
    for i in range(n_qubits):
        qc.ry(float(x1[i]), i)

    # Alternating even/odd CX entanglement
    for i in range(0, n_qubits - 1, 2):
        qc.cx(i, i + 1)
    for i in range(1, n_qubits - 1, 2):
        qc.cx(i, i + 1)

    # Adjoint U†(x2)
    for i in range(1, n_qubits - 1, 2)[::-1]:
        qc.cx(i, i + 1)
    for i in range(0, n_qubits - 1, 2)[::-1]:
        qc.cx(i, i + 1)
    for i in range(n_qubits):
        qc.ry(-float(x2[i]), i)

    if measure:
        qc.measure(range(n_qubits), range(n_qubits))

    return qc
