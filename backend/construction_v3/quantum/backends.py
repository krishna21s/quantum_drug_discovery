"""
Quantum Backends — Statevector & Shot-based Fidelity Computation (V3)
======================================================================
Identical to V2 backends — the quantum circuits are hardware-level and
don't change between classification and regression modes.
"""

import numpy as np
from qiskit_aer import AerSimulator

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_QUBITS, N_SHOTS
from quantum.circuits import build_hea_circuit


class StatevectorBackend:
    """
    Fast deterministic fidelity using statevector simulation.
    Used for bulk screening and cached kernel computation.
    """

    def __init__(self, n_qubits=N_QUBITS, n_shots=N_SHOTS):
        self.n_qubits = n_qubits
        self.n_shots  = n_shots
        self.zeros    = "0" * n_qubits
        self.sim      = AerSimulator(method="statevector")

    def fidelity(self, x1, x2) -> float:
        """Compute fidelity (overlap) between two 20-dim feature vectors."""
        qc     = build_hea_circuit(x1, x2, n_qubits=self.n_qubits, measure=True)
        counts = self.sim.run(qc, shots=self.n_shots).result().get_counts()
        return counts.get(self.zeros, 0) / self.n_shots


class ShotBackend:
    """
    Shot-based fidelity with optional noise model.
    Used for hardware-realistic final evaluation and CI estimation.
    """

    def __init__(self, n_qubits=N_QUBITS, n_shots=N_SHOTS, noise_model=None):
        self.n_qubits    = n_qubits
        self.n_shots     = n_shots
        self.noise_model = noise_model
        self.zeros       = "0" * n_qubits

        if noise_model is not None:
            self.sim = AerSimulator(noise_model=noise_model)
        else:
            self.sim = AerSimulator(method="automatic")

    def fidelity(self, x1, x2) -> float:
        return self.fidelity_with_counts(x1, x2)["fidelity"]

    def fidelity_with_counts(self, x1, x2) -> dict:
        qc          = build_hea_circuit(x1, x2, n_qubits=self.n_qubits, measure=True)
        result      = self.sim.run(qc, shots=self.n_shots).result()
        counts      = result.get_counts()
        zero_counts = counts.get(self.zeros, 0)
        return {
            "fidelity":    zero_counts / self.n_shots,
            "counts":      dict(counts),
            "n_shots":     self.n_shots,
            "zero_counts": zero_counts,
        }
