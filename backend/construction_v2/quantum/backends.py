"""
Quantum Backends — Statevector & Shot-based Fidelity Computation
================================================================
Two-mode backend system extracted from V1 core_engine_shot.py::StatefulLocalBackend.

StatevectorBackend: Deterministic, fast. Used for screening & cached bulk work.
ShotBackend:        Shot-based with optional noise model. Used for final evaluation.
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

    Internally uses shots with statevector method for consistency
    with V1 behavior (counts-based fidelity extraction).
    """

    def __init__(self, n_qubits=N_QUBITS, n_shots=N_SHOTS):
        self.n_qubits = n_qubits
        self.n_shots = n_shots
        self.sim = AerSimulator(method="statevector")
        self.zeros = "0" * n_qubits

    def fidelity(self, x1, x2):
        """
        Compute fidelity (overlap) between two feature vectors.

        Returns:
            float: Fidelity value in [0, 1]. Self-similarity returns ~1.0.
        """
        qc = build_hea_circuit(x1, x2, n_qubits=self.n_qubits, measure=True)
        counts = self.sim.run(qc, shots=self.n_shots).result().get_counts()
        return counts.get(self.zeros, 0) / self.n_shots


class ShotBackend:
    """
    Shot-based fidelity with optional noise model.
    Used for hardware-realistic final evaluation and CI estimation.

    Provides both simple fidelity and full count data for error analysis.
    """

    def __init__(self, n_qubits=N_QUBITS, n_shots=N_SHOTS, noise_model=None):
        self.n_qubits = n_qubits
        self.n_shots = n_shots
        self.noise_model = noise_model
        self.zeros = "0" * n_qubits

        if noise_model is not None:
            self.sim = AerSimulator(noise_model=noise_model)
        else:
            self.sim = AerSimulator(method="automatic")

    def fidelity(self, x1, x2):
        """Compute shot-based fidelity value."""
        return self.fidelity_with_counts(x1, x2)["fidelity"]

    def fidelity_with_counts(self, x1, x2):
        """
        Compute shot-based fidelity with full count data.

        Returns:
            dict: {
                'fidelity': float,
                'counts': dict,
                'n_shots': int,
                'zero_counts': int
            }
        """
        qc = build_hea_circuit(x1, x2, n_qubits=self.n_qubits, measure=True)
        result = self.sim.run(qc, shots=self.n_shots).result()
        counts = result.get_counts()
        zero_counts = counts.get(self.zeros, 0)

        return {
            "fidelity": zero_counts / self.n_shots,
            "counts": dict(counts),
            "n_shots": self.n_shots,
            "zero_counts": zero_counts,
        }

    def fidelity_batch(self, x1, x2_list):
        """
        Compute fidelity between x1 and each vector in x2_list.
        More efficient than calling fidelity() in a loop (potential for
        circuit batching in future Qiskit versions).

        Returns:
            list[float]: Fidelity values.
        """
        results = []
        for x2 in x2_list:
            results.append(self.fidelity(x1, x2))
        return results
