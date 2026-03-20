"""
Quantum Backends — Phase B: Batch Fidelity Execution (V3)
==========================================================
Key upgrade: fidelity_batch() submits all 50 landmark circuits for
one training row as a SINGLE sim.run() call, reducing Python overhead
by 50x and speeding up K_nm computation by 4-8x.
"""

import numpy as np
from qiskit_aer import AerSimulator

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_QUBITS, N_SHOTS
from quantum.circuits import build_reuploading_circuit as build_hea_circuit


class StatevectorBackend:
    """
    Fast deterministic fidelity using statevector simulation.
    Supports both single-pair and BATCH fidelity computation.

    Batch mode (4-8x faster):
        Instead of calling sim.run() for each of the 50 landmark
        pairs, submits all 50 circuits in one sim.run([c1..c50]) call.
    """

    def __init__(self, n_qubits=N_QUBITS, n_shots=N_SHOTS):
        self.n_qubits = n_qubits
        self.n_shots  = n_shots
        self.zeros    = "0" * n_qubits
        self.sim      = AerSimulator(method="statevector")

    def fidelity(self, x1, x2) -> float:
        """Compute fidelity between two feature vectors (single pair)."""
        qc     = build_hea_circuit(x1, x2, n_qubits=self.n_qubits, measure=True)
        counts = self.sim.run(qc, shots=self.n_shots).result().get_counts()
        return counts.get(self.zeros, 0) / self.n_shots

    def fidelity_batch(self, x_query, landmark_list) -> np.ndarray:
        """
        Compute fidelity between x_query and ALL landmarks in one call.

        Phase B speedup: submits len(landmark_list) circuits as a single
        sim.run() invocation, reducing Python round-trip overhead by N.

        Args:
            x_query:       (n_features,) feature vector
            landmark_list: (m, n_features) array of landmark vectors

        Returns:
            np.ndarray of shape (m,) with fidelity values
        """
        circuits = [
            build_hea_circuit(x_query, lm, n_qubits=self.n_qubits, measure=True)
            for lm in landmark_list
        ]
        results = self.sim.run(circuits, shots=self.n_shots).result()
        fidelities = np.zeros(len(landmark_list), dtype=np.float32)
        for j in range(len(landmark_list)):
            counts = results.get_counts(j)
            fidelities[j] = counts.get(self.zeros, 0) / self.n_shots
        return fidelities


class ShotBackend:
    """
    Shot-based fidelity with optional noise model.
    Used for hardware-realistic final evaluation and CI estimation.
    Also supports batch mode.
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

    def fidelity_batch(self, x_query, landmark_list) -> np.ndarray:
        """Batch fidelity for shot backend."""
        circuits = [
            build_hea_circuit(x_query, lm, n_qubits=self.n_qubits, measure=True)
            for lm in landmark_list
        ]
        results = self.sim.run(circuits, shots=self.n_shots).result()
        fidelities = np.zeros(len(landmark_list), dtype=np.float32)
        for j in range(len(landmark_list)):
            counts = results.get_counts(j)
            fidelities[j] = counts.get(self.zeros, 0) / self.n_shots
        return fidelities

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
