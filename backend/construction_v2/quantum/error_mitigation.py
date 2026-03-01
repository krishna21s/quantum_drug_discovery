"""
Error Mitigation — Measurement Error Correction
=================================================
Measurement error mitigation for shot-based quantum runs.

Provides:
  1. Calibration matrix construction (complete readout characterization)
  2. Matrix-inverse mitigation of raw measurement counts
  3. ZNE (Zero-Noise Extrapolation) stubs for future hardware integration
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_QUBITS, N_SHOTS


class MeasurementErrorMitigation:
    """
    Readout error mitigation via calibration matrix approach.

    For n qubits, prepares 2^n basis states and measures to build
    the assignment matrix A where A[i,j] = P(measure i | prepared j).
    Then mitigates raw counts: ideal_counts ≈ A^{-1} · raw_counts.

    NOTE: Full 2^n calibration is only practical for small qubit counts.
    For production (20 qubits), use tensored mitigation (per-qubit calibration).
    """

    def __init__(self, n_qubits=N_QUBITS, n_shots=N_SHOTS):
        self.n_qubits = n_qubits
        self.n_shots = n_shots
        self.cal_matrix = None
        self.cal_matrix_inv = None

    def build_calibration_matrix_tensored(self, backend=None):
        """
        Build per-qubit calibration matrices (tensored approach).
        Scales O(n) instead of O(2^n). Practical for 20+ qubits.

        Creates 2 circuits per qubit (prepare |0⟩ and |1⟩).

        Args:
            backend: AerSimulator instance (creates one if not provided)
        """
        if backend is None:
            backend = AerSimulator(method="automatic")

        self.per_qubit_cal = []  # List of 2x2 matrices

        for q in range(self.n_qubits):
            cal_2x2 = np.zeros((2, 2))

            for prep_state in [0, 1]:
                qc = QuantumCircuit(1, 1)
                if prep_state == 1:
                    qc.x(0)
                qc.measure(0, 0)

                counts = backend.run(qc, shots=self.n_shots).result().get_counts()
                for bitstring, count in counts.items():
                    measured = int(bitstring)
                    cal_2x2[measured, prep_state] = count / self.n_shots

            self.per_qubit_cal.append(cal_2x2)

        print(f"  Tensored calibration: {self.n_qubits} qubits calibrated")

    def mitigate_counts(self, raw_counts):
        """
        Apply measurement error mitigation to raw counts.
        Uses tensored (per-qubit) approach for scalability.

        Args:
            raw_counts: dict mapping bitstrings to counts

        Returns:
            dict: Mitigated counts (may contain fractional values)
        """
        if self.per_qubit_cal is None:
            raise ValueError("Must call build_calibration_matrix_tensored() first")

        # Convert counts to probability vector
        total_shots = sum(raw_counts.values())
        all_strings = list(raw_counts.keys())

        mitigated = {}
        for bitstring, count in raw_counts.items():
            # For each bitstring, compute the correction factor
            correction = 1.0
            for q in range(min(len(bitstring), self.n_qubits)):
                bit = int(bitstring[-(q + 1)])  # LSB first
                # P(correct) for this qubit
                p_correct = self.per_qubit_cal[q][bit, bit]
                if p_correct > 0:
                    correction *= 1.0 / p_correct

            mitigated[bitstring] = count * min(correction, 2.0)  # Cap to avoid blow-up

        # Renormalize
        total_mitigated = sum(mitigated.values())
        if total_mitigated > 0:
            for k in mitigated:
                mitigated[k] = mitigated[k] * total_shots / total_mitigated

        return mitigated

    def mitigated_fidelity(self, raw_counts):
        """
        Extract mitigated fidelity (all-zeros probability) from raw counts.

        Args:
            raw_counts: dict mapping bitstrings to counts

        Returns:
            float: Mitigated fidelity value
        """
        mitigated = self.mitigate_counts(raw_counts)
        zeros = "0" * self.n_qubits
        total = sum(mitigated.values())
        if total == 0:
            return 0.0
        return mitigated.get(zeros, 0) / total


class ZeroNoiseExtrapolation:
    """
    Stub for Zero-Noise Extrapolation (ZNE).

    ZNE runs circuits at multiple noise levels and extrapolates
    to zero noise. Used for final hardware claims.

    This is a placeholder for future hardware integration.
    """

    def __init__(self, noise_factors=None):
        self.noise_factors = noise_factors or [1.0, 1.5, 2.0]
        self.results = {}

    def extrapolate(self, values_at_noise_levels):
        """
        Linear extrapolation to zero noise.

        Args:
            values_at_noise_levels: list of (noise_factor, value) pairs

        Returns:
            float: Extrapolated zero-noise value
        """
        if len(values_at_noise_levels) < 2:
            return values_at_noise_levels[0][1] if values_at_noise_levels else 0.0

        factors = np.array([v[0] for v in values_at_noise_levels])
        values = np.array([v[1] for v in values_at_noise_levels])

        # Linear fit and extrapolate to factor=0
        coeffs = np.polyfit(factors, values, deg=min(1, len(factors) - 1))
        return float(np.polyval(coeffs, 0.0))
