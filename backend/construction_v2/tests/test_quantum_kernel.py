"""
Test Suite — Quantum Kernel Components
========================================
Tests for HEA circuit, backends, and kernel row computation.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quantum.circuits import build_hea_circuit
from quantum.backends import StatevectorBackend, ShotBackend
from config import N_QUBITS


class TestHEACircuit:
    def test_circuit_creation(self):
        x = np.random.uniform(-np.pi, np.pi, N_QUBITS)
        qc = build_hea_circuit(x, x, n_qubits=N_QUBITS, measure=True)
        assert qc.num_qubits == N_QUBITS
        assert qc.num_clbits == N_QUBITS

    def test_circuit_no_measure(self):
        x = np.random.uniform(-np.pi, np.pi, N_QUBITS)
        qc = build_hea_circuit(x, x, n_qubits=N_QUBITS, measure=False)
        assert qc.num_qubits == N_QUBITS
        assert qc.num_clbits == 0

    def test_small_circuit(self):
        """Test with fewer qubits."""
        x = np.array([0.5, 1.0, -0.5])
        qc = build_hea_circuit(x, x, n_qubits=3, measure=True)
        assert qc.num_qubits == 3


class TestStatevectorBackend:
    @pytest.fixture
    def backend(self):
        return StatevectorBackend(n_qubits=4, n_shots=1024)

    def test_self_fidelity_high(self, backend):
        """Fidelity of a vector with itself should be ~1.0."""
        x = np.array([0.5, 1.0, -0.5, 0.3])
        fid = backend.fidelity(x, x)
        assert fid > 0.95  # Should be very close to 1.0

    def test_different_vectors_lower_fidelity(self, backend):
        """Fidelity of very different vectors should be < 1.0."""
        x1 = np.array([0.0, 0.0, 0.0, 0.0])
        x2 = np.array([np.pi, np.pi, np.pi, np.pi])
        fid = backend.fidelity(x1, x2)
        assert fid < 0.9

    def test_fidelity_in_range(self, backend):
        x1 = np.random.uniform(-np.pi, np.pi, 4)
        x2 = np.random.uniform(-np.pi, np.pi, 4)
        fid = backend.fidelity(x1, x2)
        assert 0.0 <= fid <= 1.0


class TestShotBackend:
    @pytest.fixture
    def backend(self):
        return ShotBackend(n_qubits=4, n_shots=1024)

    def test_fidelity_with_counts(self, backend):
        x = np.array([0.5, 1.0, -0.5, 0.3])
        result = backend.fidelity_with_counts(x, x)
        assert "fidelity" in result
        assert "counts" in result
        assert "n_shots" in result
        assert result["n_shots"] == 1024
        assert result["fidelity"] > 0.8

    def test_fidelity_batch(self, backend):
        x1 = np.array([0.5, 1.0, -0.5, 0.3])
        x2_list = [
            np.array([0.5, 1.0, -0.5, 0.3]),
            np.array([0.0, 0.0, 0.0, 0.0]),
        ]
        results = backend.fidelity_batch(x1, x2_list)
        assert len(results) == 2
        assert results[0] > results[1]  # Self should be higher
