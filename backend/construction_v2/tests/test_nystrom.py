"""
Test Suite — Nystrom Engine
=============================
Tests for kernel reconstruction and normalization.
"""

import sys
import os
import numpy as np
import pytest
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.nystrom_engine import NystromEngine


class TestLandmarkSelection:
    def test_linspace_landmarks(self):
        engine = NystromEngine(checkpoint_dir=tempfile.mkdtemp())
        X = np.random.randn(100, 20)
        lm, idx = engine.select_landmarks(X, m=10, method="linspace")
        assert lm.shape == (10, 20)
        assert len(idx) == 10

    def test_random_landmarks(self):
        engine = NystromEngine(checkpoint_dir=tempfile.mkdtemp())
        X = np.random.randn(100, 20)
        lm, idx = engine.select_landmarks(X, m=10, method="random")
        assert lm.shape == (10, 20)
        assert len(np.unique(idx)) == 10

    def test_m_capped_at_N(self):
        engine = NystromEngine(checkpoint_dir=tempfile.mkdtemp())
        X = np.random.randn(5, 20)
        lm, idx = engine.select_landmarks(X, m=100, method="linspace")
        assert len(lm) <= 5


class TestKernelReconstruction:
    def test_svd_psd_cosine_clip(self):
        """Verify that reconstruction produces a valid kernel matrix."""
        engine = NystromEngine(checkpoint_dir=tempfile.mkdtemp())

        m, N = 10, 50
        # Create synthetic kernel matrices
        K_mm = np.eye(m) + 0.1 * np.random.randn(m, m)
        K_mm = (K_mm + K_mm.T) / 2  # Symmetrize
        np.fill_diagonal(K_mm, 1.0)

        K_nm = np.random.uniform(0, 1, (N, m))

        engine.K_mm = K_mm
        engine.K_nm = K_nm

        K_train, K_mm_inv, diag_train = engine.reconstruct_kernel()

        # Kernel should be square
        assert K_train.shape == (N, N)

        # Diagonal should be 1.0
        np.testing.assert_array_almost_equal(np.diag(K_train), np.ones(N), decimal=5)

        # Symmetric
        np.testing.assert_array_almost_equal(K_train, K_train.T, decimal=10)

        # Values in [0, 1]
        assert np.all(K_train >= 0)
        assert np.all(K_train <= 1)

        # PSD: all eigenvalues >= 0
        eigvals = np.linalg.eigvalsh(K_train)
        assert np.all(eigvals >= -1e-10)

    def test_diag_train_positive(self):
        engine = NystromEngine(checkpoint_dir=tempfile.mkdtemp())
        m, N = 5, 20
        K_mm = np.eye(m)
        K_nm = np.random.uniform(0, 1, (N, m))
        engine.K_mm = K_mm
        engine.K_nm = K_nm
        _, _, diag_train = engine.reconstruct_kernel()
        assert np.all(diag_train > 0)


class TestSingleRowPrediction:
    def test_predict_from_kernel_row_shape(self):
        """Test that single-row prediction returns a float."""
        from sklearn.svm import SVC

        engine = NystromEngine(checkpoint_dir=tempfile.mkdtemp())
        m, N = 5, 20

        # Create minimal valid state
        K_mm = np.eye(m)
        K_nm = np.random.uniform(0, 1, (N, m))
        engine.K_mm = K_mm
        engine.K_nm = K_nm
        K_train, K_mm_inv, diag_train = engine.reconstruct_kernel()

        # Train a simple SVM
        y_train = np.random.randint(0, 2, N)
        svm = SVC(kernel="precomputed", probability=True, class_weight="balanced")
        svm.fit(K_train, y_train)

        # Predict for a new kernel row
        K_new_m = np.random.uniform(0, 1, (1, m))
        prob = engine.predict_from_kernel_row(K_new_m, svm_model=svm)
        assert 0.0 <= prob <= 1.0
