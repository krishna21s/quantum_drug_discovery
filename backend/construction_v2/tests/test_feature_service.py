"""
Test Suite — Feature Service
==============================
Unit tests for feature extraction, canonical SMILES, and caching.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.feature_service import FeatureService
from config import MULTI_FP_DIM


@pytest.fixture
def feature_svc():
    return FeatureService()


class TestCanonicalSmiles:
    def test_valid_smiles(self, feature_svc):
        canon = feature_svc.canonical_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
        assert canon is not None
        assert isinstance(canon, str)
        assert len(canon) > 0

    def test_invalid_smiles_returns_none(self, feature_svc):
        assert feature_svc.canonical_smiles("INVALID") is None

    def test_canonical_consistency(self, feature_svc):
        """Same molecule, different SMILES → same canonical form."""
        c1 = feature_svc.canonical_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
        c2 = feature_svc.canonical_smiles("O=C(O)c1ccccc1OC(C)=O")
        assert c1 == c2


class TestMultiFingerprint:
    def test_aspirin_shape(self, feature_svc):
        fp = feature_svc.extract_multi_fingerprint("CC(=O)OC1=CC=CC=C1C(=O)O")
        assert fp.shape == (MULTI_FP_DIM,)
        assert fp.dtype == np.float32

    def test_invalid_smiles_returns_zeros(self, feature_svc):
        fp = feature_svc.extract_multi_fingerprint("INVALID")
        assert np.all(fp == 0)
        assert fp.shape == (MULTI_FP_DIM,)

    def test_different_molecules_different_fps(self, feature_svc):
        fp1 = feature_svc.extract_multi_fingerprint("CC(=O)OC1=CC=CC=C1C(=O)O")
        fp2 = feature_svc.extract_multi_fingerprint("C1=CC=C2C(=C1)C=CC3=CC=CC=C32")
        assert not np.array_equal(fp1, fp2)


class TestOrthogonalDescriptors:
    def test_aspirin_extraction(self, feature_svc):
        if feature_svc._selected_features is None:
            pytest.skip("No selected_features.json in checkpoints")
        desc = feature_svc.extract_orthogonal_descriptors("CC(=O)OC1=CC=CC=C1C(=O)O")
        assert desc.shape == (20,)
        assert desc.dtype == np.float64

    def test_invalid_returns_zeros(self, feature_svc):
        if feature_svc._selected_features is None:
            pytest.skip("No selected_features.json in checkpoints")
        desc = feature_svc.extract_orthogonal_descriptors("INVALID")
        assert np.all(desc == 0)


class TestBaselineScore:
    def test_aspirin_low_score(self, feature_svc):
        score = feature_svc.baseline_rule_score("CC(=O)OC1=CC=CC=C1C(=O)O")
        assert 0.0 <= score <= 1.0
        assert score < 0.5  # Aspirin should be low risk

    def test_invalid_returns_zero(self, feature_svc):
        assert feature_svc.baseline_rule_score("INVALID") == 0.0


class TestCaching:
    def test_cache_populated(self, feature_svc):
        assert feature_svc.cache_size == 0
        feature_svc.extract_multi_fingerprint("CC(=O)OC1=CC=CC=C1C(=O)O")
        assert feature_svc.cache_size == 1

    def test_cache_hit_same_result(self, feature_svc):
        fp1 = feature_svc.extract_multi_fingerprint("CC(=O)OC1=CC=CC=C1C(=O)O")
        fp2 = feature_svc.extract_multi_fingerprint("CC(=O)OC1=CC=CC=C1C(=O)O")
        assert np.array_equal(fp1, fp2)

    def test_clear_cache(self, feature_svc):
        feature_svc.extract_multi_fingerprint("CC(=O)OC1=CC=CC=C1C(=O)O")
        feature_svc.clear_cache()
        assert feature_svc.cache_size == 0
