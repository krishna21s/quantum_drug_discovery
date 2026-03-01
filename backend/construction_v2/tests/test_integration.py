"""
Test Suite — Integration Tests
================================
End-to-end tests using real checkpoints.
"""

import sys
import os
import json
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHECKPOINT_DIR, REFERENCE_MOLECULES
from services.nystrom_engine import NystromEngine


def checkpoints_exist():
    """Check if V1 checkpoints are available."""
    required = [
        "K_mm.npy",
        "K_nm.npy",
        "selected_features.json",
        "xgb_model_v2.pkl",
        "xgb_var_selector.pkl",
    ]
    return all(os.path.exists(os.path.join(str(CHECKPOINT_DIR), f)) for f in required)


@pytest.mark.skipif(not checkpoints_exist(), reason="Checkpoints not available")
class TestIntegration:
    """Full pipeline integration test using real checkpoints."""

    @pytest.fixture(scope="class")
    def pipeline_components(self):
        """Load all pipeline components once for the test class."""
        import pickle
        import pandas as pd
        from sklearn.svm import SVC
        from sklearn.preprocessing import MinMaxScaler
        from services.feature_service import FeatureService
        from services.nystrom_engine import NystromEngine
        from services.classical_router import ClassicalRouter
        from services.quantum_kernel_service import QuantumKernelService
        from quantum.backends import StatevectorBackend
        from pipeline.orchestrator import InferencePipeline

        ckpt = str(CHECKPOINT_DIR)
        feature_svc = FeatureService()
        classical_router = ClassicalRouter.from_checkpoints(feature_svc, ckpt)

        with open(os.path.join(ckpt, "selected_features.json"), "r") as f:
            selected_features = json.load(f)

        return {
            "feature_svc": feature_svc,
            "classical_router": classical_router,
            "selected_features": selected_features,
        }

    def test_xgboost_reference_molecules(self, pipeline_components):
        """Test XGBoost predictions on reference molecules."""
        router = pipeline_components["classical_router"]

        for name, (smiles, true_label) in REFERENCE_MOLECULES.items():
            result = router.predict_xgb(smiles)
            prob = result["probability"]
            assert 0.0 <= prob <= 1.0, f"Invalid probability for {name}: {prob}"
            assert result["latency_ms"] < 5000, f"XGBoost too slow for {name}"

    def test_feature_extraction_reference(self, pipeline_components):
        """Test feature extraction on reference molecules."""
        feature_svc = pipeline_components["feature_svc"]
        selected = pipeline_components["selected_features"]

        for name, (smiles, _) in REFERENCE_MOLECULES.items():
            fp = feature_svc.extract_multi_fingerprint(smiles)
            assert fp.shape[0] == 4278, f"Wrong FP shape for {name}"

            ortho = feature_svc.extract_orthogonal_descriptors(smiles, selected)
            assert ortho.shape[0] == len(selected), f"Wrong ortho shape for {name}"

    def test_nystrom_checkpoint_loading(self):
        """Test that Nystrom checkpoints load correctly."""
        nystrom = NystromEngine(str(CHECKPOINT_DIR))
        loaded = nystrom.load_checkpoints()
        assert loaded, "Failed to load Nystrom checkpoints"
        assert nystrom.K_mm is not None
        assert nystrom.K_nm is not None
        assert nystrom.K_mm.shape[0] == nystrom.K_mm.shape[1]  # Square

    def test_nystrom_reconstruction(self):
        """Test kernel reconstruction from loaded checkpoints."""
        nystrom = NystromEngine(str(CHECKPOINT_DIR))
        nystrom.load_checkpoints()
        K_train, K_mm_inv, diag_train = nystrom.reconstruct_kernel()

        # Valid kernel properties
        assert np.all(K_train >= 0)
        assert np.all(K_train <= 1)
        np.testing.assert_array_almost_equal(
            np.diag(K_train), np.ones(K_train.shape[0]), decimal=5
        )
