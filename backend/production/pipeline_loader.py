"""
Pipeline Loader — Production Pipeline Initialization
=====================================================
Extracts the ML pipeline initialization from app_v2.py into a
reusable function with no Streamlit dependency.

Returns a fully initialized InferencePipeline ready for inference.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# ── Resolve the construction_v2 package ──
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_V2_DIR = os.path.join(_BACKEND_DIR, "construction_v2")
sys.path.insert(0, _V2_DIR)

from config import CHECKPOINT_DIR, TOX21_URL
from services.feature_service import FeatureService
from services.nystrom_engine import NystromEngine
from services.classical_router import ClassicalRouter
from services.quantum_kernel_service import QuantumKernelService
from quantum.backends import StatevectorBackend, ShotBackend
from pipeline.orchestrator import InferencePipeline
from pipeline.pipeline_config import PipelineConfig

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def load_pipeline(checkpoint_dir: str | None = None):
    """
    Initialize the full V2 inference pipeline.

    This is the production equivalent of app_v2.py::load_pipeline(),
    without any Streamlit caching or UI dependencies.

    Args:
        checkpoint_dir: Override checkpoint directory path.
                        Defaults to construction_v2/checkpoints/.

    Returns:
        tuple: (InferencePipeline, FeatureService)
    """
    ckpt = checkpoint_dir or str(CHECKPOINT_DIR)
    print(f"[Pipeline] Loading from checkpoint dir: {ckpt}")

    # 1. Feature Service
    feature_svc = FeatureService()
    print("[Pipeline] FeatureService initialized")

    # 2. Classical Router (XGBoost from checkpoint)
    classical_router = ClassicalRouter.from_checkpoints(feature_svc, ckpt)
    print("[Pipeline] ClassicalRouter loaded (XGBoost)")

    # 3. Load selected features
    with open(os.path.join(ckpt, "selected_features.json"), "r") as f:
        selected_features = json.load(f)
    print(f"[Pipeline] Selected features: {len(selected_features)} orthogonal descriptors")

    # 4. Rebuild dataset scaler (same procedure as app_v2.py)
    print("[Pipeline] Downloading Tox21 dataset for scaler fitting...")
    df = pd.read_csv(TOX21_URL).dropna(subset=["NR-AR"])
    toxic = df[df["NR-AR"] == 1].head(250)
    safe = df[df["NR-AR"] == 0].head(250)
    train_df = pd.concat([toxic, safe]).sample(frac=1, random_state=42)
    y_train = train_df["NR-AR"].values

    X_train_raw = np.array(
        [
            feature_svc.extract_orthogonal_descriptors(s, selected_features)
            for s in train_df["smiles"]
        ]
    )
    scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    scaler.fit(X_train_raw)
    print(f"[Pipeline] Scaler fitted on {len(train_df)} training samples")

    # 5. Nystrom engine (load checkpoints)
    nystrom = NystromEngine(ckpt)
    nystrom.load_checkpoints()
    K_train, K_mm_inv, diag_train = nystrom.reconstruct_kernel()
    print("[Pipeline] NystromEngine loaded and kernel reconstructed")

    # 6. Train SVM on reconstructed kernel
    svm_model = SVC(
        kernel="precomputed",
        probability=True,
        class_weight="balanced",
        C=20.0,
    )
    svm_model.fit(K_train, y_train)
    print("[Pipeline] SVM trained on precomputed kernel")

    # 7. Prepare landmarks
    m = len(nystrom.K_mm)
    landmark_idx = np.linspace(0, 499, m, dtype=int)
    landmarks_raw = np.array(
        [
            feature_svc.extract_orthogonal_descriptors(s, selected_features)
            for s in train_df.iloc[landmark_idx]["smiles"]
        ]
    )
    landmarks_scaled = np.nan_to_num(scaler.transform(landmarks_raw))
    print(f"[Pipeline] {m} landmark vectors prepared")

    # 8. Quantum backends
    backend_sv = StatevectorBackend()
    backend_shot = ShotBackend()
    print("[Pipeline] Quantum backends initialized (Statevector + Shot)")

    # 9. Quantum Kernel Service
    quantum_svc = QuantumKernelService(
        backend_sv=backend_sv,
        backend_shot=backend_shot,
        nystrom_engine=nystrom,
        svm_model=svm_model,
        scaler=scaler,
        landmarks_scaled=landmarks_scaled,
        feature_service=feature_svc,
        selected_features=selected_features,
    )

    # 10. Pipeline config
    pipeline_config = PipelineConfig()

    # 11. Orchestrator
    pipeline = InferencePipeline(
        feature_service=feature_svc,
        classical_router=classical_router,
        quantum_service=quantum_svc,
        pipeline_config=pipeline_config,
    )

    print("[Pipeline] ✅ Production pipeline ready")
    return pipeline, feature_svc
