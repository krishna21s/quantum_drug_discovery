"""
Orchestrator — End-to-End Inference Pipeline
=============================================
Ties together Feature Service, Classical Router, Quantum Kernel Service,
and Calibration into a single prediction pipeline with progressive disclosure.

Provides:
  1. predict_fast() — Quick path: XGB + cached statevector (≤3s)
  2. predict_full() — Final check: shot-based + CI (15-120s)
  3. predict_batch() — Batch processing for CSV uploads
"""

import json
import time
import numpy as np
import pandas as pd

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import W_XGB, W_QML, ALERT_THRESHOLD


class InferencePipeline:
    """
    End-to-end prediction pipeline with progressive disclosure.

    Architecture:
      SMILES → FeatureService → ClassicalRouter (fast) → QuantumKernelService
      → CalibrationService → Ensemble → Result

    Returns results progressively: fast estimate first, then CI-equipped update.
    """

    def __init__(
        self,
        feature_service,
        classical_router,
        quantum_service,
        calibration_service=None,
        pipeline_config=None,
    ):
        """
        Args:
            feature_service: FeatureService instance
            classical_router: ClassicalRouter instance
            quantum_service: QuantumKernelService instance
            calibration_service: CalibrationService (optional)
            pipeline_config: PipelineConfig (optional)
        """
        self.feature_svc = feature_service
        self.classical = classical_router
        self.quantum = quantum_service
        self.calibration = calibration_service
        self.config = pipeline_config

    def predict_fast(self, smiles, progress_callback=None):
        """
        Quick path prediction: XGB + statevector quantum.
        Target latency: ≤3s (interactive SLA).

        Args:
            smiles: SMILES string
            progress_callback: Callable(step, total) for UI progress

        Returns:
            dict: {
                'smiles': str,
                'xgb_prob': float,
                'quantum_prob': float,
                'ensemble_prob': float,
                'baseline_score': float,
                'timings': dict,
                'mode': 'fast',
            }
        """
        timings = {}

        # 1. Classical XGBoost (≤50ms)
        t0 = time.time()
        xgb_result = self.classical.predict_xgb(smiles)
        timings["xgb_ms"] = (time.time() - t0) * 1000

        # 2. Quantum statevector (≤3s with cache)
        t0 = time.time()
        quantum_result = self.quantum.predict(
            smiles, mode="statevector", progress_callback=progress_callback
        )
        timings["quantum_s"] = time.time() - t0

        # 3. Baseline heuristic
        baseline_score = self.feature_svc.baseline_rule_score(smiles)

        # 4. Ensemble
        xgb_prob = xgb_result["probability"]
        q_prob = quantum_result["probability"]
        ensemble_prob = self._compute_ensemble(xgb_prob, q_prob)

        return {
            "smiles": smiles,
            "xgb_prob": xgb_prob,
            "quantum_prob": q_prob,
            "ensemble_prob": ensemble_prob,
            "baseline_score": baseline_score,
            "timings": timings,
            "mode": "fast",
            "quantum_cached": quantum_result.get("cached", False),
        }

    def predict_full(self, smiles, n_bootstrap=10, progress_callback=None):
        """
        Full prediction with shot-based quantum evaluation and CI.
        Target latency: 15-120s (acceptable for final candidates).

        Args:
            smiles: SMILES string
            n_bootstrap: Number of bootstrap repeats for CI
            progress_callback: Callable(step, total)

        Returns:
            dict: Same as predict_fast() plus CI fields
        """
        # First get fast result
        fast_result = self.predict_fast(smiles)

        # Then do shot-based evaluation
        t0 = time.time()
        shot_result = self.quantum.predict_with_ci(
            smiles, n_bootstrap=n_bootstrap, progress_callback=progress_callback
        )
        fast_result["timings"]["shot_s"] = time.time() - t0

        # Update with shot-based results
        fast_result["quantum_prob_shot"] = shot_result["probability"]
        fast_result["quantum_ci_lower"] = shot_result["ci_lower"]
        fast_result["quantum_ci_upper"] = shot_result["ci_upper"]
        fast_result["quantum_std"] = shot_result["std"]

        # Recompute ensemble with shot-based quantum
        ensemble_shot = self._compute_ensemble(
            fast_result["xgb_prob"], shot_result["probability"]
        )
        fast_result["ensemble_prob_shot"] = ensemble_shot
        fast_result["mode"] = "full"
        fast_result["n_bootstrap"] = n_bootstrap

        return fast_result

    def predict_batch(self, smiles_list, mode="fast", progress_callback=None):
        """
        Batch prediction for CSV uploads.

        Args:
            smiles_list: List of SMILES strings
            mode: 'fast' or 'full'
            progress_callback: Callable(step, total) for overall progress

        Returns:
            pd.DataFrame with predictions for all molecules
        """
        results = []
        total = len(smiles_list)

        for i, smiles in enumerate(smiles_list):
            if mode == "fast":
                result = self.predict_fast(smiles)
            else:
                result = self.predict_full(smiles)

            results.append(result)
            if progress_callback:
                progress_callback(i + 1, total)

        return pd.DataFrame(results)

    def _compute_ensemble(
        self, xgb_prob, q_prob, w_xgb=None, w_q=None, alert_threshold=None
    ):
        """
        Conservative max-alert ensemble (preserved from V1).

        If either model flags high toxicity, the ensemble is boosted
        to at least 85% of the max signal — safety-first policy.
        """
        w_xgb = w_xgb or (self.config.w_xgb if self.config else W_XGB)
        w_q = w_q or (self.config.w_qml if self.config else W_QML)
        threshold = alert_threshold or (
            self.config.alert_threshold if self.config else ALERT_THRESHOLD
        )

        ensemble_avg = w_xgb * xgb_prob + w_q * q_prob
        either_flags = xgb_prob > threshold or q_prob > threshold

        if either_flags:
            ensemble_prob = max(ensemble_avg, max(xgb_prob, q_prob) * 0.85)
        else:
            ensemble_prob = ensemble_avg

        return float(np.clip(ensemble_prob, 0, 1))

    def generate_report(self, result):
        """Generate a downloadable JSON report from a prediction result."""
        return json.dumps(result, indent=2, default=str)
