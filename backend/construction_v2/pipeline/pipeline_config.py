"""
Pipeline Configuration — Feature Flags & Latency Budgets
==========================================================
Runtime configuration for the inference pipeline.
Separates deployment-time toggles from build-time constants in config.py.
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ENABLE_GNN,
    ENABLE_SHOT_MODE,
    ENABLE_HARDWARE_CHECK,
    W_XGB,
    W_QML,
    ALERT_THRESHOLD,
    INTERACTIVE_SLA_S,
    SHOT_LATENCY_TARGET_S,
)


class PipelineConfig:
    """
    Runtime pipeline configuration.
    Can be modified at startup or via environment variables.
    """

    def __init__(self):
        # Model toggles
        self.enable_gnn = ENABLE_GNN
        self.enable_shot_mode = ENABLE_SHOT_MODE
        self.enable_hardware_check = ENABLE_HARDWARE_CHECK

        # Ensemble weights
        self.w_xgb = W_XGB
        self.w_qml = W_QML
        self.alert_threshold = ALERT_THRESHOLD

        # Latency
        self.interactive_sla_s = INTERACTIVE_SLA_S
        self.shot_latency_target_s = SHOT_LATENCY_TARGET_S

        # Shot-based evaluation
        self.shot_n_bootstrap = 10
        self.shot_n_shots = 1024

        # Thresholds for triggering final evaluation
        self.final_check_near_boundary = 0.10  # ±10% around 0.5
        self.final_check_top_k = 5  # Top K candidates in batch mode

    @classmethod
    def from_env(cls):
        """Create config from environment variables (for Docker/K8s)."""
        cfg = cls()
        cfg.enable_gnn = os.environ.get("ENABLE_GNN", "false").lower() == "true"
        cfg.enable_shot_mode = os.environ.get("ENABLE_SHOT", "true").lower() == "true"
        cfg.enable_hardware_check = (
            os.environ.get("ENABLE_HW", "false").lower() == "true"
        )
        return cfg

    def should_trigger_final_check(self, ensemble_prob):
        """
        Determine if a molecule should undergo shot-based final evaluation.
        Triggers when probability is near the decision boundary.
        """
        if not self.enable_shot_mode:
            return False
        return abs(ensemble_prob - 0.5) < self.final_check_near_boundary

    def to_dict(self):
        return {
            "enable_gnn": self.enable_gnn,
            "enable_shot_mode": self.enable_shot_mode,
            "enable_hardware_check": self.enable_hardware_check,
            "w_xgb": self.w_xgb,
            "w_qml": self.w_qml,
            "alert_threshold": self.alert_threshold,
            "interactive_sla_s": self.interactive_sla_s,
            "shot_n_bootstrap": self.shot_n_bootstrap,
        }
