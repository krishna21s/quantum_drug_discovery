"""
Monitoring — Metrics Recording & Dashboards
=============================================
Records model metrics (AUC, Brier, latency) and operational
metrics (cache hits, worker utilization) for observability.
"""

import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHECKPOINT_DIR


class MetricsRecorder:
    """
    Records and aggregates model and operational metrics.

    Stores metrics in-memory and optionally persists to JSON for audit.
    """

    def __init__(self, persist_dir=None):
        self.persist_dir = Path(persist_dir or CHECKPOINT_DIR)
        self._metrics = defaultdict(list)
        self._counters = defaultdict(int)
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # INFERENCE METRICS
    # ------------------------------------------------------------------

    def record_prediction(self, model_name, probability, latency_s, cached=False):
        """Record a single prediction event."""
        self._metrics[f"{model_name}_probs"].append(probability)
        self._metrics[f"{model_name}_latency"].append(latency_s)
        self._counters[f"{model_name}_total"] += 1
        if cached:
            self._counters[f"{model_name}_cache_hits"] += 1

    def record_ensemble(self, xgb_prob, q_prob, ensemble_prob):
        """Record ensemble prediction components."""
        self._metrics["disagreement"].append(abs(xgb_prob - q_prob))
        self._metrics["ensemble_probs"].append(ensemble_prob)

    # ------------------------------------------------------------------
    # MODEL EVALUATION METRICS
    # ------------------------------------------------------------------

    def compute_model_metrics(self, y_true, y_pred_proba, model_name):
        """Compute AUC, Brier, calibration error for a model."""
        from sklearn.metrics import roc_auc_score, brier_score_loss

        metrics = {}
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))
        except Exception:
            metrics["roc_auc"] = None

        metrics["brier"] = float(brier_score_loss(y_true, y_pred_proba))
        metrics["mean_prediction"] = float(np.mean(y_pred_proba))
        metrics["std_prediction"] = float(np.std(y_pred_proba))
        metrics["model"] = model_name
        metrics["timestamp"] = datetime.now().isoformat()

        self._metrics[f"{model_name}_eval"].append(metrics)
        return metrics

    # ------------------------------------------------------------------
    # OPERATIONAL METRICS
    # ------------------------------------------------------------------

    def cache_hit_ratio(self, model_name):
        """Compute cache hit ratio for a model."""
        total = self._counters.get(f"{model_name}_total", 0)
        hits = self._counters.get(f"{model_name}_cache_hits", 0)
        return hits / total if total > 0 else 0.0

    def mean_latency(self, model_name):
        """Average latency for a model."""
        latencies = self._metrics.get(f"{model_name}_latency", [])
        return np.mean(latencies) if latencies else 0.0

    def disagreement_rate(self, threshold=0.2):
        """Fraction of predictions where |XGB - Quantum| > threshold."""
        disag = self._metrics.get("disagreement", [])
        if not disag:
            return 0.0
        return float(np.mean(np.array(disag) > threshold))

    # ------------------------------------------------------------------
    # SUMMARY & PERSISTENCE
    # ------------------------------------------------------------------

    def summary(self):
        """Generate a metrics summary dict."""
        return {
            "uptime_s": time.time() - self._start_time,
            "counters": dict(self._counters),
            "xgb_mean_latency_ms": self.mean_latency("xgboost") * 1000,
            "quantum_mean_latency_s": self.mean_latency("quantum"),
            "cache_hit_ratio_quantum": self.cache_hit_ratio("quantum"),
            "disagreement_rate": self.disagreement_rate(),
            "timestamp": datetime.now().isoformat(),
        }

    def save(self, filename="metrics_log.json"):
        """Persist metrics to JSON."""
        path = self.persist_dir / filename
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)

    def reset(self):
        """Clear all recorded metrics."""
        self._metrics.clear()
        self._counters.clear()
        self._start_time = time.time()
