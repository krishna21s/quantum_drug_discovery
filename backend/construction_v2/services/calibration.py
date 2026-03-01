"""
Calibration Service — Platt/Isotonic Calibrators
=================================================
Manages calibration for all model predictions to ensure
probabilities are well-calibrated and comparable.
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.base import BaseEstimator, ClassifierMixin

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHECKPOINT_DIR


class ProbabilityCalibrator(BaseEstimator, ClassifierMixin):
    """
    Wraps a raw probability scorer to calibrate it with Platt scaling
    or isotonic regression.
    """

    def __init__(self, method="sigmoid"):
        """
        Args:
            method: 'sigmoid' (Platt scaling) or 'isotonic'
        """
        self.method = method
        self._is_fitted = False
        self._a = 0.0  # Platt scale
        self._b = 0.0  # Platt offset

    def fit(self, raw_probs, true_labels):
        """
        Fit calibration from raw probabilities to true labels.

        Args:
            raw_probs: np.ndarray of uncalibrated probabilities
            true_labels: np.ndarray of binary labels (0/1)
        """
        raw_probs = np.asarray(raw_probs).flatten()
        true_labels = np.asarray(true_labels).flatten()

        if self.method == "sigmoid":
            # Platt scaling: logistic regression on probabilities
            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(C=1.0, max_iter=1000)
            lr.fit(raw_probs.reshape(-1, 1), true_labels)
            self._lr = lr
        elif self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            ir.fit(raw_probs, true_labels)
            self._ir = ir
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self._is_fitted = True
        return self

    def calibrate(self, raw_prob):
        """
        Calibrate a single probability value.

        Args:
            raw_prob: float or np.ndarray of raw probabilities

        Returns:
            Calibrated probability (float or ndarray)
        """
        if not self._is_fitted:
            return raw_prob  # Passthrough if not fitted

        raw_prob = np.atleast_1d(raw_prob)

        if self.method == "sigmoid":
            result = self._lr.predict_proba(raw_prob.reshape(-1, 1))[:, 1]
        elif self.method == "isotonic":
            result = self._ir.predict(raw_prob)

        return float(result[0]) if len(result) == 1 else result

    def save(self, path):
        """Save calibrator to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """Load calibrator from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)


class CalibrationService:
    """
    Manages calibrators for each model in the ensemble.
    Supports fit, apply, save/load, and reliability curve generation.
    """

    def __init__(self, checkpoint_dir=None):
        self.checkpoint_dir = Path(checkpoint_dir or CHECKPOINT_DIR)
        self.calibrators = {}  # model_name → ProbabilityCalibrator

    def fit_calibrator(self, model_name, raw_probs, true_labels, method="sigmoid"):
        """
        Fit a calibrator for a given model.

        Args:
            model_name: e.g., 'xgboost', 'gnn', 'quantum_sv', 'quantum_shot'
            raw_probs: Array of raw model probabilities
            true_labels: Array of true binary labels
            method: 'sigmoid' or 'isotonic'
        """
        cal = ProbabilityCalibrator(method=method)
        cal.fit(raw_probs, true_labels)
        self.calibrators[model_name] = cal
        return cal

    def apply(self, model_name, raw_prob):
        """
        Apply calibration to a raw probability.

        Returns raw_prob unchanged if no calibrator is fitted for this model.
        """
        if model_name in self.calibrators:
            return self.calibrators[model_name].calibrate(raw_prob)
        return raw_prob

    def reliability_curve(self, model_name, probs, labels, n_bins=10):
        """
        Compute reliability (calibration) curve data.

        Returns:
            tuple: (prob_true, prob_pred) arrays for plotting
        """
        prob_true, prob_pred = calibration_curve(
            labels, probs, n_bins=n_bins, strategy="uniform"
        )
        return prob_true, prob_pred

    def save_all(self):
        """Save all calibrators to checkpoint directory."""
        for name, cal in self.calibrators.items():
            path = self.checkpoint_dir / f"calibrator_{name}.pkl"
            cal.save(str(path))

    def load_all(self):
        """Load all calibrators from checkpoint directory."""
        import glob

        for path in self.checkpoint_dir.glob("calibrator_*.pkl"):
            name = path.stem.replace("calibrator_", "")
            self.calibrators[name] = ProbabilityCalibrator.load(str(path))

    @property
    def available_calibrators(self):
        return list(self.calibrators.keys())
