"""
Uncertainty Estimation — Bootstrap CI for Shot-Based Predictions
================================================================
Provides standalone uncertainty estimation that can wrap any
prediction service, not just the quantum kernel.
"""

import numpy as np
import time


class UncertaintyEstimator:
    """
    Bootstrap confidence interval estimator.

    Can wrap any prediction function to produce mean ± CI
    from repeated stochastic evaluations.
    """

    def __init__(self, confidence_level=0.95):
        """
        Args:
            confidence_level: CI confidence level (default 0.95 for 95% CI)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def bootstrap_prediction(self, predict_fn, n_repeats=10):
        """
        Run predict_fn() multiple times and compute bootstrap statistics.

        Args:
            predict_fn: Callable that returns a float (probability)
            n_repeats: Number of bootstrap repeats

        Returns:
            dict: {
                'mean': float,
                'std': float,
                'ci_lower': float,
                'ci_upper': float,
                'raw_values': list[float],
                'n_repeats': int,
                'confidence_level': float,
            }
        """
        values = [predict_fn() for _ in range(n_repeats)]
        arr = np.array(values)

        lower_pct = 100 * self.alpha / 2
        upper_pct = 100 * (1 - self.alpha / 2)

        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_lower": float(np.percentile(arr, lower_pct)),
            "ci_upper": float(np.percentile(arr, upper_pct)),
            "raw_values": values,
            "n_repeats": n_repeats,
            "confidence_level": self.confidence_level,
        }

    @staticmethod
    def format_ci(mean, ci_lower, ci_upper):
        """Format a CI as a human-readable string."""
        return f"{mean:.2%} [{ci_lower:.2%}, {ci_upper:.2%}]"
