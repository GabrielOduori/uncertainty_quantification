"""
Conformal Prediction Module

Provides distribution-free uncertainty quantification with finite-sample coverage guarantees.

Key Advantage:
    P(y* ∈ C(x*)) ≥ 1-α for ANY test point

This guarantee holds regardless of:
- Model misspecification
- Distribution assumptions
- Sample size (finite-sample guarantee)

Implementation based on:
- Vovk et al. (2005): Algorithmic Learning in a Random World
- Shafer & Vovk (2008): A Tutorial on Conformal Prediction
- Angelopoulos & Bates (2021): Gentle Introduction to Conformal Prediction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import numpy.typing as npt
from loguru import logger
from scipy.stats import norm


@dataclass
class ConformalIntervals:
    """Container for conformal prediction intervals."""

    means: npt.NDArray[np.float64]  # Point predictions [N]
    lower_bounds: npt.NDArray[np.float64]  # Lower bounds [N]
    upper_bounds: npt.NDArray[np.float64]  # Upper bounds [N]
    widths: npt.NDArray[np.float64]  # Interval widths [N]
    coverage_level: float  # Nominal coverage (e.g., 0.95)
    calibrated_quantile: float  # Calibrated nonconformity quantile

    def summary(self) -> dict:
        """Summarize interval statistics."""
        return {
            "mean_width": float(np.mean(self.widths)),
            "median_width": float(np.median(self.widths)),
            "min_width": float(np.min(self.widths)),
            "max_width": float(np.max(self.widths)),
            "coverage_level": self.coverage_level,
            "quantile": self.calibrated_quantile,
        }

    def __repr__(self) -> str:
        stats = self.summary()
        return (
            f"ConformalIntervals(\n"
            f"  Coverage: {stats['coverage_level']:.1%}\n"
            f"  Mean width: {stats['mean_width']:.4f}\n"
            f"  Calibrated quantile: {stats['quantile']:.4f}\n"
            f")"
        )


class ConformalPredictionWrapper:
    """
    Wrap any probabilistic model with conformal prediction intervals.

    Provides distribution-free finite-sample coverage guarantees by
    calibrating on a held-out calibration set.

    Algorithm (Split Conformal Prediction):
    1. Split data: Train / Calibration / Test
    2. Train model on Train set
    3. Compute nonconformity scores on Calibration set
    4. Find quantile q̂ such that coverage ≥ 1-α
    5. Prediction set: Ĉ(x) = [ŷ(x) - q̂·σ̂(x), ŷ(x) + q̂·σ̂(x)]
    """

    def __init__(
        self,
        base_model: Any,
        alpha: float = 0.05,
        score_type: str = "normalized"
    ):
        """
        Initialize conformal prediction wrapper.

        Args:
            base_model: Trained probabilistic model with predict_f method
            alpha: Miscoverage rate (1-α is coverage level)
            score_type: Type of nonconformity score
                       'normalized': |y - ŷ| / σ̂ (adaptive to uncertainty)
                       'absolute': |y - ŷ| (fixed width)
        """
        self.base_model = base_model
        self.alpha = alpha
        self.score_type = score_type
        self.quantile: Optional[float] = None
        self.is_calibrated = False

        logger.info(
            f"Initialized ConformalPredictionWrapper with α={alpha}, "
            f"score_type={score_type}"
        )

    def calibrate(
        self,
        X_cal: npt.NDArray[np.float64],
        y_cal: npt.NDArray[np.float64]
    ):
        """
        Calibrate nonconformity scores on calibration set.

        Computes the (1-α)-quantile of nonconformity scores, adjusted for
        finite-sample guarantee.

        Args:
            X_cal: Calibration features [N_cal x D]
            y_cal: Calibration targets [N_cal]
        """
        logger.info(f"Calibrating on {len(X_cal)} samples with α={self.alpha}")

        # Get model predictions on calibration set
        mean_cal, var_cal = self.base_model.predict_f(X_cal)

        # Handle both tensors (GPflow) and numpy arrays (mock models)
        if hasattr(mean_cal, 'numpy'):
            mean_cal = mean_cal.numpy().flatten()
            std_cal = np.sqrt(var_cal.numpy().flatten())
        else:
            mean_cal = np.asarray(mean_cal).flatten()
            std_cal = np.sqrt(np.asarray(var_cal).flatten())

        # Compute nonconformity scores
        if self.score_type == "normalized":
            # Adaptive: normalized by predicted uncertainty
            scores = np.abs(y_cal - mean_cal) / (std_cal + 1e-10)
        elif self.score_type == "absolute":
            # Fixed: absolute residuals
            scores = np.abs(y_cal - mean_cal)
        else:
            raise ValueError(f"Unknown score_type: {self.score_type}")

        # Compute quantile with finite-sample correction
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile = float(np.quantile(scores, q_level))

        self.is_calibrated = True

        logger.info(
            f"Calibration complete. Quantile q̂={self.quantile:.4f} "
            f"at level {q_level:.4f}"
        )

    def predict_with_conformal_intervals(
        self,
        X_test: npt.NDArray[np.float64]
    ) -> ConformalIntervals:
        """
        Predict with conformal prediction intervals.

        Guarantee: P(y ∈ [lower, upper]) ≥ 1-α

        Args:
            X_test: Test locations [N x D]

        Returns:
            ConformalIntervals with finite-sample coverage guarantee

        Raises:
            RuntimeError: If not calibrated yet
        """
        if not self.is_calibrated:
            raise RuntimeError("Must call calibrate() before prediction")

        logger.info(f"Predicting with conformal intervals for {len(X_test)} points")

        # Get base model predictions
        mean, var = self.base_model.predict_f(X_test)

        # Handle both tensors (GPflow) and numpy arrays (mock models)
        if hasattr(mean, 'numpy'):
            mean = mean.numpy().flatten()
            std = np.sqrt(var.numpy().flatten())
        else:
            mean = np.asarray(mean).flatten()
            std = np.sqrt(np.asarray(var).flatten())

        # Construct conformal intervals
        if self.score_type == "normalized":
            # Adaptive intervals (width depends on local uncertainty)
            half_width = self.quantile * std
        elif self.score_type == "absolute":
            # Fixed-width intervals
            half_width = self.quantile * np.ones_like(mean)
        else:
            raise ValueError(f"Unknown score_type: {self.score_type}")

        lower = mean - half_width
        upper = mean + half_width
        widths = 2 * half_width

        return ConformalIntervals(
            means=mean,
            lower_bounds=lower,
            upper_bounds=upper,
            widths=widths,
            coverage_level=1 - self.alpha,
            calibrated_quantile=self.quantile
        )

    def compare_with_gaussian_intervals(
        self,
        X_test: npt.NDArray[np.float64],
        y_test: npt.NDArray[np.float64]
    ) -> dict:
        """
        Compare conformal intervals with standard Gaussian intervals.

        Args:
            X_test: Test features [N x D]
            y_test: True test values [N]

        Returns:
            Dictionary comparing both methods
        """
        # Conformal intervals
        conformal_intervals = self.predict_with_conformal_intervals(X_test)
        conformal_coverage = np.mean(
            (y_test >= conformal_intervals.lower_bounds) &
            (y_test <= conformal_intervals.upper_bounds)
        )
        conformal_width = np.mean(conformal_intervals.widths)

        # Gaussian intervals
        mean, var = self.base_model.predict_f(X_test)

        # Handle both tensors (GPflow) and numpy arrays (mock models)
        if hasattr(mean, 'numpy'):
            mean = mean.numpy().flatten()
            std = np.sqrt(var.numpy().flatten())
        else:
            mean = np.asarray(mean).flatten()
            std = np.sqrt(np.asarray(var).flatten())

        z = norm.ppf(1 - self.alpha / 2)
        gaussian_lower = mean - z * std
        gaussian_upper = mean + z * std
        gaussian_coverage = np.mean(
            (y_test >= gaussian_lower) & (y_test <= gaussian_upper)
        )
        gaussian_width = np.mean(2 * z * std)

        return {
            "conformal_coverage": float(conformal_coverage),
            "conformal_width": float(conformal_width),
            "gaussian_coverage": float(gaussian_coverage),
            "gaussian_width": float(gaussian_width),
            "target_coverage": 1 - self.alpha,
            "conformal_achieves_target": conformal_coverage >= (1 - self.alpha),
            "gaussian_achieves_target": gaussian_coverage >= (1 - self.alpha),
        }


class AdaptiveConformalPredictor:
    """
    Adaptive conformal prediction for non-stationary environments.

    Updates quantile online as new data arrives, suitable for air quality
    monitoring where conditions change over time.
    """

    def __init__(
        self,
        base_model: Any,
        alpha: float = 0.05,
        window_size: int = 100
    ):
        """
        Initialize adaptive conformal predictor.

        Args:
            base_model: Trained probabilistic model
            alpha: Miscoverage rate
            window_size: Number of recent observations for quantile update
        """
        self.base_model = base_model
        self.alpha = alpha
        self.window_size = window_size
        self.recent_scores: list = []
        self.quantile: Optional[float] = None

        logger.info(
            f"Initialized AdaptiveConformalPredictor with "
            f"window_size={window_size}"
        )

    def update(self, x: npt.NDArray[np.float64], y: float):
        """
        Update quantile with new observation.

        Args:
            x: Feature vector [D]
            y: True value (scalar)
        """
        # Get prediction
        mean, var = self.base_model.predict_f(x.reshape(1, -1))
        mean = float(mean.numpy().flatten()[0])
        std = float(np.sqrt(var.numpy().flatten()[0]))

        # Compute nonconformity score
        score = abs(y - mean) / (std + 1e-10)

        # Add to recent scores
        self.recent_scores.append(score)
        if len(self.recent_scores) > self.window_size:
            self.recent_scores.pop(0)

        # Update quantile
        if len(self.recent_scores) >= 10:  # Minimum for stable quantile
            n = len(self.recent_scores)
            q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            self.quantile = float(np.quantile(self.recent_scores, q_level))

    def predict(
        self,
        X_test: npt.NDArray[np.float64]
    ) -> ConformalIntervals:
        """
        Predict with adaptive conformal intervals.

        Args:
            X_test: Test features [N x D]

        Returns:
            ConformalIntervals using current quantile
        """
        if self.quantile is None:
            raise RuntimeError("Must update with at least 10 observations first")

        mean, var = self.base_model.predict_f(X_test)
        mean = mean.numpy().flatten()
        std = np.sqrt(var.numpy().flatten())

        half_width = self.quantile * std
        lower = mean - half_width
        upper = mean + half_width

        return ConformalIntervals(
            means=mean,
            lower_bounds=lower,
            upper_bounds=upper,
            widths=2 * half_width,
            coverage_level=1 - self.alpha,
            calibrated_quantile=self.quantile
        )


def evaluate_conformal_coverage(
    conformal_intervals: ConformalIntervals,
    y_test: npt.NDArray[np.float64]
) -> dict:
    """
    Evaluate actual coverage of conformal intervals.

    Args:
        conformal_intervals: Predicted intervals
        y_test: True test values [N]

    Returns:
        Dictionary with coverage statistics
    """
    # Check coverage
    within_bounds = (
        (y_test >= conformal_intervals.lower_bounds) &
        (y_test <= conformal_intervals.upper_bounds)
    )
    actual_coverage = float(np.mean(within_bounds))

    # Compute miscoverage
    miscoverage = 1 - actual_coverage
    target_coverage = conformal_intervals.coverage_level

    return {
        "actual_coverage": actual_coverage,
        "target_coverage": target_coverage,
        "miscoverage": miscoverage,
        "achieves_target": actual_coverage >= target_coverage,
        "coverage_error": actual_coverage - target_coverage,
        "mean_width": float(np.mean(conformal_intervals.widths)),
    }


if __name__ == "__main__":
    # Example usage and testing
    logger.info("Testing ConformalPredictionWrapper")

    # Mock model for testing
    class MockModel:
        def predict_f(self, X):
            n = len(X)
            mean = np.sin(X[:, 0]) + np.random.randn(n) * 0.1
            var = 0.5 + 0.3 * np.abs(X[:, 0])  # Heteroscedastic

            class MockTensor:
                def __init__(self, data):
                    self.data = data
                def numpy(self):
                    return self.data

            return MockTensor(mean), MockTensor(var)

    # Generate data
    np.random.seed(42)
    n_cal = 200
    n_test = 100

    X_cal = np.random.randn(n_cal, 3)
    y_cal = np.sin(X_cal[:, 0]) + np.random.randn(n_cal) * 0.5

    X_test = np.random.randn(n_test, 3)
    y_test = np.sin(X_test[:, 0]) + np.random.randn(n_test) * 0.5

    # Initialize and calibrate
    mock_model = MockModel()
    conformal = ConformalPredictionWrapper(mock_model, alpha=0.05)
    conformal.calibrate(X_cal, y_cal)

    # Predict with conformal intervals
    intervals = conformal.predict_with_conformal_intervals(X_test)
    print(intervals)

    # Evaluate coverage
    coverage_stats = evaluate_conformal_coverage(intervals, y_test)
    print("\nCoverage evaluation:")
    for key, value in coverage_stats.items():
        print(f"  {key}: {value}")

    # Compare with Gaussian
    comparison = conformal.compare_with_gaussian_intervals(X_test, y_test)
    print("\nConformal vs Gaussian comparison:")
    for key, value in comparison.items():
        print(f"  {key}: {value}")
