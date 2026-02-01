"""
Calibration Evaluation Module

Provides tools for evaluating probabilistic calibration quality:
- Prediction Interval Coverage Probability (PICP)
- Expected Calibration Error (ECE)
- Continuous Ranked Probability Score (CRPS)
- Sharpness metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from loguru import logger
from scipy.stats import norm


@dataclass
class CalibrationResults:
    """Container for calibration evaluation results."""

    picp: Dict[str, float]  # Coverage at different levels
    ece: float  # Expected Calibration Error
    crps: float  # Continuous Ranked Probability Score
    sharpness: float  # Average prediction interval width
    is_calibrated: bool  # Overall calibration flag
    details: Dict[str, any]  # Additional diagnostic info

    def __repr__(self) -> str:
        return (
            f"CalibrationResults(\n"
            f"  PICP(95%): {self.picp.get('95%', 0.0):.3f}\n"
            f"  ECE: {self.ece:.4f}\n"
            f"  CRPS: {self.crps:.4f}\n"
            f"  Sharpness: {self.sharpness:.4f}\n"
            f"  Calibrated: {self.is_calibrated}\n"
            f")"
        )


class CalibrationEvaluator:
    """
    Comprehensive calibration evaluation for probabilistic predictions.

    Evaluates whether uncertainty estimates are well-calibrated, meaning that
    predicted confidence intervals contain the true values at the expected rates.
    """

    def __init__(
        self,
        confidence_levels: Optional[List[float]] = None,
        ece_bins: int = 10,
        calibration_tolerance: float = 0.05,
    ):
        """
        Initialize calibration evaluator.

        Args:
            confidence_levels: Levels to evaluate coverage (default: [0.5, 0.68, 0.95, 0.99])
            ece_bins: Number of bins for ECE calculation
            calibration_tolerance: Tolerance for calibration (|observed - expected| < tol)
        """
        self.confidence_levels = confidence_levels or [0.50, 0.68, 0.95, 0.99]
        self.ece_bins = ece_bins
        self.tolerance = calibration_tolerance

        logger.info(
            f"Initialized CalibrationEvaluator with {len(self.confidence_levels)} "
            f"confidence levels and {ece_bins} ECE bins"
        )

    def evaluate(
        self,
        predictions: npt.NDArray[np.float64],
        uncertainties: npt.NDArray[np.float64],
        actuals: npt.NDArray[np.float64],
    ) -> CalibrationResults:
        """
        Comprehensive calibration evaluation.

        Args:
            predictions: Predictive means [N]
            uncertainties: Predictive standard deviations [N]
            actuals: True values [N]

        Returns:
            CalibrationResults with all metrics
        """
        logger.info(f"Evaluating calibration for {len(predictions)} predictions")

        # Prediction Interval Coverage Probability
        picp = self._compute_picp_multi_level(predictions, uncertainties, actuals)

        # Expected Calibration Error
        ece = self._compute_ece(predictions, uncertainties, actuals)

        # Continuous Ranked Probability Score
        crps = compute_crps(predictions, uncertainties, actuals)

        # Sharpness
        sharpness = float(np.mean(uncertainties))

        # Check if calibrated
        is_calibrated = self._check_calibration(picp)

        # Additional diagnostics
        details = {
            "n_samples": len(predictions),
            "mean_abs_error": float(np.mean(np.abs(actuals - predictions))),
            "rmse": float(np.sqrt(np.mean((actuals - predictions) ** 2))),
            "normalized_residuals": self._compute_normalized_residuals(
                predictions, uncertainties, actuals
            ),
        }

        return CalibrationResults(
            picp=picp,
            ece=ece,
            crps=crps,
            sharpness=sharpness,
            is_calibrated=is_calibrated,
            details=details,
        )

    def _compute_picp_multi_level(
        self,
        predictions: npt.NDArray[np.float64],
        uncertainties: npt.NDArray[np.float64],
        actuals: npt.NDArray[np.float64],
    ) -> Dict[str, float]:
        """Compute PICP at multiple confidence levels."""
        picp_dict = {}

        for alpha in self.confidence_levels:
            coverage = compute_picp(predictions, uncertainties, actuals, alpha)
            picp_dict[f"{int(alpha * 100)}%"] = coverage

        return picp_dict

    def _compute_ece(
        self,
        predictions: npt.NDArray[np.float64],
        uncertainties: npt.NDArray[np.float64],
        actuals: npt.NDArray[np.float64],
    ) -> float:
        """Compute Expected Calibration Error."""
        return compute_ece(predictions, uncertainties, actuals, n_bins=self.ece_bins)

    def _check_calibration(self, picp: Dict[str, float]) -> bool:
        """Check if model is well-calibrated based on PICP."""
        # Check 95% coverage specifically
        coverage_95 = picp.get("95%", 0.0)
        is_calibrated = abs(coverage_95 - 0.95) < self.tolerance

        if not is_calibrated:
            logger.warning(
                f"Model is not well-calibrated: "
                f"95% PICP = {coverage_95:.3f} (expected 0.95)"
            )

        return is_calibrated

    def _compute_normalized_residuals(
        self,
        predictions: npt.NDArray[np.float64],
        uncertainties: npt.NDArray[np.float64],
        actuals: npt.NDArray[np.float64],
    ) -> Dict[str, float]:
        """Compute normalized residuals (z-scores)."""
        z_scores = (actuals - predictions) / uncertainties

        return {
            "mean": float(np.mean(z_scores)),
            "std": float(np.std(z_scores)),
            "abs_mean": float(np.mean(np.abs(z_scores))),
        }


def compute_picp(
    predictions: npt.NDArray[np.float64],
    uncertainties: npt.NDArray[np.float64],
    actuals: npt.NDArray[np.float64],
    alpha: float = 0.95,
) -> float:
    """
    Compute Prediction Interval Coverage Probability.

    The fraction of true values falling within the α confidence interval.

    Args:
        predictions: Predictive means [N]
        uncertainties: Predictive standard deviations [N]
        actuals: True values [N]
        alpha: Confidence level (default: 0.95 for 95% intervals)

    Returns:
        Coverage probability (should be ≈ alpha for well-calibrated model)
    """
    # Z-score for two-sided interval
    z = norm.ppf((1 + alpha) / 2)

    # Compute bounds
    lower = predictions - z * uncertainties
    upper = predictions + z * uncertainties

    # Count how many actuals fall within bounds
    within_bounds = (actuals >= lower) & (actuals <= upper)
    coverage = float(np.mean(within_bounds))

    return coverage


def compute_ece(
    predictions: npt.NDArray[np.float64],
    uncertainties: npt.NDArray[np.float64],
    actuals: npt.NDArray[np.float64],
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error.

    Measures average deviation between predicted and observed coverage across
    multiple confidence levels.

    Args:
        predictions: Predictive means [N]
        uncertainties: Predictive standard deviations [N]
        actuals: True values [N]
        n_bins: Number of confidence levels to evaluate

    Returns:
        ECE score (lower is better, 0 is perfect)
    """
    alphas = np.linspace(0.1, 0.9, n_bins)
    calibration_errors = []

    for alpha in alphas:
        observed_coverage = compute_picp(predictions, uncertainties, actuals, alpha)
        error = abs(observed_coverage - alpha)
        calibration_errors.append(error)

    ece = float(np.mean(calibration_errors))
    return ece


def compute_crps(
    predictions: npt.NDArray[np.float64],
    uncertainties: npt.NDArray[np.float64],
    actuals: npt.NDArray[np.float64],
) -> float:
    """
    Compute Continuous Ranked Probability Score for Gaussian predictions.

    CRPS is a proper scoring rule that evaluates both accuracy and uncertainty.
    Lower is better.

    Formula for Gaussian:
        CRPS = σ * [φ(z) - z(2Φ(z) - 1) + 1/√π]
    where z = (y - μ) / σ

    Args:
        predictions: Predictive means [N]
        uncertainties: Predictive standard deviations [N]
        actuals: True values [N]

    Returns:
        Mean CRPS across all predictions
    """
    # Normalized residuals
    z = (actuals - predictions) / uncertainties

    # CRPS formula
    crps_values = uncertainties * (
        z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)
    )

    return float(np.mean(crps_values))


def compute_sharpness(
    uncertainties: npt.NDArray[np.float64], alpha: float = 0.95
) -> float:
    """
    Compute sharpness (average prediction interval width).

    Sharpness measures how confident predictions are. Lower is "sharper" (more confident).
    However, sharpness should not be minimized without maintaining calibration!

    Args:
        uncertainties: Predictive standard deviations [N]
        alpha: Confidence level for interval width

    Returns:
        Average interval width
    """
    z = norm.ppf((1 + alpha) / 2)
    interval_widths = 2 * z * uncertainties
    return float(np.mean(interval_widths))


def plot_calibration_curve(
    predictions: npt.NDArray[np.float64],
    uncertainties: npt.NDArray[np.float64],
    actuals: npt.NDArray[np.float64],
    ax: Optional[any] = None,
) -> any:
    """
    Plot calibration curve (reliability diagram).

    Args:
        predictions: Predictive means [N]
        uncertainties: Predictive standard deviations [N]
        actuals: True values [N]
        ax: Matplotlib axis (created if None)

    Returns:
        Matplotlib axis with calibration plot
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Compute coverage at multiple levels
    alphas = np.linspace(0.05, 0.95, 20)
    observed_coverage = [
        compute_picp(predictions, uncertainties, actuals, alpha) for alpha in alphas
    ]

    # Plot
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=2)
    ax.plot(alphas, observed_coverage, "o-", label="Observed", linewidth=2, markersize=6)

    # Shaded tolerance region
    tolerance = 0.05
    ax.fill_between(
        alphas,
        alphas - tolerance,
        alphas + tolerance,
        alpha=0.2,
        color="green",
        label=f"±{tolerance:.0%} tolerance",
    )

    ax.set_xlabel("Expected Coverage", fontsize=12)
    ax.set_ylabel("Observed Coverage", fontsize=12)
    ax.set_title("Calibration Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    return ax


if __name__ == "__main__":
    # Example usage
    logger.info("Testing CalibrationEvaluator")

    # Simulate well-calibrated predictions
    np.random.seed(42)
    n = 1000

    predictions = np.random.randn(n) * 10 + 50
    true_sigma = 5.0
    uncertainties = true_sigma * np.ones(n)
    actuals = predictions + np.random.randn(n) * true_sigma

    # Evaluate
    evaluator = CalibrationEvaluator()
    results = evaluator.evaluate(predictions, uncertainties, actuals)

    print(results)
    print("\nDetailed PICP:")
    for level, coverage in results.picp.items():
        expected = float(level.strip("%")) / 100
        print(f"  {level}: {coverage:.3f} (expected {expected:.3f})")
