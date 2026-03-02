"""
Second-Order Uncertainty Quantification Module

Quantifies uncertainty about uncertainty estimates themselves.

Key Concept:
    First-order: p(y|x, θ) - uncertainty in predictions given parameters
    Second-order: p(θ|D) - uncertainty in the parameters themselves

This module provides tools to:
1. Estimate credible intervals on variance predictions
2. Identify unreliable uncertainty estimates
3. Quantify meta-uncertainty in probabilistic forecasts

Based on:
- Depeweg et al. (2018): Decomposition of Uncertainty in Bayesian Deep Learning
- Osband et al. (2018): Randomized Prior Functions for Deep RL
- Recent work on second-order probability (2024)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from loguru import logger


@dataclass
class SecondOrderUncertainty:
    """Container for second-order uncertainty estimates."""

    mean_variance: npt.NDArray[np.float64]  # E[Var(y|x,θ)] over θ
    std_variance: npt.NDArray[np.float64]  # SD[Var(y|x,θ)] - meta-uncertainty!
    variance_lower: npt.NDArray[np.float64]  # 2.5th percentile
    variance_upper: npt.NDArray[np.float64]  # 97.5th percentile
    coefficient_of_variation: npt.NDArray[np.float64]  # std / mean

    def summary(self) -> Dict[str, float]:
        """Compute summary statistics."""
        return {
            "mean_variance": float(np.mean(self.mean_variance)),
            "mean_std_variance": float(np.mean(self.std_variance)),
            "mean_cv": float(np.mean(self.coefficient_of_variation)),
            "max_cv": float(np.max(self.coefficient_of_variation)),
        }

    def identify_unreliable_estimates(
        self, cv_threshold: float = 0.3
    ) -> npt.NDArray[np.bool_]:
        """
        Flag predictions where uncertainty estimates are unreliable.

        Args:
            cv_threshold: Coefficient of variation threshold
                         CV > 0.3 means uncertainty estimate has high meta-uncertainty

        Returns:
            Boolean mask of unreliable predictions
        """
        return self.coefficient_of_variation > cv_threshold

    def __repr__(self) -> str:
        stats = self.summary()
        return (
            f"SecondOrderUncertainty(\n"
            f"  Mean variance: {stats['mean_variance']:.4f}\n"
            f"  Mean std of variance: {stats['mean_std_variance']:.4f}\n"
            f"  Mean CV: {stats['mean_cv']:.4f}\n"
            f"  Max CV: {stats['max_cv']:.4f}\n"
            f")"
        )


class SecondOrderAnalyzer:
    """
    Analyze second-order uncertainty using bootstrap ensemble.

    Quantifies: "How certain are we about our uncertainty estimates?"
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize second-order analyzer.

        Args:
            confidence_level: Level for credible intervals (e.g., 0.95 for 95% CI)
        """
        self.confidence_level = confidence_level
        logger.info(
            f"Initialized SecondOrderAnalyzer with "
            f"confidence_level={confidence_level}"
        )

    def analyze_from_ensemble(
        self,
        ensemble_models: List[Any],
        X_test: npt.NDArray[np.float64]
    ) -> SecondOrderUncertainty:
        """
        Analyze second-order uncertainty from bootstrap ensemble.

        Args:
            ensemble_models: List of trained models from bootstrap
            X_test: Test locations [N x D]

        Returns:
            SecondOrderUncertainty with meta-uncertainty estimates
        """
        logger.info(
            f"Analyzing second-order uncertainty from {len(ensemble_models)} models"
        )

        # Collect variance predictions from each model
        variance_samples = []

        for model in ensemble_models:
            _, var = model.predict_f(X_test)
            variance_samples.append(var.numpy().flatten())

        variance_samples = np.array(variance_samples)  # [n_ensemble, n_test]

        # First-order statistics
        mean_variance = np.mean(variance_samples, axis=0)
        std_variance = np.std(variance_samples, axis=0)

        # Credible intervals on variance
        alpha = 1 - self.confidence_level
        variance_lower = np.percentile(variance_samples, 100 * alpha / 2, axis=0)
        variance_upper = np.percentile(variance_samples, 100 * (1 - alpha / 2), axis=0)

        # Coefficient of variation (normalized meta-uncertainty)
        cv = std_variance / (mean_variance + 1e-10)

        return SecondOrderUncertainty(
            mean_variance=mean_variance,
            std_variance=std_variance,
            variance_lower=variance_lower,
            variance_upper=variance_upper,
            coefficient_of_variation=cv
        )

    def compute_prediction_uncertainty_bands(
        self,
        ensemble_models: List[Any],
        X_test: npt.NDArray[np.float64]
    ) -> Dict[str, npt.NDArray[np.float64]]:
        """
        Compute uncertainty bands that account for second-order uncertainty.

        Returns prediction intervals that incorporate both:
        1. Predictive uncertainty (first-order)
        2. Uncertainty about the uncertainty (second-order)

        Args:
            ensemble_models: List of trained models
            X_test: Test locations [N x D]

        Returns:
            Dictionary with conservative uncertainty bands
        """
        logger.info("Computing prediction uncertainty bands with second-order UQ")

        # Collect predictions from ensemble
        means = []
        stds = []

        for model in ensemble_models:
            mean, var = model.predict_f(X_test)
            means.append(mean.numpy().flatten())
            stds.append(np.sqrt(var.numpy().flatten()))

        means = np.array(means)
        stds = np.array(stds)

        # Conservative approach: use upper bound of uncertainty
        alpha = 1 - self.confidence_level
        mean_prediction = np.mean(means, axis=0)
        std_prediction = np.percentile(stds, 100 * (1 - alpha / 2), axis=0)

        # Construct conservative intervals
        z = 1.96  # For 95% intervals
        lower = mean_prediction - z * std_prediction
        upper = mean_prediction + z * std_prediction

        return {
            "mean": mean_prediction,
            "std": std_prediction,
            "lower": lower,
            "upper": upper,
        }

    def spatial_analysis_of_meta_uncertainty(
        self,
        second_order_unc: SecondOrderUncertainty,
        X_test: npt.NDArray[np.float64],
        distance_to_nearest_train: npt.NDArray[np.float64]
    ) -> Dict[str, Any]:
        """
        Analyze how meta-uncertainty varies spatially.

        Hypothesis: Meta-uncertainty increases far from training data.

        Args:
            second_order_unc: SecondOrderUncertainty object
            X_test: Test locations [N x D]
            distance_to_nearest_train: Distance to nearest training point [N]

        Returns:
            Dictionary with spatial analysis results
        """
        logger.info("Analyzing spatial patterns in meta-uncertainty")

        # Bin by distance
        distance_bins = [0, 1, 2, 5, 10, np.inf]
        bin_labels = ["0-1km", "1-2km", "2-5km", "5-10km", ">10km"]

        results = {}

        for i, (d_min, d_max) in enumerate(zip(distance_bins[:-1], distance_bins[1:])):
            mask = (distance_to_nearest_train >= d_min) & (distance_to_nearest_train < d_max)

            if np.sum(mask) > 0:
                results[bin_labels[i]] = {
                    "n_points": int(np.sum(mask)),
                    "mean_cv": float(np.mean(second_order_unc.coefficient_of_variation[mask])),
                    "median_cv": float(np.median(second_order_unc.coefficient_of_variation[mask])),
                    "fraction_unreliable": float(
                        np.mean(second_order_unc.coefficient_of_variation[mask] > 0.3)
                    ),
                }

        return results


class MetaUncertaintyVisualizer:
    """
    Visualization tools for second-order uncertainty.
    """

    @staticmethod
    def plot_variance_credible_intervals(
        second_order_unc: SecondOrderUncertainty,
        indices: Optional[npt.NDArray[np.int64]] = None,
        ax: Optional[Any] = None
    ) -> Any:
        """
        Plot variance predictions with credible intervals.

        Shows:
        - Mean variance estimate
        - 95% credible interval on variance
        - Highlights unreliable estimates

        Args:
            second_order_unc: SecondOrderUncertainty object
            indices: Subset of points to plot (default: first 50)
            ax: Matplotlib axis

        Returns:
            Matplotlib axis
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        if indices is None:
            indices = np.arange(min(50, len(second_order_unc.mean_variance)))

        x = np.arange(len(indices))

        # Plot mean variance with error bands
        ax.plot(
            x,
            second_order_unc.mean_variance[indices],
            'o-',
            label='Mean variance',
            color='blue',
            markersize=4
        )

        # Credible intervals
        ax.fill_between(
            x,
            second_order_unc.variance_lower[indices],
            second_order_unc.variance_upper[indices],
            alpha=0.3,
            color='blue',
            label='95% credible interval'
        )

        # Highlight unreliable estimates
        unreliable = second_order_unc.identify_unreliable_estimates()[indices]
        if np.any(unreliable):
            ax.scatter(
                x[unreliable],
                second_order_unc.mean_variance[indices][unreliable],
                color='red',
                s=100,
                marker='x',
                linewidths=3,
                label='Unreliable (CV > 0.3)',
                zorder=5
            )

        ax.set_xlabel('Prediction Index', fontsize=12)
        ax.set_ylabel('Variance', fontsize=12)
        ax.set_title('Variance Estimates with Second-Order Uncertainty', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        return ax

    @staticmethod
    def plot_cv_histogram(
        second_order_unc: SecondOrderUncertainty,
        ax: Optional[Any] = None
    ) -> Any:
        """
        Plot histogram of coefficient of variation.

        Args:
            second_order_unc: SecondOrderUncertainty object
            ax: Matplotlib axis

        Returns:
            Matplotlib axis
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        cv = second_order_unc.coefficient_of_variation

        ax.hist(cv, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(0.3, color='red', linestyle='--', linewidth=2, label='Unreliable threshold (CV=0.3)')
        ax.axvline(np.median(cv), color='green', linestyle='-', linewidth=2, label=f'Median CV={np.median(cv):.3f}')

        ax.set_xlabel('Coefficient of Variation', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Meta-Uncertainty (CV of Variance)', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')

        # Add statistics text box
        stats_text = (
            f"Mean CV: {np.mean(cv):.3f}\n"
            f"Median CV: {np.median(cv):.3f}\n"
            f"Unreliable: {np.mean(cv > 0.3):.1%}"
        )
        ax.text(
            0.95, 0.95, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10
        )

        return ax


def propagate_second_order_to_decision(
    second_order_unc: SecondOrderUncertainty,
    threshold: float,
    mean_prediction: npt.NDArray[np.float64]
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Propagate second-order uncertainty to decision-making.

    Computes conservative exceedance probabilities that account for
    meta-uncertainty in variance estimates.

    Args:
        second_order_unc: SecondOrderUncertainty object
        threshold: Decision threshold (e.g., PM2.5 = 35 μg/m³)
        mean_prediction: Mean predictions [N]

    Returns:
        Dictionary with conservative and optimistic exceedance probabilities
    """
    from scipy.stats import norm

    # Optimistic: use lower bound of uncertainty
    z_optimistic = (threshold - mean_prediction) / np.sqrt(second_order_unc.variance_lower + 1e-10)
    prob_exceed_optimistic = 1 - norm.cdf(z_optimistic)

    # Conservative: use upper bound of uncertainty
    z_conservative = (threshold - mean_prediction) / np.sqrt(second_order_unc.variance_upper + 1e-10)
    prob_exceed_conservative = 1 - norm.cdf(z_conservative)

    # Mean estimate
    z_mean = (threshold - mean_prediction) / np.sqrt(second_order_unc.mean_variance + 1e-10)
    prob_exceed_mean = 1 - norm.cdf(z_mean)

    return {
        "optimistic": prob_exceed_optimistic,
        "mean": prob_exceed_mean,
        "conservative": prob_exceed_conservative,
        "uncertainty_range": prob_exceed_conservative - prob_exceed_optimistic,
    }


if __name__ == "__main__":
    # Example usage and testing
    logger.info("Testing SecondOrderAnalyzer")

    # Mock ensemble for testing
    class MockModel:
        def __init__(self, noise_level):
            self.noise_level = noise_level

        def predict_f(self, X):
            n = len(X)
            mean = np.random.randn(n)
            # Different models have different variance predictions
            var = self.noise_level * np.random.gamma(2, 1, n)

            class MockTensor:
                def __init__(self, data):
                    self.data = data
                def numpy(self):
                    return self.data

            return MockTensor(mean), MockTensor(var)

    # Create ensemble with varying noise levels
    np.random.seed(42)
    ensemble = [MockModel(noise_level=0.8 + 0.4 * i / 10) for i in range(10)]

    # Test data
    X_test = np.random.randn(100, 3)

    # Analyze second-order uncertainty
    analyzer = SecondOrderAnalyzer()
    second_order = analyzer.analyze_from_ensemble(ensemble, X_test)

    print(second_order)
    print(f"\nUnreliable estimates: {np.sum(second_order.identify_unreliable_estimates())}/100")

    # Test spatial analysis
    distance_to_train = np.random.gamma(2, 1, 100)
    spatial_analysis = analyzer.spatial_analysis_of_meta_uncertainty(
        second_order, X_test, distance_to_train
    )

    print("\nSpatial analysis:")
    for bin_name, stats in spatial_analysis.items():
        print(f"  {bin_name}: {stats}")
