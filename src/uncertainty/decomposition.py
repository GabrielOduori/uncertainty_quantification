"""
Uncertainty Decomposition Module

Provides tools for decomposing total uncertainty into epistemic and aleatoric components.
Supports both GP-based and hybrid GAM-SSM models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple

import numpy as np
import numpy.typing as npt
from loguru import logger


class ProbabilisticModel(Protocol):
    """Protocol for probabilistic models with uncertainty quantification."""

    def predict(
        self, X: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Predict mean and variance.

        Args:
            X: Input features [N x D]

        Returns:
            mean: Predictive mean [N]
            variance: Predictive variance [N]
        """
        ...


@dataclass
class UncertaintyComponents:
    """Container for decomposed uncertainty components."""

    total: npt.NDArray[np.float64]
    epistemic: npt.NDArray[np.float64]
    aleatoric: npt.NDArray[np.float64]
    epistemic_fraction: npt.NDArray[np.float64]
    aleatoric_fraction: npt.NDArray[np.float64]

    def summary_stats(self) -> Dict[str, float]:
        """Compute summary statistics across all predictions."""
        return {
            "mean_total_std": float(np.mean(self.total)),
            "mean_epistemic_std": float(np.mean(self.epistemic)),
            "mean_aleatoric_std": float(np.mean(self.aleatoric)),
            "avg_epistemic_fraction": float(np.mean(self.epistemic_fraction)),
            "avg_aleatoric_fraction": float(np.mean(self.aleatoric_fraction)),
            "median_epistemic_fraction": float(np.median(self.epistemic_fraction)),
        }

    def __repr__(self) -> str:
        stats = self.summary_stats()
        return (
            f"UncertaintyComponents(\n"
            f"  Total: μ={stats['mean_total_std']:.3f}\n"
            f"  Epistemic: μ={stats['mean_epistemic_std']:.3f} "
            f"({stats['avg_epistemic_fraction']:.1%})\n"
            f"  Aleatoric: μ={stats['mean_aleatoric_std']:.3f} "
            f"({stats['avg_aleatoric_fraction']:.1%})\n"
            f")"
        )


class UncertaintyDecomposer:
    """
    Decompose predictive uncertainty into epistemic and aleatoric components.

    Supports multiple model types:
    - Sparse Variational GPs (SVGPs)
    - Hybrid GAM-SSM-LUR models
    - Custom models implementing ProbabilisticModel protocol
    """

    def __init__(self, model_type: str = "svgp"):
        """
        Initialize decomposer.

        Args:
            model_type: Type of model ('svgp', 'gam_ssm', 'custom')
        """
        self.model_type = model_type
        logger.info(f"Initialized UncertaintyDecomposer for {model_type} models")

    def decompose_svgp(
        self,
        model: any,  # GPflow SVGP model
        X_test: npt.NDArray[np.float64],
        include_noise: bool = True,
    ) -> UncertaintyComponents:
        """
        Decompose uncertainty for Sparse Variational GP models.

        Args:
            model: Trained SVGP model (GPflow)
            X_test: Test locations [N x D]
            include_noise: Whether to include observation noise in aleatoric

        Returns:
            UncertaintyComponents with decomposed uncertainties
        """
        logger.info(f"Decomposing uncertainty for {len(X_test)} test points")

        # Get GP predictions (f-distribution, no noise)
        f_mean, f_var = model.predict_f(X_test)
        f_mean = f_mean.numpy().flatten()
        f_var = f_var.numpy().flatten()

        # Epistemic uncertainty from GP posterior
        sigma_epistemic = np.sqrt(f_var)

        # Aleatoric uncertainty from likelihood
        if include_noise:
            try:
                # Try to get noise variance from likelihood
                noise_var = model.likelihood.variance.numpy()
                sigma_aleatoric = np.sqrt(noise_var) * np.ones_like(sigma_epistemic)
            except AttributeError:
                logger.warning("Could not extract noise variance, setting aleatoric=0")
                sigma_aleatoric = np.zeros_like(sigma_epistemic)
        else:
            sigma_aleatoric = np.zeros_like(sigma_epistemic)

        # Total uncertainty (independent sources add variances)
        sigma_total = np.sqrt(f_var + sigma_aleatoric**2)

        # Compute fractions
        var_total = sigma_total**2
        epistemic_fraction = f_var / var_total
        aleatoric_fraction = sigma_aleatoric**2 / var_total

        return UncertaintyComponents(
            total=sigma_total,
            epistemic=sigma_epistemic,
            aleatoric=sigma_aleatoric,
            epistemic_fraction=epistemic_fraction,
            aleatoric_fraction=aleatoric_fraction,
        )

    def decompose_gam_ssm(
        self,
        spatial_std: npt.NDArray[np.float64],
        temporal_std: npt.NDArray[np.float64],
        correlation: float = 0.0,
    ) -> UncertaintyComponents:
        """
        Decompose uncertainty for GAM-SSM-LUR hybrid models.

        Args:
            spatial_std: Spatial component uncertainty [N]
            temporal_std: Temporal component uncertainty [N]
            correlation: Correlation between spatial and temporal errors

        Returns:
            UncertaintyComponents treating spatial/temporal as epistemic/aleatoric analogs
        """
        logger.info(
            f"Decomposing GAM-SSM uncertainty with correlation={correlation:.3f}"
        )

        # Combine with correlation
        var_spatial = spatial_std**2
        var_temporal = temporal_std**2
        covar = 2 * correlation * spatial_std * temporal_std

        # Total variance
        var_total = var_spatial + var_temporal + covar

        # Ensure non-negative
        var_total = np.maximum(var_total, var_spatial + var_temporal * 0.5)
        sigma_total = np.sqrt(var_total)

        # For GAM-SSM, treat spatial as "epistemic-like" and temporal as "aleatoric-like"
        # (This is an approximation; both have epistemic and aleatoric components)
        epistemic_fraction = var_spatial / var_total
        aleatoric_fraction = var_temporal / var_total

        return UncertaintyComponents(
            total=sigma_total,
            epistemic=spatial_std,  # Spatial uncertainty
            aleatoric=temporal_std,  # Temporal uncertainty
            epistemic_fraction=epistemic_fraction,
            aleatoric_fraction=aleatoric_fraction,
        )

    def decompose(
        self, model: any, X_test: npt.NDArray[np.float64], **kwargs: any
    ) -> UncertaintyComponents:
        """
        Generic decomposition dispatcher.

        Args:
            model: Trained probabilistic model
            X_test: Test locations
            **kwargs: Model-specific arguments

        Returns:
            UncertaintyComponents

        Raises:
            ValueError: If model_type is unknown
        """
        if self.model_type == "svgp":
            return self.decompose_svgp(model, X_test, **kwargs)
        elif self.model_type == "gam_ssm":
            return self.decompose_gam_ssm(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


def decompose_epistemic_aleatoric(
    predictions: npt.NDArray[np.float64],
    total_variance: npt.NDArray[np.float64],
    aleatoric_variance: npt.NDArray[np.float64],
) -> UncertaintyComponents:
    """
    Decompose uncertainty given total and aleatoric variances.

    Utility function for quick decomposition when variances are known.

    Args:
        predictions: Predictive means [N]
        total_variance: Total predictive variance [N]
        aleatoric_variance: Aleatoric (noise) variance [N]

    Returns:
        UncertaintyComponents
    """
    # Epistemic = Total - Aleatoric
    epistemic_variance = np.maximum(total_variance - aleatoric_variance, 0.0)

    sigma_total = np.sqrt(total_variance)
    sigma_epistemic = np.sqrt(epistemic_variance)
    sigma_aleatoric = np.sqrt(aleatoric_variance)

    epistemic_fraction = epistemic_variance / total_variance
    aleatoric_fraction = aleatoric_variance / total_variance

    return UncertaintyComponents(
        total=sigma_total,
        epistemic=sigma_epistemic,
        aleatoric=sigma_aleatoric,
        epistemic_fraction=epistemic_fraction,
        aleatoric_fraction=aleatoric_fraction,
    )


def test_independence_assumption(
    residuals_spatial: npt.NDArray[np.float64],
    residuals_temporal: npt.NDArray[np.float64],
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Test if spatial and temporal residuals are independent.

    Args:
        residuals_spatial: Spatial residuals (y - μ_spatial)
        residuals_temporal: Temporal residuals (y - μ_temporal)
        alpha: Significance level for hypothesis test

    Returns:
        Dictionary with correlation, p-value, and independence flag
    """
    from scipy.stats import pearsonr

    correlation, p_value = pearsonr(residuals_spatial, residuals_temporal)

    is_independent = abs(correlation) < 0.1  # Rule of thumb

    if not is_independent:
        logger.warning(
            f"Independence assumption violated: ρ={correlation:.3f}, p={p_value:.3e}"
        )

    return {
        "correlation": float(correlation),
        "p_value": float(p_value),
        "is_independent": bool(is_independent),
        "conclusion": "independent" if is_independent else "correlated",
    }


if __name__ == "__main__":
    # Example usage
    logger.info("Testing UncertaintyDecomposer")

    # Simulate data
    n_test = 100
    X_test = np.random.randn(n_test, 3)

    # Simulate decomposition
    total_var = np.random.gamma(2, 2, n_test)
    aleatoric_var = np.random.gamma(1, 1, n_test)

    components = decompose_epistemic_aleatoric(
        predictions=np.random.randn(n_test),
        total_variance=total_var,
        aleatoric_variance=aleatoric_var,
    )

    print(components)
    print("\nSummary Statistics:")
    for key, value in components.summary_stats().items():
        print(f"  {key}: {value:.4f}")
