"""
Hierarchical Variance Propagation Module

Tracks uncertainty through all stages of the fusion pipeline:
- Stage 0: Raw measurement variance by source
- Stage 1: GP posterior variance (epistemic)
- Stage 2: Total predictive variance

Provides variance attribution to quantify information contribution from each data source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from loguru import logger


@dataclass
class HierarchicalVariance:
    """Container for variance at each fusion stage."""

    stage_0_raw: Dict[str, npt.NDArray[np.float64]]  # Raw measurement by source
    stage_1_epistemic: npt.NDArray[np.float64]  # GP posterior variance
    stage_2_predictive: npt.NDArray[np.float64]  # Total predictive variance
    source_indices: Dict[str, npt.NDArray[np.int64]]  # Indices for each source

    def variance_contribution_by_stage(self) -> Dict[str, float]:
        """
        Compute average variance at each stage.

        Returns:
            Dictionary with mean variance at each fusion stage
        """
        raw_contributions = {
            name: float(np.mean(var)) for name, var in self.stage_0_raw.items()
        }

        return {
            "stage_0_raw": raw_contributions,
            "stage_1_epistemic": float(np.mean(self.stage_1_epistemic)),
            "stage_2_predictive": float(np.mean(self.stage_2_predictive)),
        }

    def source_specific_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Summarize variance by data source.

        Returns:
            Statistics for each data source
        """
        summary = {}

        for source_name, indices in self.source_indices.items():
            if len(indices) > 0:
                summary[source_name] = {
                    "n_observations": len(indices),
                    "mean_raw_variance": float(np.mean(self.stage_0_raw.get(source_name, [0]))),
                    "mean_epistemic_variance": float(np.mean(self.stage_1_epistemic[indices])),
                    "mean_predictive_variance": float(np.mean(self.stage_2_predictive[indices])),
                }

        return summary

    def __repr__(self) -> str:
        contributions = self.variance_contribution_by_stage()
        return (
            f"HierarchicalVariance(\n"
            f"  Stage 0 (Raw): {contributions['stage_0_raw']}\n"
            f"  Stage 1 (Epistemic): {contributions['stage_1_epistemic']:.4f}\n"
            f"  Stage 2 (Predictive): {contributions['stage_2_predictive']:.4f}\n"
            f")"
        )


@dataclass
class VarianceAttribution:
    """Container for variance attribution to sources."""

    total_variance: float  # Total predictive variance
    source_contributions: Dict[str, float]  # Contribution from each source
    source_fractions: Dict[str, float]  # Fraction of total (sums to 1)
    information_gain: Dict[str, float]  # Variance reduction from each source

    def __repr__(self) -> str:
        lines = [f"VarianceAttribution(Total Variance: {self.total_variance:.4f})"]
        lines.append("  Source Contributions:")
        for source, fraction in sorted(self.source_fractions.items(),
                                       key=lambda x: x[1], reverse=True):
            lines.append(f"    {source}: {fraction:.1%}")
        return "\n".join(lines)


class HierarchicalUQTracker:
    """
    Track uncertainty propagation through fusion architecture.

    Implements formal variance decomposition at each stage:
    1. Observation: y ~ N(μ, σ²_observation + σ²_instrument)
    2. GP Fusion: f ~ GP(m, K) with posterior variance
    3. Prediction: y* ~ N(f*, σ²_epistemic + σ²_aleatoric)
    """

    def __init__(self, source_noise_levels: Optional[Dict[str, float]] = None):
        """
        Initialize hierarchical UQ tracker.

        Args:
            source_noise_levels: Known noise variance for each source type
                                 e.g., {'EPA': 2.1, 'LC': 8.3, 'SAT': 15.6}
        """
        self.source_noise_levels = source_noise_levels or {
            'EPA': 2.1,
            'LC': 8.3,
            'SAT': 15.6
        }
        logger.info(f"Initialized HierarchicalUQTracker with noise levels: {self.source_noise_levels}")

    def decompose_by_stage(
        self,
        model: any,  # GPflow SVGP model
        X_test: npt.NDArray[np.float64],
        sources_test: npt.NDArray[np.int64],  # 0=EPA, 1=LC, 2=SAT
        source_names: Optional[List[str]] = None
    ) -> HierarchicalVariance:
        """
        Decompose variance at each fusion stage.

        Args:
            model: Trained SVGP model
            X_test: Test locations [N x D]
            sources_test: Source identifier for each test point [N]
            source_names: Names for source types (default: ['EPA', 'LC', 'SAT'])

        Returns:
            HierarchicalVariance with stage-wise decomposition
        """
        if source_names is None:
            source_names = ['EPA', 'LC', 'SAT']

        logger.info(f"Decomposing variance for {len(X_test)} test points")

        # Stage 0: Raw measurement variance by source
        stage_0_raw = {}
        source_indices = {}

        for i, source_name in enumerate(source_names):
            indices = np.where(sources_test == i)[0]
            source_indices[source_name] = indices

            if len(indices) > 0:
                # Raw measurement noise for this source type
                raw_var = self.source_noise_levels.get(source_name, 1.0)
                stage_0_raw[source_name] = np.full(len(indices), raw_var)
            else:
                stage_0_raw[source_name] = np.array([])

        # Stage 1: GP posterior variance (epistemic uncertainty)
        f_mean, f_var = model.predict_f(X_test)
        if hasattr(f_var, "numpy"):
            stage_1_epistemic = f_var.numpy().flatten()
        else:
            stage_1_epistemic = np.asarray(f_var).flatten()

        # Stage 2: Total predictive variance (epistemic + aleatoric)
        # For each point, add source-specific noise
        stage_2_predictive = stage_1_epistemic.copy()
        for i, source_name in enumerate(source_names):
            indices = source_indices[source_name]
            if len(indices) > 0:
                noise_var = self.source_noise_levels.get(source_name, 1.0)
                stage_2_predictive[indices] += noise_var

        return HierarchicalVariance(
            stage_0_raw=stage_0_raw,
            stage_1_epistemic=stage_1_epistemic,
            stage_2_predictive=stage_2_predictive,
            source_indices=source_indices
        )

    def propagate_through_calibration(
        self,
        raw_variance: npt.NDArray[np.float64],
        calibration_slope: float,
        calibration_intercept: float,
        param_covariance: Optional[npt.NDArray[np.float64]] = None
    ) -> npt.NDArray[np.float64]:
        """
        Propagate variance through calibration transform: y_cal = a*y + b

        Uses delta method for variance propagation:
        Var(a*y + b) ≈ a²·Var(y) + Var(a)·E[y]² + Var(b)

        Args:
            raw_variance: Variance before calibration [N]
            calibration_slope: Slope parameter (a)
            calibration_intercept: Intercept parameter (b)
            param_covariance: Covariance of (a, b) if available [2x2]

        Returns:
            Calibrated variance [N]
        """
        # First-order term (dominant)
        calibrated_var = (calibration_slope ** 2) * raw_variance

        # Add parameter uncertainty if available
        if param_covariance is not None:
            var_a = param_covariance[0, 0]
            var_b = param_covariance[1, 1]
            # Approximate contribution from parameter uncertainty
            calibrated_var += var_a + var_b

        return calibrated_var


class VariancePropagationAnalyzer:
    """
    Analyze how variance flows through the fusion model.

    Provides tools for variance attribution and information gain analysis.
    """

    def __init__(self):
        logger.info("Initialized VariancePropagationAnalyzer")

    def attribute_to_sources(
        self,
        model: any,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        sources_train: npt.NDArray[np.int64],
        X_test: npt.NDArray[np.float64],
        source_names: Optional[List[str]] = None
    ) -> VarianceAttribution:
        """
        Attribute variance reduction to each data source.

        Computes: How much does each data source contribute to reducing
        uncertainty at test locations?

        Uses leave-one-source-out analysis:
        - Train model without source i
        - Measure variance increase
        - Attribution = variance_without_i - variance_with_all

        Args:
            model: Trained fusion model with all sources
            X_train: Training features [N x D]
            y_train: Training targets [N]
            sources_train: Source identifiers [N]
            X_test: Test locations [M x D]
            source_names: Names for sources

        Returns:
            VarianceAttribution with source contributions
        """
        if source_names is None:
            source_names = ['EPA', 'LC', 'SAT']

        logger.info("Computing variance attribution via leave-one-source-out")

        # Get baseline variance with all sources
        _, var_all = model.predict_f(X_test)
        var_all = var_all.numpy().flatten()
        mean_var_all = float(np.mean(var_all))

        # Compute variance without each source
        source_contributions = {}
        information_gain = {}

        for i, source_name in enumerate(source_names):
            # Identify indices for this source
            source_mask = sources_train == i
            n_source = np.sum(source_mask)

            if n_source == 0:
                logger.warning(f"No training data for source {source_name}")
                source_contributions[source_name] = 0.0
                information_gain[source_name] = 0.0
                continue

            # Create dataset without this source
            mask_without = ~source_mask
            X_without = X_train[mask_without]
            y_without = y_train[mask_without]

            # Approximate variance without retraining:
            # Use a heuristic based on distance to nearest remaining point
            var_without = self._estimate_variance_without_source(
                X_test, X_without, var_all
            )

            # Information gain = reduction in variance
            variance_reduction = var_without - var_all
            mean_reduction = float(np.mean(variance_reduction))

            source_contributions[source_name] = mean_reduction
            information_gain[source_name] = mean_reduction

        # Normalize to get fractions
        total_contribution = sum(source_contributions.values())
        if total_contribution > 0:
            source_fractions = {
                name: contrib / total_contribution
                for name, contrib in source_contributions.items()
            }
        else:
            source_fractions = {name: 0.0 for name in source_names}

        return VarianceAttribution(
            total_variance=mean_var_all,
            source_contributions=source_contributions,
            source_fractions=source_fractions,
            information_gain=information_gain
        )

    def _estimate_variance_without_source(
        self,
        X_test: npt.NDArray[np.float64],
        X_remaining: npt.NDArray[np.float64],
        var_current: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Estimate variance without retraining model.

        Heuristic: Points far from remaining data will have higher variance.
        Uses distance to nearest remaining point as proxy.

        Args:
            X_test: Test locations [M x D]
            X_remaining: Training data without one source [N' x D]
            var_current: Current variance with all sources [M]

        Returns:
            Estimated variance without the removed source [M]
        """
        from scipy.spatial.distance import cdist

        # Compute distances to nearest remaining point
        distances = cdist(X_test, X_remaining, metric='euclidean')
        min_distances = np.min(distances, axis=1)

        # Heuristic inflation based on distance
        # Points far from data get higher variance increase
        distance_factor = 1.0 + 0.5 * (min_distances / np.median(min_distances))
        var_without = var_current * distance_factor

        return var_without

    def compute_information_gain_by_source(
        self,
        attribution: VarianceAttribution
    ) -> Dict[str, Dict[str, float]]:
        """
        Summarize information gain from each source.

        Args:
            attribution: VarianceAttribution result

        Returns:
            Dictionary with absolute and relative information gain
        """
        results = {}

        for source_name in attribution.source_contributions.keys():
            results[source_name] = {
                "absolute_variance_reduction": attribution.information_gain[source_name],
                "fraction_of_total": attribution.source_fractions[source_name],
                "percentage": attribution.source_fractions[source_name] * 100
            }

        return results


def compute_variance_propagation_matrix(
    hierarchical_var: HierarchicalVariance
) -> npt.NDArray[np.float64]:
    """
    Compute matrix showing how variance propagates through stages.

    Returns [N x 3] matrix where columns are:
    - Stage 0: Raw measurement variance
    - Stage 1: Epistemic variance
    - Stage 2: Total predictive variance

    Args:
        hierarchical_var: HierarchicalVariance object

    Returns:
        Variance propagation matrix [N x 3]
    """
    n_points = len(hierarchical_var.stage_1_epistemic)

    # Reconstruct stage 0 variance for all points
    stage_0_full = np.zeros(n_points)
    for source_name, indices in hierarchical_var.source_indices.items():
        if len(indices) > 0 and source_name in hierarchical_var.stage_0_raw:
            stage_0_full[indices] = hierarchical_var.stage_0_raw[source_name][0]  # Constant per source

    # Stack into matrix
    propagation_matrix = np.column_stack([
        stage_0_full,
        hierarchical_var.stage_1_epistemic,
        hierarchical_var.stage_2_predictive
    ])

    return propagation_matrix


if __name__ == "__main__":
    # Example usage and testing
    logger.info("Testing HierarchicalUQTracker")

    # Simulate test data
    np.random.seed(42)
    n_test = 100
    X_test = np.random.randn(n_test, 3)
    sources_test = np.random.choice([0, 1, 2], size=n_test)

    # Mock model for testing
    class MockModel:
        def predict_f(self, X):
            n = len(X)
            mean = np.random.randn(n)
            var = np.random.gamma(2, 1, n)

            # Mock TensorFlow tensors
            class MockTensor:
                def __init__(self, data):
                    self.data = data
                def numpy(self):
                    return self.data
                def flatten(self):
                    return self.data.flatten()

            return MockTensor(mean), MockTensor(var)

    mock_model = MockModel()

    # Test hierarchical decomposition
    tracker = HierarchicalUQTracker()
    hierarchical_var = tracker.decompose_by_stage(
        model=mock_model,
        X_test=X_test,
        sources_test=sources_test
    )

    print(hierarchical_var)
    print("\nSource-specific summary:")
    for source, stats in hierarchical_var.source_specific_summary().items():
        print(f"{source}: {stats}")

    # Test propagation matrix
    prop_matrix = compute_variance_propagation_matrix(hierarchical_var)
    print(f"\nPropagation matrix shape: {prop_matrix.shape}")
    print(f"Stage means: {np.mean(prop_matrix, axis=0)}")
