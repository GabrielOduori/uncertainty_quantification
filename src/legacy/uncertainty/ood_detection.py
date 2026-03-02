"""
Out-of-Distribution (OOD) Detection Module

Provides tools for detecting when predictions are unreliable due to:
- Spatial extrapolation (far from training data)
- Temporal concept drift (model becoming stale)
- Domain shift
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from loguru import logger
from scipy.spatial import cKDTree
from shapely.geometry import Point
from shapely.ops import unary_union


class RiskLevel(Enum):
    """Risk levels for predictions."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class OODResult:
    """Result from OOD detection."""

    is_ood: bool
    ood_score: float  # >1.0 indicates OOD
    risk_level: RiskLevel
    warnings: List[str]
    details: Dict[str, any]

    def __repr__(self) -> str:
        return (
            f"OODResult(\n"
            f"  OOD: {self.is_ood}\n"
            f"  Score: {self.ood_score:.3f}\n"
            f"  Risk: {self.risk_level.value}\n"
            f"  Warnings: {len(self.warnings)}\n"
            f")"
        )


class SpatialOODDetector:
    """
    Detect spatial out-of-distribution points.

    Flags test points that are too far from training data based on:
    - Distance to nearest training point (normalized by lengthscale)
    - Spatial domain of applicability
    """

    def __init__(
        self,
        X_train: npt.NDArray[np.float64],
        lengthscales: npt.NDArray[np.float64],
        threshold: float = 2.5,
    ):
        """
        Initialize spatial OOD detector.

        Args:
            X_train: Training locations [N x D] (e.g., lat, lon)
            lengthscales: Kernel lengthscales [D] from trained model
            threshold: Number of lengthscales for OOD flagging (default: 2.5)
        """
        self.X_train = X_train
        self.lengthscales = lengthscales
        self.threshold = threshold

        # Build spatial index for fast nearest-neighbor queries
        X_train_normalized = X_train / lengthscales
        self.tree = cKDTree(X_train_normalized)

        # Create domain of applicability polygon
        self._create_reliable_zone()

        logger.info(
            f"Initialized SpatialOODDetector with {len(X_train)} training points, "
            f"threshold={threshold} lengthscales"
        )

    def _create_reliable_zone(self) -> None:
        """Create polygon representing reliable prediction zone."""
        try:
            # Buffer distance in same units as X_train
            buffer_distance = self.threshold * np.mean(self.lengthscales)

            # Create circles around each training point
            circles = [Point(x).buffer(buffer_distance) for x in self.X_train]

            # Union of all circles
            self.reliable_zone = unary_union(circles)

            area_km2 = self.reliable_zone.area * 111**2  # Approx deg² to km²
            logger.debug(f"Reliable zone area: {area_km2:.1f} km²")

        except Exception as e:
            logger.warning(f"Could not create spatial polygon: {e}")
            self.reliable_zone = None

    def detect(
        self, X_test: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
        """
        Detect OOD points in test set.

        Args:
            X_test: Test locations [M x D]

        Returns:
            ood_flags: Boolean array [M] indicating OOD points
            ood_scores: Continuous OOD scores [M] (>1.0 means OOD)
        """
        # Normalize by lengthscales
        X_test_normalized = X_test / self.lengthscales

        # Find distance to nearest training point
        distances, nearest_indices = self.tree.query(X_test_normalized)

        # OOD score in lengthscale units
        ood_scores = distances / self.threshold

        # Binary OOD flags
        ood_flags = ood_scores > 1.0

        logger.info(
            f"Detected {ood_flags.sum()} / {len(X_test)} OOD points "
            f"({ood_flags.mean():.1%})"
        )

        return ood_flags, ood_scores

    def adjust_uncertainty(
        self,
        sigma_base: npt.NDArray[np.float64],
        ood_scores: npt.NDArray[np.float64],
        inflation_rate: float = 1.0,
    ) -> npt.NDArray[np.float64]:
        """
        Inflate uncertainty for OOD predictions.

        Args:
            sigma_base: Base uncertainty estimates [N]
            ood_scores: OOD scores from detect() [N]
            inflation_rate: Aggressiveness of inflation (default: 1.0)

        Returns:
            sigma_adjusted: Inflated uncertainties [N]
        """
        # Exponential inflation for OOD points
        inflation_factor = 1.0 + inflation_rate * np.maximum(0, ood_scores - 1.0)

        sigma_adjusted = sigma_base * inflation_factor

        logger.debug(
            f"Uncertainty inflation: "
            f"mean factor = {np.mean(inflation_factor):.2f}, "
            f"max factor = {np.max(inflation_factor):.2f}"
        )

        return sigma_adjusted


class TemporalDriftDetector:
    """
    Detect temporal concept drift (model becoming stale).

    Monitors prediction errors over time to identify when the model
    is no longer valid for current conditions.
    """

    def __init__(self, window_size: int = 24, drift_threshold: float = 1.5):
        """
        Initialize drift detector.

        Args:
            window_size: Number of recent predictions for moving average (e.g., 24 hours)
            drift_threshold: Multiplier over baseline error to flag drift
        """
        self.window_size = window_size
        self.threshold = drift_threshold

        self.errors: List[float] = []
        self.timestamps: List[any] = []
        self.baseline_error: Optional[float] = None
        self.drift_detected = False

        logger.info(
            f"Initialized TemporalDriftDetector "
            f"(window={window_size}, threshold={drift_threshold})"
        )

    def update(
        self, prediction: float, actual: float, timestamp: any
    ) -> bool:
        """
        Update drift detector with new prediction-actual pair.

        Args:
            prediction: Model prediction
            actual: True observed value
            timestamp: Timestamp of observation

        Returns:
            True if drift detected
        """
        error = abs(prediction - actual)

        self.errors.append(error)
        self.timestamps.append(timestamp)

        # Establish baseline from first window
        if len(self.errors) == self.window_size:
            self.baseline_error = float(np.median(self.errors[: self.window_size]))
            logger.info(f"Baseline error established: {self.baseline_error:.3f}")

        # Detect drift if enough data
        if len(self.errors) > self.window_size and self.baseline_error is not None:
            recent_errors = self.errors[-self.window_size :]
            moving_avg = np.mean(recent_errors)

            if moving_avg > self.threshold * self.baseline_error:
                if not self.drift_detected:
                    logger.warning(
                        f"DRIFT DETECTED: moving avg error = {moving_avg:.3f}, "
                        f"baseline = {self.baseline_error:.3f}"
                    )
                self.drift_detected = True
                return True

        return False

    def get_drift_score(self) -> float:
        """
        Get current drift score (>1.0 indicates drift).

        Returns:
            Drift score
        """
        if len(self.errors) < self.window_size or self.baseline_error is None:
            return 0.0

        recent_errors = self.errors[-self.window_size :]
        moving_avg = np.mean(recent_errors)

        drift_score = moving_avg / self.baseline_error

        return float(drift_score)

    def reset(self) -> None:
        """Reset detector (e.g., after model retraining)."""
        self.errors = []
        self.timestamps = []
        self.baseline_error = None
        self.drift_detected = False
        logger.info("Drift detector reset")


class OODWarningSystem:
    """
    Integrated OOD warning system combining spatial and temporal detection.

    Provides comprehensive risk assessment for predictions.
    """

    def __init__(
        self,
        X_train: npt.NDArray[np.float64],
        lengthscales: npt.NDArray[np.float64],
        spatial_threshold: float = 2.5,
        temporal_window: int = 24,
    ):
        """
        Initialize integrated OOD system.

        Args:
            X_train: Training locations for spatial OOD
            lengthscales: Kernel lengthscales
            spatial_threshold: Spatial OOD threshold
            temporal_window: Temporal drift window size
        """
        self.spatial_detector = SpatialOODDetector(
            X_train, lengthscales, threshold=spatial_threshold
        )
        self.drift_detector = TemporalDriftDetector(window_size=temporal_window)

        logger.info("Initialized integrated OODWarningSystem")

    def evaluate(
        self,
        X_test: npt.NDArray[np.float64],
        prediction: float,
        actual: Optional[float] = None,
        timestamp: Optional[any] = None,
    ) -> OODResult:
        """
        Evaluate if prediction is trustworthy.

        Args:
            X_test: Test location [D]
            prediction: Model prediction
            actual: True value (if available for drift detection)
            timestamp: Timestamp (if available)

        Returns:
            OODResult with comprehensive assessment
        """
        warnings = []
        details = {}
        risk_level = RiskLevel.LOW

        # Spatial OOD check
        ood_flags, ood_scores = self.spatial_detector.detect(X_test.reshape(1, -1))
        spatial_ood = ood_flags[0]
        spatial_score = ood_scores[0]

        details["spatial_ood_score"] = float(spatial_score)

        if spatial_ood:
            warnings.append(
                f"SPATIAL OOD: Prediction {spatial_score:.2f}× threshold "
                f"from training data"
            )
            risk_level = RiskLevel.MEDIUM

        # Temporal drift check (if ground truth available)
        if actual is not None and timestamp is not None:
            drift_detected = self.drift_detector.update(prediction, actual, timestamp)
            drift_score = self.drift_detector.get_drift_score()

            details["temporal_drift_score"] = float(drift_score)

            if drift_detected:
                warnings.append(
                    f"TEMPORAL DRIFT: Model may be stale (drift score = {drift_score:.2f})"
                )
                if risk_level == RiskLevel.MEDIUM:
                    risk_level = RiskLevel.HIGH
                else:
                    risk_level = RiskLevel.MEDIUM

        # Combined OOD score
        ood_score = spatial_score
        if "temporal_drift_score" in details:
            ood_score = max(spatial_score, details["temporal_drift_score"])

        # Determine overall risk
        if ood_score > 3.0:
            risk_level = RiskLevel.CRITICAL
        elif ood_score > 2.0 and len(warnings) > 1:
            risk_level = RiskLevel.HIGH

        is_ood = len(warnings) > 0

        return OODResult(
            is_ood=is_ood,
            ood_score=ood_score,
            risk_level=risk_level,
            warnings=warnings,
            details=details,
        )

    def generate_report(self) -> Dict[str, any]:
        """
        Generate comprehensive OOD detection report.

        Returns:
            Dictionary with system status and statistics
        """
        report = {
            "spatial": {
                "n_training_points": len(self.spatial_detector.X_train),
                "threshold_lengthscales": self.spatial_detector.threshold,
            },
            "temporal": {
                "window_size": self.drift_detector.window_size,
                "baseline_error": self.drift_detector.baseline_error,
                "current_drift_score": self.drift_detector.get_drift_score(),
                "drift_detected": self.drift_detector.drift_detected,
                "n_observations": len(self.drift_detector.errors),
            },
        }

        if self.spatial_detector.reliable_zone is not None:
            report["spatial"]["reliable_area_km2"] = float(
                self.spatial_detector.reliable_zone.area * 111**2
            )

        return report


if __name__ == "__main__":
    # Example usage
    logger.info("Testing OOD Detection")

    # Simulate training data (2D spatial)
    np.random.seed(42)
    X_train = np.random.randn(100, 2) * 0.1 + [34.05, -118.25]  # LA Basin
    lengthscales = np.array([0.05, 0.05])  # ~5km

    # Initialize OOD system
    ood_system = OODWarningSystem(
        X_train=X_train, lengthscales=lengthscales, spatial_threshold=2.5
    )

    # Test on in-distribution point
    X_in = np.array([[34.05, -118.25]])  # Center of training data
    result_in = ood_system.evaluate(X_in, prediction=35.0)
    print("In-distribution test:")
    print(result_in)

    # Test on OOD point
    X_out = np.array([[34.5, -117.5]])  # Far from training data
    result_out = ood_system.evaluate(X_out, prediction=40.0)
    print("\nOut-of-distribution test:")
    print(result_out)

    # Generate report
    report = ood_system.generate_report()
    print("\nSystem Report:")
    import json

    print(json.dumps(report, indent=2))
