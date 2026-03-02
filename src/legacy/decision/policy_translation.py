"""
Policy Translation Module

Translates uncertainty quantification outputs into actionable decisions for:
- Health advisories and alerts
- Sensor network design and optimization
- Regulatory compliance monitoring
- Public communication

Addresses research objective: "provide interpretable and actionable outputs
for decision-making and policy"
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from scipy.stats import norm


class AlertLevel(Enum):
    """Air quality alert levels based on EPA AQI."""
    GOOD = "Good"
    MODERATE = "Moderate"
    UNHEALTHY_SENSITIVE = "Unhealthy for Sensitive Groups"
    UNHEALTHY = "Unhealthy"
    VERY_UNHEALTHY = "Very Unhealthy"
    HAZARDOUS = "Hazardous"


class CertaintyLevel(Enum):
    """Certainty level for exceedance probabilities."""
    CERTAIN = "Certain"  # P > 90%
    LIKELY = "Likely"  # 70% < P ≤ 90%
    POSSIBLE = "Possible"  # 30% < P ≤ 70%
    UNLIKELY = "Unlikely"  # P ≤ 30%


@dataclass
class ExceedanceProbability:
    """Container for exceedance probability with uncertainty."""

    location_id: int
    threshold: float
    threshold_name: str  # e.g., "Unhealthy"
    probability: float  # P(PM2.5 > threshold)
    certainty: CertaintyLevel
    mean_prediction: float
    std_prediction: float


@dataclass
class HealthAlert:
    """Health alert with uncertainty-aware messaging."""

    location_id: int
    alert_level: AlertLevel
    certainty: CertaintyLevel
    probability: float
    mean_pm25: float
    uncertainty_pm25: float
    message: str
    recommended_actions: List[str]


@dataclass
class SensorPlacementRecommendation:
    """Recommendation for new sensor placement."""

    location: npt.NDArray[np.float64]  # [lat, lon]
    expected_variance_reduction: float
    information_gain: float
    priority_rank: int
    rationale: str


class PolicyTranslator:
    """
    Translate UQ outputs to policy-relevant decisions.

    Provides tools for:
    1. Computing exceedance probabilities for health thresholds
    2. Generating uncertainty-aware health alerts
    3. Identifying high-value monitoring locations
    4. Creating interpretable uncertainty visualizations
    """

    # EPA PM2.5 thresholds (μg/m³)
    THRESHOLDS = {
        "Good": 12.0,
        "Moderate": 35.4,
        "Unhealthy for Sensitive Groups": 55.4,
        "Unhealthy": 150.4,
        "Very Unhealthy": 250.4,
        "Hazardous": 500.4,
    }

    def __init__(self):
        """Initialize policy translator."""
        logger.info("Initialized PolicyTranslator")

    def compute_exceedance_probabilities(
        self,
        predictions: npt.NDArray[np.float64],
        uncertainties: npt.NDArray[np.float64],
        threshold: float = 35.4,
        threshold_name: str = "Moderate"
    ) -> List[ExceedanceProbability]:
        """
        Compute probability of exceeding threshold.

        P(PM2.5 > threshold) accounting for prediction uncertainty.

        Args:
            predictions: Mean predictions [N]
            uncertainties: Prediction standard deviations [N]
            threshold: PM2.5 threshold value
            threshold_name: Name of threshold level

        Returns:
            List of ExceedanceProbability objects
        """
        logger.info(f"Computing exceedance probabilities for threshold={threshold}")

        # Compute z-scores
        z = (threshold - predictions) / (uncertainties + 1e-10)
        probabilities = 1 - norm.cdf(z)

        # Create exceedance objects
        results = []
        for i, (pred, std, prob) in enumerate(zip(predictions, uncertainties, probabilities)):
            # Determine certainty level
            if prob > 0.9:
                certainty = CertaintyLevel.CERTAIN
            elif prob > 0.7:
                certainty = CertaintyLevel.LIKELY
            elif prob > 0.3:
                certainty = CertaintyLevel.POSSIBLE
            else:
                certainty = CertaintyLevel.UNLIKELY

            results.append(ExceedanceProbability(
                location_id=i,
                threshold=threshold,
                threshold_name=threshold_name,
                probability=float(prob),
                certainty=certainty,
                mean_prediction=float(pred),
                std_prediction=float(std)
            ))

        return results

    def generate_health_alerts(
        self,
        predictions: npt.NDArray[np.float64],
        uncertainties: npt.NDArray[np.float64],
        alert_threshold_prob: float = 0.3
    ) -> List[HealthAlert]:
        """
        Generate uncertainty-aware health alerts.

        Only generates alerts when exceedance probability is significant.

        Args:
            predictions: Mean PM2.5 predictions [N]
            uncertainties: Prediction uncertainties [N]
            alert_threshold_prob: Minimum probability to issue alert

        Returns:
            List of HealthAlert objects
        """
        logger.info("Generating health alerts")

        alerts = []

        for i, (mean, std) in enumerate(zip(predictions, uncertainties)):
            # Check each threshold level
            for level_name, threshold in self.THRESHOLDS.items():
                if level_name == "Good":
                    continue  # No alert for good air quality

                # Compute exceedance probability
                z = (threshold - mean) / (std + 1e-10)
                prob = float(1 - norm.cdf(z))

                # Only alert if probability exceeds threshold
                if prob < alert_threshold_prob:
                    continue

                # Determine certainty
                if prob > 0.9:
                    certainty = CertaintyLevel.CERTAIN
                elif prob > 0.7:
                    certainty = CertaintyLevel.LIKELY
                elif prob > 0.3:
                    certainty = CertaintyLevel.POSSIBLE
                else:
                    continue

                # Map to AlertLevel
                try:
                    alert_level = AlertLevel(level_name)
                except ValueError:
                    continue

                # Create message
                message = self._create_alert_message(
                    level_name, certainty, prob, mean, std
                )

                # Recommended actions
                actions = self._get_recommended_actions(alert_level, certainty)

                alerts.append(HealthAlert(
                    location_id=i,
                    alert_level=alert_level,
                    certainty=certainty,
                    probability=prob,
                    mean_pm25=float(mean),
                    uncertainty_pm25=float(std),
                    message=message,
                    recommended_actions=actions
                ))

                # Only issue one alert per location (highest level)
                break

        return alerts

    def _create_alert_message(
        self,
        level_name: str,
        certainty: CertaintyLevel,
        probability: float,
        mean: float,
        std: float
    ) -> str:
        """Create human-readable alert message."""
        certainty_phrases = {
            CertaintyLevel.CERTAIN: "is very likely",
            CertaintyLevel.LIKELY: "is likely",
            CertaintyLevel.POSSIBLE: "may be",
        }

        phrase = certainty_phrases.get(certainty, "may be")

        message = (
            f"Air quality {phrase} {level_name.lower()} "
            f"({probability:.0%} probability). "
            f"Predicted PM2.5: {mean:.1f} ± {std:.1f} μg/m³."
        )

        return message

    def _get_recommended_actions(
        self,
        alert_level: AlertLevel,
        certainty: CertaintyLevel
    ) -> List[str]:
        """Get recommended actions based on alert level and certainty."""

        base_actions = {
            AlertLevel.MODERATE: [
                "Unusually sensitive people should consider reducing prolonged outdoor exertion."
            ],
            AlertLevel.UNHEALTHY_SENSITIVE: [
                "People with respiratory or heart disease, elderly, and children should limit prolonged outdoor exertion.",
                "Everyone else should limit prolonged outdoor exertion."
            ],
            AlertLevel.UNHEALTHY: [
                "Everyone should avoid prolonged outdoor exertion.",
                "People with respiratory or heart disease, elderly, and children should avoid all outdoor exertion."
            ],
            AlertLevel.VERY_UNHEALTHY: [
                "Everyone should avoid all outdoor exertion.",
                "People with respiratory or heart disease should remain indoors."
            ],
            AlertLevel.HAZARDOUS: [
                "Everyone should remain indoors with windows closed.",
                "Use air purifiers if available.",
                "Seek medical attention if experiencing symptoms."
            ],
        }

        actions = base_actions.get(alert_level, [])

        # Add uncertainty-specific guidance
        if certainty == CertaintyLevel.POSSIBLE:
            actions.append("Monitor air quality updates closely as conditions are uncertain.")

        return actions

    def identify_high_value_sensor_locations(
        self,
        X_candidate: npt.NDArray[np.float64],
        current_variance: npt.NDArray[np.float64],
        top_n: int = 10,
        variance_reduction_threshold: float = 0.15
    ) -> List[SensorPlacementRecommendation]:
        """
        Identify optimal locations for new sensor deployment.

        Prioritizes locations where:
        1. Current uncertainty is high
        2. New sensor would provide most information gain
        3. Strategic coverage of undermonitored areas

        Args:
            X_candidate: Candidate locations [M x D]
            current_variance: Variance at each candidate location [M]
            top_n: Number of top recommendations to return
            variance_reduction_threshold: Minimum reduction to recommend (fraction)

        Returns:
            List of SensorPlacementRecommendation objects, ranked by priority
        """
        logger.info(f"Identifying top {top_n} sensor placement locations")

        # Compute information gain (approximation based on current variance)
        # Higher variance = more valuable to add sensor
        information_gain = current_variance

        # Expected variance reduction (heuristic)
        # Assume new sensor reduces variance by 50% in local area
        expected_reduction = 0.5 * current_variance

        # Filter by threshold
        meets_threshold = expected_reduction > variance_reduction_threshold

        # Rank by information gain
        rankings = np.argsort(information_gain)[::-1]

        recommendations = []
        rank = 1

        for idx in rankings:
            if rank > top_n:
                break

            if not meets_threshold[idx]:
                continue

            # Create rationale
            if current_variance[idx] > np.percentile(current_variance, 75):
                rationale = "High uncertainty area - critical coverage gap"
            elif current_variance[idx] > np.percentile(current_variance, 50):
                rationale = "Moderate uncertainty - would improve spatial resolution"
            else:
                rationale = "Strategic location for network optimization"

            recommendations.append(SensorPlacementRecommendation(
                location=X_candidate[idx],
                expected_variance_reduction=float(expected_reduction[idx]),
                information_gain=float(information_gain[idx]),
                priority_rank=rank,
                rationale=rationale
            ))

            rank += 1

        return recommendations

    def create_decision_summary_report(
        self,
        predictions: npt.NDArray[np.float64],
        uncertainties: npt.NDArray[np.float64],
        locations: Optional[npt.NDArray[np.float64]] = None
    ) -> pd.DataFrame:
        """
        Create comprehensive decision support report.

        Args:
            predictions: Mean predictions [N]
            uncertainties: Prediction uncertainties [N]
            locations: Spatial locations [N x 2] (optional)

        Returns:
            DataFrame with decision-relevant summaries
        """
        n = len(predictions)

        # Compute exceedances for key thresholds
        moderate_exceed = self.compute_exceedance_probabilities(
            predictions, uncertainties,
            threshold=35.4, threshold_name="Moderate"
        )
        unhealthy_exceed = self.compute_exceedance_probabilities(
            predictions, uncertainties,
            threshold=55.4, threshold_name="Unhealthy for Sensitive"
        )

        # Build dataframe
        data = {
            "location_id": np.arange(n),
            "mean_pm25": predictions,
            "std_pm25": uncertainties,
            "lower_95ci": predictions - 1.96 * uncertainties,
            "upper_95ci": predictions + 1.96 * uncertainties,
            "prob_exceed_moderate": [ex.probability for ex in moderate_exceed],
            "prob_exceed_unhealthy": [ex.probability for ex in unhealthy_exceed],
            "certainty_moderate": [ex.certainty.value for ex in moderate_exceed],
            "certainty_unhealthy": [ex.certainty.value for ex in unhealthy_exceed],
        }

        if locations is not None:
            data["latitude"] = locations[:, 0]
            data["longitude"] = locations[:, 1]

        df = pd.DataFrame(data)

        return df

    def format_uncertainty_for_public(
        self,
        mean: float,
        std: float,
        confidence_level: float = 0.95
    ) -> str:
        """
        Format uncertainty in plain language for public communication.

        Args:
            mean: Mean prediction
            std: Standard deviation
            confidence_level: Confidence level for interval

        Returns:
            Plain-language uncertainty statement
        """
        z = norm.ppf((1 + confidence_level) / 2)
        lower = mean - z * std
        upper = mean + z * std

        # Categorize uncertainty
        cv = std / (mean + 1e-10)  # Coefficient of variation

        if cv < 0.15:
            certainty_phrase = "with high confidence"
        elif cv < 0.30:
            certainty_phrase = "with moderate confidence"
        else:
            certainty_phrase = "with considerable uncertainty"

        statement = (
            f"PM2.5 is predicted to be {mean:.1f} μg/m³ {certainty_phrase}. "
            f"The true value is {confidence_level:.0%} likely to be between "
            f"{lower:.1f} and {upper:.1f} μg/m³."
        )

        return statement


if __name__ == "__main__":
    # Example usage and testing
    logger.info("Testing PolicyTranslator")

    # Simulate predictions
    np.random.seed(42)
    n_locations = 50

    predictions = np.random.gamma(shape=5, scale=10, size=n_locations)  # PM2.5 values
    uncertainties = np.random.gamma(shape=2, scale=3, size=n_locations)  # Uncertainties
    locations = np.random.randn(n_locations, 2) * 0.1 + [34.05, -118.25]  # LA Basin

    translator = PolicyTranslator()

    # Test exceedance probabilities
    exceedances = translator.compute_exceedance_probabilities(
        predictions, uncertainties, threshold=35.4, threshold_name="Moderate"
    )

    print(f"Computed {len(exceedances)} exceedance probabilities")
    print(f"Example: Location 0 - P(exceed) = {exceedances[0].probability:.2%}, "
          f"Certainty = {exceedances[0].certainty.value}")

    # Test health alerts
    alerts = translator.generate_health_alerts(predictions, uncertainties)
    print(f"\nGenerated {len(alerts)} health alerts")
    if len(alerts) > 0:
        print(f"Example alert: {alerts[0].message}")
        print(f"Actions: {alerts[0].recommended_actions}")

    # Test sensor placement
    sensor_recs = translator.identify_high_value_sensor_locations(
        locations, uncertainties**2, top_n=5
    )
    print(f"\nTop {len(sensor_recs)} sensor placement recommendations:")
    for rec in sensor_recs:
        print(f"  Rank {rec.priority_rank}: Location {rec.location} - {rec.rationale}")

    # Test decision report
    report = translator.create_decision_summary_report(
        predictions, uncertainties, locations
    )
    print(f"\nDecision summary report shape: {report.shape}")
    print(report.head())

    # Test public communication
    public_msg = translator.format_uncertainty_for_public(mean=45.2, std=8.3)
    print(f"\nPublic communication example:\n{public_msg}")
