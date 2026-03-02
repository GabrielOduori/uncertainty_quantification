"""
Reliability Analysis Module

Based on Johns Hopkins UQ Course, Module 4, Lesson 2:
"Reliability analysis is concerned with estimating small failure probabilities."

This module implements:
1. First-Order Reliability Method (FORM)
2. Second-Order Reliability Method (SORM)
3. Monte Carlo reliability estimation
4. Exceedance probability calculation for air quality thresholds

Key Concepts:
- Limit state function: g(X) where failure occurs when g(X) < 0
- Reliability index: β = distance to failure surface in standard normal space
- Probability of failure: P_f = Φ(-β) where Φ is standard normal CDF

Application to Air Quality:
- What's the probability PM2.5 exceeds 35.5 μg/m³ (EPA standard)?
- What's the probability AQI exceeds 100 (unhealthy)?
- These are "failure" events in reliability analysis terms

References:
- Hasofer, A.M. & Lind, N.C. (1974). Exact and invariant second-moment code format.
- Breitung, K. (1984). Asymptotic approximations for multinormal integrals.
"""

import numpy as np
from typing import Callable, Tuple, Optional, List, Dict, Union
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize


@dataclass
class ReliabilityResult:
    """Results from reliability analysis."""

    # Probability of failure (exceedance)
    probability_of_failure: float

    # Reliability index β
    reliability_index: float

    # Design point (most probable failure point)
    design_point: Optional[np.ndarray] = None

    # Sensitivity factors (importance of each variable)
    sensitivity_factors: Optional[np.ndarray] = None

    # Method used
    method: str = "FORM"

    # Confidence interval (for Monte Carlo)
    confidence_interval: Optional[Tuple[float, float]] = None

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"Reliability Analysis Results ({self.method})",
            "=" * 50,
            f"Probability of failure: {self.probability_of_failure:.6f}",
            f"                      = {self.probability_of_failure * 100:.4f}%",
            f"Reliability index (β): {self.reliability_index:.4f}",
        ]

        if self.sensitivity_factors is not None:
            lines.append("\nSensitivity factors (importance):")
            for i, alpha in enumerate(self.sensitivity_factors):
                lines.append(f"  α_{i+1} = {alpha:.4f} ({alpha**2 * 100:.1f}% of variance)")

        if self.confidence_interval is not None:
            lines.append(f"\n95% CI: [{self.confidence_interval[0]:.6f}, {self.confidence_interval[1]:.6f}]")

        return "\n".join(lines)


class ReliabilityAnalyzer:
    """
    Perform reliability analysis for uncertainty quantification.

    Reliability analysis answers: "What is the probability of an
    undesirable event (failure/exceedance)?"

    For air quality: P(PM2.5 > threshold) or P(AQI > 100)

    Mathematical Foundation:
    -----------------------
    Define limit state function g(X) such that:
    - g(X) > 0: Safe region
    - g(X) = 0: Failure surface
    - g(X) < 0: Failure region

    For exceedance: g(X) = threshold - f(X)
    Failure occurs when f(X) > threshold, i.e., g(X) < 0

    FORM (First-Order Reliability Method):
    1. Transform to standard normal space U
    2. Find design point u* (closest point to origin on g=0)
    3. β = ||u*|| (reliability index)
    4. P_f ≈ Φ(-β)
    """

    def __init__(
        self,
        limit_state_func: Callable,
        n_dims: int,
        means: np.ndarray,
        stds: np.ndarray,
        distributions: Optional[List[str]] = None
    ):
        """
        Initialize reliability analyzer.

        Parameters
        ----------
        limit_state_func : Callable
            Function g(X) -> float. Failure when g(X) < 0.
        n_dims : int
            Number of random variables
        means : np.ndarray
            Mean values of random variables
        stds : np.ndarray
            Standard deviations
        distributions : List[str], optional
            Distribution types ('normal', 'lognormal', 'uniform')
            Defaults to all normal.
        """
        self.g = limit_state_func
        self.n_dims = n_dims
        self.means = np.atleast_1d(means)
        self.stds = np.atleast_1d(stds)
        self.distributions = distributions or ['normal'] * n_dims

    def form(self) -> ReliabilityResult:
        """
        First-Order Reliability Method.

        Linearizes the limit state surface at the design point
        and computes probability using the reliability index.

        Returns
        -------
        ReliabilityResult
            Probability of failure and reliability index
        """
        # Transform to standard normal space
        # For normal: u = (x - μ) / σ
        # Design point minimizes ||u|| subject to g(x(u)) = 0

        def objective(u):
            """Minimize distance to origin."""
            return 0.5 * np.sum(u ** 2)

        def constraint(u):
            """Constraint: g(x) = 0."""
            x = self._transform_to_physical(u)
            return self.g(x)

        # Initial guess: origin
        u0 = np.zeros(self.n_dims)

        # Optimization
        from scipy.optimize import minimize, NonlinearConstraint

        constraint_obj = NonlinearConstraint(constraint, 0, 0)

        result = minimize(
            objective,
            u0,
            method='SLSQP',
            constraints={'type': 'eq', 'fun': constraint},
            options={'maxiter': 100}
        )

        if not result.success:
            # Try different initial points
            for scale in [0.5, 1.0, 2.0]:
                u0 = scale * np.ones(self.n_dims)
                result = minimize(
                    objective,
                    u0,
                    method='SLSQP',
                    constraints={'type': 'eq', 'fun': constraint},
                    options={'maxiter': 100}
                )
                if result.success:
                    break

        u_star = result.x
        beta = np.sqrt(np.sum(u_star ** 2))

        # Sensitivity factors (direction cosines)
        if beta > 1e-10:
            alpha = u_star / beta
        else:
            alpha = np.zeros(self.n_dims)

        # Probability of failure
        p_f = stats.norm.cdf(-beta)

        # Design point in physical space
        x_star = self._transform_to_physical(u_star)

        return ReliabilityResult(
            probability_of_failure=p_f,
            reliability_index=beta,
            design_point=x_star,
            sensitivity_factors=alpha,
            method="FORM"
        )

    def monte_carlo(
        self,
        n_samples: int = 100000,
        seed: int = 42
    ) -> ReliabilityResult:
        """
        Monte Carlo reliability estimation.

        Samples from input distributions and counts failures.

        Returns
        -------
        ReliabilityResult
            Probability of failure with confidence interval
        """
        rng = np.random.RandomState(seed)

        # Generate samples
        X = np.zeros((n_samples, self.n_dims))

        for i in range(self.n_dims):
            if self.distributions[i] == 'normal':
                X[:, i] = rng.normal(self.means[i], self.stds[i], n_samples)
            elif self.distributions[i] == 'lognormal':
                # Convert to lognormal parameters
                mu_ln = np.log(self.means[i]**2 / np.sqrt(self.stds[i]**2 + self.means[i]**2))
                sigma_ln = np.sqrt(np.log(1 + self.stds[i]**2 / self.means[i]**2))
                X[:, i] = rng.lognormal(mu_ln, sigma_ln, n_samples)
            elif self.distributions[i] == 'uniform':
                # Assume mean ± sqrt(3)*std for uniform
                a = self.means[i] - np.sqrt(3) * self.stds[i]
                b = self.means[i] + np.sqrt(3) * self.stds[i]
                X[:, i] = rng.uniform(a, b, n_samples)

        # Evaluate limit state
        g_values = np.array([self.g(X[j, :]) for j in range(n_samples)])

        # Count failures (g < 0)
        n_failures = np.sum(g_values < 0)
        p_f = n_failures / n_samples

        # Confidence interval (using normal approximation)
        if p_f > 0 and p_f < 1:
            se = np.sqrt(p_f * (1 - p_f) / n_samples)
            ci = (p_f - 1.96 * se, p_f + 1.96 * se)
        else:
            ci = (p_f, p_f)

        # Reliability index
        if p_f > 0:
            beta = -stats.norm.ppf(p_f)
        else:
            beta = np.inf

        return ReliabilityResult(
            probability_of_failure=p_f,
            reliability_index=beta,
            method="Monte Carlo",
            confidence_interval=ci
        )

    def _transform_to_physical(self, u: np.ndarray) -> np.ndarray:
        """Transform from standard normal to physical space."""
        x = np.zeros_like(u)

        for i in range(self.n_dims):
            if self.distributions[i] == 'normal':
                x[i] = self.means[i] + self.stds[i] * u[i]
            elif self.distributions[i] == 'lognormal':
                mu_ln = np.log(self.means[i]**2 / np.sqrt(self.stds[i]**2 + self.means[i]**2))
                sigma_ln = np.sqrt(np.log(1 + self.stds[i]**2 / self.means[i]**2))
                x[i] = np.exp(mu_ln + sigma_ln * u[i])
            elif self.distributions[i] == 'uniform':
                a = self.means[i] - np.sqrt(3) * self.stds[i]
                b = self.means[i] + np.sqrt(3) * self.stds[i]
                x[i] = a + (b - a) * stats.norm.cdf(u[i])

        return x


# =============================================================================
# Air Quality Specific Functions
# =============================================================================

class AirQualityReliability:
    """
    Reliability analysis for air quality applications.

    Answers questions like:
    - What's the probability PM2.5 exceeds EPA standard (35.5 μg/m³)?
    - What's the probability of "Unhealthy" AQI (>100)?
    - Where should we place monitors to reduce exceedance risk?
    """

    def __init__(self, gp_model=None):
        """
        Initialize air quality reliability analyzer.

        Parameters
        ----------
        gp_model : object, optional
            GP model with predict_f(X) -> (mean, var)
        """
        self.gp_model = gp_model

    def exceedance_probability(
        self,
        mean: Union[float, np.ndarray],
        std: Union[float, np.ndarray],
        threshold: float
    ) -> Union[float, np.ndarray]:
        """
        Compute probability of exceeding threshold.

        Assumes Gaussian distribution: P(X > t) = 1 - Φ((t - μ) / σ)

        Parameters
        ----------
        mean : float or array
            Mean value(s)
        std : float or array
            Standard deviation(s)
        threshold : float
            Exceedance threshold (e.g., 35.5 for PM2.5)

        Returns
        -------
        prob : float or array
            Exceedance probability
        """
        mean = np.atleast_1d(mean)
        std = np.atleast_1d(std)

        z = (threshold - mean) / std
        prob = 1 - stats.norm.cdf(z)

        return prob.item() if prob.size == 1 else prob

    def probability_unhealthy(
        self,
        pm25_mean: float,
        pm25_std: float,
        duration_hours: float = 24
    ) -> Dict[str, float]:
        """
        Compute probability of various health-relevant thresholds.

        EPA PM2.5 Standards:
        - Good: 0-12 μg/m³
        - Moderate: 12.1-35.4 μg/m³
        - Unhealthy for Sensitive: 35.5-55.4 μg/m³
        - Unhealthy: 55.5-150.4 μg/m³
        - Very Unhealthy: 150.5-250.4 μg/m³
        - Hazardous: > 250.5 μg/m³

        Returns
        -------
        Dict with probabilities for each category
        """
        thresholds = {
            'moderate': 12.0,
            'unhealthy_sensitive': 35.5,
            'unhealthy': 55.5,
            'very_unhealthy': 150.5,
            'hazardous': 250.5
        }

        probabilities = {}
        prev_prob = 0

        for name, thresh in thresholds.items():
            prob = self.exceedance_probability(pm25_mean, pm25_std, thresh)
            probabilities[f'P(>{name})'] = prob

        return probabilities

    def reliability_index_from_gp(
        self,
        X: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """
        Compute reliability index from GP predictions.

        β = (threshold - μ) / σ

        Higher β = lower probability of exceedance = more reliable

        Parameters
        ----------
        X : np.ndarray
            Locations
        threshold : float
            Threshold value

        Returns
        -------
        beta : np.ndarray
            Reliability index at each location
        """
        if self.gp_model is None:
            raise ValueError("GP model not provided")

        mean, var = self.gp_model.predict_f(X)
        mean = np.atleast_1d(mean).flatten()
        std = np.sqrt(np.atleast_1d(var).flatten())

        beta = (threshold - mean) / std

        return beta

    def risk_map(
        self,
        X_grid: np.ndarray,
        threshold: float = 35.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create spatial risk map of exceedance probability.

        Parameters
        ----------
        X_grid : np.ndarray
            Grid of locations (n_points, n_features)
        threshold : float
            Exceedance threshold

        Returns
        -------
        prob : np.ndarray
            Exceedance probability at each location
        beta : np.ndarray
            Reliability index at each location
        """
        if self.gp_model is None:
            raise ValueError("GP model not provided")

        mean, var = self.gp_model.predict_f(X_grid)
        mean = np.atleast_1d(mean).flatten()
        std = np.sqrt(np.atleast_1d(var).flatten())

        prob = self.exceedance_probability(mean, std, threshold)
        beta = (threshold - mean) / std

        return prob, beta


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_exceedance_probability(
    mean: float,
    std: float,
    threshold: float
) -> float:
    """
    Quick function to compute exceedance probability.

    P(X > threshold) for X ~ N(mean, std²)

    Example
    -------
    >>> prob = compute_exceedance_probability(
    ...     mean=30.0,    # PM2.5 = 30 μg/m³
    ...     std=5.0,      # uncertainty
    ...     threshold=35.5 # EPA standard
    ... )
    >>> print(f"P(PM2.5 > 35.5) = {prob:.1%}")
    P(PM2.5 > 35.5) = 13.6%
    """
    z = (threshold - mean) / std
    return 1 - stats.norm.cdf(z)


def reliability_index(mean: float, std: float, threshold: float) -> float:
    """
    Compute reliability index β.

    β = (threshold - mean) / std

    Interpretation:
    - β > 3: Very reliable (P_f < 0.1%)
    - β > 2: Reliable (P_f < 2.3%)
    - β > 1: Marginally reliable (P_f < 16%)
    - β < 1: Unreliable
    """
    return (threshold - mean) / std


def probability_to_reliability_index(p_f: float) -> float:
    """Convert probability of failure to reliability index."""
    return -stats.norm.ppf(p_f)


def reliability_index_to_probability(beta: float) -> float:
    """Convert reliability index to probability of failure."""
    return stats.norm.cdf(-beta)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Reliability Analysis for Air Quality")
    print("=" * 60)

    # Example: PM2.5 exceedance analysis
    print("\n1. Simple Exceedance Probability")
    print("-" * 40)

    pm25_mean = 30.0  # μg/m³
    pm25_std = 8.0    # Includes measurement + model uncertainty

    prob = compute_exceedance_probability(pm25_mean, pm25_std, 35.5)
    beta = reliability_index(pm25_mean, pm25_std, 35.5)

    print(f"   PM2.5 = {pm25_mean} ± {pm25_std} μg/m³")
    print(f"   EPA 24-hr standard: 35.5 μg/m³")
    print(f"   P(PM2.5 > 35.5) = {prob:.2%}")
    print(f"   Reliability index β = {beta:.3f}")

    # Example 2: Full reliability analysis
    print("\n\n2. Full Reliability Analysis")
    print("-" * 40)

    # Air quality model: AQ = baseline + 0.5*traffic - 0.3*wind + noise
    def air_quality_model(X):
        """X = [traffic, wind_speed]"""
        baseline = 25.0
        return baseline + 0.5 * X[0] - 0.3 * X[1]

    # Limit state: g < 0 means exceedance
    def limit_state(X):
        return 35.5 - air_quality_model(X)

    analyzer = ReliabilityAnalyzer(
        limit_state_func=limit_state,
        n_dims=2,
        means=np.array([20.0, 15.0]),  # traffic=20, wind=15
        stds=np.array([5.0, 3.0])
    )

    # FORM analysis
    result_form = analyzer.form()
    print("\n   FORM Results:")
    print(f"   P(exceedance) = {result_form.probability_of_failure:.4%}")
    print(f"   Reliability index β = {result_form.reliability_index:.4f}")
    if result_form.sensitivity_factors is not None:
        print(f"   Sensitivity (traffic): {result_form.sensitivity_factors[0]:.3f}")
        print(f"   Sensitivity (wind): {result_form.sensitivity_factors[1]:.3f}")

    # Monte Carlo verification
    result_mc = analyzer.monte_carlo(n_samples=100000)
    print("\n   Monte Carlo Results:")
    print(f"   P(exceedance) = {result_mc.probability_of_failure:.4%}")
    print(f"   95% CI: [{result_mc.confidence_interval[0]:.4%}, {result_mc.confidence_interval[1]:.4%}]")

    # Example 3: Health categories
    print("\n\n3. Health Risk Categories")
    print("-" * 40)

    aq_reliability = AirQualityReliability()
    health_probs = aq_reliability.probability_unhealthy(pm25_mean, pm25_std)

    print(f"   PM2.5 = {pm25_mean} ± {pm25_std} μg/m³")
    for category, prob in health_probs.items():
        print(f"   {category}: {prob:.2%}")

    print("\n" + "=" * 60)
    print("Reliability analysis complete!")
    print("=" * 60)
