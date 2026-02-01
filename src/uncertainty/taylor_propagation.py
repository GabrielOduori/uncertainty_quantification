"""
Taylor Series Uncertainty Propagation Module

Based on Johns Hopkins UQ Course, Module 3, Lesson 2:
"The Taylor series expansion allows us to analytically estimate the moments
of a function of random variables."

This module implements:
1. First-order Taylor expansion (Delta method / Linear error propagation)
2. Second-order Taylor expansion (includes curvature effects)
3. General nonlinear uncertainty propagation
4. Gradient and Hessian computation (analytical and numerical)

Mathematical Foundation:
-----------------------
For a function Y = g(X) where X is a random vector with mean μ and covariance Σ:

FIRST-ORDER APPROXIMATION:
    g(X) ≈ g(μ) + ∇g(μ)ᵀ(X - μ)

    E[Y] ≈ g(μ)
    Var[Y] ≈ ∇g(μ)ᵀ Σ ∇g(μ)

    For scalar input: Var[Y] ≈ (dg/dx)² σ²

SECOND-ORDER APPROXIMATION:
    g(X) ≈ g(μ) + ∇g(μ)ᵀ(X - μ) + ½(X - μ)ᵀH(X - μ)

    E[Y] ≈ g(μ) + ½ tr(HΣ)
    Var[Y] ≈ ∇g(μ)ᵀ Σ ∇g(μ) + ½ tr((HΣ)²)

    Where H = ∂²g/∂xᵢ∂xⱼ is the Hessian matrix.

References:
- Ku, H.H. (1966). Notes on the use of propagation of error formulas. NIST.
- JCGM 100:2008. Guide to the expression of uncertainty in measurement (GUM).
"""

import numpy as np
from typing import Callable, Tuple, Optional, Union, List
from dataclasses import dataclass
from scipy.optimize import approx_fprime


@dataclass
class PropagationResult:
    """Results from Taylor series uncertainty propagation."""

    # Mean of output
    mean: float

    # Variance of output
    variance: float

    # Standard deviation
    std: float

    # Gradient at mean (sensitivity coefficients)
    gradient: np.ndarray

    # Hessian at mean (if second-order)
    hessian: Optional[np.ndarray] = None

    # Order of approximation used
    order: int = 1

    # Individual variance contributions from each input
    variance_contributions: Optional[np.ndarray] = None

    # Input names for reference
    input_names: Optional[List[str]] = None

    @property
    def sensitivity_coefficients(self) -> np.ndarray:
        """Squared gradient components (variance weights)."""
        return self.gradient ** 2

    def contribution_summary(self) -> str:
        """Summarize variance contributions from each input."""
        if self.variance_contributions is None:
            return "Variance contributions not computed."

        lines = ["Variance Contribution Analysis", "=" * 40]
        total = np.sum(self.variance_contributions)

        names = self.input_names or [f"X{i}" for i in range(len(self.gradient))]

        for i, (name, contrib) in enumerate(zip(names, self.variance_contributions)):
            pct = 100 * contrib / total if total > 0 else 0
            lines.append(f"{name}: {contrib:.6f} ({pct:.1f}%)")

        lines.append(f"\nTotal variance: {total:.6f}")
        lines.append(f"Standard deviation: {np.sqrt(total):.6f}")

        return "\n".join(lines)


class TaylorPropagator:
    """
    Propagate uncertainty through a function using Taylor series expansion.

    This is THE classic method for analytical uncertainty propagation,
    used in metrology, physics, and engineering for decades.

    The "GUM" (Guide to the expression of Uncertainty in Measurement)
    is based on this method.

    Example
    -------
    >>> # Function: f(x, y) = x * y
    >>> def f(X):
    ...     return X[0] * X[1]
    >>>
    >>> # Input means and uncertainties
    >>> mu = np.array([10.0, 5.0])  # x=10, y=5
    >>> sigma = np.array([0.5, 0.2])  # σ_x=0.5, σ_y=0.2
    >>>
    >>> propagator = TaylorPropagator(f)
    >>> result = propagator.propagate(mu, sigma)
    >>>
    >>> print(f"f(x,y) = {result.mean:.2f} ± {result.std:.2f}")
    >>> # f(x,y) = 50.00 ± 3.20
    """

    def __init__(
        self,
        func: Callable,
        gradient_func: Optional[Callable] = None,
        hessian_func: Optional[Callable] = None,
        input_names: Optional[List[str]] = None
    ):
        """
        Initialize Taylor propagator.

        Parameters
        ----------
        func : Callable
            Function g(X) -> Y where X is array-like
        gradient_func : Callable, optional
            Analytical gradient ∇g(X). If None, computed numerically.
        hessian_func : Callable, optional
            Analytical Hessian H(X). If None, computed numerically.
        input_names : List[str], optional
            Names for input variables
        """
        self.func = func
        self.gradient_func = gradient_func
        self.hessian_func = hessian_func
        self.input_names = input_names

    def propagate(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        correlation: Optional[np.ndarray] = None,
        order: int = 1
    ) -> PropagationResult:
        """
        Propagate uncertainty through the function.

        Parameters
        ----------
        mean : np.ndarray
            Mean values of input variables μ = [μ₁, μ₂, ..., μₙ]
        std : np.ndarray
            Standard deviations of inputs σ = [σ₁, σ₂, ..., σₙ]
        correlation : np.ndarray, optional
            Correlation matrix (n × n). If None, assumes independent inputs.
        order : int
            1 for first-order (linear), 2 for second-order

        Returns
        -------
        PropagationResult
            Propagated mean, variance, and analysis
        """
        mean = np.atleast_1d(mean).astype(float)
        std = np.atleast_1d(std).astype(float)
        n = len(mean)

        # Build covariance matrix
        if correlation is None:
            # Independent inputs: diagonal covariance
            cov = np.diag(std ** 2)
        else:
            # Correlated inputs
            cov = np.outer(std, std) * correlation

        # Evaluate function at mean
        f_mean = float(self.func(mean))

        # Compute gradient
        gradient = self._compute_gradient(mean)

        if order == 1:
            # First-order approximation
            return self._first_order(f_mean, gradient, cov, mean, std)
        elif order == 2:
            # Second-order approximation
            hessian = self._compute_hessian(mean)
            return self._second_order(f_mean, gradient, hessian, cov, mean, std)
        else:
            raise ValueError(f"Order must be 1 or 2, got {order}")

    def _first_order(
        self,
        f_mean: float,
        gradient: np.ndarray,
        cov: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray
    ) -> PropagationResult:
        """
        First-order Taylor expansion (Delta method).

        E[Y] ≈ g(μ)
        Var[Y] ≈ ∇g(μ)ᵀ Σ ∇g(μ)
        """
        # Output mean (just the function value at input mean)
        output_mean = f_mean

        # Output variance: ∇gᵀ Σ ∇g
        output_variance = gradient @ cov @ gradient

        # Individual contributions (for independent inputs)
        variance_contributions = (gradient ** 2) * (np.diag(cov))

        return PropagationResult(
            mean=output_mean,
            variance=output_variance,
            std=np.sqrt(output_variance),
            gradient=gradient,
            order=1,
            variance_contributions=variance_contributions,
            input_names=self.input_names
        )

    def _second_order(
        self,
        f_mean: float,
        gradient: np.ndarray,
        hessian: np.ndarray,
        cov: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray
    ) -> PropagationResult:
        """
        Second-order Taylor expansion.

        E[Y] ≈ g(μ) + ½ tr(HΣ)
        Var[Y] ≈ ∇g(μ)ᵀ Σ ∇g(μ) + ½ tr((HΣ)²)
        """
        # Mean correction: add ½ tr(HΣ)
        mean_correction = 0.5 * np.trace(hessian @ cov)
        output_mean = f_mean + mean_correction

        # First-order variance term
        var_first_order = gradient @ cov @ gradient

        # Second-order variance correction: ½ tr((HΣ)²)
        H_cov = hessian @ cov
        var_correction = 0.5 * np.trace(H_cov @ H_cov)

        output_variance = var_first_order + var_correction

        # Individual contributions (approximate)
        variance_contributions = (gradient ** 2) * (np.diag(cov))

        return PropagationResult(
            mean=output_mean,
            variance=output_variance,
            std=np.sqrt(max(0, output_variance)),
            gradient=gradient,
            hessian=hessian,
            order=2,
            variance_contributions=variance_contributions,
            input_names=self.input_names
        )

    def _compute_gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient at point x."""
        if self.gradient_func is not None:
            return np.atleast_1d(self.gradient_func(x))
        else:
            # Numerical gradient using finite differences
            eps = np.sqrt(np.finfo(float).eps)
            return approx_fprime(x, self.func, eps)

    def _compute_hessian(self, x: np.ndarray) -> np.ndarray:
        """Compute Hessian at point x."""
        if self.hessian_func is not None:
            return np.atleast_2d(self.hessian_func(x))
        else:
            # Numerical Hessian using finite differences
            n = len(x)
            hessian = np.zeros((n, n))
            eps = np.cbrt(np.finfo(float).eps)

            f0 = self.func(x)

            for i in range(n):
                for j in range(i, n):
                    x_pp = x.copy()
                    x_pm = x.copy()
                    x_mp = x.copy()
                    x_mm = x.copy()

                    x_pp[i] += eps
                    x_pp[j] += eps

                    x_pm[i] += eps
                    x_pm[j] -= eps

                    x_mp[i] -= eps
                    x_mp[j] += eps

                    x_mm[i] -= eps
                    x_mm[j] -= eps

                    hessian[i, j] = (
                        self.func(x_pp) - self.func(x_pm) -
                        self.func(x_mp) + self.func(x_mm)
                    ) / (4 * eps * eps)

                    hessian[j, i] = hessian[i, j]

            return hessian


# =============================================================================
# Common Uncertainty Propagation Formulas (Analytical)
# =============================================================================

class CommonFormulas:
    """
    Analytical formulas for common operations.

    These are the formulas you'll find in physics/engineering textbooks
    for "error propagation" or "propagation of uncertainties."

    All formulas assume INDEPENDENT inputs unless otherwise noted.
    """

    @staticmethod
    def addition(
        means: List[float],
        stds: List[float],
        coefficients: Optional[List[float]] = None
    ) -> Tuple[float, float]:
        """
        Y = a₁X₁ + a₂X₂ + ... + aₙXₙ

        E[Y] = a₁μ₁ + a₂μ₂ + ... + aₙμₙ
        Var[Y] = a₁²σ₁² + a₂²σ₂² + ... + aₙ²σₙ²

        (Variances add in quadrature for sums)
        """
        means = np.array(means)
        stds = np.array(stds)

        if coefficients is None:
            coefficients = np.ones(len(means))
        else:
            coefficients = np.array(coefficients)

        mean_y = np.sum(coefficients * means)
        var_y = np.sum((coefficients ** 2) * (stds ** 2))

        return mean_y, np.sqrt(var_y)

    @staticmethod
    def multiplication(
        means: List[float],
        stds: List[float]
    ) -> Tuple[float, float]:
        """
        Y = X₁ × X₂ × ... × Xₙ

        E[Y] ≈ μ₁ × μ₂ × ... × μₙ

        For relative uncertainties (CV = σ/μ):
        (σᵧ/μᵧ)² ≈ (σ₁/μ₁)² + (σ₂/μ₂)² + ...

        (Relative uncertainties add in quadrature for products)
        """
        means = np.array(means)
        stds = np.array(stds)

        mean_y = np.prod(means)

        # Relative uncertainties
        rel_vars = (stds / means) ** 2
        rel_var_y = np.sum(rel_vars)

        std_y = np.abs(mean_y) * np.sqrt(rel_var_y)

        return mean_y, std_y

    @staticmethod
    def division(
        mean_num: float, std_num: float,
        mean_den: float, std_den: float
    ) -> Tuple[float, float]:
        """
        Y = X₁ / X₂

        E[Y] ≈ μ₁ / μ₂
        (σᵧ/μᵧ)² ≈ (σ₁/μ₁)² + (σ₂/μ₂)²
        """
        mean_y = mean_num / mean_den

        rel_var = (std_num / mean_num) ** 2 + (std_den / mean_den) ** 2
        std_y = np.abs(mean_y) * np.sqrt(rel_var)

        return mean_y, std_y

    @staticmethod
    def power(mean_x: float, std_x: float, n: float) -> Tuple[float, float]:
        """
        Y = X^n

        E[Y] ≈ μₓⁿ
        σᵧ ≈ |n| × μₓⁿ⁻¹ × σₓ = |n| × (σₓ/μₓ) × μₓⁿ
        """
        mean_y = mean_x ** n
        std_y = np.abs(n) * (std_x / mean_x) * np.abs(mean_y)

        return mean_y, std_y

    @staticmethod
    def logarithm(mean_x: float, std_x: float, base: str = 'natural') -> Tuple[float, float]:
        """
        Y = ln(X) or log₁₀(X)

        For ln(X):
            E[Y] ≈ ln(μₓ)
            σᵧ ≈ σₓ / μₓ

        For log₁₀(X):
            E[Y] ≈ log₁₀(μₓ)
            σᵧ ≈ σₓ / (μₓ × ln(10))
        """
        if base == 'natural':
            mean_y = np.log(mean_x)
            std_y = std_x / mean_x
        else:  # base 10
            mean_y = np.log10(mean_x)
            std_y = std_x / (mean_x * np.log(10))

        return mean_y, std_y

    @staticmethod
    def exponential(mean_x: float, std_x: float) -> Tuple[float, float]:
        """
        Y = exp(X)

        E[Y] ≈ exp(μₓ) × (1 + σₓ²/2)  [second-order]
        E[Y] ≈ exp(μₓ)                 [first-order]
        σᵧ ≈ exp(μₓ) × σₓ
        """
        mean_y = np.exp(mean_x)
        std_y = mean_y * std_x

        return mean_y, std_y

    @staticmethod
    def trigonometric(
        mean_x: float,
        std_x: float,
        func: str = 'sin'
    ) -> Tuple[float, float]:
        """
        Y = sin(X), cos(X), or tan(X)

        Uses dY/dX evaluated at μₓ
        """
        if func == 'sin':
            mean_y = np.sin(mean_x)
            derivative = np.cos(mean_x)
        elif func == 'cos':
            mean_y = np.cos(mean_x)
            derivative = -np.sin(mean_x)
        elif func == 'tan':
            mean_y = np.tan(mean_x)
            derivative = 1 / np.cos(mean_x) ** 2
        else:
            raise ValueError(f"Unknown function: {func}")

        std_y = np.abs(derivative) * std_x

        return mean_y, std_y


# =============================================================================
# GP-Specific Taylor Propagation
# =============================================================================

class GPTaylorPropagator:
    """
    Taylor series propagation specifically for Gaussian Process predictions.

    When a GP gives you f(x) ~ N(μ(x), σ²(x)), and you apply a transformation
    g(f), this class propagates uncertainty through g.

    Common use cases:
    - Log-transformed GP: f ~ GP, then exp(f) for positive outputs
    - Threshold exceedance: P(f > threshold)
    - Nonlinear air quality indices
    """

    def __init__(self, gp_model):
        """
        Initialize GP Taylor propagator.

        Parameters
        ----------
        gp_model : object
            GP model with predict_f(X) -> (mean, var) method
        """
        self.gp_model = gp_model

    def propagate_transformation(
        self,
        X: np.ndarray,
        transform: Callable,
        transform_derivative: Optional[Callable] = None,
        transform_second_derivative: Optional[Callable] = None,
        order: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate GP uncertainty through a transformation.

        If f(x) ~ N(μ, σ²) and Y = g(f), then:
        - First-order: E[Y] ≈ g(μ), Var[Y] ≈ g'(μ)² σ²
        - Second-order: E[Y] ≈ g(μ) + ½g''(μ)σ², Var[Y] ≈ g'(μ)² σ² + ½g''(μ)² σ⁴

        Parameters
        ----------
        X : np.ndarray
            Input locations
        transform : Callable
            Transformation g(f)
        transform_derivative : Callable, optional
            g'(f). If None, computed numerically.
        transform_second_derivative : Callable, optional
            g''(f). If None, computed numerically.
        order : int
            1 or 2

        Returns
        -------
        mean_Y : np.ndarray
            Transformed mean
        var_Y : np.ndarray
            Transformed variance
        """
        # Get GP predictions
        mean_f, var_f = self.gp_model.predict_f(X)
        mean_f = np.atleast_1d(mean_f).flatten()
        var_f = np.atleast_1d(var_f).flatten()
        std_f = np.sqrt(var_f)

        # Compute derivatives
        if transform_derivative is None:
            # Numerical derivative
            eps = 1e-6
            g_prime = (transform(mean_f + eps) - transform(mean_f - eps)) / (2 * eps)
        else:
            g_prime = transform_derivative(mean_f)

        if order == 1:
            # First-order
            mean_Y = transform(mean_f)
            var_Y = g_prime ** 2 * var_f

        else:
            # Second-order
            if transform_second_derivative is None:
                eps = 1e-5
                g_double_prime = (
                    transform(mean_f + eps) - 2 * transform(mean_f) + transform(mean_f - eps)
                ) / (eps ** 2)
            else:
                g_double_prime = transform_second_derivative(mean_f)

            mean_Y = transform(mean_f) + 0.5 * g_double_prime * var_f
            var_Y = g_prime ** 2 * var_f + 0.5 * g_double_prime ** 2 * var_f ** 2

        return mean_Y, np.maximum(var_Y, 0)

    def propagate_exp(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate through exponential: Y = exp(f).

        Common for log-transformed GPs to ensure positive outputs.

        Exact for log-normal:
        E[Y] = exp(μ + σ²/2)
        Var[Y] = exp(2μ + σ²)(exp(σ²) - 1)
        """
        mean_f, var_f = self.gp_model.predict_f(X)
        mean_f = np.atleast_1d(mean_f).flatten()
        var_f = np.atleast_1d(var_f).flatten()

        # Log-normal formulas (exact)
        mean_Y = np.exp(mean_f + var_f / 2)
        var_Y = np.exp(2 * mean_f + var_f) * (np.exp(var_f) - 1)

        return mean_Y, var_Y

    def exceedance_probability(
        self,
        X: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """
        Compute P(f(x) > threshold) for each x.

        Since f(x) ~ N(μ(x), σ²(x)):
        P(f > t) = 1 - Φ((t - μ) / σ)

        where Φ is the standard normal CDF.

        Parameters
        ----------
        X : np.ndarray
            Input locations
        threshold : float
            Threshold value (e.g., 35.5 for PM2.5 EPA standard)

        Returns
        -------
        prob : np.ndarray
            Exceedance probability at each location
        """
        from scipy.stats import norm

        mean_f, var_f = self.gp_model.predict_f(X)
        mean_f = np.atleast_1d(mean_f).flatten()
        std_f = np.sqrt(np.atleast_1d(var_f).flatten())

        # Standardize
        z = (threshold - mean_f) / std_f

        # P(f > threshold) = 1 - P(f < threshold) = 1 - Φ(z)
        prob = 1 - norm.cdf(z)

        return prob


# =============================================================================
# Convenience Functions
# =============================================================================

def propagate_uncertainty(
    func: Callable,
    means: Union[float, np.ndarray],
    stds: Union[float, np.ndarray],
    order: int = 1,
    correlation: Optional[np.ndarray] = None
) -> PropagationResult:
    """
    Quick function to propagate uncertainty through any function.

    Parameters
    ----------
    func : Callable
        Function Y = f(X)
    means : array-like
        Mean values of inputs
    stds : array-like
        Standard deviations of inputs
    order : int
        1 for first-order, 2 for second-order
    correlation : np.ndarray, optional
        Correlation matrix

    Returns
    -------
    PropagationResult
        Propagated uncertainty

    Example
    -------
    >>> # Area of rectangle: A = length × width
    >>> def area(X):
    ...     return X[0] * X[1]
    >>>
    >>> result = propagate_uncertainty(
    ...     area,
    ...     means=[10.0, 5.0],      # length=10, width=5
    ...     stds=[0.5, 0.2]         # uncertainties
    ... )
    >>> print(f"Area = {result.mean:.1f} ± {result.std:.2f}")
    Area = 50.0 ± 3.20
    """
    propagator = TaylorPropagator(func)
    return propagator.propagate(
        np.atleast_1d(means),
        np.atleast_1d(stds),
        correlation=correlation,
        order=order
    )


def combine_uncertainties(
    stds: List[float],
    operation: str = 'add',
    means: Optional[List[float]] = None
) -> float:
    """
    Combine uncertainties for common operations.

    Parameters
    ----------
    stds : List[float]
        Standard deviations to combine
    operation : str
        'add' (sum), 'multiply' (product), or 'quadrature'
    means : List[float], optional
        Mean values (required for multiplication)

    Returns
    -------
    float
        Combined uncertainty

    Example
    -------
    >>> # Sum of two measurements
    >>> combined = combine_uncertainties([0.5, 0.3], operation='add')
    >>> print(f"Combined uncertainty: {combined:.3f}")
    Combined uncertainty: 0.583
    """
    stds = np.array(stds)

    if operation == 'add' or operation == 'quadrature':
        # Root sum of squares
        return np.sqrt(np.sum(stds ** 2))

    elif operation == 'multiply':
        if means is None:
            raise ValueError("Means required for multiplication")
        means = np.array(means)
        # Relative uncertainties add in quadrature
        rel_vars = (stds / means) ** 2
        rel_std = np.sqrt(np.sum(rel_vars))
        return np.abs(np.prod(means)) * rel_std

    else:
        raise ValueError(f"Unknown operation: {operation}")


# =============================================================================
# Example Usage and Verification
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Taylor Series Uncertainty Propagation Examples")
    print("=" * 60)

    # Example 1: Simple product
    print("\n1. Product: Y = X₁ × X₂")
    print("-" * 40)

    def product(X):
        return X[0] * X[1]

    result = propagate_uncertainty(
        product,
        means=[10.0, 5.0],
        stds=[0.5, 0.2]
    )

    print(f"   X₁ = 10.0 ± 0.5")
    print(f"   X₂ = 5.0 ± 0.2")
    print(f"   Y = {result.mean:.2f} ± {result.std:.3f}")
    print(f"   Gradient: {result.gradient}")
    print(result.contribution_summary())

    # Verify with analytical formula
    analytical_mean, analytical_std = CommonFormulas.multiplication(
        [10.0, 5.0], [0.5, 0.2]
    )
    print(f"\n   Analytical: Y = {analytical_mean:.2f} ± {analytical_std:.3f}")

    # Example 2: Nonlinear function
    print("\n\n2. Nonlinear: Y = X₁² + sin(X₂)")
    print("-" * 40)

    def nonlinear(X):
        return X[0]**2 + np.sin(X[1])

    # First-order
    result1 = propagate_uncertainty(
        nonlinear,
        means=[2.0, np.pi/4],
        stds=[0.1, 0.05],
        order=1
    )

    # Second-order
    result2 = propagate_uncertainty(
        nonlinear,
        means=[2.0, np.pi/4],
        stds=[0.1, 0.05],
        order=2
    )

    print(f"   X₁ = 2.0 ± 0.1")
    print(f"   X₂ = π/4 ± 0.05")
    print(f"   First-order:  Y = {result1.mean:.4f} ± {result1.std:.4f}")
    print(f"   Second-order: Y = {result2.mean:.4f} ± {result2.std:.4f}")

    # Example 3: Air quality index calculation
    print("\n\n3. Air Quality Application: AQI from PM2.5")
    print("-" * 40)

    def pm25_to_aqi(pm25):
        """Simplified AQI calculation for PM2.5."""
        # Simplified linear interpolation for demonstration
        if isinstance(pm25, np.ndarray):
            pm25 = pm25[0]
        if pm25 <= 12.0:
            return 50 * pm25 / 12.0
        elif pm25 <= 35.4:
            return 50 + 50 * (pm25 - 12.0) / (35.4 - 12.0)
        else:
            return 100 + 100 * (pm25 - 35.4) / (55.4 - 35.4)

    # PM2.5 measurement with uncertainty
    pm25_mean = 25.0  # μg/m³
    pm25_std = 3.0    # measurement uncertainty

    result = propagate_uncertainty(
        pm25_to_aqi,
        means=[pm25_mean],
        stds=[pm25_std]
    )

    print(f"   PM2.5 = {pm25_mean} ± {pm25_std} μg/m³")
    print(f"   AQI = {result.mean:.1f} ± {result.std:.1f}")

    print("\n" + "=" * 60)
    print("Taylor series propagation complete!")
    print("=" * 60)
