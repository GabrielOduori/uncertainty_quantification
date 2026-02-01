"""
Global Sensitivity Analysis Module

Based on Johns Hopkins UQ Course, Module 4, Lesson 3:
"Global sensitivity analysis aims to identify which random variables make
the most significant contributions to uncertainty in the output of the model."

This module implements:
1. Sobol indices (variance-based sensitivity)
2. Morris method (screening)
3. Correlation-based sensitivity
4. Permutation feature importance

References:
- Saltelli, A. (2002). Making best use of model evaluations to compute sensitivity indices.
- Sobol, I.M. (2001). Global sensitivity indices for nonlinear mathematical models.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from scipy import stats


@dataclass
class SensitivityResults:
    """Results from global sensitivity analysis."""

    # First-order Sobol indices (main effects)
    first_order: np.ndarray  # S_i for each input

    # Total-order Sobol indices (includes interactions)
    total_order: np.ndarray  # S_Ti for each input

    # Feature names
    feature_names: List[str]

    # Confidence intervals (if computed)
    first_order_ci: Optional[np.ndarray] = None
    total_order_ci: Optional[np.ndarray] = None

    # Interaction indices (second-order)
    interaction_indices: Optional[np.ndarray] = None

    def get_ranking(self) -> List[Tuple[str, float]]:
        """Return features ranked by total-order sensitivity."""
        ranked = sorted(
            zip(self.feature_names, self.total_order),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked

    def get_important_features(self, threshold: float = 0.05) -> List[str]:
        """Return features with sensitivity above threshold."""
        return [
            name for name, si in zip(self.feature_names, self.total_order)
            if si >= threshold
        ]

    def summary(self) -> str:
        """Generate summary string."""
        lines = ["Global Sensitivity Analysis Results", "=" * 40]
        lines.append(f"\n{'Feature':<20} {'First-Order':<15} {'Total-Order':<15}")
        lines.append("-" * 50)

        for name, s1, st in zip(self.feature_names, self.first_order, self.total_order):
            lines.append(f"{name:<20} {s1:<15.4f} {st:<15.4f}")

        lines.append("-" * 50)
        lines.append(f"\nSum of first-order: {np.sum(self.first_order):.4f}")
        lines.append(f"(Should be ≤ 1.0, remainder is interactions)")

        lines.append(f"\nMost important feature: {self.feature_names[np.argmax(self.total_order)]}")
        lines.append(f"Least important feature: {self.feature_names[np.argmin(self.total_order)]}")

        return "\n".join(lines)


class GlobalSensitivityAnalyzer:
    """
    Perform global sensitivity analysis on a model.

    Identifies which input variables contribute most to output uncertainty.
    This is crucial for:
    - Understanding which measurements matter most
    - Prioritizing data collection efforts
    - Simplifying models by fixing unimportant inputs

    Mathematical Foundation (Sobol Indices):
    -----------------------------------------
    For a model Y = f(X₁, X₂, ..., Xₐ), the variance can be decomposed:

    Var[Y] = ∑ᵢ Vᵢ + ∑ᵢ<ⱼ Vᵢⱼ + ... + V₁₂...ₐ

    First-order Sobol index:
        Sᵢ = Vᵢ / Var[Y] = Var[E[Y|Xᵢ]] / Var[Y]

    Total-order Sobol index:
        Sᵀᵢ = E[Var[Y|X₋ᵢ]] / Var[Y]

    Where X₋ᵢ denotes all inputs except Xᵢ.

    Interpretation:
    - Sᵢ: Main effect of input i alone
    - Sᵀᵢ: Total effect including all interactions
    - Sᵀᵢ - Sᵢ: Effect due to interactions
    """

    def __init__(
        self,
        model: Callable,
        feature_names: Optional[List[str]] = None,
        n_samples: int = 1000,
        seed: int = 42
    ):
        """
        Initialize sensitivity analyzer.

        Parameters
        ----------
        model : Callable
            Function that takes X (n_samples, n_features) and returns predictions
        feature_names : List[str], optional
            Names for each input feature
        n_samples : int
            Number of Monte Carlo samples for estimation
        seed : int
            Random seed for reproducibility
        """
        self.model = model
        self.feature_names = feature_names
        self.n_samples = n_samples
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def analyze(
        self,
        bounds: List[Tuple[float, float]],
        method: str = 'sobol',
        calc_second_order: bool = False
    ) -> SensitivityResults:
        """
        Perform global sensitivity analysis.

        Parameters
        ----------
        bounds : List[Tuple[float, float]]
            Lower and upper bounds for each input: [(low1, high1), (low2, high2), ...]
        method : str
            'sobol' for Sobol indices, 'morris' for Morris screening
        calc_second_order : bool
            Whether to compute second-order (interaction) indices

        Returns
        -------
        SensitivityResults
            Sensitivity indices and rankings
        """
        d = len(bounds)

        if self.feature_names is None:
            self.feature_names = [f"X{i}" for i in range(d)]

        if method == 'sobol':
            return self._sobol_analysis(bounds, calc_second_order)
        elif method == 'morris':
            return self._morris_analysis(bounds)
        elif method == 'correlation':
            return self._correlation_analysis(bounds)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _sobol_analysis(
        self,
        bounds: List[Tuple[float, float]],
        calc_second_order: bool = False
    ) -> SensitivityResults:
        """
        Compute Sobol sensitivity indices using Saltelli's method.

        This implements the algorithm from:
        Saltelli, A. (2002). Making best use of model evaluations to compute
        sensitivity indices. Computer Physics Communications.
        """
        d = len(bounds)
        N = self.n_samples

        # Generate two independent sample matrices A and B
        # Each row is a sample point, each column is an input dimension
        A = self._generate_samples(bounds, N)
        B = self._generate_samples(bounds, N)

        # Evaluate model at A and B
        f_A = self._evaluate_model(A)
        f_B = self._evaluate_model(B)

        # Total variance
        f_all = np.concatenate([f_A, f_B])
        var_total = np.var(f_all)

        if var_total < 1e-10:
            # No variance in output
            return SensitivityResults(
                first_order=np.zeros(d),
                total_order=np.zeros(d),
                feature_names=self.feature_names
            )

        # Compute first-order and total-order indices
        first_order = np.zeros(d)
        total_order = np.zeros(d)

        for i in range(d):
            # Create AB_i: A with column i replaced by B's column i
            AB_i = A.copy()
            AB_i[:, i] = B[:, i]

            # Create BA_i: B with column i replaced by A's column i
            BA_i = B.copy()
            BA_i[:, i] = A[:, i]

            # Evaluate model
            f_AB_i = self._evaluate_model(AB_i)
            f_BA_i = self._evaluate_model(BA_i)

            # First-order index: S_i = V_i / V(Y)
            # V_i = Var[E[Y|X_i]] ≈ (1/N) * sum(f_B * (f_AB_i - f_A))
            first_order[i] = np.mean(f_B * (f_AB_i - f_A)) / var_total

            # Total-order index: S_Ti = E[V(Y|X_{-i})] / V(Y)
            # ≈ (1/2N) * sum((f_A - f_AB_i)^2)
            total_order[i] = 0.5 * np.mean((f_A - f_AB_i)**2) / var_total

        # Clip to valid range [0, 1]
        first_order = np.clip(first_order, 0, 1)
        total_order = np.clip(total_order, 0, 1)

        # Compute interaction indices if requested
        interaction_indices = None
        if calc_second_order:
            interaction_indices = self._compute_interactions(
                A, B, f_A, f_B, var_total, bounds
            )

        return SensitivityResults(
            first_order=first_order,
            total_order=total_order,
            feature_names=self.feature_names,
            interaction_indices=interaction_indices
        )

    def _morris_analysis(
        self,
        bounds: List[Tuple[float, float]],
        num_trajectories: int = 10,
        num_levels: int = 4
    ) -> SensitivityResults:
        """
        Morris method (Elementary Effects) for screening.

        Computationally cheaper than Sobol, good for initial screening.

        For each input i, compute:
        - μ*_i: Mean of absolute elementary effects (importance)
        - σ_i: Std of elementary effects (interactions/nonlinearity)
        """
        d = len(bounds)

        # Generate Morris trajectories
        delta = 1.0 / (num_levels - 1)

        elementary_effects = [[] for _ in range(d)]

        for _ in range(num_trajectories):
            # Start with random base point
            x_base = self.rng.randint(0, num_levels, size=d) / (num_levels - 1)

            # Scale to bounds
            x_current = np.array([
                bounds[i][0] + x_base[i] * (bounds[i][1] - bounds[i][0])
                for i in range(d)
            ])

            f_current = self._evaluate_model(x_current.reshape(1, -1))[0]

            # Random order of inputs
            order = self.rng.permutation(d)

            for i in order:
                # Perturb input i
                x_new = x_current.copy()

                # Determine direction
                if x_base[i] + delta <= 1:
                    x_base[i] += delta
                else:
                    x_base[i] -= delta

                x_new[i] = bounds[i][0] + x_base[i] * (bounds[i][1] - bounds[i][0])

                f_new = self._evaluate_model(x_new.reshape(1, -1))[0]

                # Elementary effect
                ee = (f_new - f_current) / (delta * (bounds[i][1] - bounds[i][0]))
                elementary_effects[i].append(ee)

                x_current = x_new
                f_current = f_new

        # Compute statistics
        mu_star = np.array([np.mean(np.abs(ee)) for ee in elementary_effects])
        sigma = np.array([np.std(ee) for ee in elementary_effects])

        # Normalize to get pseudo-indices
        total = np.sum(mu_star)
        if total > 0:
            first_order = mu_star / total
            total_order = (mu_star + sigma) / (total + np.sum(sigma))
        else:
            first_order = np.zeros(d)
            total_order = np.zeros(d)

        return SensitivityResults(
            first_order=first_order,
            total_order=total_order,
            feature_names=self.feature_names
        )

    def _correlation_analysis(
        self,
        bounds: List[Tuple[float, float]]
    ) -> SensitivityResults:
        """
        Simple correlation-based sensitivity analysis.

        Compute Pearson and Spearman correlation between each input and output.
        Fast but only captures linear/monotonic relationships.
        """
        d = len(bounds)

        # Generate samples
        X = self._generate_samples(bounds, self.n_samples)
        Y = self._evaluate_model(X)

        # Compute correlations
        pearson = np.zeros(d)
        spearman = np.zeros(d)

        for i in range(d):
            pearson[i] = np.abs(np.corrcoef(X[:, i], Y)[0, 1])
            spearman[i] = np.abs(stats.spearmanr(X[:, i], Y)[0])

        # Handle NaN
        pearson = np.nan_to_num(pearson)
        spearman = np.nan_to_num(spearman)

        # Normalize
        total_pearson = np.sum(pearson)
        total_spearman = np.sum(spearman)

        if total_pearson > 0:
            first_order = pearson / total_pearson
        else:
            first_order = np.zeros(d)

        if total_spearman > 0:
            total_order = spearman / total_spearman
        else:
            total_order = np.zeros(d)

        return SensitivityResults(
            first_order=first_order,
            total_order=total_order,
            feature_names=self.feature_names
        )

    def _generate_samples(
        self,
        bounds: List[Tuple[float, float]],
        n: int
    ) -> np.ndarray:
        """Generate uniform samples within bounds."""
        d = len(bounds)
        samples = np.zeros((n, d))

        for i, (low, high) in enumerate(bounds):
            samples[:, i] = self.rng.uniform(low, high, n)

        return samples

    def _evaluate_model(self, X: np.ndarray) -> np.ndarray:
        """Evaluate model at given points."""
        if X.ndim == 1:
            X = X.reshape(1, -1)

        result = self.model(X)

        # Handle tuple output (mean, var) from GP
        if isinstance(result, tuple):
            result = result[0]

        return np.asarray(result).flatten()

    def _compute_interactions(
        self,
        A: np.ndarray,
        B: np.ndarray,
        f_A: np.ndarray,
        f_B: np.ndarray,
        var_total: float,
        bounds: List[Tuple[float, float]]
    ) -> np.ndarray:
        """Compute second-order interaction indices."""
        d = len(bounds)
        S_ij = np.zeros((d, d))

        # This is computationally expensive
        for i in range(d):
            for j in range(i + 1, d):
                # Create matrix with columns i and j from B
                AB_ij = A.copy()
                AB_ij[:, i] = B[:, i]
                AB_ij[:, j] = B[:, j]

                f_AB_ij = self._evaluate_model(AB_ij)

                # Second-order index
                AB_i = A.copy()
                AB_i[:, i] = B[:, i]
                f_AB_i = self._evaluate_model(AB_i)

                AB_j = A.copy()
                AB_j[:, j] = B[:, j]
                f_AB_j = self._evaluate_model(AB_j)

                V_ij = np.mean(f_B * (f_AB_ij - f_AB_i - f_AB_j + f_A))
                S_ij[i, j] = V_ij / var_total
                S_ij[j, i] = S_ij[i, j]

        return np.clip(S_ij, 0, 1)


class GPSensitivityAnalyzer(GlobalSensitivityAnalyzer):
    """
    Sensitivity analysis specifically for Gaussian Process models.

    Takes advantage of GP structure to also analyze:
    - Sensitivity of uncertainty (not just mean)
    - Lengthscale-based sensitivity
    """

    def __init__(
        self,
        gp_model,
        feature_names: Optional[List[str]] = None,
        n_samples: int = 1000,
        seed: int = 42
    ):
        """
        Initialize GP sensitivity analyzer.

        Parameters
        ----------
        gp_model : object
            GP model with predict_f(X) -> (mean, var) method
        """
        self.gp_model = gp_model

        # Create wrapper for mean prediction
        def mean_predictor(X):
            mean, _ = gp_model.predict_f(X)
            return mean

        super().__init__(mean_predictor, feature_names, n_samples, seed)

    def analyze_variance_sensitivity(
        self,
        bounds: List[Tuple[float, float]]
    ) -> SensitivityResults:
        """
        Analyze which inputs most affect predictive VARIANCE.

        This tells you which inputs, when varied, cause the most
        change in uncertainty - useful for understanding where
        the model is most/least confident.
        """
        # Create wrapper for variance prediction
        def var_predictor(X):
            _, var = self.gp_model.predict_f(X)
            return var

        # Temporarily swap predictor
        original_model = self.model
        self.model = var_predictor

        # Run analysis
        results = self.analyze(bounds, method='sobol')

        # Restore
        self.model = original_model

        return results

    def lengthscale_sensitivity(self) -> Dict[str, float]:
        """
        Infer sensitivity from GP lengthscales.

        Shorter lengthscale → function varies more rapidly with that input
        → that input is more important.

        This is a fast approximation that doesn't require Monte Carlo.
        """
        try:
            lengthscales = self.gp_model.get_lengthscales()
            lengthscales = np.atleast_1d(lengthscales)
        except:
            return {}

        # Inverse lengthscale as importance
        importance = 1.0 / (np.array(lengthscales) + 1e-6)

        # Normalize
        importance = importance / np.sum(importance)

        if self.feature_names is None:
            names = [f"X{i}" for i in range(len(lengthscales))]
        else:
            names = self.feature_names[:len(lengthscales)]

        return dict(zip(names, importance))


# =============================================================================
# Convenience Functions
# =============================================================================

def analyze_sensitivity(
    model: Callable,
    bounds: List[Tuple[float, float]],
    feature_names: Optional[List[str]] = None,
    method: str = 'sobol',
    n_samples: int = 1000
) -> SensitivityResults:
    """
    Quick function to perform global sensitivity analysis.

    Parameters
    ----------
    model : Callable
        Function f(X) -> Y
    bounds : List[Tuple[float, float]]
        Input bounds [(low1, high1), ...]
    feature_names : List[str], optional
        Names for inputs
    method : str
        'sobol', 'morris', or 'correlation'
    n_samples : int
        Number of Monte Carlo samples

    Returns
    -------
    SensitivityResults
        Sensitivity indices

    Example
    -------
    >>> def model(X):
    ...     return X[:, 0]**2 + 0.5*X[:, 1]
    >>> bounds = [(0, 1), (0, 1)]
    >>> results = analyze_sensitivity(model, bounds, ['x1', 'x2'])
    >>> print(results.summary())
    """
    analyzer = GlobalSensitivityAnalyzer(
        model, feature_names, n_samples
    )
    return analyzer.analyze(bounds, method)


def rank_feature_importance(
    model: Callable,
    bounds: List[Tuple[float, float]],
    feature_names: List[str]
) -> List[Tuple[str, float]]:
    """
    Get ranked list of feature importance.

    Returns list of (feature_name, importance) tuples, sorted by importance.
    """
    results = analyze_sensitivity(model, bounds, feature_names)
    return results.get_ranking()
