"""
Bootstrap Ensemble for Hyperparameter Uncertainty Quantification

Implements bootstrap ensemble method to quantify uncertainty in GP hyperparameters.
Addresses Research Question 2: How much do point estimates underestimate uncertainty?

Key Innovation:
    Total Uncertainty = Within-model variance + Between-model variance
    σ²_total(x*) = E[σ²_within(x*)] + Var[μ_between(x*)]

This decomposes uncertainty into:
- Within-model: Standard GP predictive variance (epistemic + aleatoric)
- Between-model: Uncertainty due to hyperparameter estimation
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from loguru import logger

try:
    import gpflow
    import tensorflow as tf
    GPFLOW_AVAILABLE = True
except ImportError:
    GPFLOW_AVAILABLE = False
    logger.warning("GPflow not available. Ensemble will use mock models for testing.")


@dataclass
class EnsembleUncertainty:
    """Container for ensemble uncertainty decomposition."""

    mean_prediction: npt.NDArray[np.float64]  # Ensemble mean [N]
    within_model_variance: npt.NDArray[np.float64]  # E[Var] - GP predictive var
    between_model_variance: npt.NDArray[np.float64]  # Var[E] - Hyperparameter uncertainty
    total_variance: npt.NDArray[np.float64]  # Total = within + between
    hyperparameter_fraction: npt.NDArray[np.float64]  # between / total

    def summary_stats(self) -> Dict[str, float]:
        """Compute summary statistics."""
        return {
            "mean_within_std": float(np.mean(np.sqrt(self.within_model_variance))),
            "mean_between_std": float(np.mean(np.sqrt(self.between_model_variance))),
            "mean_total_std": float(np.mean(np.sqrt(self.total_variance))),
            "mean_hyperparameter_fraction": float(np.mean(self.hyperparameter_fraction)),
            "median_hyperparameter_fraction": float(np.median(self.hyperparameter_fraction)),
        }

    def __repr__(self) -> str:
        stats = self.summary_stats()
        return (
            f"EnsembleUncertainty(\n"
            f"  Within-model σ: {stats['mean_within_std']:.4f}\n"
            f"  Between-model σ: {stats['mean_between_std']:.4f}\n"
            f"  Total σ: {stats['mean_total_std']:.4f}\n"
            f"  Hyperparameter fraction: {stats['mean_hyperparameter_fraction']:.1%}\n"
            f")"
        )


@dataclass
class HyperparameterDistribution:
    """Distribution of hyperparameters across ensemble."""

    lengthscales: npt.NDArray[np.float64]  # [n_ensemble, n_dims]
    output_scales: npt.NDArray[np.float64]  # [n_ensemble]
    noise_variances: Dict[str, npt.NDArray[np.float64]]  # {source: [n_ensemble]}

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Summarize hyperparameter distributions."""
        summary = {
            "lengthscales_mean": np.mean(self.lengthscales, axis=0).tolist(),
            "lengthscales_std": np.std(self.lengthscales, axis=0).tolist(),
            "output_scale_mean": float(np.mean(self.output_scales)),
            "output_scale_std": float(np.std(self.output_scales)),
        }

        for source, values in self.noise_variances.items():
            summary[f"noise_{source}_mean"] = float(np.mean(values))
            summary[f"noise_{source}_std"] = float(np.std(values))

        return summary


class BootstrapSVGPEnsemble:
    """
    Bootstrap ensemble for SVGP models.

    Quantifies hyperparameter uncertainty by training multiple models
    on bootstrap samples of the training data.
    """

    def __init__(
        self,
        n_ensemble: int = 10,
        bootstrap_fraction: float = 0.8,
        n_inducing: int = 500,
        parallel: bool = True,
        n_workers: Optional[int] = None
    ):
        """
        Initialize bootstrap ensemble.

        Args:
            n_ensemble: Number of models in ensemble (10 is good balance for 4 days)
            bootstrap_fraction: Fraction of data for each bootstrap sample
            n_inducing: Number of inducing points for SVGP
            parallel: Whether to train models in parallel
            n_workers: Number of parallel workers (None = auto)
        """
        self.n_ensemble = n_ensemble
        self.bootstrap_fraction = bootstrap_fraction
        self.n_inducing = n_inducing
        self.parallel = parallel
        self.n_workers = n_workers
        self.models: List[Any] = []

        logger.info(
            f"Initialized BootstrapSVGPEnsemble with {n_ensemble} models, "
            f"bootstrap_fraction={bootstrap_fraction}"
        )

    def fit(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        sources_train: npt.NDArray[np.int64],
        max_iter: int = 1000,
        verbose: bool = False
    ):
        """
        Train ensemble of models on bootstrap samples.

        Args:
            X_train: Training features [N x D]
            y_train: Training targets [N]
            sources_train: Source identifiers [N]
            max_iter: Maximum optimization iterations per model
            verbose: Print training progress
        """
        logger.info(f"Training ensemble of {self.n_ensemble} SVGP models")

        n_samples = len(X_train)
        n_bootstrap = int(n_samples * self.bootstrap_fraction)

        if self.parallel and self.n_ensemble > 1:
            # Parallel training
            self.models = self._fit_parallel(
                X_train, y_train, sources_train, n_bootstrap, max_iter, verbose
            )
        else:
            # Sequential training
            self.models = []
            for i in range(self.n_ensemble):
                model = self._fit_single_bootstrap(
                    X_train, y_train, sources_train, n_bootstrap, max_iter, i, verbose
                )
                self.models.append(model)

        logger.info(f"Ensemble training complete. {len(self.models)} models trained.")

    def _fit_single_bootstrap(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        sources_train: npt.NDArray[np.int64],
        n_bootstrap: int,
        max_iter: int,
        model_id: int,
        verbose: bool
    ) -> Any:
        """Train a single bootstrap model."""
        # Bootstrap sample
        indices = np.random.choice(len(X_train), n_bootstrap, replace=True)
        X_boot = X_train[indices]
        y_boot = y_train[indices]
        sources_boot = sources_train[indices]

        if verbose:
            logger.info(f"Training model {model_id + 1}/{self.n_ensemble}")

        # Train model
        if GPFLOW_AVAILABLE:
            model = self._build_and_train_svgp(
                X_boot, y_boot, sources_boot, max_iter
            )
        else:
            # Mock model for testing
            model = self._build_mock_model(X_boot, y_boot)

        return model

    def _fit_parallel(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        sources_train: npt.NDArray[np.int64],
        n_bootstrap: int,
        max_iter: int,
        verbose: bool
    ) -> List[Any]:
        """Train ensemble in parallel."""
        logger.info(f"Training {self.n_ensemble} models in parallel")

        models = []
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for i in range(self.n_ensemble):
                future = executor.submit(
                    self._fit_single_bootstrap,
                    X_train, y_train, sources_train, n_bootstrap, max_iter, i, verbose
                )
                futures.append(future)

            for i, future in enumerate(as_completed(futures)):
                try:
                    model = future.result()
                    models.append(model)
                    logger.info(f"Model {i+1}/{self.n_ensemble} completed")
                except Exception as e:
                    logger.error(f"Model {i+1} training failed: {e}")

        return models

    def _build_and_train_svgp(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        sources_train: npt.NDArray[np.int64],
        max_iter: int
    ) -> Any:
        """Build and train a single SVGP model."""
        # Initialize inducing points
        n_inducing_actual = min(self.n_inducing, len(X_train) // 2)
        indices = np.random.choice(len(X_train), n_inducing_actual, replace=False)
        Z = X_train[indices].copy()

        # Define kernel
        kernel = gpflow.kernels.Matern52(
            lengthscales=np.ones(X_train.shape[1])
        )

        # Build SVGP model
        model = gpflow.models.SVGP(
            kernel=kernel,
            likelihood=gpflow.likelihoods.Gaussian(),
            inducing_variable=Z,
            num_data=len(X_train)
        )

        # Optimize
        optimizer = gpflow.optimizers.Scipy()
        opt_result = optimizer.minimize(
            model.training_loss_closure((X_train, y_train[:, None])),
            variables=model.trainable_variables,
            options=dict(maxiter=max_iter)
        )

        return model

    def _build_mock_model(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64]
    ) -> Any:
        """Build mock model for testing without GPflow."""
        class MockSVGP:
            def __init__(self, X, y):
                self.X_train = X
                self.y_train = y
                self.lengthscales = np.random.gamma(2, 0.5, X.shape[1])
                self.variance = np.random.gamma(2, 1.0)
                self.noise_variance = np.random.gamma(2, 0.5)

            def predict_f(self, X_test):
                n = len(X_test)
                mean = np.random.randn(n)
                var = np.random.gamma(2, 1, n)

                class MockTensor:
                    def __init__(self, data):
                        self.data = data
                    def numpy(self):
                        return self.data

                return MockTensor(mean), MockTensor(var)

        return MockSVGP(X_train, y_train)

    def predict_with_full_uncertainty(
        self,
        X_test: npt.NDArray[np.float64]
    ) -> EnsembleUncertainty:
        """
        Predict with full uncertainty decomposition.

        Uses Law of Total Variance:
            Var(Y) = E[Var(Y|θ)] + Var[E(Y|θ)]
                   = within-model + between-model

        Args:
            X_test: Test locations [N x D]

        Returns:
            EnsembleUncertainty with decomposed variance
        """
        logger.info(f"Predicting with ensemble for {len(X_test)} test points")

        means = []
        variances = []

        # Get predictions from each model
        for i, model in enumerate(self.models):
            mean, var = model.predict_f(X_test)
            means.append(mean.numpy().flatten())
            variances.append(var.numpy().flatten())

        means = np.array(means)  # [n_ensemble, n_test]
        variances = np.array(variances)  # [n_ensemble, n_test]

        # Law of total variance
        within_model_var = np.mean(variances, axis=0)  # E[Var(Y|θ)]
        mean_prediction = np.mean(means, axis=0)  # E[E(Y|θ)]
        between_model_var = np.var(means, axis=0)  # Var[E(Y|θ)]

        # Total uncertainty
        total_var = within_model_var + between_model_var

        # Fraction due to hyperparameter uncertainty
        hyperparameter_fraction = between_model_var / (total_var + 1e-10)

        return EnsembleUncertainty(
            mean_prediction=mean_prediction,
            within_model_variance=within_model_var,
            between_model_variance=between_model_var,
            total_variance=total_var,
            hyperparameter_fraction=hyperparameter_fraction
        )

    def get_hyperparameter_distribution(self) -> HyperparameterDistribution:
        """
        Extract posterior distribution over hyperparameters.

        Returns:
            HyperparameterDistribution containing samples from each model
        """
        lengthscales = []
        output_scales = []
        noise_variances = {'all': []}

        for model in self.models:
            if hasattr(model, 'kernel'):
                # Real GPflow model
                lengthscales.append(model.kernel.lengthscales.numpy())
                output_scales.append(model.kernel.variance.numpy())
                noise_variances['all'].append(model.likelihood.variance.numpy())
            else:
                # Mock model
                lengthscales.append(model.lengthscales)
                output_scales.append(model.variance)
                noise_variances['all'].append(model.noise_variance)

        return HyperparameterDistribution(
            lengthscales=np.array(lengthscales),
            output_scales=np.array(output_scales),
            noise_variances={k: np.array(v) for k, v in noise_variances.items()}
        )

    def quantify_underestimation(
        self,
        point_model: Any,
        X_test: npt.NDArray[np.float64]
    ) -> Dict[str, float]:
        """
        Quantify how much point estimates underestimate uncertainty.

        Addresses Research Question 2: Expected 10-30% underestimation.

        Args:
            point_model: Single SVGP model trained on full dataset
            X_test: Test locations [N x D]

        Returns:
            Dictionary with underestimation statistics
        """
        logger.info("Quantifying underestimation from point estimates")

        # Point estimate uncertainty
        _, var_point = point_model.predict_f(X_test)
        sigma_point = np.sqrt(var_point.numpy().flatten())

        # Ensemble uncertainty (with hyperparameter uncertainty)
        ensemble_unc = self.predict_with_full_uncertainty(X_test)
        sigma_ensemble = np.sqrt(ensemble_unc.total_variance)

        # Compute underestimation
        underestimation_ratio = sigma_point / sigma_ensemble
        underestimation_pct = (1 - underestimation_ratio) * 100

        # Filter out any infinite or nan values
        valid_mask = np.isfinite(underestimation_pct)
        underestimation_pct = underestimation_pct[valid_mask]

        return {
            'mean_underestimation_pct': float(np.mean(underestimation_pct)),
            'median_underestimation_pct': float(np.median(underestimation_pct)),
            'std_underestimation_pct': float(np.std(underestimation_pct)),
            'min_underestimation_pct': float(np.min(underestimation_pct)),
            'max_underestimation_pct': float(np.max(underestimation_pct)),
            'fraction_underestimated': float(np.mean(underestimation_pct > 0)),
        }


if __name__ == "__main__":
    # Example usage and testing
    logger.info("Testing BootstrapSVGPEnsemble")

    # Simulate data
    np.random.seed(42)
    n_train = 500
    n_test = 100
    d = 3

    X_train = np.random.randn(n_train, d)
    y_train = np.random.randn(n_train)
    sources_train = np.random.choice([0, 1, 2], size=n_train)

    X_test = np.random.randn(n_test, d)

    # Build ensemble
    ensemble = BootstrapSVGPEnsemble(n_ensemble=5, parallel=False)
    ensemble.fit(X_train, y_train, sources_train, max_iter=100, verbose=True)

    # Get predictions with full uncertainty
    ensemble_unc = ensemble.predict_with_full_uncertainty(X_test)
    print(ensemble_unc)

    # Get hyperparameter distribution
    hyperparam_dist = ensemble.get_hyperparameter_distribution()
    print("\nHyperparameter distribution:")
    for key, value in hyperparam_dist.summary().items():
        print(f"  {key}: {value}")

    # Mock point model for testing underestimation
    point_model = ensemble.models[0]
    underestimation = ensemble.quantify_underestimation(point_model, X_test)
    print("\nUnderestimation analysis:")
    for key, value in underestimation.items():
        print(f"  {key}: {value:.2f}")
