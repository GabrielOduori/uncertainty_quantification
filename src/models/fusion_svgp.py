"""
FusionSVGP — Sparse Variational GP for multi-source NO₂ fusion.

Reconstructs the exact GPyTorch architecture used during training so that
``model.load_state_dict(checkpoint['model_state_dict'], strict=False)``
correctly restores all kernel hyperparameters and variational parameters.

State-dict key mapping
----------------------
covar_module.spatial_kernel.*           MaternKernel(nu=1.5, ARD, Interval[0.05,0.70])
covar_module.temporal_kernel.*          MaternKernel(nu=0.5, ARD, Interval[0.05,2.0])
covar_module.covariate_kernel.*         RBFKernel(ARD, Positive/softplus)
covar_module.outputscale_param          raw scalar — softplus → output scale
variational_strategy.inducing_points    [M, D]
variational_strategy._variational_distribution.*  CholeskyVariationalDistribution
mean_module.*                           GridPriorMean
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import gpytorch
from gpytorch.constraints import Interval, Positive
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from linear_operator.operators import DenseLinearOperator


# ---------------------------------------------------------------------------
# Composite kernel
# ---------------------------------------------------------------------------

class _FusionKernel(gpytorch.kernels.Kernel):
    """
    k(x,x') = softplus(outputscale_param)
               × k_spatial(x,x') × k_temporal(x,x') × k_covariate(x,x')

    active_dims convention (matches the checkpoint):
        [0, 1]      → lat, lon   (spatial)
        [2]         → timestamp  (temporal)
        [3, …, 18]  → 16 wind/traffic covariates
    """

    def __init__(self, n_covariates: int = 16):
        super().__init__()

        covariate_dims = list(range(3, 3 + n_covariates))

        self.spatial_kernel = MaternKernel(
            nu=1.5,
            ard_num_dims=2,
            active_dims=[0, 1],
            lengthscale_constraint=Interval(0.05, 0.70),
        )
        self.temporal_kernel = MaternKernel(
            nu=0.5,
            ard_num_dims=1,
            active_dims=[2],
            lengthscale_constraint=Interval(0.05, 2.0),
        )
        self.covariate_kernel = RBFKernel(
            ard_num_dims=n_covariates,
            active_dims=covariate_dims,
            lengthscale_constraint=Positive(),
        )
        # Raw scalar output scale — matches checkpoint key covar_module.outputscale_param
        self.outputscale_param = nn.Parameter(torch.zeros(()))  # 0-dim, matches checkpoint

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params):
        scale = F.softplus(self.outputscale_param)
        k = (
            self.spatial_kernel(x1, x2).to_dense()
            * self.temporal_kernel(x1, x2).to_dense()
            * self.covariate_kernel(x1, x2).to_dense()
        )
        k_scaled = scale * k
        if diag:
            return k_scaled.diagonal(dim1=-2, dim2=-1)
        return DenseLinearOperator(k_scaled)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class FusionSVGP(ApproximateGP):
    """
    Sparse Variational GP with LUR prior mean and separable
    Matérn-3/2 × Matérn-1/2 × RBF kernel.

    Parameters
    ----------
    n_inducing : int
        Number of inducing points (300 in the trained checkpoint).
    n_covariates : int
        Number of wind/traffic covariate columns (16 in the checkpoint).
    prior_mean : GridPriorMean
        Constructed prior mean module (must be passed in after building it
        from the checkpoint's grid arrays).
    All other keyword arguments are accepted and ignored for API compatibility
    with the original FusionGP constructor signature.
    """

    def __init__(
        self,
        n_inducing: int,
        n_covariates: int,
        prior_mean,
        # The following args are accepted for API compatibility but unused here
        spatial_kernel_type=None,
        temporal_kernel_type=None,
        spatial_ard=True,
        learn_inducing_locations=True,
        sources=None,
        initial_noise=None,
        initial_lengthscales=None,
        **kwargs,
    ):
        n_features = 3 + n_covariates  # lat, lon, time + covariates

        variational_distribution = CholeskyVariationalDistribution(n_inducing)
        inducing_points = torch.zeros(n_inducing, n_features)
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)

        self.mean_module = prior_mean
        self.covar_module = _FusionKernel(n_covariates=n_covariates)

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)
