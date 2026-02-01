"""
Models module for uncertainty quantification.

Provides model wrappers and ensemble methods for UQ.
"""

from .ensemble import (
    BootstrapSVGPEnsemble,
    EnsembleUncertainty,
    HyperparameterDistribution,
)

__all__ = [
    "BootstrapSVGPEnsemble",
    "EnsembleUncertainty",
    "HyperparameterDistribution",
]
