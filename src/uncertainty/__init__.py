"""
Uncertainty Quantification Module for Air Quality Sensor Fusion

This module provides tools for quantifying, decomposing, and validating uncertainty
in probabilistic air quality models.
"""

from .decomposition import UncertaintyDecomposer, decompose_epistemic_aleatoric
from .calibration import CalibrationEvaluator, compute_picp, compute_crps, compute_ece
from .ood_detection import SpatialOODDetector, TemporalDriftDetector, OODWarningSystem
from .hierarchical import (
    HierarchicalUQTracker,
    VariancePropagationAnalyzer,
    HierarchicalVariance,
    VarianceAttribution,
)
from .conformal import (
    ConformalPredictionWrapper,
    AdaptiveConformalPredictor,
    evaluate_conformal_coverage,
)
from .second_order import (
    SecondOrderAnalyzer,
    SecondOrderUncertainty,
    MetaUncertaintyVisualizer,
)

# New modules from Johns Hopkins UQ Course
from .taylor_propagation import (
    TaylorPropagator,
    GPTaylorPropagator,
    CommonFormulas,
    propagate_uncertainty,
    combine_uncertainties,
)
from .sensitivity_analysis import (
    GlobalSensitivityAnalyzer,
    GPSensitivityAnalyzer,
    SensitivityResults,
    analyze_sensitivity,
    rank_feature_importance,
)
from .reliability_analysis import (
    ReliabilityAnalyzer,
    AirQualityReliability,
    ReliabilityResult,
    compute_exceedance_probability,
    reliability_index,
)

try:
    from .metrics import UncertaintyMetrics, EntropyCalculator, InformationGain
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

__all__ = [
    # Decomposition
    "UncertaintyDecomposer",
    "decompose_epistemic_aleatoric",
    # Calibration
    "CalibrationEvaluator",
    "compute_picp",
    "compute_crps",
    "compute_ece",
    # OOD Detection
    "SpatialOODDetector",
    "TemporalDriftDetector",
    "OODWarningSystem",
    # Hierarchical
    "HierarchicalUQTracker",
    "VariancePropagationAnalyzer",
    "HierarchicalVariance",
    "VarianceAttribution",
    # Conformal
    "ConformalPredictionWrapper",
    "AdaptiveConformalPredictor",
    "evaluate_conformal_coverage",
    # Second-Order
    "SecondOrderAnalyzer",
    "SecondOrderUncertainty",
    "MetaUncertaintyVisualizer",
    # Taylor Series Propagation (NEW - from JHU UQ Course)
    "TaylorPropagator",
    "GPTaylorPropagator",
    "CommonFormulas",
    "propagate_uncertainty",
    "combine_uncertainties",
    # Sensitivity Analysis (NEW - from JHU UQ Course)
    "GlobalSensitivityAnalyzer",
    "GPSensitivityAnalyzer",
    "SensitivityResults",
    "analyze_sensitivity",
    "rank_feature_importance",
    # Reliability Analysis (NEW - from JHU UQ Course)
    "ReliabilityAnalyzer",
    "AirQualityReliability",
    "ReliabilityResult",
    "compute_exceedance_probability",
    "reliability_index",
]

if METRICS_AVAILABLE:
    __all__.extend(["UncertaintyMetrics", "EntropyCalculator", "InformationGain"])

__version__ = "0.1.0"
