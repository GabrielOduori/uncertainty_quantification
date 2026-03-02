"""
Integration module for bridging FusionGP pipeline outputs to the UQ system.
"""

from .fusiongp_adapter import (
    FusionGPDataAdapter,
    FusionGPModelAdapter,
    load_uq_datasets,
    SOURCE_ENCODING,
)

__all__ = [
    "FusionGPDataAdapter",
    "FusionGPModelAdapter",
    "load_uq_datasets",
    "SOURCE_ENCODING",
]
