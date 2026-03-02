"""
GridPriorMean — GPyTorch mean module backed by the LUR grid.

Stores the LUR lat/lon grid and NO₂ values as non-trainable buffers.
For each query point, the mean is the LUR value at the nearest grid cell,
scaled and shifted by learnable scalar parameters.

State-dict keys (must match checkpoint):
    mean_module.grid_coords   [G, 2]  — normalised lat/lon of each LUR cell
    mean_module.grid_values   [G]     — LUR NO₂ value at each cell
    mean_module.scale         []      — learnable multiplier (init 1.0)
    mean_module.bias          []      — learnable offset   (init 0.0)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import gpytorch


class GridPriorMean(gpytorch.means.Mean):
    """
    LUR-grid prior mean with learnable scale and bias.

    Parameters
    ----------
    grid_coords : array-like [G, 2]
        Normalised (lat, lon) of each LUR grid cell.
    grid_values : array-like [G]
        LUR NO₂ prediction at each grid cell (normalised units).
    scalers : ignored
        Accepted for API compatibility with the original FusionGP signature.
    learnable_bias : bool
        Whether to register ``bias`` as a trainable parameter.
    """

    def __init__(self, grid_coords, grid_values, scalers=None, learnable_bias=True):
        super().__init__()

        coords = np.asarray(grid_coords, dtype=np.float32)
        values = np.asarray(grid_values, dtype=np.float32)

        self.register_buffer("grid_coords", torch.from_numpy(coords))
        self.register_buffer("grid_values", torch.from_numpy(values))

        self.scale = nn.Parameter(torch.ones(()))   # 0-dim, matches checkpoint
        if learnable_bias:
            self.bias = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer("bias", torch.zeros(()))

        # Build KDTree once at construction for O(N log G) nearest-neighbour lookup
        from scipy.spatial import cKDTree  # noqa: PLC0415
        self._tree = cKDTree(coords)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [N, D]
            Feature matrix; columns 0 and 1 are normalised lat/lon.

        Returns
        -------
        Tensor [N]  — prior mean at each query point.
        """
        coords_np = x[:, :2].detach().cpu().numpy()
        _, idx = self._tree.query(coords_np, k=1)
        lur_vals = self.grid_values[idx]          # [N], on same device as buffer
        return self.scale.squeeze() * lur_vals + self.bias.squeeze()
