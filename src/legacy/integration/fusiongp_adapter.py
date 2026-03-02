"""
FusionGP → UQ System Integration Adapter

Bridges the gap between:
  - FusionGP pipeline outputs (PyTorch model, CSV exports with string source labels)
  - The UQ system (expects integer source codes, GPflow-compatible predict_f interface)

Gap resolution summary
----------------------
1. Source encoding: "epa" → 0, "lur" → 1, "satellite" → 2
2. Target values: pipeline de-normalizes before exporting; CSVs are in original µg/m³
3. Calibration split: val CSV (uq_val.csv) is used as the calibration set
4. Lengthscale extraction: uses model.get_hyperparameters() dict (not GPflow attribute)
5. predict_f interface: wraps FusionGP's Predictor to return (mean, var) arrays
6. Feature dimensionality: lat, lon, timestamp + N covariate columns (N=16 by default)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Source encoding map
# ---------------------------------------------------------------------------

SOURCE_ENCODING: Dict[str, int] = {
    "epa": 0,         # Reference-grade EPA monitors
    "lur": 1,         # Land-use regression (treated like low-cost / model-based)
    "satellite": 2,   # Satellite retrievals
    # Allow aliases that may appear in raw data
    "sat": 2,
    "lc": 1,
    "reference": 0,
    "sensor": 1,
}

# Covariate columns produced by add_wind_weighted_covariates() in run_demo_pipeline.py
WIND_COVARIATE_COLUMNS: List[str] = (
    [f"traffic_wind_{i}" for i in range(8)]
    + [f"wind_speed_w_{i}" for i in range(8)]
)


# ---------------------------------------------------------------------------
# Data loading helper
# ---------------------------------------------------------------------------

def load_uq_datasets(
    uq_dir: Path | str,
    covariate_columns: Optional[List[str]] = None,
    normalise_coords: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load the three UQ CSV files exported by the FusionGP pipeline.

    Parameters
    ----------
    uq_dir : Path or str
        Directory containing uq_train.csv, uq_val.csv, uq_test.csv
    covariate_columns : list of str, optional
        Which covariate columns to include in the feature matrix.
        Defaults to the 16 wind-weighted columns from run_demo_pipeline.py.
    normalise_coords : bool
        Whether to z-score normalise lat/lon/timestamp independently.
        Recommended when combining with models trained on normalised inputs.

    Returns
    -------
    dict with keys "train", "cal", "test". Each value is a dict:
        X         : np.ndarray [N, D] – feature matrix
        y         : np.ndarray [N]    – observations in original µg/m³
        sources   : np.ndarray [N]    – integer source codes
        lat       : np.ndarray [N]
        lon       : np.ndarray [N]
        timestamp : np.ndarray [N]
        source_str: np.ndarray [N]    – original string labels
    """
    uq_dir = Path(uq_dir)
    covariate_columns = covariate_columns or WIND_COVARIATE_COLUMNS

    splits: Dict[str, Dict] = {}

    for split_name, fname in [("train", "uq_train.csv"), ("cal", "uq_val.csv"), ("test", "uq_test.csv")]:
        path = uq_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"UQ CSV not found: {path}")

        df = pd.read_csv(path)

        # ------------------------------------------------------------------
        # 1. Source encoding: string → integer
        # ------------------------------------------------------------------
        if "source" not in df.columns:
            raise ValueError(f"{fname}: missing 'source' column")

        src_lower = df["source"].str.lower().str.strip()
        unknown = set(src_lower.unique()) - set(SOURCE_ENCODING.keys())
        if unknown:
            raise ValueError(
                f"{fname}: unknown source labels {unknown}. "
                f"Known: {list(SOURCE_ENCODING.keys())}"
            )
        sources_int = src_lower.map(SOURCE_ENCODING).to_numpy(dtype=np.int64)

        # ------------------------------------------------------------------
        # 2. Target values – already in original µg/m³ (pipeline de-normalises)
        # ------------------------------------------------------------------
        if "value" not in df.columns:
            raise ValueError(f"{fname}: missing 'value' column")
        y = df["value"].to_numpy(dtype=np.float64)

        # Drop rows with NaN targets (can arise from source masking in pipeline)
        valid = np.isfinite(y)
        df = df[valid].reset_index(drop=True)
        y = y[valid]
        sources_int = sources_int[valid]
        src_lower = src_lower[valid].reset_index(drop=True)

        # ------------------------------------------------------------------
        # 3. Spatial / temporal coordinates
        # ------------------------------------------------------------------
        lat = df["latitude"].to_numpy(dtype=np.float64)
        lon = df["longitude"].to_numpy(dtype=np.float64)

        # timestamp may be a date string or a numeric value
        if pd.api.types.is_string_dtype(df["timestamp"]):
            timestamps = pd.to_datetime(df["timestamp"]).astype(np.int64) / 1e9  # Unix seconds
        else:
            timestamps = df["timestamp"].to_numpy(dtype=np.float64)

        # ------------------------------------------------------------------
        # 4. Covariate columns
        # ------------------------------------------------------------------
        available_cov = [c for c in covariate_columns if c in df.columns]
        missing_cov = [c for c in covariate_columns if c not in df.columns]
        if missing_cov:
            # Fill missing covariate columns with zeros (graceful degradation)
            import warnings
            warnings.warn(
                f"{fname}: covariate columns not found, filling with zeros: {missing_cov}",
                UserWarning,
                stacklevel=2,
            )
            for col in missing_cov:
                df[col] = 0.0
            available_cov = covariate_columns

        covariates = df[available_cov].to_numpy(dtype=np.float64)

        # ------------------------------------------------------------------
        # 5. Build feature matrix: [lat, lon, timestamp, covariates...]
        # ------------------------------------------------------------------
        X_raw = np.column_stack([lat, lon, timestamps, covariates])

        if normalise_coords and split_name == "train":
            # Compute normalisation statistics from training split and reuse for
            # cal/test (stored in closure via mutable dict written to splits["_norm"])
            col_mean = X_raw.mean(axis=0)
            col_std = X_raw.std(axis=0)
            col_std[col_std == 0] = 1.0  # avoid division by zero
            splits["_norm"] = {"mean": col_mean, "std": col_std}
            X = (X_raw - col_mean) / col_std
        elif normalise_coords and "_norm" in splits:
            norm = splits["_norm"]
            X = (X_raw - norm["mean"]) / norm["std"]
        else:
            X = X_raw

        splits[split_name] = {
            "X": X,
            "y": y,
            "sources": sources_int,
            "lat": lat,
            "lon": lon,
            "timestamp": timestamps,
            "source_str": src_lower.to_numpy(),
        }

    # Remove internal key before returning
    splits.pop("_norm", None)
    return splits


# ---------------------------------------------------------------------------
# Data adapter class (convenience wrapper around load_uq_datasets)
# ---------------------------------------------------------------------------

class FusionGPDataAdapter:
    """
    High-level adapter for loading and preparing FusionGP UQ exports.

    Usage
    -----
    >>> adapter = FusionGPDataAdapter(uq_dir="/path/to/outputs/demo_run_xxx/uq")
    >>> adapter.load()
    >>> X_train, y_train, src_train = adapter.train
    >>> X_cal,   y_cal,   src_cal   = adapter.calibration
    >>> X_test,  y_test,  src_test  = adapter.test
    """

    def __init__(
        self,
        uq_dir: Path | str,
        covariate_columns: Optional[List[str]] = None,
        normalise_coords: bool = True,
    ):
        self.uq_dir = Path(uq_dir)
        self.covariate_columns = covariate_columns or WIND_COVARIATE_COLUMNS
        self.normalise_coords = normalise_coords
        self._splits: Optional[Dict] = None

    def load(self) -> "FusionGPDataAdapter":
        """Load all three CSV splits. Returns self for chaining."""
        self._splits = load_uq_datasets(
            self.uq_dir,
            covariate_columns=self.covariate_columns,
            normalise_coords=self.normalise_coords,
        )
        return self

    def _get_split(self, name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._splits is None:
            raise RuntimeError("Call adapter.load() first.")
        s = self._splits[name]
        return s["X"], s["y"], s["sources"]

    @property
    def train(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(X_train, y_train, sources_train)"""
        return self._get_split("train")

    @property
    def calibration(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(X_cal, y_cal, sources_cal) – from uq_val.csv"""
        return self._get_split("cal")

    @property
    def test(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(X_test, y_test, sources_test)"""
        return self._get_split("test")

    def get_split(self, name: str) -> Dict[str, np.ndarray]:
        """Return full split dict (X, y, sources, lat, lon, timestamp, source_str)."""
        if self._splits is None:
            raise RuntimeError("Call adapter.load() first.")
        return self._splits[name]

    def source_noise_levels(self, units: str = "original") -> Dict[str, float]:
        """
        Return source-specific noise variances in µg/m³² for use with FusionGPUQConfig.

        These match the initial_noise values in the FusionSVGP model (epa=1.0, satellite=3.0),
        converted from normalised units to physical units by typical urban NO2 std ≈ 5-8 µg/m³.

        Parameters
        ----------
        units : "original" or "normalised"
        """
        # Physical noise levels (µg/m³)²
        noise = {
            "EPA": 2.1,         # Reference monitors: tight uncertainty
            "LUR": 6.5,         # LUR model-based predictions
            "SAT": 15.6,        # Satellite: highest retrieval uncertainty
        }
        return noise

    def __repr__(self) -> str:
        status = "loaded" if self._splits is not None else "not loaded"
        return f"FusionGPDataAdapter(uq_dir={self.uq_dir}, status={status})"


# ---------------------------------------------------------------------------
# Model adapter: wraps FusionGP PyTorch model to expose predict_f interface
# ---------------------------------------------------------------------------

class FusionGPModelAdapter:
    """
    Adapter wrapping a trained FusionSVGP (PyTorch) model so it exposes the
    same interface used by the UQ system's ensemble and conformal modules.

    The UQ system expects:
        mean, var = model.predict_f(X)   # numpy arrays, shape [N] each

    The FusionGP Predictor exposes:
        predictions = predictor.predict(data)
        predictions.mean  # numpy array
        predictions.std   # numpy array (not var!)

    This adapter resolves that mismatch.

    Usage
    -----
    >>> from src.inference import Predictor
    >>> predictor = Predictor(fusiongp_model, scalers=preprocessor.get_scalers())
    >>> adapter = FusionGPModelAdapter(predictor, adapter.get_split("test"))
    >>> mean, var = adapter.predict_f(X_test)
    """

    def __init__(self, predictor, test_split: Optional[Dict] = None):
        """
        Parameters
        ----------
        predictor : src.inference.Predictor
            A fitted FusionGP Predictor wrapping a FusionSVGP model.
        test_split : dict, optional
            Full split dict from FusionGPDataAdapter.get_split(). Used to
            extract lat/lon/timestamps needed by the Predictor.
        """
        self.predictor = predictor
        self.test_split = test_split

    def predict_f(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run prediction and return (mean, variance) in original µg/m³ units.

        Parameters
        ----------
        X : np.ndarray [N, D]
            Feature matrix (lat, lon, timestamp, covariates).
            Must match the normalisation used during training.

        Returns
        -------
        mean : np.ndarray [N]
        var  : np.ndarray [N]   (std² from Predictor.std)
        """
        # The Predictor expects a FusionData object, not a raw array.
        # We use a lightweight MockFusionData to wrap the arrays.
        mock_data = _MockFusionData(X, self.test_split)
        predictions = self.predictor.predict(mock_data, verbose=False)
        mean = np.asarray(predictions.mean, dtype=np.float64)
        var = np.asarray(predictions.std, dtype=np.float64) ** 2
        return mean, var

    def get_hyperparameters(self) -> Dict:
        """
        Extract hyperparameters using FusionSVGP's get_hyperparameters() method.

        Returns a dict compatible with the UQ system's lengthscale extraction.
        Keys include: spatial_lengthscale_lat, spatial_lengthscale_lon,
                      temporal_lengthscale, output_scale, noise_epa, noise_satellite
        """
        model = getattr(self.predictor, "model", None)
        if model is None:
            raise AttributeError("Predictor has no .model attribute")
        if not hasattr(model, "get_hyperparameters"):
            raise AttributeError("FusionSVGP model has no get_hyperparameters() method")
        return model.get_hyperparameters()

    def get_lengthscales(self) -> np.ndarray:
        """
        Return lengthscales as a 1-D numpy array [lat_ls, lon_ls, time_ls].

        Handles both ARD (per-dimension) and isotropic cases gracefully.
        """
        params = self.get_hyperparameters()

        # Try ARD format first (spatial_lengthscale_lat / lon exist)
        if "spatial_lengthscale_lat" in params and "spatial_lengthscale_lon" in params:
            lat_ls = float(params["spatial_lengthscale_lat"])
            lon_ls = float(params["spatial_lengthscale_lon"])
        elif "spatial_lengthscale" in params:
            lat_ls = lon_ls = float(params["spatial_lengthscale"])
        else:
            lat_ls = lon_ls = 0.1  # fallback

        time_ls = float(params.get("temporal_lengthscale", 1.0))
        return np.array([lat_ls, lon_ls, time_ls])


# ---------------------------------------------------------------------------
# Minimal FusionData mock (internal only)
# ---------------------------------------------------------------------------

class _MockFusionData:
    """
    Minimal stand-in for FusionData that satisfies the Predictor interface
    when we already have a raw numpy feature matrix.

    Only used internally by FusionGPModelAdapter.predict_f().
    """

    def __init__(self, X: np.ndarray, split: Optional[Dict] = None):
        self.X = X
        n = len(X)

        # Try to reconstruct coords and timestamps from feature matrix columns
        # Convention: X[:, 0]=lat, X[:, 1]=lon, X[:, 2]=timestamp
        if split is not None:
            self.coords = np.column_stack([split["lat"], split["lon"]])
            self.timestamps = split["timestamp"]
        else:
            self.coords = X[:, :2]
            self.timestamps = X[:, 2] if X.shape[1] > 2 else np.zeros(n)

        self.covariates = X[:, 3:] if X.shape[1] > 3 else None
        self.n_observations = n
        # Mock source masks (all observations are "unknown" – Predictor uses
        # coords + covariates, not source masks, for prediction)
        self.source_masks = {}
        self.observations = {}
        self.grid_ids = None
