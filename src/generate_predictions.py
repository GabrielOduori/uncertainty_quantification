"""
Generate model predictions from the saved FusionGP checkpoint.

Predictions are already committed to data/predictions_val.csv and
data/predictions_test.csv.  run() returns immediately if both files
exist, so this script is only needed when regenerating from scratch.

To regenerate:
    python src/generate_predictions.py --force

No external repository dependency — FusionSVGP and GridPriorMean are
implemented in src/models/ using GPyTorch primitives.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Ensure this repo's src/ is importable when running as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import FusionSVGP, GridPriorMean  # noqa: E402

DATA_DIR  = REPO_ROOT / "data"
CKPT      = DATA_DIR / "checkpoints" / "best_model.pt"
PRED_VAL  = DATA_DIR / "predictions_val.csv"
PRED_TEST = DATA_DIR / "predictions_test.csv"

FEATURE_COLS = (
    ["latitude", "longitude", "timestamp"]
    + [f"traffic_wind_{i}" for i in range(8)]
    + [f"wind_speed_w_{i}" for i in range(8)]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def derive_target_scalers(train_csv: Path) -> dict:
    """Compute per-source (mean, std) from training observations."""
    df = pd.read_csv(train_csv)
    scalers = {}
    for src in ["epa", "satellite"]:
        obs = df[df["source"] == src]["value"].dropna()
        if len(obs) == 0:
            obs = df[df["source"] == "lur"]["value"].dropna()
        scalers[src] = {"mean": float(obs.mean()), "std": float(obs.std())}
        print(f"  Scaler {src}: mean={scalers[src]['mean']:.2f}, "
              f"std={scalers[src]['std']:.2f}, n={len(obs)}")
    return scalers


def build_model_shell(ckpt: dict) -> FusionSVGP:
    """Reconstruct FusionSVGP from checkpoint state-dict shapes and load weights."""
    sd = ckpt["model_state_dict"]

    n_inducing   = sd["variational_strategy.inducing_points"].shape[0]
    n_covariates = sd["covar_module.covariate_kernel.raw_lengthscale"].shape[1]
    grid_coords  = sd["mean_module.grid_coords"].numpy()
    grid_values  = sd["mean_module.grid_values"].numpy()

    print(f"Architecture: n_inducing={n_inducing}, "
          f"n_covariates={n_covariates}, n_grid_cells={len(grid_coords)}")

    prior_mean = GridPriorMean(
        grid_coords=grid_coords,
        grid_values=grid_values,
        learnable_bias=True,
    )
    model = FusionSVGP(
        n_inducing=n_inducing,
        n_covariates=n_covariates,
        prior_mean=prior_mean,
    )
    _result = model.load_state_dict(sd, strict=False)
    if _result.unexpected_keys:
        print(f"  Unexpected keys (ignored): {_result.unexpected_keys}")
    model.eval()
    return model


def predict_batch(
    model: FusionSVGP,
    X: np.ndarray,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Run GP posterior prediction in batches. Returns (mean, var) normalised."""
    means, vars_ = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x_batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32)
            try:
                dist = model(x_batch)
                means.append(dist.mean.numpy())
                vars_.append(dist.variance.numpy())
            except Exception as e:
                print(f"  Batch {i} failed: {e} — using prior mean fallback")
                means.append(np.zeros(len(x_batch)))
                vars_.append(np.ones(len(x_batch)))
    return np.concatenate(means), np.concatenate(vars_)


def run_predictions_for_split(
    csv_path: Path,
    model: FusionSVGP,
    scalers: dict,
    out_path: Path,
    split_name: str,
) -> None:
    """Load a UQ CSV, predict, inverse-transform to µg/m³, and save."""
    df = pd.read_csv(csv_path)
    df_pred = df[df["source"].isin({"epa", "satellite", "lur"})].copy()
    X = df_pred[FEATURE_COLS].to_numpy(dtype=np.float32)

    print(f"\nPredicting {split_name}: {len(df_pred):,} rows...")
    mean_norm, var_norm = predict_batch(model, X)

    # Inverse-transform from normalised EPA space to µg/m³
    epa_mean = scalers["epa"]["mean"]
    epa_std  = scalers["epa"]["std"]
    mean_ug  = mean_norm * epa_std + epa_mean
    std_ug   = np.sqrt(np.maximum(var_norm, 0)) * epa_std

    result = df_pred[["latitude", "longitude", "timestamp",
                       "grid_id", "source", "value"]].copy()
    result["pred_mean"]     = mean_ug
    result["pred_std"]      = std_ug
    result["pred_lower_90"] = mean_ug - 1.645 * std_ug
    result["pred_upper_90"] = mean_ug + 1.645 * std_ug
    result["pred_lower_95"] = mean_ug - 1.960 * std_ug
    result["pred_upper_95"] = mean_ug + 1.960 * std_ug
    result.to_csv(out_path, index=False)

    print(f"  Saved → {out_path}")
    print(f"  mean: {mean_ug.min():.2f} – {mean_ug.max():.2f} µg/m³  "
          f"std: {std_ug.min():.4f} – {std_ug.max():.4f}")

    epa_rows = result[result["source"] == "epa"]
    if len(epa_rows) > 0:
        y_true = epa_rows["value"].to_numpy()
        mu     = epa_rows["pred_mean"].to_numpy()
        rmse   = float(np.sqrt(np.mean((y_true - mu) ** 2)))
        cov95  = float(np.mean(
            (y_true >= epa_rows["pred_lower_95"].to_numpy()) &
            (y_true <= epa_rows["pred_upper_95"].to_numpy())
        ))
        print(f"  EPA check (n={len(epa_rows)}): RMSE={rmse:.3f}, Coverage_95={cov95:.3f}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(force: bool = False) -> bool:
    """
    Ensure predictions exist in data/.

    Returns True immediately (without loading the model) if both
    predictions_val.csv and predictions_test.csv already exist and
    force=False.  Called by run.py --stage predictions.
    """
    already_done = PRED_VAL.exists() and PRED_TEST.exists()

    if already_done and not force:
        print("Predictions already present — skipping regeneration.")
        print(f"  {PRED_VAL}")
        print(f"  {PRED_TEST}")
        return True

    print("=" * 60)
    print("FusionGP Prediction Generator")
    print("=" * 60)

    if not CKPT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT}")

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"Checkpoint: epoch={ckpt['epoch']}, "
          f"val_loss={ckpt['metrics'].get('val_loss', 'n/a'):.4g}")

    print("\nDeriving target scalers...")
    scalers = derive_target_scalers(DATA_DIR / "uq" / "uq_train.csv")

    print("\nRebuilding model...")
    model = build_model_shell(ckpt)
    print("  Model loaded.")

    run_predictions_for_split(
        DATA_DIR / "uq" / "uq_val.csv",  model, scalers, PRED_VAL,  "val/calibration",
    )
    run_predictions_for_split(
        DATA_DIR / "uq" / "uq_test.csv", model, scalers, PRED_TEST, "test",
    )
    print("\nDone.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate FusionGP predictions.")
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate predictions even if they already exist in data/.",
    )
    run(force=parser.parse_args().force)


if __name__ == "__main__":
    main()
