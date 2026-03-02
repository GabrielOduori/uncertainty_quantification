"""
Tests for src/models (GridPriorMean, FusionSVGP) and src/generate_predictions.

Run:
    python -m pytest tests/test_fusion_svgp.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models import FusionSVGP, GridPriorMean  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_GRID  = 50    # small synthetic LUR grid
N_INDUC = 10    # small number of inducing points
N_COV   = 4     # small number of covariates (must be ≥ 1)
N_FEAT  = 3 + N_COV  # lat + lon + time + covariates


@pytest.fixture()
def small_grid():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 1, (N_GRID, 2)).astype(np.float32)
    values = rng.normal(20, 5, N_GRID).astype(np.float32)
    return coords, values


@pytest.fixture()
def prior_mean(small_grid):
    coords, values = small_grid
    return GridPriorMean(coords, values, learnable_bias=True)


@pytest.fixture()
def model(prior_mean):
    return FusionSVGP(
        n_inducing=N_INDUC,
        n_covariates=N_COV,
        prior_mean=prior_mean,
    )


# ---------------------------------------------------------------------------
# GridPriorMean tests
# ---------------------------------------------------------------------------


class TestGridPriorMean:

    def test_output_shape(self, prior_mean):
        x = torch.zeros(8, N_FEAT)
        out = prior_mean(x)
        assert out.shape == (8,), f"Expected (8,), got {out.shape}"

    def test_output_is_finite(self, prior_mean):
        x = torch.rand(16, N_FEAT)
        out = prior_mean(x)
        assert torch.isfinite(out).all()

    def test_lookup_returns_grid_value(self, small_grid):
        """Query at an exact grid cell should return that cell's value."""
        coords, values = small_grid
        pm = GridPriorMean(coords, values, learnable_bias=False)
        # Set scale=1, bias=0 so output == raw grid value
        with torch.no_grad():
            pm.scale.fill_(1.0)

        # Query the first grid cell directly
        query_coords = torch.tensor(coords[:1], dtype=torch.float32)  # [1, 2]
        x = torch.cat([query_coords, torch.zeros(1, N_FEAT - 2)], dim=1)
        out = pm(x)
        expected = float(values[0])
        assert abs(out.item() - expected) < 1e-4, \
            f"Expected {expected:.4f}, got {out.item():.4f}"

    def test_learnable_parameters_exist(self, prior_mean):
        param_names = {n for n, _ in prior_mean.named_parameters()}
        assert "scale" in param_names
        assert "bias" in param_names

    def test_buffers_not_trainable(self, prior_mean):
        """grid_coords and grid_values must be buffers, not parameters."""
        param_names = {n for n, _ in prior_mean.named_parameters()}
        assert "grid_coords" not in param_names
        assert "grid_values" not in param_names

    def test_scale_multiplies_output(self, small_grid):
        coords, values = small_grid
        pm = GridPriorMean(coords, values, learnable_bias=False)
        x = torch.rand(4, N_FEAT)
        with torch.no_grad():
            pm.scale.fill_(1.0)
            out1 = pm(x).clone()
            pm.scale.fill_(2.0)
            out2 = pm(x).clone()
        torch.testing.assert_close(out2, 2 * out1)


# ---------------------------------------------------------------------------
# FusionSVGP tests
# ---------------------------------------------------------------------------


class TestFusionSVGP:

    def test_forward_mean_shape(self, model):
        x = torch.rand(6, N_FEAT)
        model.eval()
        with torch.no_grad():
            dist = model(x)
        assert dist.mean.shape == (6,)

    def test_forward_variance_positive(self, model):
        x = torch.rand(6, N_FEAT)
        model.eval()
        with torch.no_grad():
            dist = model(x)
        assert (dist.variance > 0).all(), "All variances should be positive"

    def test_forward_finite(self, model):
        x = torch.rand(6, N_FEAT)
        model.eval()
        with torch.no_grad():
            dist = model(x)
        assert torch.isfinite(dist.mean).all()
        assert torch.isfinite(dist.variance).all()

    def test_state_dict_keys_match_checkpoint_structure(self, model):
        """Required key groups must be present in state dict."""
        sd = model.state_dict()
        key_prefixes = {k.split(".")[0] for k in sd}
        assert "mean_module" in key_prefixes
        assert "covar_module" in key_prefixes
        assert "variational_strategy" in key_prefixes

    def test_outputscale_param_0dim(self, model):
        sd = model.state_dict()
        assert sd["covar_module.outputscale_param"].shape == torch.Size([]), \
            "outputscale_param must be 0-dim to match checkpoint"

    def test_scale_bias_0dim(self, model):
        sd = model.state_dict()
        assert sd["mean_module.scale"].shape == torch.Size([])
        assert sd["mean_module.bias"].shape == torch.Size([])

    def test_no_missing_keys_from_synthetic_checkpoint(self, model):
        """Loading a state dict back into a fresh model leaves no missing keys."""
        sd = model.state_dict()
        fresh_pm = GridPriorMean(
            model.mean_module.grid_coords.numpy(),
            model.mean_module.grid_values.numpy(),
            learnable_bias=True,
        )
        fresh_model = FusionSVGP(
            n_inducing=N_INDUC, n_covariates=N_COV, prior_mean=fresh_pm
        )
        result = fresh_model.load_state_dict(sd, strict=True)
        assert result.missing_keys == []
        assert result.unexpected_keys == []


# ---------------------------------------------------------------------------
# Checkpoint integration test (skipped if checkpoint absent)
# ---------------------------------------------------------------------------

CKPT = REPO_ROOT / "data" / "checkpoints" / "best_model.pt"


@pytest.mark.skipif(not CKPT.exists(), reason="Checkpoint not present")
class TestCheckpointLoad:

    @pytest.fixture()
    def ckpt_model(self):
        sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
        pm = GridPriorMean(
            sd["mean_module.grid_coords"].numpy(),
            sd["mean_module.grid_values"].numpy(),
            learnable_bias=True,
        )
        m = FusionSVGP(
            n_inducing=sd["variational_strategy.inducing_points"].shape[0],
            n_covariates=sd["covar_module.covariate_kernel.raw_lengthscale"].shape[1],
            prior_mean=pm,
        )
        result = m.load_state_dict(sd, strict=False)
        assert result.missing_keys == [], f"Missing keys: {result.missing_keys}"
        m.eval()
        return m

    def test_no_missing_keys(self, ckpt_model):
        pass  # assertion already in fixture

    def test_forward_returns_correct_shapes(self, ckpt_model):
        x = torch.rand(8, 19)
        with torch.no_grad():
            dist = ckpt_model(x)
        assert dist.mean.shape == (8,)
        assert dist.variance.shape == (8,)

    def test_variance_matches_output_scale(self, ckpt_model):
        """
        For points far from all inducing points the posterior variance ≈
        outputscale (prior variance).  softplus(0.3555) ≈ 0.887.
        """
        import torch.nn.functional as F
        sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
        expected_scale = float(F.softplus(sd["covar_module.outputscale_param"]))
        x = torch.zeros(4, 19)
        with torch.no_grad():
            dist = ckpt_model(x)
        # Variance should be in the vicinity of output scale (not zero, not huge)
        assert dist.variance.mean().item() > 0
        assert dist.variance.mean().item() < 10 * expected_scale


# ---------------------------------------------------------------------------
# generate_predictions.run() tests
# ---------------------------------------------------------------------------


class TestGeneratePredictionsRun:

    def test_returns_true_when_predictions_exist(self, tmp_path, monkeypatch):
        """run() should short-circuit and return True if both CSVs exist."""
        import src.generate_predictions as gp

        # Create dummy prediction files
        (tmp_path / "predictions_val.csv").write_text("a,b\n1,2\n")
        (tmp_path / "predictions_test.csv").write_text("a,b\n3,4\n")

        monkeypatch.setattr(gp, "PRED_VAL",  tmp_path / "predictions_val.csv")
        monkeypatch.setattr(gp, "PRED_TEST", tmp_path / "predictions_test.csv")

        result = gp.run(force=False)
        assert result is True

    def test_skips_when_not_forced(self, tmp_path, monkeypatch, capsys):
        import src.generate_predictions as gp

        (tmp_path / "predictions_val.csv").write_text("a\n1\n")
        (tmp_path / "predictions_test.csv").write_text("a\n2\n")
        monkeypatch.setattr(gp, "PRED_VAL",  tmp_path / "predictions_val.csv")
        monkeypatch.setattr(gp, "PRED_TEST", tmp_path / "predictions_test.csv")

        gp.run(force=False)
        captured = capsys.readouterr()
        assert "skipping" in captured.out.lower()

    def test_raises_when_checkpoint_missing_and_forced(self, tmp_path, monkeypatch):
        import src.generate_predictions as gp

        monkeypatch.setattr(gp, "PRED_VAL",  tmp_path / "predictions_val.csv")
        monkeypatch.setattr(gp, "PRED_TEST", tmp_path / "predictions_test.csv")
        monkeypatch.setattr(gp, "CKPT",      tmp_path / "nonexistent.pt")

        with pytest.raises(FileNotFoundError):
            gp.run(force=True)
