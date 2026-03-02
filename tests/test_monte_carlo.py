# pip install gpytorch torch pandas numpy

from __future__ import annotations
import math
import numpy as np
import pandas as pd
import torch
import gpytorch

torch.set_default_dtype(torch.float64)


# -----------------------------
# 1) Your data -> tensors
# -----------------------------
def make_tensors_from_dataframe(
    df: pd.DataFrame,
    *,
    space_col: str,
    time_col: str,
    y_col: str,
    source_col: str | None = None,
    source_noise_var: dict | None = None,
):
    """
    df must contain:
      - space_col: a numeric spatial coordinate (or feature)
      - time_col:  a numeric time coordinate (e.g., hours since start)
      - y_col:     observed value
      - source_col (optional): categorical source label

    If source_col is given and source_noise_var is provided, we use fixed per-source noise variances.
    Otherwise, we use a single learnable noise term (homoscedastic Gaussian likelihood).
    """
    # Basic numeric arrays
    s = df[space_col].to_numpy(dtype=float)
    t = df[time_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    # X: (n, 2) with columns [space, time]
    X = np.column_stack([s, t])

    # Standardize inputs (recommended for stable lengthscales)
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True)
    X_std[X_std == 0] = 1.0
    Xz = (X - X_mean) / X_std

    # Standardize outputs too (optional but often helps optimization)
    y_mean = y.mean()
    y_std = y.std() if y.std() > 0 else 1.0
    yz = (y - y_mean) / y_std

    X_train = torch.tensor(Xz)
    y_train = torch.tensor(yz)

    if source_col is None:
        return (X_train, y_train, None, None, (X_mean, X_std, y_mean, y_std))

    src = df[source_col].astype("category")
    src_codes = torch.tensor(src.cat.codes.to_numpy(), dtype=torch.long)
    src_levels = list(src.cat.categories)

    if source_noise_var is None:
        # user can still use learnable noise; we just return src info for later use if desired
        return (X_train, y_train, src_codes, src_levels, (X_mean, X_std, y_mean, y_std))

    # Build fixed per-observation noise variance vector
    # noise variances must be provided in ORIGINAL y-units; we scale them into standardized y-units.
    noise_var = np.empty(len(df), dtype=float)
    for i, lab in enumerate(src_levels):
        if lab not in source_noise_var:
            raise ValueError(f"Missing noise variance for source '{lab}' in source_noise_var.")
        noise_var[src.cat.codes.to_numpy() == i] = source_noise_var[lab]

    # Convert noise variance to standardized-y scale: Var((y - mean)/std) = Var(y)/std^2
    noise_var_z = noise_var / (y_std**2)
    noise_var_t = torch.tensor(noise_var_z, dtype=X_train.dtype)

    return (X_train, y_train, src_codes, src_levels, (X_mean, X_std, y_mean, y_std), noise_var_t)


# -----------------------------
# 2) Separable GP model
# -----------------------------
class SeparableSTExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        # Separable kernel: k = k_space * k_time
        # Here, both are RBF; you can swap to MaternKernel, etc.
        k_space = gpytorch.kernels.RBFKernel(active_dims=(0,))
        k_time = gpytorch.kernels.RBFKernel(active_dims=(1,))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.ProductKernel(k_space, k_time)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        cov_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, cov_x)


# -----------------------------
# 3) Train + predict
# -----------------------------
def fit_gp(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    noise_var: torch.Tensor | None = None,   # fixed noise per point if provided
    training_iter: int = 200,
    lr: float = 0.1,
    seed: int = 0,
):
    torch.manual_seed(seed)

    if noise_var is None:
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
    else:
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=noise_var,
            learn_additional_noise=False,
        )

    model = SeparableSTExactGP(X_train, y_train, likelihood)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()

    return model, likelihood


@torch.no_grad()
def predict_gp(
    model,
    likelihood,
    X_test: torch.Tensor,
):
    model.eval()
    likelihood.eval()

    pred_f = model(X_test)                    # latent f
    pred_y = likelihood(pred_f)               # predictive y (adds observation noise if GaussianLikelihood)

    return {
        "f_mean": pred_f.mean,
        "f_var": pred_f.variance,
        "y_mean": pred_y.mean,
        "y_var": pred_y.variance,
    }


# -----------------------------
# 4) Example "run it" script
# -----------------------------
if __name__ == "__main__":
    # Example expected dataframe columns:
    #   - space: numeric coordinate (or you can use one projected coordinate; for 2D see note below)
    #   - time: numeric (e.g., hours since first timestamp)
    #   - value: observation
    #   - source: label like "sat", "lcs", "ref" (optional)
    #
    # Replace this with: df = pd.read_csv("your_data.csv") etc.
    n = 300
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "space": rng.uniform(0, 10, n),
        "time":  rng.uniform(0, 48, n),
        "value": np.nan,
        "source": rng.choice(["lcs", "ref", "sat"], size=n, p=[0.6, 0.2, 0.2]),
    })
    # synthetic signal
    f = np.sin(df["space"].to_numpy() / 2.0) + 0.3 * np.cos(df["time"].to_numpy() / 3.0)
    noise = np.where(df["source"].to_numpy() == "ref", 0.05,
             np.where(df["source"].to_numpy() == "sat", 0.15, 0.25))
    df["value"] = f + noise * rng.normal(size=n)

    # Provide per-source noise variances (in original y units^2) if you want "fusion-style" known errors.
    # If you don't know these, omit source_noise_var and the model learns one global noise.
    source_noise_var = {"ref": 0.05**2, "sat": 0.15**2, "lcs": 0.25**2}

    X_train, y_train, src_id, src_levels, scaler, noise_var = make_tensors_from_dataframe(
        df,
        space_col="space",
        time_col="time",
        y_col="value",
        source_col="source",
        source_noise_var=source_noise_var,
    )

    model, likelihood = fit_gp(X_train, y_train, noise_var=noise_var, training_iter=200, lr=0.1)

    # Build a prediction grid
    space_grid = np.linspace(df["space"].min(), df["space"].max(), 80)
    time_grid = np.linspace(df["time"].min(), df["time"].max(), 80)
    X_test_raw = np.column_stack([space_grid, time_grid])

    # Apply same standardization used for training X
    X_mean, X_std, y_mean, y_std = scaler
    X_test = torch.tensor((X_test_raw - X_mean) / X_std)

    pred = predict_gp(model, likelihood, X_test)

    # Convert predictions back to original y-units
    yhat = (pred["y_mean"].cpu().numpy() * y_std) + y_mean
    ysd = (np.sqrt(pred["y_var"].cpu().numpy()) * y_std)

    print("First 5 predictions (mean, sd):")
    for i in range(5):
        print(float(yhat[i]), float(ysd[i]))