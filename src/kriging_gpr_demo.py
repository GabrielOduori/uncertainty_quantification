"""
Kriging / Gaussian Process Regression Demo
==========================================

Implements the Kriging predictor as defined in:
  Sacks et al. (1989); Santner et al. (2003); Rasmussen & Williams (2006)

GPR surrogate model (Sacks et al., 1989, Eq. 1):
    Y(x, ω) = F(x) + Z(x, ω)
    F(x) = β^T f(x)               — regression trend
    E[Z(x₁)Z(x₂)] = σ²_Z R(x₁,x₂;θ)  — GP covariance

Kriging predictor (Sacks et al., 1989):
    ŷ(x) = f(x)^T β̂ + r(x)^T R⁻¹ (Y − F β̂)
    σ²_ŷ(x) = σ²_Z (1 − r(x)^T R⁻¹ r(x) + t(x)^T (F^T R⁻¹ F)⁻¹ t(x))
    t(x) = F^T R⁻¹ r(x) − f(x)

Hyperparameters: preferred method is LOO cross-validation (Bachoc, 2013)
    θ̂ = argmin ε_LOO(θ | Y)
Noiseless MLE fallback (Sacks et al., 1989):
    β̂ = (F^T R⁻¹ F)⁻¹ F^T R⁻¹ Y
    σ̂²_Z = (1/n)(Y − Fβ̂)^T R⁻¹ (Y − Fβ̂)

Noisy case (Kennedy & O'Hagan, 2001; Rasmussen & Williams, 2006):
    y = h(x) + ε,  ε ~ N(0, σ²_n)
    R̃ = (1−τ)R + τI  (modified correlation matrix)

Outputs:
    output/kriging_gpr_demo/<timestamp>/
        gpr_kriging_panel.png   — 4-panel visualisation
        gpr_temporal_uq.png     — temporal UQ from Day 1 to Day 29
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import solve, cholesky
from scipy.optimize import minimize

ROOT_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Correlation functions  R(x₁, x₂; θ)
# ---------------------------------------------------------------------------

def corr_gaussian(x1: float, x2: float, theta: float) -> float:
    """Gaussian (squared-exponential) correlation: exp(−(x₁−x₂)²/(2θ²))"""
    return float(np.exp(-((x1 - x2) ** 2) / (2 * theta ** 2)))


def corr_matrix(X: np.ndarray, theta: float, noise_ratio: float = 0.0) -> np.ndarray:
    """Build n×n correlation matrix R̃ = R + τI for noisy GPR (Kennedy & O'Hagan, 2001)."""
    n = len(X)
    R = np.array([[corr_gaussian(X[i], X[j], theta) for j in range(n)] for i in range(n)])
    return R + noise_ratio * np.eye(n)   # R̃ = R + τI


def corr_vector(x: float, X: np.ndarray, theta: float) -> np.ndarray:
    """Build n-vector r(x) of correlations between x and training points."""
    return np.array([corr_gaussian(x, xi, theta) for xi in X])


# ---------------------------------------------------------------------------
# Kriging predictor  (JHU Eqs. 6-8)
# ---------------------------------------------------------------------------

def kriging_fit(X: np.ndarray, Y: np.ndarray, theta: float,
                noise_ratio: float = 0.0) -> dict:
    """
    Fit Kriging / GPR model for given θ (Sacks et al., 1989).

    Constant mean (Ordinary Kriging): f(x) = 1,  F = [1, 1, ..., 1]^T
    θ should be selected by LOO-CV via _loo_cv_theta() before calling this.

    Returns dict with beta_hat, sigma2_z, R_inv, model params.
    """
    n  = len(X)
    F  = np.ones((n, 1))          # constant trend basis (f(x) = 1)
    R  = corr_matrix(X, theta, noise_ratio)
    R_inv = np.linalg.inv(R)

    # β̂ = (F^T R⁻¹ F)⁻¹ F^T R⁻¹ Y   (Sacks et al., 1989)
    FtRiF = F.T @ R_inv @ F
    beta_hat = np.linalg.solve(FtRiF, F.T @ R_inv @ Y)

    # σ̂²_Z = (1/n)(Y − Fβ̂)^T R⁻¹ (Y − Fβ̂)   (Sacks et al., 1989)
    resid    = Y - (F @ beta_hat).ravel()
    sigma2_z = float((resid @ R_inv @ resid) / n)

    return dict(X=X, Y=Y, F=F, R_inv=R_inv, FtRiF=FtRiF,
                beta_hat=beta_hat, sigma2_z=sigma2_z, theta=theta)


def kriging_predict(x_pred: np.ndarray, model: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Kriging predictor at new points x_pred (JHU Eqs. 6-8).

    Returns:
        mu    — posterior mean ŷ(x)
        sigma — posterior std  σ_ŷ(x)
    """
    X, F, R_inv = model["X"], model["F"], model["R_inv"]
    FtRiF  = model["FtRiF"]
    beta   = model["beta_hat"]
    sz2    = model["sigma2_z"]
    theta  = model["theta"]
    Y      = model["Y"]

    mu    = np.zeros(len(x_pred))
    var   = np.zeros(len(x_pred))
    resid = Y - (F @ beta).ravel()

    for i, x in enumerate(x_pred):
        r  = corr_vector(x, X, theta)           # r(x) — correlation with training pts
        f  = np.array([1.0])                     # f(x) — trend basis at x

        # ŷ(x) = f^T β̂ + r^T R⁻¹ (Y − Fβ̂)   (Sacks et al., 1989)
        mu[i] = float(f @ beta) + float(r @ R_inv @ resid)

        # t(x) = F^T R⁻¹ r(x) − f(x)          (Sacks et al., 1989)
        t = (F.T @ R_inv @ r).ravel() - f

        # σ²_ŷ(x) = σ²_Z (1 − r^T R⁻¹ r + t^T (F^T R⁻¹ F)⁻¹ t)  (Sacks et al., 1989)
        reduction = float(r @ R_inv @ r)
        extra     = float(t @ np.linalg.solve(FtRiF, t))
        var[i]    = sz2 * max(0.0, 1.0 - reduction + extra)

    return mu, np.sqrt(var)


def prior_samples(x_grid: np.ndarray, theta: float, sigma2_z: float,
                  n_samples: int = 5, seed: int = 0) -> np.ndarray:
    """Draw sample paths from the GP prior N(0, σ²_Z R)."""
    rng = np.random.default_rng(seed)
    R   = corr_matrix(x_grid, theta)
    L   = np.linalg.cholesky(R + 1e-8 * np.eye(len(x_grid)))
    return (L @ rng.standard_normal((len(x_grid), n_samples))).T * np.sqrt(sigma2_z)


def posterior_samples(x_grid: np.ndarray, model: dict, n_samples: int = 5,
                      seed: int = 42) -> np.ndarray:
    """Draw sample paths from the GP posterior."""
    rng   = np.random.default_rng(seed)
    mu, _ = kriging_predict(x_grid, model)

    # Build posterior covariance matrix at grid points
    n = len(x_grid)
    cov = np.zeros((n, n))
    X, R_inv, F = model["X"], model["R_inv"], model["F"]
    FtRiF = model["FtRiF"]
    sz2   = model["sigma2_z"]
    theta = model["theta"]

    for i in range(n):
        ri = corr_vector(x_grid[i], X, theta)
        ti = (F.T @ R_inv @ ri).ravel() - np.array([1.0])
        for j in range(i, n):
            rj  = corr_vector(x_grid[j], X, theta)
            tj  = (F.T @ R_inv @ rj).ravel() - np.array([1.0])
            rij = corr_gaussian(x_grid[i], x_grid[j], theta)
            c   = sz2 * (rij - ri @ R_inv @ rj
                         + ti @ np.linalg.solve(FtRiF, tj))
            cov[i, j] = cov[j, i] = c

    cov += 1e-8 * np.eye(n)
    L    = np.linalg.cholesky(cov)
    z    = rng.standard_normal((n, n_samples))
    return (mu[:, None] + L @ z).T


# ---------------------------------------------------------------------------
# LOO cross-validation theta selection  (Bachoc, 2013)
# ---------------------------------------------------------------------------

def _loo_cv_theta(X: np.ndarray, Y: np.ndarray,
                  noise_ratio: float = 0.0,
                  theta_candidates: np.ndarray | None = None) -> float:
    """
    Select correlation length θ by leave-one-out cross-validation.

    For each candidate θ, fits the model on n-1 points and predicts the
    held-out point; returns the θ that minimises mean LOO squared error.

    Reference: Bachoc (2013), Santner et al. (2003).
    """
    if theta_candidates is None:
        theta_candidates = np.logspace(-2, 1, 30)
    if len(X) < 3:
        return float(theta_candidates[len(theta_candidates) // 2])

    best_err, best_th = np.inf, float(theta_candidates[len(theta_candidates) // 2])
    for th in theta_candidates:
        errs = []
        for k in range(len(X)):
            mask = np.arange(len(X)) != k
            if mask.sum() < 2:
                continue
            model_k = kriging_fit(X[mask], Y[mask], th, noise_ratio)
            mu_k, _ = kriging_predict(np.array([X[k]]), model_k)
            errs.append((Y[k] - mu_k[0]) ** 2)
        loo = float(np.mean(errs)) if errs else np.inf
        if loo < best_err:
            best_err, best_th = loo, float(th)
    return best_th


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_kriging_panel(output_dir: Path) -> None:
    """
    Four-panel Kriging visualisation (JHU-style).

    Panel 1: GP prior samples
    Panel 2: GP posterior (JHU slide 8) — ŷ(x) ± 2σ, sample paths, obs
    Panel 3: Posterior variance σ²_ŷ(x) — shows zero at observation points
    Panel 4: Effect of correlation length θ on posterior uncertainty
    """

    # ── Setup: 1D problem with two noisy observations ──────────────────────
    x_obs  = np.array([1.2, 2.0])
    y_obs  = np.array([-1.0, 0.2])
    x_grid = np.linspace(0, 6, 300)
    theta  = 1.0       # correlation length
    sigma2 = 1.0       # GP variance

    model_noiseless = kriging_fit(x_obs, y_obs, theta, noise_ratio=0.0)
    model_noisy     = kriging_fit(x_obs, y_obs, theta, noise_ratio=0.10)

    mu_nl, sig_nl = kriging_predict(x_grid, model_noiseless)
    mu_n,  sig_n  = kriging_predict(x_grid, model_noisy)

    COLORS = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # ── Panel 1: GP Prior samples ──────────────────────────────────────────
    ax = axes[0, 0]
    prior_s = prior_samples(x_grid, theta, sigma2, n_samples=5, seed=7)
    prior_mu  = np.zeros(len(x_grid))
    prior_sig = np.sqrt(sigma2 * np.diag(corr_matrix(x_grid, theta)))
    ax.fill_between(x_grid, prior_mu - 2*prior_sig, prior_mu + 2*prior_sig,
                    alpha=0.15, color="steelblue", label="Prior ±2σ")
    for k, s in enumerate(prior_s):
        ax.plot(x_grid, s, color=COLORS[k], lw=1.5, alpha=0.85)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set(title="GP Prior  f(·) ~ GP(0, σ²_Z R(·,·;θ))",
           xlabel="x", ylabel="f(x)", xlim=(0, 6))
    ax.text(0.05, 0.92, f"θ = {theta},  σ²_Z = {sigma2}",
            transform=ax.transAxes, fontsize=9, color="grey")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

    # ── Panel 2: GP Posterior (JHU slide 8 style) ─────────────────────────
    ax = axes[0, 1]
    post_s = posterior_samples(x_grid, model_noiseless, n_samples=5, seed=42)
    ax.fill_between(x_grid, mu_nl - 2*sig_nl, mu_nl + 2*sig_nl,
                    alpha=0.20, color="steelblue", label="Posterior ±2σ")
    ax.fill_between(x_grid, mu_nl - sig_nl,   mu_nl + sig_nl,
                    alpha=0.30, color="steelblue")
    for k, s in enumerate(post_s):
        ax.plot(x_grid, s, color=COLORS[k], lw=1.5, alpha=0.85)
    ax.plot(x_grid, mu_nl, "b-", lw=2.0, label="Posterior mean ŷ(x)")
    # Observation lines (JHU slide 8 style — vertical bars)
    for xi, yi in zip(x_obs, y_obs):
        ax.axvline(xi, color="grey", lw=0.8, ls=":")
        ax.plot(xi, yi, "ko", ms=8, zorder=10)
    ax.set(title="GP Posterior  Y(x)|X,Y ~ N(ŷ(x), σ²_ŷ(x))",
           xlabel="x", ylabel="f(x)", xlim=(0, 6))
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

    # ── Panel 3: Posterior variance σ²_ŷ(x) ──────────────────────────────
    ax = axes[1, 0]
    var_nl = sig_nl ** 2
    var_n  = sig_n  ** 2
    ax.plot(x_grid, var_nl, "b-",  lw=2, label="Noiseless (Kriging interpolant)")
    ax.plot(x_grid, var_n,  "r--", lw=2, label="Noisy (τ = 0.10)")
    ax.fill_between(x_grid, 0, var_nl, alpha=0.12, color="steelblue")
    for xi in x_obs:
        ax.axvline(xi, color="grey", lw=0.8, ls=":", label="_")
    ax.set(title="Kriging Variance  σ²_ŷ(x) = σ²_Z(1 − r^T R⁻¹r + t^T(F^TR⁻¹F)⁻¹t)",
           xlabel="x", ylabel="σ²_ŷ(x)",
           xlim=(0, 6))
    ax.text(0.38, 0.88, "σ² → 0 at\nobservation\npoints",
            transform=ax.transAxes, fontsize=8, ha="center",
            color="steelblue",
            bbox=dict(fc="white", ec="steelblue", alpha=0.7, boxstyle="round"))
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

    # ── Panel 4: Effect of correlation length θ ────────────────────────────
    ax = axes[1, 1]
    for theta_k, color, ls in [(0.3, "#e41a1c", "--"),
                                 (1.0, "#377eb8", "-"),
                                 (3.0, "#4daf4a", ":")]:
        m_k     = kriging_fit(x_obs, y_obs, theta_k, noise_ratio=0.0)
        mu_k, sig_k = kriging_predict(x_grid, m_k)
        ax.fill_between(x_grid, mu_k - 2*sig_k, mu_k + 2*sig_k,
                        alpha=0.10, color=color)
        ax.plot(x_grid, mu_k, color=color, lw=2, ls=ls,
                label=f"θ = {theta_k}")
    ax.plot(x_obs, y_obs, "ko", ms=8, zorder=10, label="Observations")
    ax.set(title="Effect of Correlation Length θ on ŷ(x)",
           xlabel="x", ylabel="ŷ(x)", xlim=(0, 6))
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Kriging — GPR for Surrogate Uncertainty Propagation\n"
        "Y(x,ω) = F(x) + Z(x,ω),   F(x) = β^T f(x),   "
        "E[Z(x₁)Z(x₂)] = σ²_Z R(x₁,x₂;θ)\n"
        "(Sacks et al., 1989;  Rasmussen & Williams, 2006)",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    out = output_dir / "gpr_kriging_panel.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kriging panel → {out}")


def plot_temporal_gpr(output_dir: Path) -> None:
    """
    Temporal GPR: fit on synthetic daily NO₂ values (Day 1 → Day 29)
    to show how the Kriging surrogate propagates uncertainty over time.

    Mimics the structure of the Dublin problem:
      x = day (normalised to [0,1])
      y = observed NO₂ at sparse monitoring stations
      Predict over full temporal grid with uncertainty bands
    """
    rng = np.random.default_rng(0)

    # Simulate 6 EPA-like sparse observations scattered over 29 days
    # (in normalised time units matching uq_test.csv)
    t_obs = np.array([0.0, 0.107, 0.25, 0.50, 0.75, 1.0])
    # True underlying signal (unknown — what we're predicting)
    true_signal = lambda t: 28 + 4 * np.sin(2 * np.pi * t) - 2 * t
    y_obs = true_signal(t_obs) + rng.normal(0, 2.5, size=len(t_obs))

    t_grid = np.linspace(0, 1, 300)
    days   = np.round(t_grid * 28).astype(int) + 1   # Day 1 to Day 29

    # Select θ by LOO-CV (Bachoc, 2013) — preferred over fixed MLE value
    noise_ratio = 0.05
    theta_opt = _loo_cv_theta(t_obs, y_obs, noise_ratio=noise_ratio)
    print(f"  LOO-CV selected θ = {theta_opt:.4f}")
    model_gpr = kriging_fit(t_obs, y_obs, theta=theta_opt, noise_ratio=noise_ratio)
    mu, sig   = kriging_predict(t_grid, model_gpr)
    sz2 = model_gpr["sigma2_z"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    # ── Panel 1: GPR fit with uncertainty bands ────────────────────────────
    ax = axes[0]
    post_s = posterior_samples(t_grid, model_gpr, n_samples=5, seed=99)
    for s in post_s:
        ax.plot(days, s, lw=1.0, alpha=0.55)
    ax.fill_between(days, mu - 2*sig, mu + 2*sig,
                    alpha=0.20, color="steelblue", label="Posterior ±2σ (95%)")
    ax.fill_between(days, mu - sig, mu + sig,
                    alpha=0.30, color="steelblue", label="Posterior ±1σ (68%)")
    ax.plot(days, mu, "b-", lw=2.5, label="GP surrogate mean ŷ(t)")
    ax.plot(np.round(t_obs * 28).astype(int) + 1, y_obs,
            "ko", ms=8, zorder=10, label="EPA observations")
    ax.axhline(25, color="red", lw=1, ls="--", label="WHO daily (25 µg/m³)")
    ax.set(title=f"GP Surrogate for Temporal Uncertainty Propagation\n"
                  f"Y(t,ω) = F(t) + Z(t,ω),  θ = {theta_opt:.3f} (LOO-CV)",
           xlabel="Day (Jun 1 → Jun 29)", ylabel="NO₂ (µg/m³)")
    ax.legend(fontsize=9, loc="upper right"); ax.grid(True, alpha=0.3)

    # ── Panel 2: Posterior variance σ²_ŷ(t) over time ────────────────────
    ax = axes[1]
    ax.fill_between(days, 0, sig**2, alpha=0.3, color="steelblue")
    ax.plot(days, sig**2, "b-", lw=2, label="σ²_ŷ(t)  (Sacks et al., 1989)")
    ax.plot(days, sig,    "g--", lw=1.5, label="σ_ŷ(t) = √σ²_ŷ(t)")
    obs_days = np.round(t_obs * 28).astype(int) + 1
    for od in obs_days:
        ax.axvline(od, color="grey", lw=0.8, ls=":", alpha=0.7)
    ax.text(0.5, 0.85,
            "Uncertainty collapses to 0\nat observation points\n"
            "(Kriging interpolation property)",
            transform=ax.transAxes, fontsize=9, ha="center",
            bbox=dict(fc="white", ec="steelblue", alpha=0.8, boxstyle="round"))
    ax.set(title="Kriging Variance σ²_ŷ(t) — Day 1 to Day 29\n"
                  "σ²_ŷ(t) = σ²_Z (1 − r(t)^T R⁻¹ r(t) + t(t)^T (F^T R⁻¹ F)⁻¹ t(t))",
           xlabel="Day (Jun 1 → Jun 29)", ylabel="Variance / Std")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.suptitle(
        "GPR Temporal Uncertainty Propagation — Dublin NO₂ (Jun 2023)\n"
        "Surrogate model: ŷ(t) and σ²_ŷ(t) quantify prediction uncertainty at each day",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    out = output_dir / "gpr_temporal_uq.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Temporal GPR → {out}")


# ---------------------------------------------------------------------------
# Callable entry point (used by run.py)
# ---------------------------------------------------------------------------

def run(tables_dir: Path, figures_dir: Path) -> dict:
    """
    Run Kriging / GPR demo figures.

    Args:
        tables_dir: Directory for CSV outputs (none produced by this stage).
        figures_dir: Directory for PNG outputs.

    Returns:
        dict with status.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Kriging / GPR Demo")
    print("=" * 60)

    print("\n[1] Kriging panel (prior, posterior, variance, θ effect)")
    plot_kriging_panel(figures_dir)

    print("\n[2] Temporal GPR (Day 1 → Day 29)")
    plot_temporal_gpr(figures_dir)

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Main (standalone)
# ---------------------------------------------------------------------------

def main():
    run_ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT_DIR / "output" / "kriging_gpr_demo" / run_ts
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")
    run(tables_dir=output_dir, figures_dir=output_dir)


if __name__ == "__main__":
    main()
