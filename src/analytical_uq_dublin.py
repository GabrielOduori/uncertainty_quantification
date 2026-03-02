"""
GP Surrogate Uncertainty Quantification — Dublin NO₂
=====================================================

Methodology (JHU Module 3.4 — Surrogate Models for Uncertainty Propagation)
---------------------------------------------------------------------------
The NO₂ field is modelled as a Gaussian Process surrogate of the form:

    Y(x, ω) = F(x) + Z(x, ω)                                    [Eq. 1]

where
  • F(x) = β^T f(x)  is the regression trend (here: LUR-based GridPriorMean)
  • Z(x, ω)          is a zero-mean GP with covariance
                     E[Z(x₁)Z(x₂)] = σ²_Z · R(x₁, x₂; θ)      [Eq. 3]
  • R(·,·; θ)        is the separable spatiotemporal kernel:
                     k_spatial (Matérn-ARD, ℓ_s ≈ 0.54)
                     × k_temporal (Exponential, ℓ_t ≈ 0.15)
                     × k_covariate (ARD, ℓ_c ≈ 0.47)

Multi-source heteroscedastic noise (noisy GPR, JHU §2.2):

    y_src = h(x) + ε_src,  ε_src ~ N(0, σ²_src)                [Eq. 19]

    σ²_EPA ≈ 0.24 µg²/m⁶  (learned via MLE, raw = −1.22)
    σ²_sat ≈ 9.00 µg²/m⁶  (learned via MLE, raw = +1.10)

The Kriging predictor conditional distribution is (JHU Eqs. 5–7):

    Y(x) | X, Y ~ N( ŷ(x), σ²_ŷ(x) )

    ŷ(x)     = f(x)^T β + r(x)^T R⁻¹ (Y − Fβ)
    σ²_ŷ(x)  = σ²_Z (1 − r(x)^T R⁻¹ r(x) + t(x)^T (F^T R⁻¹ F)⁻¹ t(x))

In practice, the full O(n³) inversion is replaced by a Sparse Variational
GP (SVGP) with 300 inducing points, giving the same posterior form at
O(m²n) cost where m = 300 ≪ n.

Temporal uncertainty propagation: pred_mean μ*(x,t) and pred_std σ*(x,t)
are evaluated for all 29 daily time steps (Jun 1–29 2023), then aggregated
spatially to give daily uncertainty envelopes.

Inputs (data/):
  predictions_val.csv   — GP posterior on val set (conformal calibration)
  predictions_test.csv  — GP posterior on test set (evaluation)

References:
  Rasmussen, C.E. & Williams, C.K.I. (2006). Gaussian Processes for
    Machine Learning. MIT Press.
  Sacks, J. et al. (1989). Design and analysis of computer experiments.
    Statist. Sci., 4(4), 409-423.  [Kriging for surrogate modelling]
  Angelopoulos, A. & Bates, S. (2021). A Gentle Introduction to Conformal
    Prediction and Distribution-Free Uncertainty Quantification.
    arXiv:2107.07511.
  Gneiting, T. & Raftery, A.E. (2007). Strictly proper scoring rules,
    prediction, and estimation. J. Amer. Stat. Assoc., 102(477), 359-378.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR  = Path(__file__).resolve().parent.parent / "data"
ROOT_DIR  = Path(__file__).resolve().parent.parent

# WHO/EU NO₂ thresholds (µg/m³)
WHO_ANNUAL = 10.0
WHO_DAILY  = 25.0
EU_ANNUAL  = 40.0

# Date range of the study period
DATE_START = datetime(2023, 6, 1)
DATE_END   = datetime(2023, 6, 29)
N_DAYS     = 29   # timestamps 0, 1/28, …, 28/28


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ts_to_date(ts: float) -> datetime:
    """Map normalised timestamp ∈ [0,1] to actual date."""
    day_idx = int(round(ts * (N_DAYS - 1)))
    return DATE_START + timedelta(days=day_idx)


def calibration_metrics(y_true: np.ndarray,
                        mu: np.ndarray,
                        sigma: np.ndarray | float) -> dict:
    """PICP, ECE, CRPS, NLL for a Gaussian predictive distribution.

    Reference: Gneiting & Raftery (2007).
    """
    sig = np.full_like(y_true, float(sigma)) if np.isscalar(sigma) else np.asarray(sigma, float)
    results: dict = {}

    # alpha = tail probability (significance level); label = nominal coverage %
    # 95% PI: alpha=0.05, z=1.96; 90% PI: alpha=0.10, z=1.645; 50% PI: alpha=0.50, z=0.674
    for alpha, label in [(0.50, 50), (0.10, 90), (0.05, 95)]:
        z   = stats.norm.ppf(1 - alpha / 2)
        lo  = mu - z * sig
        hi  = mu + z * sig
        cov = float(np.mean((y_true >= lo) & (y_true <= hi)))
        results[f"coverage_{label}"] = cov
        results[f"ece_{label}"]      = abs(cov - (1 - alpha))  # 1-alpha = nominal coverage

    z    = (y_true - mu) / np.maximum(sig, 1e-8)
    crps = sig * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    results["crps"]      = float(np.mean(crps))
    results["nll"]       = float(np.mean(0.5 * np.log(2 * np.pi * sig**2) + 0.5 * z**2))
    results["sharpness"] = float(np.mean(sig))
    results["rmse"]      = float(np.sqrt(np.mean((y_true - mu) ** 2)))
    results["bias"]      = float(np.mean(mu - y_true))
    results["mae"]       = float(np.mean(np.abs(y_true - mu)))
    return results


def crps_gaussian(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Pointwise CRPS for a Gaussian predictive distribution."""
    sig = np.maximum(sigma, 1e-8)
    z   = (y_true - mu) / sig
    return sig * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def conformal_calibrate(cal_y: np.ndarray,
                        cal_mu: np.ndarray,
                        cal_sigma: np.ndarray,
                        alpha: float = 0.05) -> float:
    """Split conformal quantile q̂  s.t. P(|y − μ| ≤ q̂·σ) ≥ 1−α.

    Reference: Angelopoulos & Bates (2021).
    """
    n      = len(cal_y)
    scores = np.abs(cal_y - cal_mu) / np.maximum(cal_sigma, 1e-8)
    level  = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level))


def exceedance_prob(mu: np.ndarray, sigma: np.ndarray, threshold: float) -> np.ndarray:
    """P(NO₂ > threshold) = 1 − Φ((threshold − μ) / σ)."""
    return 1 - stats.norm.cdf((threshold - mu) / np.maximum(sigma, 1e-8))


# ===========================================================================
# 1. Load GP predictions
# ===========================================================================

def load_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load GP posterior predictions for val and test sets."""
    val  = pd.read_csv(DATA_DIR / "predictions_val.csv")
    test = pd.read_csv(DATA_DIR / "predictions_test.csv")

    for df in [val, test]:
        df["date"] = pd.to_datetime(
            df["timestamp"].apply(ts_to_date)
        )

    print(f"GP predictions loaded:")
    print(f"  Val:  {len(val):,} rows  |  dates: {val['date'].min().date()} → {val['date'].max().date()}")
    print(f"  Test: {len(test):,} rows  |  dates: {test['date'].min().date()} → {test['date'].max().date()}")
    print(f"  Test sources: {test['source'].value_counts().to_dict()}")
    return val, test


# ===========================================================================
# 2. Conformal recalibration on GP intervals
# ===========================================================================

def conformal_recalibrate(val: pd.DataFrame) -> tuple[float, float]:
    """Use satellite val set to calibrate GP prediction intervals.

    The GP posterior σ*(x) is already a probabilistic estimate, but it may
    be over- or under-dispersed. Conformal prediction scales σ by a data-
    driven factor q̂ to guarantee marginal coverage ≥ 1−α.
    """
    cal = val[val["source"] == "satellite"].copy()
    cal = cal[cal["value"].notna() & cal["pred_std"].notna()]

    y_cal  = cal["value"].to_numpy()
    mu_cal = cal["pred_mean"].to_numpy()
    s_cal  = cal["pred_std"].to_numpy()
    q50 = conformal_calibrate(y_cal, mu_cal, s_cal, alpha=0.50)
    q90 = conformal_calibrate(y_cal, mu_cal, s_cal, alpha=0.10)
    q95 = conformal_calibrate(y_cal, mu_cal, s_cal, alpha=0.05)
    print(f"  n_cal (satellite val): {len(cal):,}")
    print(f"  q̂ (50%): {q50:.4f}×σ_GP   →  conformal PI width = {2*q50 * cal['pred_std'].mean():.2f} µg/m³")
    print(f"  q̂ (90%): {q90:.4f}×σ_GP   →  conformal PI width = {2*q90 * cal['pred_std'].mean():.2f} µg/m³")
    print(f"  q̂ (95%): {q95:.4f}×σ_GP   →  conformal PI width = {2*q95 * cal['pred_std'].mean():.2f} µg/m³")
    return q50, q90, q95


# ===========================================================================
# 3. Temporal UQ sweep  (Day 1 → Day 29)
# ===========================================================================

def temporal_uq(test: pd.DataFrame, q95: float) -> pd.DataFrame:
    """
    For each day t ∈ {Jun 1, …, Jun 29}:

      μ̄(t)   = spatial mean GP prediction over Dublin grid
      σ̄(t)   = spatial mean GP uncertainty (surrogate variance)
      P_exc(t) = fraction of grid cells with P(NO₂ > WHO_daily) > 0.5
      PICP(t)  = empirical coverage on EPA obs that day  (NaN if no EPA)
      CRPS(t)  = mean CRPS on EPA obs that day           (NaN if no EPA)

    The spatial average σ̄(t) reflects how much the GP surrogate uncertainty
    varies through the study period — driven by data density and covariate
    variability.
    """
    lur = test[test["source"] == "lur"].copy()
    epa = test[(test["source"] == "epa") & test["value"].notna()].copy()

    rows = []
    for date, grp in lur.groupby("date"):
        mu  = grp["pred_mean"].to_numpy()
        sig = grp["pred_std"].to_numpy() * q95   # conformal-scaled σ* = q̂95 · σ_GP

        p_exc = exceedance_prob(mu, sig, WHO_DAILY)

        row = {
            "date":       date,
            "n_grid":     len(grp),
            "mean_no2":   float(mu.mean()),
            "sigma_mean": float(sig.mean()),   # conformal effective σ = q̂95 · σ_GP
            "pi95_width": float((2 * sig).mean()),  # conformal PI width = 2 · q̂95 · σ_GP
            "p_exc_who25_frac": float((p_exc > 0.5).mean()),
            "p_exc_who25_mean": float(p_exc.mean()),
        }

        # EPA metrics for this day (if available)
        epa_day = epa[epa["date"] == date]
        if len(epa_day) > 0:
            y    = epa_day["value"].to_numpy()
            mu_e = epa_day["pred_mean"].to_numpy()
            sig_raw_e = epa_day["pred_std"].to_numpy()
            # Conformal PI: mu ± q̂95 · σ  (q̂95 replaces the Gaussian 1.96)
            lo = mu_e - q95 * sig_raw_e
            hi = mu_e + q95 * sig_raw_e
            row["picp_95"]  = float(np.mean((y >= lo) & (y <= hi)))
            row["crps_epa"] = float(crps_gaussian(y, mu_e, sig_raw_e * q95).mean())
            row["n_epa"]    = len(epa_day)
        else:
            row["picp_95"]  = np.nan
            row["crps_epa"] = np.nan
            row["n_epa"]    = 0

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


# ===========================================================================
# 4. Source uncertainty decomposition  (Law of Total Variance)
# ===========================================================================

def uncertainty_decomposition(test: pd.DataFrame) -> dict:
    """
    Law of Total Variance across observation sources:

      Var[Y] = E[Var[Y|source]]  +  Var[E[Y|source]]
                ─────────────────     ────────────────
                within (aleatoric)     between (epistemic)
    """
    stats_: dict = {}
    for src in ["epa", "satellite", "lur"]:
        v = test[test["source"] == src]["value"].dropna()
        if len(v) == 0:
            continue
        stats_[src] = {"n": len(v), "mean": float(v.mean()), "std": float(v.std())}

    total_n    = sum(s["n"] for s in stats_.values())
    within_var = sum(s["n"] * s["std"] ** 2 for s in stats_.values()) / total_n
    means      = np.array([s["mean"] for s in stats_.values()])
    between_var = float(np.var(means))
    total_var   = within_var + between_var

    return {
        "source_stats":       stats_,
        "within_var":         float(within_var),
        "between_var":        float(between_var),
        "total_var":          float(total_var),
        "aleatoric_fraction": within_var  / total_var if total_var else 0.0,
        "epistemic_fraction": between_var / total_var if total_var else 0.0,
    }


# ===========================================================================
# 5. Figures
# ===========================================================================

def plot_temporal(tdf: pd.DataFrame, output_dir: Path) -> None:
    """Six-panel temporal UQ figure: Day 1 → Day 29."""
    dates = tdf["date"].dt.to_pydatetime()

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    fmt = mdates.DateFormatter("%b %d")

    # ── Panel 1: Daily mean NO₂ with ±1σ uncertainty band ─────────────────
    ax = axes[0, 0]
    mu  = tdf["mean_no2"].to_numpy()
    sig = tdf["sigma_mean"].to_numpy()
    ax.fill_between(dates, mu - sig, mu + sig, alpha=0.25, color="steelblue", label="±1σ_GP (conformal)")
    ax.fill_between(dates, mu - 2*sig, mu + 2*sig, alpha=0.10, color="steelblue", label="±2σ_GP")
    ax.plot(dates, mu, "b-o", ms=4, lw=1.5, label="GP mean")
    ax.axhline(WHO_DAILY, color="red", lw=1, ls="--", label=f"WHO daily ({WHO_DAILY} µg/m³)")
    ax.set(title="Daily Mean NO₂ ± GP Uncertainty\n(spatial average over Dublin grid)",
           ylabel="NO₂ (µg/m³)")
    ax.xaxis.set_major_formatter(fmt)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ── Panel 2: Daily mean posterior standard deviation ──────────────────
    ax = axes[0, 1]
    ax.plot(dates, sig, "g-o", ms=4, lw=1.5)
    ax.fill_between(dates, sig * 0.95, sig * 1.05, alpha=0.2, color="green")
    ax.set(title="Daily Mean GP Uncertainty σ̄(t)\n(spatial mean of posterior std)",
           ylabel="σ (µg/m³)")
    ax.xaxis.set_major_formatter(fmt)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ── Panel 3: 95% PI width over time ───────────────────────────────────
    ax = axes[1, 0]
    pi_vals = tdf["pi95_width"].to_numpy()
    ax.plot(dates, pi_vals, "m-o", ms=4, lw=1.5)
    ax.set(title="Daily 95% Prediction Interval Width\n(conformal-scaled GP intervals)",
           ylabel="PI width (µg/m³)",
           ylim=(0, max(pi_vals) * 1.25))   # force zero baseline — removes offset notation
    ax.yaxis.get_major_formatter().set_useOffset(False)
    ax.xaxis.set_major_formatter(fmt)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ── Panel 4: Exceedance probability over time ──────────────────────────
    ax = axes[1, 1]
    ax.plot(dates, tdf["p_exc_who25_mean"].to_numpy(), "r-o", ms=4, lw=1.5,
            label="Mean P(NO₂ > 25 µg/m³)")
    ax.plot(dates, tdf["p_exc_who25_frac"].to_numpy(), "r--s", ms=4, lw=1.5,
            label="Frac. cells P > 0.5")
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.set(title=f"Daily Exceedance Probability\nP(NO₂ > {WHO_DAILY} µg/m³ [WHO daily limit])",
           ylabel="Probability / Fraction", ylim=(0, 1))
    ax.xaxis.set_major_formatter(fmt)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ── Panel 5: PICP per day (EPA ground truth) ──────────────────────────
    ax = axes[2, 0]
    epa_mask = tdf["n_epa"] > 0
    ax.scatter(tdf.loc[epa_mask, "date"].dt.to_pydatetime(),
               tdf.loc[epa_mask, "picp_95"].to_numpy(),
               c="steelblue", s=60, zorder=5, label="Observed PICP")
    ax.axhline(0.95, color="black", lw=1.5, ls="--", label="Nominal 95%")
    ax.set(title="Daily PI Coverage (PICP₉₅) vs EPA Ground Truth\n(only days with EPA obs)",
           ylabel="Empirical coverage", ylim=(0, 1.05))
    ax.xaxis.set_major_formatter(fmt)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ── Panel 6: CRPS per day (EPA ground truth) ──────────────────────────
    ax = axes[2, 1]
    ax.scatter(tdf.loc[epa_mask, "date"].dt.to_pydatetime(),
               tdf.loc[epa_mask, "crps_epa"].to_numpy(),
               c="darkorange", s=60, zorder=5)
    ax.set(title="Daily CRPS vs EPA Ground Truth\n(lower = better probabilistic forecast)",
           ylabel="CRPS (µg/m³)")
    ax.xaxis.set_major_formatter(fmt)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.suptitle(
        "GP Surrogate Uncertainty Propagation — Dublin NO₂  (June 2023)\n"
        "FusionSVGP posterior p(f*|X,y,x*) = N(μ*(x*), σ*²(x*))  +  conformal recalibration",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    out = output_dir / "temporal_uq_june2023.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Temporal plot → {out}")


def plot_spatial_panels(test: pd.DataFrame, epa_test: pd.DataFrame,
                        q50: float, q90: float, q95: float,
                        output_dir: Path) -> None:
    """Static spatial/calibration panel figure."""
    epa_gp  = epa_test.copy()
    y_true  = epa_gp["value"].to_numpy()
    mu_gp   = epa_gp["pred_mean"].to_numpy()
    sig_raw = epa_gp["pred_std"].to_numpy()          # raw GP σ
    sig_gp  = sig_raw * q95                          # conformal-scaled: q̂95 · σ

    # Calibration curve — GP raw
    levels, empirical_raw = [], []
    for alpha in np.linspace(0.01, 0.99, 50):
        z   = stats.norm.ppf(1 - alpha / 2)
        cov = np.mean((y_true >= mu_gp - z * epa_gp["pred_std"].to_numpy())
                    & (y_true <= mu_gp + z * epa_gp["pred_std"].to_numpy()))
        levels.append(1 - alpha)
        empirical_raw.append(cov)

    # Calibration curve — GP conformal
    empirical_conf = []
    for alpha in np.linspace(0.01, 0.99, 50):
        z   = stats.norm.ppf(1 - alpha / 2)
        cov = np.mean((y_true >= mu_gp - z * sig_gp) & (y_true <= mu_gp + z * sig_gp))
        empirical_conf.append(cov)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Panel 1: Raw GP calibration
    ax = axes[0, 0]
    ax.plot(levels, empirical_raw,  "o-", ms=3, color="tomato",    label="GP raw")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Ideal")
    ax.fill_between(levels, levels, empirical_raw, alpha=0.15, color="tomato")
    ax.set(xlabel="Nominal coverage", ylabel="Empirical coverage",
           title=f"GP Surrogate Calibration (raw, n={len(y_true)} EPA)", xlim=(0,1), ylim=(0,1))
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel 2: Conformal GP calibration
    ax = axes[0, 1]
    ax.plot(levels, empirical_conf, "o-", ms=3, color="steelblue", label=f"GP + conformal (q̂={q95:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Ideal")
    ax.fill_between(levels, levels, empirical_conf, alpha=0.15, color="steelblue")
    ax.set(xlabel="Nominal coverage", ylabel="Empirical coverage",
           title="GP + Conformal Calibration", xlim=(0,1), ylim=(0,1))
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel 3: Spatial mean NO₂ field (all LUR test rows, averaged over time)
    ax = axes[0, 2]
    lur_all = test[test["source"] == "lur"].copy()
    lur_mean = lur_all.groupby("grid_id").agg(
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        pred_mean=("pred_mean", "mean"),
    ).reset_index()
    sc = ax.scatter(lur_mean["longitude"], lur_mean["latitude"],
                    c=lur_mean["pred_mean"], cmap="YlOrRd", s=2, alpha=0.6)
    ax.scatter(epa_gp["longitude"], epa_gp["latitude"],
               c="blue", s=30, marker="^", zorder=5, label="EPA monitors")
    plt.colorbar(sc, ax=ax, label="NO₂ (µg/m³)")
    ax.set(title="Time-Averaged GP Mean NO₂\n(Jun 1–29, blue = EPA)",
           xlabel="Longitude (normalised)", ylabel="Latitude (normalised)")
    ax.legend(fontsize=8)

    # Panel 4: Prediction intervals (sorted by truth)
    ax = axes[1, 0]
    idx = np.argsort(y_true)
    x   = np.arange(len(y_true))
    lo_raw  = mu_gp - 1.96 * sig_raw    # raw GP 95% PI (Gaussian ±1.96σ)
    hi_raw  = mu_gp + 1.96 * sig_raw
    lo_conf = mu_gp - sig_gp            # conformal 95% PI: mu ± q̂95·σ
    hi_conf = mu_gp + sig_gp
    ax.fill_between(x, lo_conf[idx], hi_conf[idx],
                    alpha=0.35, color="steelblue", label=f"95% conformal PI (q̂={q95:.4f}·σ)")
    ax.fill_between(x, lo_raw[idx], hi_raw[idx],
                    alpha=0.15, color="tomato", label="95% raw GP PI (±1.96σ)")
    ax.plot(x, y_true[idx], "ko", ms=5, label="EPA truth", zorder=4)
    ax.plot(x, mu_gp[idx],  "b-", lw=1,  label="GP mean")
    ax.set(title="GP Predictions vs EPA Truth\n(sorted by observed value)",
           xlabel="Index (sorted)", ylabel="NO₂ (µg/m³)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 5: Spatial exceedance map (time-averaged)
    ax = axes[1, 1]
    lur_exc = lur_all.groupby("grid_id").agg(
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        pred_mean=("pred_mean", "mean"),
        pred_std=("pred_std", "mean"),
    ).reset_index()
    sig_cal = lur_exc["pred_std"].to_numpy() * q95
    p_exc = exceedance_prob(lur_exc["pred_mean"].to_numpy(), sig_cal, WHO_DAILY)
    sc2 = ax.scatter(lur_exc["longitude"], lur_exc["latitude"],
                     c=p_exc, cmap="RdYlGn_r", s=2, alpha=0.6, vmin=0, vmax=1)
    plt.colorbar(sc2, ax=ax, label=f"P(NO₂ > {WHO_DAILY} µg/m³)")
    ax.set(title=f"Time-Averaged Exceedance Probability\nP(NO₂ > {WHO_DAILY} µg/m³ [WHO daily])",
           xlabel="Longitude (normalised)", ylabel="Latitude (normalised)")

    # Panel 6: Coverage bar chart (raw GP vs conformal vs ideal)
    # Raw GP: Gaussian z-scores on raw σ;  Conformal: use calibrated quantiles q̂ directly
    ax = axes[1, 2]
    labels = ["50%", "90%", "95%"]
    nominal = [0.50, 0.90, 0.95]
    raw_cov, conf_cov = [], []
    for alpha, q_conf in zip([0.50, 0.10, 0.05], [q50, q90, q95]):
        z = stats.norm.ppf(1 - alpha / 2)
        raw_cov.append(float(np.mean(
            (y_true >= mu_gp - z * sig_raw)
          & (y_true <= mu_gp + z * sig_raw)
        )))
        # Conformal PI: mu ± q̂_alpha · σ  (no extra z multiplier)
        conf_cov.append(float(np.mean(
            (y_true >= mu_gp - q_conf * sig_raw)
          & (y_true <= mu_gp + q_conf * sig_raw)
        )))
    x_pos = np.arange(len(labels)); w = 0.25
    ax.bar(x_pos - w, raw_cov,  w, color="tomato",    alpha=0.8, label="GP raw")
    ax.bar(x_pos,     conf_cov, w, color="steelblue", alpha=0.8, label="GP + Conformal")
    ax.bar(x_pos + w, nominal,  w, color="silver",    alpha=0.8, label="Ideal")
    ax.set_xticks(x_pos); ax.set_xticklabels([f"PI-{l}" for l in labels])
    ax.set(ylabel="Empirical coverage", ylim=(0, 1.1),
           title="Coverage: GP Raw vs Conformal vs Ideal")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "GP Surrogate UQ — Spatial & Calibration Analysis\n"
        "Dublin NO₂, June 2023  (FusionSVGP + conformal recalibration)",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    out = output_dir / "spatial_uq_panel.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Spatial panel → {out}")


# ===========================================================================
# Main
# ===========================================================================

def run(tables_dir: Path, figures_dir: Path) -> dict:
    """
    Run the full analytical UQ pipeline and write outputs to the given dirs.

    tables_dir  — destination for CSV files
    figures_dir — destination for PNG files
    Returns a dict of key scalar metrics for the pipeline report.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("GP Surrogate UQ  —  Dublin NO₂  (June 1–29, 2023)")
    print("=" * 65)

    # 1. Load GP predictions ────────────────────────────────────────────────
    print("\n[1] Loading GP posterior predictions")
    print("-" * 50)
    val_pred, test_pred = load_predictions()

    epa_test = test_pred[(test_pred["source"] == "epa") & test_pred["value"].notna()].copy()
    print(f"  EPA test observations: {len(epa_test)}")

    # 2. Conformal recalibration ────────────────────────────────────────────
    print("\n[2] Conformal recalibration of GP intervals")
    print("-" * 50)
    q50, q90, q95 = conformal_recalibrate(val_pred)

    # 3. Calibration metrics ────────────────────────────────────────────────
    print("\n[3] GP calibration metrics on EPA test set (n={})".format(len(epa_test)))
    print("-" * 50)
    y_true  = epa_test["value"].to_numpy()
    mu_gp   = epa_test["pred_mean"].to_numpy()
    sig_gp  = epa_test["pred_std"].to_numpy()

    raw_m  = calibration_metrics(y_true, mu_gp, sig_gp)
    conf_m = calibration_metrics(y_true, mu_gp, sig_gp * q95)   # CRPS / NLL only
    # Correct conformal coverage: PI = mu ± q̂ · σ  (q̂ replaces the Gaussian z-score)
    conf_picp_95 = float(np.mean(np.abs(y_true - mu_gp) / np.maximum(sig_gp, 1e-8) <= q95))
    conf_picp_90 = float(np.mean(np.abs(y_true - mu_gp) / np.maximum(sig_gp, 1e-8) <= q90))
    conf_picp_50 = float(np.mean(np.abs(y_true - mu_gp) / np.maximum(sig_gp, 1e-8) <= q50))
    print("  GP raw:")
    for k, v in raw_m.items():
        print(f"    {k:<22}: {v:.4f}")
    print("  GP + conformal (q̂95 · σ  prediction intervals):")
    print(f"    {'coverage_95 (conf)':<22}: {conf_picp_95:.4f}")
    print(f"    {'coverage_90 (conf)':<22}: {conf_picp_90:.4f}")
    print(f"    {'coverage_50 (conf)':<22}: {conf_picp_50:.4f}")
    for k in ["crps", "rmse", "bias"]:
        print(f"    {k:<22}: {conf_m[k]:.4f}")

    # 4. Temporal UQ (Jun 1 → Jun 29) ──────────────────────────────────────
    print("\n[4] Temporal uncertainty propagation (Day 1 → Day 29)")
    print("-" * 50)
    tdf = temporal_uq(test_pred, q95)
    print(tdf[["date", "mean_no2", "sigma_mean", "pi95_width",
               "p_exc_who25_mean", "picp_95", "crps_epa", "n_epa"]].to_string(index=False))
    tdf.to_csv(tables_dir / "temporal_uq_daily.csv", index=False)
    print(f"\n  Temporal table → {tables_dir}/temporal_uq_daily.csv")

    # 5. Source uncertainty decomposition ──────────────────────────────────
    print("\n[5] Law-of-Total-Variance source decomposition")
    print("-" * 50)
    decomp = uncertainty_decomposition(test_pred)
    for src, s in decomp["source_stats"].items():
        print(f"  {src:<12}: mean={s['mean']:6.2f}, std={s['std']:5.2f}, n={s['n']:,}")
    print(f"  Within-source  (aleatoric): {decomp['within_var']:.2f}  ({decomp['aleatoric_fraction']:.1%})")
    print(f"  Between-source (epistemic): {decomp['between_var']:.2f}  ({decomp['epistemic_fraction']:.1%})")

    # 6. Exceedance summary ─────────────────────────────────────────────────
    print("\n[6] Exceedance probabilities (conformal-scaled GP)")
    print("-" * 50)
    lur_all = test_pred[test_pred["source"] == "lur"].copy()
    sig_cal = lur_all["pred_std"].to_numpy() * q95
    for threshold, name in [(WHO_ANNUAL, "WHO annual (10)"),
                             (WHO_DAILY,  "WHO daily  (25)"),
                             (EU_ANNUAL,  "EU limit   (40)")]:
        p = exceedance_prob(lur_all["pred_mean"].to_numpy(), sig_cal, threshold)
        print(f"  {name}: mean P={p.mean():.3f},  % cells P>0.5: {(p>0.5).mean()*100:.1f}%")

    # 7. Figures ────────────────────────────────────────────────────────────
    print("\n[7] Generating figures")
    print("-" * 50)
    plot_temporal(tdf, figures_dir)
    plot_spatial_panels(test_pred, epa_test, q50, q90, q95, figures_dir)

    # 8. Summary tables ─────────────────────────────────────────────────────
    print("\n[8] Dissertation summary table")
    print("=" * 65)
    lo_r = mu_gp - 1.96 * sig_gp           # raw GP 95% PI
    hi_r = mu_gp + 1.96 * sig_gp
    lo_c = mu_gp - q95 * sig_gp            # conformal 95% PI: mu ± q̂95 · σ
    hi_c = mu_gp + q95 * sig_gp
    rows = [
        {
            "Model":          "GP Surrogate (raw)",
            "RMSE (µg/m³)":   f"{raw_m['rmse']:.2f}",
            "MAE (µg/m³)":    f"{raw_m['mae']:.2f}",
            "Bias (µg/m³)":   f"{raw_m['bias']:.2f}",
            "CRPS":           f"{raw_m['crps']:.3f}",
            "Coverage 95%":   f"{raw_m['coverage_95']:.3f}",
            "PI-95 width":    f"{float(np.mean(hi_r - lo_r)):.2f}",
        },
        {
            "Model":          "GP + Conformal",
            "RMSE (µg/m³)":   f"{raw_m['rmse']:.2f}",
            "MAE (µg/m³)":    f"{raw_m['mae']:.2f}",
            "Bias (µg/m³)":   f"{raw_m['bias']:.2f}",
            "CRPS":           f"{conf_m['crps']:.3f}",
            "Coverage 95%":   f"{conf_picp_95:.3f}",
            "PI-95 width":    f"{float(np.mean(hi_c - lo_c)):.2f}",
        },
    ]
    df_out = pd.DataFrame(rows)
    print(df_out.to_string(index=False))
    df_out.to_csv(tables_dir / "uq_results_table.csv", index=False)

    decomp_rows = [{"source": k, **v} for k, v in decomp["source_stats"].items()]
    pd.DataFrame(decomp_rows).to_csv(
        tables_dir / "source_uncertainty_decomposition.csv", index=False)

    print(f"\n  Tables → {tables_dir}")
    print(f"  Figures → {figures_dir}")
    print("=" * 65)

    return {
        "q95":            round(float(q95), 4),
        "picp_raw":       round(float(raw_m["coverage_95"]), 4),
        "picp_conf":      round(float(conf_picp_95), 4),
        "crps_conf":      round(float(conf_m["crps"]), 3),
        "rmse":           round(float(raw_m["rmse"]), 2),
        "epistemic_frac": f"{decomp['epistemic_fraction']:.1%}",
        "mean_no2":       round(float(tdf["mean_no2"].mean()), 2),
        "pi95_width":     round(float(tdf["pi95_width"].mean()), 1),
    }


def main():
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out    = ROOT_DIR / "output" / "analytical_uq_dublin" / run_ts
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out}")
    run(tables_dir=out, figures_dir=out)


if __name__ == "__main__":
    main()
