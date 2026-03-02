"""
Monte Carlo Uncertainty Quantification — Dublin NO₂
=====================================================

Three MC methods applied to the real multi-source NO₂ dataset.

Method 1 — Parametric Bootstrap MC
    Resample EPA training residuals (n=34) B=2000 times.
    Propagate σ_LUR uncertainty into:
      a) PICP distribution (with bootstrap 95% CI)
      b) CRPS distribution
      c) Exceedance probability fields with bootstrap CI

Method 2 — GP Posterior Sampling
    Draw L=5000 samples from N(μ_LUR, σ_LUR²) at each test location.
    Empirical statistics vs analytical Gaussian:
      a) MC coverage vs analytical coverage
      b) Posterior sample paths over time
      c) MC exceedance P(NO₂ > τ) vs analytical formula

Method 3 — Monte Carlo Variance Decomposition
    Draw joint field samples for each source.
    Compute within/between variance empirically.
    Compare to analytical Law of Total Variance.

Mathematical formulation:
    σ_b     ~ Bootstrap(EPA_residuals),  b = 1,...,B
    f*(l)   ~ N(μ_LUR(s*), σ_b²),       l = 1,...,L
    P_MC(f > τ | s*) = (1/L) Σ_l I[f*(l) > τ]
    σ²_total = E[Var[f|source]] + Var[E[f|source]]   (Law of Total Variance)

Outputs → results/uq_monte_carlo/
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "uq"
ROOT_DIR = Path(__file__).resolve().parent.parent

WHO_ANNUAL = 10.0   # µg/m³  WHO 2021 annual guideline
EU_ANNUAL  = 40.0   # µg/m³  EU limit
WHO_DAILY  = 25.0   # µg/m³  WHO 24-hour guideline

B_FULL    = 2000    # bootstrap replications for scalar metrics
B_GRID    = 500     # bootstrap replications for full grid (100K cells)
L_SAMPLES = 5000    # posterior samples per location
ALPHA     = 0.05    # significance level for 95% intervals

RNG = np.random.default_rng(42)


# ===========================================================================
# 1. Load data
# ===========================================================================

def load_data():
    train = pd.read_csv(DATA_DIR / "uq_train.csv")
    val   = pd.read_csv(DATA_DIR / "uq_val.csv")
    test  = pd.read_csv(DATA_DIR / "uq_test.csv")

    for df in [train, val, test]:
        valid = df["source"].isin({"epa", "satellite", "lur"})
        df.drop(df.index[~valid], inplace=True)
        df.reset_index(drop=True, inplace=True)

    epa_train = train[(train["source"] == "epa") & train["lur_no2"].notna()].copy()
    epa_test  = test[(test["source"] == "epa")
                     & test["value"].notna()
                     & test["lur_no2"].notna()].copy()
    lur_test  = test[test["lur_no2"].notna()].copy()

    print(f"EPA train: {len(epa_train)} | EPA test: {len(epa_test)} | LUR grid: {len(lur_test):,}")
    return train, val, test, epa_train, epa_test, lur_test


# ===========================================================================
# 2. Method 1 — Parametric Bootstrap MC (scalar metrics)
# ===========================================================================

def bootstrap_mc_metrics(epa_train: pd.DataFrame,
                          epa_test: pd.DataFrame,
                          B: int = B_FULL) -> dict:
    """
    Bootstrap EPA training residuals B times.

    Each replicate b:
      1. Resample residuals with replacement → σ_b
      2. Compute PICP₉₅, CRPS, PI width against EPA test truth

    Returns bootstrap distributions and 95% CIs.
    """
    residuals = (epa_train["value"] - epa_train["lur_no2"]).to_numpy()
    y_true    = epa_test["value"].to_numpy()
    mu_lur    = epa_test["lur_no2"].to_numpy()
    n_res     = len(residuals)

    z_crit = stats.norm.ppf(1 - ALPHA / 2)  # 1.960 for 95%

    sigma_boot = np.empty(B)
    picp50     = np.empty(B)
    picp90     = np.empty(B)
    picp95     = np.empty(B)
    crps_boot  = np.empty(B)
    width_boot = np.empty(B)

    for b in range(B):
        res_b   = RNG.choice(residuals, size=n_res, replace=True)
        sigma_b = float(res_b.std())
        sigma_boot[b] = sigma_b

        # PICP at three nominal levels
        for level, arr in [(0.50, picp50), (0.90, picp90), (0.95, picp95)]:
            zz = stats.norm.ppf(1 - (1 - level) / 2)
            lo = mu_lur - zz * sigma_b
            hi = mu_lur + zz * sigma_b
            arr[b] = float(np.mean((y_true >= lo) & (y_true <= hi)))

        # CRPS (Gaussian, Gneiting & Raftery 2007)
        z_norm     = (y_true - mu_lur) / max(sigma_b, 1e-8)
        crps_i     = sigma_b * (
            z_norm * (2 * stats.norm.cdf(z_norm) - 1)
            + 2 * stats.norm.pdf(z_norm)
            - 1.0 / np.sqrt(np.pi)
        )
        crps_boot[b]  = float(crps_i.mean())

        # PI-95 width
        width_boot[b] = 2 * z_crit * sigma_b

    def _ci(arr):
        return np.percentile(arr, [2.5, 97.5])

    return {
        "sigma_boot": sigma_boot,
        "picp50":     picp50,
        "picp90":     picp90,
        "picp95":     picp95,
        "crps_boot":  crps_boot,
        "width_boot": width_boot,
        # Summary statistics
        "sigma_mean":  sigma_boot.mean(),
        "sigma_ci":    _ci(sigma_boot),
        "picp95_mean": picp95.mean(),
        "picp95_ci":   _ci(picp95),
        "crps_mean":   crps_boot.mean(),
        "crps_ci":     _ci(crps_boot),
        "width_mean":  width_boot.mean(),
        "width_ci":    _ci(width_boot),
    }


# ===========================================================================
# 3. Method 2 — GP Posterior Sampling (per-location)
# ===========================================================================

def posterior_sampling(epa_test: pd.DataFrame,
                        sigma_lur: float,
                        L: int = L_SAMPLES) -> dict:
    """
    Draw L samples from N(μ_LUR, σ²) at every EPA test location.

    Returns empirical vs analytical coverage, exceedance, and sample array.
    """
    y_true = epa_test["value"].to_numpy()
    mu_lur = epa_test["lur_no2"].to_numpy()
    ts     = epa_test["timestamp"].to_numpy()   # for time-series plot
    n      = len(mu_lur)

    # samples shape: (L, n)
    samples = RNG.normal(loc=mu_lur, scale=sigma_lur, size=(L, n))

    mc_q025 = np.percentile(samples, 2.5,  axis=0)
    mc_q975 = np.percentile(samples, 97.5, axis=0)

    mc_coverage   = float(np.mean((y_true >= mc_q025) & (y_true <= mc_q975)))
    anal_coverage = float(np.mean(
        (y_true >= mu_lur - 1.96 * sigma_lur)
        & (y_true <= mu_lur + 1.96 * sigma_lur)
    ))

    # Exceedance probability per location
    # MC:       P̂(f > τ) = fraction of samples > τ
    # Analytical: 1 − Φ((τ − μ) / σ)
    exceed_mc   = (samples > WHO_ANNUAL).mean(axis=0)
    exceed_anal = 1.0 - stats.norm.cdf((WHO_ANNUAL - mu_lur) / sigma_lur)

    # Posterior variance components (per location, across L draws)
    mc_mean_per_loc = samples.mean(axis=0)
    mc_var_per_loc  = samples.var(axis=0)   # aleatoric per location

    return {
        "samples":      samples,           # (L, n)
        "timestamps":   ts,
        "y_true":       y_true,
        "mu_lur":       mu_lur,
        "mc_q025":      mc_q025,
        "mc_q975":      mc_q975,
        "mc_coverage":  mc_coverage,
        "anal_coverage": anal_coverage,
        "exceed_mc":    exceed_mc,
        "exceed_anal":  exceed_anal,
        "mc_var_per_loc": mc_var_per_loc,
    }


# ===========================================================================
# 4. Method 3 — MC Variance Decomposition
# ===========================================================================

def mc_variance_decomposition(test: pd.DataFrame,
                               L: int = L_SAMPLES) -> dict:
    """
    Law of Total Variance via Monte Carlo.

    For each source s, draw samples from N(μ_s, σ_s²).
    Within-source variance:  E[Var[f | source]]  (aleatoric)
    Between-source variance: Var[E[f | source]]  (epistemic)

    Compare MC estimate to analytical calculation.
    """
    source_stats = {}
    for src in ["epa", "satellite", "lur"]:
        vals = test[test["source"] == src]["value"].dropna()
        if len(vals) >= 2:
            source_stats[src] = {
                "mean": float(vals.mean()),
                "std":  float(vals.std()),
                "n":    len(vals),
            }

    # Draw L samples per source, using that source's empirical mean + std
    mc_within_vars = []
    mc_source_means = []

    for src, d in source_stats.items():
        # (L,) samples of the source mean (parametric)
        sample_means = RNG.normal(loc=d["mean"], scale=d["std"] / np.sqrt(d["n"]), size=L)
        # within-source variance: d["std"]² (constant across draws)
        mc_within_vars.append(d["std"] ** 2)
        mc_source_means.append(sample_means)

    # MC within-source variance (weighted by sample size)
    total_n = sum(d["n"] for d in source_stats.values())
    within_var_mc = sum(
        d["n"] * mc_within_vars[i]
        for i, (_, d) in enumerate(source_stats.items())
    ) / total_n

    # MC between-source variance: variance of the L source-mean draws, averaged
    source_mean_matrix = np.array(mc_source_means)  # (S, L)
    between_var_mc = float(source_mean_matrix.var(axis=0).mean())

    total_var_mc   = within_var_mc + between_var_mc

    # Analytical
    total_n_anal = sum(d["n"] for d in source_stats.values())
    within_var_anal  = sum(d["n"] * d["std"] ** 2 for d in source_stats.values()) / total_n_anal
    means_anal       = np.array([d["mean"] for d in source_stats.values()])
    between_var_anal = float(np.var(means_anal))
    total_var_anal   = within_var_anal + between_var_anal

    return {
        "source_stats":      source_stats,
        # MC
        "within_var_mc":     float(within_var_mc),
        "between_var_mc":    float(between_var_mc),
        "total_var_mc":      float(total_var_mc),
        "aleatoric_mc":      within_var_mc  / total_var_mc if total_var_mc else 0,
        "epistemic_mc":      between_var_mc / total_var_mc if total_var_mc else 0,
        # Analytical (Law of Total Variance)
        "within_var_anal":   float(within_var_anal),
        "between_var_anal":  float(between_var_anal),
        "total_var_anal":    float(total_var_anal),
        "aleatoric_anal":    within_var_anal  / total_var_anal if total_var_anal else 0,
        "epistemic_anal":    between_var_anal / total_var_anal if total_var_anal else 0,
        # MC bootstrap CIs on between-source variance
        "between_var_mc_ci": np.percentile(source_mean_matrix.var(axis=0), [2.5, 97.5]).tolist(),
    }


# ===========================================================================
# 5. Bootstrap exceedance probability on full LUR grid
# ===========================================================================

def bootstrap_grid_exceedance(epa_train: pd.DataFrame,
                               lur_test: pd.DataFrame,
                               threshold: float = WHO_ANNUAL,
                               B: int = B_GRID) -> dict:
    """
    For each of B bootstrap σ_b values, compute P(NO₂ > τ) at every LUR grid cell.

    Returns: per-cell mean, lower and upper 95% bootstrap CI of exceedance probability.
    """
    residuals = (epa_train["value"] - epa_train["lur_no2"]).to_numpy()
    mu        = lur_test["lur_no2"].to_numpy()
    n_cells   = len(mu)
    n_res     = len(residuals)

    exceed_boot = np.empty((B, n_cells), dtype=np.float32)

    for b in range(B):
        sigma_b       = float(RNG.choice(residuals, size=n_res, replace=True).std())
        exceed_boot[b] = (1.0 - stats.norm.cdf((threshold - mu) / max(sigma_b, 1e-8))).astype(np.float32)

    return {
        "mean":  exceed_boot.mean(axis=0),
        "lower": np.percentile(exceed_boot, 2.5,  axis=0),
        "upper": np.percentile(exceed_boot, 97.5, axis=0),
        "lat":   lur_test["latitude"].to_numpy(),
        "lon":   lur_test["longitude"].to_numpy(),
    }


# ===========================================================================
# 6. Plots
# ===========================================================================

def make_plots(boot: dict, post: dict, decomp: dict, grid_exc: dict,
               sigma_lur_point: float, output_dir: Path = None, run_ts: str = "") -> None:

    fig, axes = plt.subplots(3, 3, figsize=(17, 14))

    # ── [0,0] Bootstrap distribution of σ_LUR ────────────────────────────
    ax = axes[0, 0]
    ax.hist(boot["sigma_boot"], bins=50, color="steelblue", alpha=0.7, edgecolor="none")
    lo, hi = boot["sigma_ci"]
    ax.axvline(boot["sigma_mean"], color="navy", lw=2, label=f"Bootstrap mean = {boot['sigma_mean']:.2f}")
    ax.axvline(sigma_lur_point, color="tomato", lw=2, linestyle="--",
               label=f"Point estimate = {sigma_lur_point:.2f}")
    ax.axvspan(lo, hi, alpha=0.15, color="steelblue",
               label=f"95% CI [{lo:.2f}, {hi:.2f}]")
    ax.set(xlabel="σ_LUR (µg/m³)", ylabel="Bootstrap frequency",
           title="Bootstrap MC: σ_LUR Distribution\n(B=2000 resamples of EPA residuals)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── [0,1] Bootstrap PICP₉₅ distribution ──────────────────────────────
    ax = axes[0, 1]
    ax.hist(boot["picp95"] * 100, bins=40, color="tomato", alpha=0.7, edgecolor="none")
    lo95, hi95 = boot["picp95_ci"]
    ax.axvline(boot["picp95_mean"] * 100, color="darkred", lw=2,
               label=f"MC mean = {boot['picp95_mean']*100:.1f}%")
    ax.axvline(95, color="black", lw=2, linestyle="--", label="Target = 95%")
    ax.axvspan(lo95 * 100, hi95 * 100, alpha=0.15, color="tomato",
               label=f"95% CI [{lo95*100:.1f}%, {hi95*100:.1f}%]")
    ax.set(xlabel="PICP₉₅ (%)", ylabel="Bootstrap frequency",
           title="Bootstrap MC: PICP₉₅ Distribution\n(propagated from σ_LUR uncertainty)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── [0,2] Bootstrap CRPS distribution ────────────────────────────────
    ax = axes[0, 2]
    ax.hist(boot["crps_boot"], bins=40, color="mediumpurple", alpha=0.7, edgecolor="none")
    lo_c, hi_c = boot["crps_ci"]
    ax.axvline(boot["crps_mean"], color="indigo", lw=2,
               label=f"MC mean = {boot['crps_mean']:.3f}")
    ax.axvspan(lo_c, hi_c, alpha=0.15, color="mediumpurple",
               label=f"95% CI [{lo_c:.3f}, {hi_c:.3f}]")
    ax.set(xlabel="CRPS (µg/m³)", ylabel="Bootstrap frequency",
           title="Bootstrap MC: CRPS Distribution\n(proper scoring rule, lower=better)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── [1,0] Posterior sample paths (time-sorted) ───────────────────────
    ax = axes[1, 0]
    ts    = post["timestamps"]
    y_t   = post["y_true"]
    mu_l  = post["mu_lur"]
    order = np.argsort(ts)
    n_show = min(50, post["samples"].shape[0])
    for i in range(n_show):
        ax.plot(ts[order], post["samples"][i, order], lw=0.5, alpha=0.25, color="steelblue")
    ax.plot(ts[order], mu_l[order], "k-", lw=2, zorder=5, label="LUR mean (μ*)")
    ax.fill_between(ts[order], post["mc_q025"][order], post["mc_q975"][order],
                    alpha=0.3, color="steelblue", label="MC 95% CI")
    ax.scatter(ts, y_t, c="tomato", s=40, zorder=6, label="EPA truth", edgecolors="white", lw=0.5)
    ax.set(xlabel="Timestamp (normalised)", ylabel="NO₂ (µg/m³)",
           title=f"Method 2: {n_show} Posterior Samples (time-sorted)\n"
                 f"MC coverage 95% = {post['mc_coverage']*100:.1f}% "
                 f"(analytical: {post['anal_coverage']*100:.1f}%)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── [1,1] MC vs Analytical exceedance (per EPA location) ─────────────
    ax = axes[1, 1]
    ax.scatter(post["exceed_anal"] * 100, post["exceed_mc"] * 100,
               s=60, color="darkorange", edgecolors="white", lw=0.5, zorder=3)
    lim = [0, 100]
    ax.plot(lim, lim, "k--", lw=1.5, label="Perfect agreement")
    ax.set(xlabel="Analytical P(NO₂ > 10) (%)", ylabel="MC P(NO₂ > 10) (%)",
           title="MC vs Analytical Exceedance\n(per EPA test location, WHO threshold 10 µg/m³)",
           xlim=lim, ylim=lim)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── [1,2] MC Variance Decomposition vs Analytical ────────────────────
    ax = axes[1, 2]
    methods  = ["MC\n(Method 3)", "Analytical\n(Law of Total Var.)"]
    alea_vals = [decomp["aleatoric_mc"] * 100, decomp["aleatoric_anal"] * 100]
    epis_vals = [decomp["epistemic_mc"] * 100, decomp["epistemic_anal"] * 100]
    x = np.arange(len(methods))
    w = 0.4
    b1 = ax.bar(x - w/2, alea_vals, w, color="steelblue", alpha=0.8, label="Aleatoric (within-source)")
    b2 = ax.bar(x + w/2, epis_vals, w, color="tomato",    alpha=0.8, label="Epistemic (between-source)")
    for bar, val in zip(list(b1) + list(b2), alea_vals + epis_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set(xticks=x, xticklabels=methods, ylabel="% of total variance",
           ylim=(0, 80),
           title="Variance Decomposition:\nMC vs Analytical (Law of Total Variance)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # ── [2,0] Bootstrap PICP calibration curve with CI band ──────────────
    ax = axes[2, 0]
    levels = np.linspace(0.01, 0.99, 50)
    # For each nominal level, collect PICP₊ over B bootstrap σ values
    picp_matrix = np.empty((B_FULL, len(levels)))
    mu_lur_epa  = post["mu_lur"]
    y_true_epa  = post["y_true"]
    for b_idx, sigma_b in enumerate(boot["sigma_boot"]):
        for j, nom in enumerate(levels):
            zz  = stats.norm.ppf(1 - (1 - nom) / 2)
            lo_ = mu_lur_epa - zz * sigma_b
            hi_ = mu_lur_epa + zz * sigma_b
            picp_matrix[b_idx, j] = float(np.mean((y_true_epa >= lo_) & (y_true_epa <= hi_)))
    picp_mean  = picp_matrix.mean(axis=0)
    picp_lo_b  = np.percentile(picp_matrix, 2.5,  axis=0)
    picp_hi_b  = np.percentile(picp_matrix, 97.5, axis=0)
    ax.fill_between(levels, picp_lo_b, picp_hi_b, alpha=0.25, color="steelblue",
                    label="95% bootstrap CI")
    ax.plot(levels, picp_mean, color="steelblue", lw=2, label="MC mean calibration")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Ideal")
    ax.set(xlabel="Nominal coverage", ylabel="Empirical coverage",
           title="Bootstrap Calibration Curve\n(MC 95% CI from σ_LUR uncertainty)",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── [2,1] Grid exceedance: mean probability map ───────────────────────
    ax = axes[2, 1]
    sc = ax.scatter(grid_exc["lon"], grid_exc["lat"],
                    c=grid_exc["mean"], cmap="RdYlGn_r",
                    s=0.5, alpha=0.6, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="P(NO₂ > 10 µg/m³)")
    ax.set(xlabel="Longitude (norm.)", ylabel="Latitude (norm.)",
           title=f"Bootstrap MC Exceedance\nMean P(NO₂ > {WHO_ANNUAL:.0f} µg/m³) "
                 f"[B={B_GRID}]")

    # ── [2,2] Grid exceedance: CI width map ───────────────────────────────
    ax = axes[2, 2]
    ci_width = grid_exc["upper"] - grid_exc["lower"]
    sc2 = ax.scatter(grid_exc["lon"], grid_exc["lat"],
                     c=ci_width, cmap="plasma",
                     s=0.5, alpha=0.6, vmin=0)
    plt.colorbar(sc2, ax=ax, label="Bootstrap 95% CI width")
    ax.set(xlabel="Longitude (norm.)", ylabel="Latitude (norm.)",
           title=f"Bootstrap MC Exceedance CI Width\n"
                 f"Uncertainty on P(NO₂ > {WHO_ANNUAL:.0f}) from σ_LUR variability")

    ts_label = f"  [{run_ts}]" if run_ts else ""
    fig.suptitle(f"Monte Carlo UQ Analysis — Multi-source NO₂, Dublin{ts_label}\n"
                 "Methods: Bootstrap MC | Posterior Sampling | MC Variance Decomposition",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = (output_dir or ROOT_DIR) / "mc_uq_panel.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Panel plot saved → {out_path}")


# ===========================================================================
# 7. Save results table
# ===========================================================================

def save_results(boot: dict, post: dict, decomp: dict, sigma_lur_point: float,
                 output_dir: Path = None, run_ts: str = "") -> None:
    rows = [
        # ── Bootstrap MC metrics ──────────────────────────────────────────
        {"Method": "Bootstrap MC", "Metric": "σ_LUR point estimate (µg/m³)",
         "Value": f"{sigma_lur_point:.3f}", "95% CI": "—"},
        {"Method": "Bootstrap MC", "Metric": "σ_LUR bootstrap mean (µg/m³)",
         "Value": f"{boot['sigma_mean']:.3f}",
         "95% CI": f"[{boot['sigma_ci'][0]:.2f}, {boot['sigma_ci'][1]:.2f}]"},
        {"Method": "Bootstrap MC", "Metric": "PICP₉₅ (%)",
         "Value": f"{boot['picp95_mean']*100:.1f}",
         "95% CI": f"[{boot['picp95_ci'][0]*100:.1f}, {boot['picp95_ci'][1]*100:.1f}]"},
        {"Method": "Bootstrap MC", "Metric": "CRPS (µg/m³)",
         "Value": f"{boot['crps_mean']:.3f}",
         "95% CI": f"[{boot['crps_ci'][0]:.3f}, {boot['crps_ci'][1]:.3f}]"},
        {"Method": "Bootstrap MC", "Metric": "PI-95 width (µg/m³)",
         "Value": f"{boot['width_mean']:.2f}",
         "95% CI": f"[{boot['width_ci'][0]:.2f}, {boot['width_ci'][1]:.2f}]"},
        # ── Posterior sampling ────────────────────────────────────────────
        {"Method": "Posterior Sampling", "Metric": "MC coverage 95% (%)",
         "Value": f"{post['mc_coverage']*100:.1f}", "95% CI": "—"},
        {"Method": "Posterior Sampling", "Metric": "Analytical coverage 95% (%)",
         "Value": f"{post['anal_coverage']*100:.1f}", "95% CI": "—"},
        {"Method": "Posterior Sampling", "Metric": "MC–Analytical diff (pp)",
         "Value": f"{(post['mc_coverage'] - post['anal_coverage'])*100:.2f}", "95% CI": "—"},
        {"Method": "Posterior Sampling", "Metric": "MC mean P(NO₂ > 10)",
         "Value": f"{post['exceed_mc'].mean():.3f}", "95% CI": "—"},
        {"Method": "Posterior Sampling", "Metric": "Analytical mean P(NO₂ > 10)",
         "Value": f"{post['exceed_anal'].mean():.3f}", "95% CI": "—"},
        # ── MC Variance Decomposition ─────────────────────────────────────
        {"Method": "MC Var. Decomp.", "Metric": "Epistemic fraction MC (%)",
         "Value": f"{decomp['epistemic_mc']*100:.1f}", "95% CI": "—"},
        {"Method": "MC Var. Decomp.", "Metric": "Aleatoric fraction MC (%)",
         "Value": f"{decomp['aleatoric_mc']*100:.1f}", "95% CI": "—"},
        {"Method": "MC Var. Decomp.", "Metric": "Epistemic fraction Analytical (%)",
         "Value": f"{decomp['epistemic_anal']*100:.1f}", "95% CI": "—"},
        {"Method": "MC Var. Decomp.", "Metric": "Aleatoric fraction Analytical (%)",
         "Value": f"{decomp['aleatoric_anal']*100:.1f}", "95% CI": "—"},
    ]
    if run_ts:
        rows.insert(0, {"Method": "Run metadata", "Metric": "run_timestamp",
                        "Value": run_ts, "95% CI": "—"})
    df_out = pd.DataFrame(rows)
    out_path = (output_dir or ROOT_DIR) / "mc_uq_results.csv"
    df_out.to_csv(out_path, index=False)
    print(f"  Results table saved → {out_path}")
    return df_out


# ===========================================================================
# 8. Main
# ===========================================================================

def run(tables_dir: Path, figures_dir: Path) -> dict:
    """
    Run the full Monte Carlo UQ pipeline and write outputs to the given dirs.

    tables_dir  — destination for CSV files
    figures_dir — destination for PNG files
    Returns a dict of key scalar metrics for the pipeline report.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Monte Carlo UQ Analysis  —  Dublin Multi-source NO₂")
    print("=" * 65)

    train, val, test, epa_train, epa_test, lur_test = load_data()

    residuals_full  = (epa_train["value"] - epa_train["lur_no2"]).to_numpy()
    sigma_lur_point = float(residuals_full.std())
    print(f"\nPoint estimate σ_LUR = {sigma_lur_point:.3f} µg/m³  (n={len(epa_train)} EPA train)")

    # Method 1: Parametric Bootstrap MC ────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"[Method 1] Parametric Bootstrap MC  (B={B_FULL})")
    print(f"{'─'*50}")
    boot = bootstrap_mc_metrics(epa_train, epa_test, B=B_FULL)
    print(f"  σ_LUR:  {boot['sigma_mean']:.3f}  95% CI [{boot['sigma_ci'][0]:.2f}, {boot['sigma_ci'][1]:.2f}] µg/m³")
    print(f"  PICP₉₅: {boot['picp95_mean']*100:.1f}%  95% CI [{boot['picp95_ci'][0]*100:.1f}%, {boot['picp95_ci'][1]*100:.1f}%]")
    print(f"  CRPS:   {boot['crps_mean']:.3f}  95% CI [{boot['crps_ci'][0]:.3f}, {boot['crps_ci'][1]:.3f}] µg/m³")
    print(f"  PI-95 width: {boot['width_mean']:.2f}  95% CI [{boot['width_ci'][0]:.2f}, {boot['width_ci'][1]:.2f}] µg/m³")

    # Method 2: Posterior Sampling ─────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"[Method 2] GP Posterior Sampling  (L={L_SAMPLES})")
    print(f"{'─'*50}")
    post = posterior_sampling(epa_test, sigma_lur_point, L=L_SAMPLES)
    print(f"  MC coverage 95%:       {post['mc_coverage']*100:.1f}%")
    print(f"  Analytical coverage:   {post['anal_coverage']*100:.1f}%")
    print(f"  Difference:            {(post['mc_coverage']-post['anal_coverage'])*100:.2f} pp")
    print(f"  MC mean P(NO₂>10):     {post['exceed_mc'].mean():.3f}")
    print(f"  Analytical P(NO₂>10):  {post['exceed_anal'].mean():.3f}")
    print(f"  Max MC–Analytical diff: {np.abs(post['exceed_mc'] - post['exceed_anal']).max():.4f}")

    # Method 3: MC Variance Decomposition ──────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"[Method 3] MC Variance Decomposition  (L={L_SAMPLES})")
    print(f"{'─'*50}")
    decomp = mc_variance_decomposition(test, L=L_SAMPLES)
    print(f"  MC:         epistemic={decomp['epistemic_mc']*100:.1f}%  "
          f"aleatoric={decomp['aleatoric_mc']*100:.1f}%  "
          f"total={decomp['total_var_mc']:.2f} µg²/m⁶")
    print(f"  Analytical: epistemic={decomp['epistemic_anal']*100:.1f}%  "
          f"aleatoric={decomp['aleatoric_anal']*100:.1f}%  "
          f"total={decomp['total_var_anal']:.2f} µg²/m⁶")
    print(f"  Between-source var MC 95% CI: "
          f"[{decomp['between_var_mc_ci'][0]:.2f}, {decomp['between_var_mc_ci'][1]:.2f}] µg²/m⁶")

    # Bootstrap grid exceedance ────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"[Grid Exceedance] Bootstrap over full LUR grid  "
          f"(B={B_GRID}, {len(lur_test):,} cells)")
    print(f"{'─'*50}")
    grid_exc = bootstrap_grid_exceedance(epa_train, lur_test, threshold=WHO_ANNUAL, B=B_GRID)
    pct_high = float(np.mean(grid_exc["mean"] > 0.5)) * 100
    ci_width_median = float(np.median(grid_exc["upper"] - grid_exc["lower"]))
    print(f"  Mean P(NO₂>10):              {grid_exc['mean'].mean():.3f}")
    print(f"  % cells with mean P > 0.5:   {pct_high:.1f}%")
    print(f"  Median bootstrap CI width:   {ci_width_median:.4f}")

    # Figure ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("[Plots]  Generating 3×3 panel figure...")
    make_plots(boot, post, decomp, grid_exc, sigma_lur_point,
               output_dir=figures_dir, run_ts="")

    # Results table ────────────────────────────────────────────────────────
    print("\n[Results] Saving summary table...")
    df_out = save_results(boot, post, decomp, sigma_lur_point,
                          output_dir=tables_dir, run_ts="")
    print(df_out.to_string(index=False))

    # Rename mc_uq_results.csv → mc_results.csv for canonical layout
    src = tables_dir / "mc_uq_results.csv"
    dst = tables_dir / "mc_results.csv"
    if src.exists() and not dst.exists():
        src.rename(dst)

    print(f"\n  Tables  → {tables_dir}")
    print(f"  Figures → {figures_dir}")
    print(f"{'='*65}")

    return {
        "sigma_lur":            round(float(boot["sigma_mean"]), 3),
        "sigma_lur_ci":         f"[{boot['sigma_ci'][0]:.2f}, {boot['sigma_ci'][1]:.2f}]",
        "picp95_mean":          f"{boot['picp95_mean']*100:.1f}%",
        "crps_mean":            round(float(boot["crps_mean"]), 3),
        "epistemic_mc":         f"{decomp['epistemic_mc']*100:.1f}%",
        "pct_exceed_who_annual": f"{pct_high:.1f}%",
    }


def main():
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out    = ROOT_DIR / "output" / "mc_uq_dublin" / run_ts
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out}")
    run(tables_dir=out, figures_dir=out)


if __name__ == "__main__":
    main()
