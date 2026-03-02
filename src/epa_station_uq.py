"""
EPA Observations vs GP Predictions — Dublin NO₂ (June 2023)
============================================================

For each of the 9 EPA monitoring stations, plots the FusionGP
surrogate predictions against EPA ground-truth observations
from Day 1 (Jun 1) to Day 29 (Jun 29, 2023).

Each station panel shows:
  • GP pred_mean time series with ±1σ and ±2σ conformal prediction
    intervals  (q̂ = 2.81 × σ_GP, calibrated on satellite val set)
  • EPA observed values: train = black triangles, test = red circles
  • WHO daily threshold (25 µg/m³)
  • PICP: empirical 95% coverage on test EPA observations

GP prediction source:
  predictions_val.csv + predictions_test.csv  (LUR rows at EPA grid cells)
  Conformal q̂ = 2.8121 from analytical_uq_dublin.py

Output: output/epa_station_uq/<timestamp>/epa_station_temporal_uq.png
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

ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "data"

WHO_DAILY  = 25.0
DATE_START = datetime(2023, 6, 1)
N_DAYS     = 29
Q_HAT_95   = 2.8121   # conformal quantile from satellite val calibration
Q_HAT_90   = 2.7688


def ts_to_day(ts: float) -> int:
    return int(round(ts * (N_DAYS - 1)))


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data():
    """
    Returns:
        gp_lur  — LUR GP predictions at ALL grid cells (val + test, all 29 days)
        epa_train — EPA observations from training split (value, lur_no2)
        epa_test  — EPA observations from test split (value, pred_mean, pred_std)
    """
    val   = pd.read_csv(DATA_DIR / "predictions_val.csv")
    test  = pd.read_csv(DATA_DIR / "predictions_test.csv")
    train = pd.read_csv(DATA_DIR / "uq" / "uq_train.csv")

    for df in [val, test, train]:
        df["day"]  = df["timestamp"].apply(ts_to_day)
        df["date"] = pd.to_datetime(
            df["day"].apply(lambda d: DATE_START + timedelta(days=d))
        )

    # LUR GP predictions from val + test (all grid cells, all available days)
    gp_lur = pd.concat([
        val[val["source"] == "lur"]
           [["grid_id", "day", "date", "pred_mean", "pred_std"]],
        test[test["source"] == "lur"]
            [["grid_id", "day", "date", "pred_mean", "pred_std"]],
    ], ignore_index=True).drop_duplicates(subset=["grid_id", "day"])

    for df in [train, test]:
        df["t"] = df["day"] / (N_DAYS - 1)

    epa_train = (train[train["source"] == "epa"]
                 [["grid_id", "day", "date", "t", "value"]]
                 .dropna(subset=["value"]).copy())

    epa_test = (test[test["source"] == "epa"]
                [["grid_id", "day", "date", "t", "value", "pred_mean", "pred_std"]]
                .dropna(subset=["value"]).copy())

    return gp_lur, epa_train, epa_test


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _temporal_kriging(t_obs, y_obs, t_grid, tau=0.08):
    """1-D temporal Kriging with LOO-CV theta selection (exponential correlation)."""
    def corr(t1, t2, th): return np.exp(-abs(t1 - t2) / th)

    def fit(T, Y, th):
        n  = len(T)
        R  = np.array([[corr(T[i], T[j], th) for j in range(n)] for i in range(n)])
        R += tau * np.eye(n)
        Ri = np.linalg.inv(R)
        F  = np.ones((n, 1))
        FRF = F.T @ Ri @ F
        beta = np.linalg.solve(FRF, F.T @ Ri @ Y)
        res  = Y - (F @ beta).ravel()
        sz2  = float(res @ Ri @ res) / n
        return Ri, FRF, beta, sz2

    def predict(t_pt, T, Y, Ri, FRF, beta, sz2, th):
        r   = np.array([corr(t_pt, ti, th) for ti in T])
        f   = np.array([1.0])
        tv  = (np.ones((1, len(T))) @ Ri @ r).ravel() - f
        mu  = float(f @ beta) + float(r @ Ri @ (Y - (np.ones((len(T), 1)) @ beta).ravel()))
        var = sz2 * max(0.0, 1.0 - r @ Ri @ r + tv @ np.linalg.solve(FRF, tv))
        return mu, np.sqrt(var)

    # LOO-CV theta search
    best_err, best_th = np.inf, 0.3
    for th in np.logspace(-2, 1, 25):
        if len(t_obs) < 3:
            break
        errs = []
        for k in range(len(t_obs)):
            mask = np.arange(len(t_obs)) != k
            if mask.sum() < 2:
                continue
            Ri, FRF, beta, sz2 = fit(t_obs[mask], y_obs[mask], th)
            mu_k, _ = predict(t_obs[k], t_obs[mask], y_obs[mask], Ri, FRF, beta, sz2, th)
            errs.append((y_obs[k] - mu_k) ** 2)
        loo = np.mean(errs) if errs else np.inf
        if loo < best_err:
            best_err, best_th = loo, th

    Ri, FRF, beta, sz2 = fit(t_obs, y_obs, best_th)
    mu_grid  = np.zeros(len(t_grid))
    sig_grid = np.zeros(len(t_grid))
    for i, t in enumerate(t_grid):
        mu_grid[i], sig_grid[i] = predict(t, t_obs, y_obs, Ri, FRF, beta, sz2, best_th)
    return mu_grid, sig_grid, best_th


def plot_epa_vs_predictions(gp_lur, epa_train, epa_test, output_dir: Path):

    all_grids = sorted(
        set(epa_train["grid_id"].unique()) | set(epa_test["grid_id"].unique())
    )
    n = len(all_grids)

    # Full temporal grid Day 1 → Day 29
    t_grid    = np.linspace(0, 1, 200)
    date_grid = [DATE_START + timedelta(days=int(round(t * (N_DAYS - 1))))
                 for t in t_grid]

    with plt.style.context("fivethirtyeight"):
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(6.0 * ncols, 4.8 * nrows),
                                 sharey=False)
        axes = axes.ravel()

        fmt   = mdates.DateFormatter("%b %d")
        ticks = mdates.DayLocator(interval=7)
        legend_done = False

        for idx, gid in enumerate(all_grids):
            ax = axes[idx]

            gp = gp_lur[gp_lur["grid_id"] == gid].sort_values("day")
            tr = epa_train[epa_train["grid_id"] == gid].sort_values("day")
            te = epa_test[epa_test["grid_id"] == gid].sort_values("day")

            # All EPA observations combined for temporal Kriging
            t_all = np.concatenate([tr["t"].values, te["t"].values])
            y_all = np.concatenate([tr["value"].values, te["value"].values])
            _, uniq = np.unique(np.round(t_all * 28).astype(int), return_index=True)
            t_all, y_all = t_all[uniq], y_all[uniq]
            sort_i = np.argsort(t_all)
            t_all, y_all = t_all[sort_i], y_all[sort_i]

            # ── Temporal Kriging on EPA observations ───────────────────────
            if len(t_all) >= 2:
                mu_k, sig_k, th_opt = _temporal_kriging(t_all, y_all, t_grid)
            else:
                mu_k   = np.full(len(t_grid), y_all[0])
                sig_k  = np.full(len(t_grid), 8.0)
                th_opt = np.nan

            # 95% band (notebook style: tab:blue fill + solid mean line)
            ax.fill_between(date_grid,
                            mu_k - 1.96 * sig_k,
                            mu_k + 1.96 * sig_k,
                            color="tab:blue", alpha=0.25,
                            label="95% Interval")
            ax.plot(date_grid, mu_k, color="tab:blue", lw=2.0,
                    label="GPR Mean")

            # ── FusionGP pred_mean — dashed reference (like f(x) in notebook)
            if len(gp) > 0:
                gp_mu    = gp["pred_mean"].values
                gp_dates = gp["date"].values
                ax.plot(gp_dates, gp_mu, color="black", lw=1.5,
                        linestyle="dashed", label=r"FusionGP $\hat{y}(x,t)$")

            # ── WHO threshold ──────────────────────────────────────────────
            ax.axhline(WHO_DAILY, color="tab:red", lw=1.2, ls=":",
                       label=f"WHO daily (25 µg/m³)")

            # ── Training obs: black dots (notebook style) ──────────────────
            all_dates = np.concatenate([tr["date"].values, te["date"].values])
            all_vals  = np.concatenate([tr["value"].values, te["value"].values])
            ax.scatter(all_dates, all_vals,
                       color="black", s=80, zorder=10,
                       label="EPA Observations")

            # ── PICP annotation ────────────────────────────────────────────
            if len(te) > 0 and len(gp) > 0:
                gp_idx  = gp.set_index("day")
                covered = sum(
                    1 for _, r in te.iterrows()
                    if r["day"] in gp_idx.index
                    # Conformal PI: mu ± Q_HAT_95 · σ  (no extra 1.96 factor)
                    and (gp_idx.loc[r["day"], "pred_mean"]
                         - Q_HAT_95 * gp_idx.loc[r["day"], "pred_std"])
                        <= r["value"]
                    <= (gp_idx.loc[r["day"], "pred_mean"]
                        + Q_HAT_95 * gp_idx.loc[r["day"], "pred_std"])
                )
                picp = covered / len(te)
                ax.text(0.03, 0.96,
                        f"PICP = {picp:.0%}",
                        transform=ax.transAxes, fontsize=9, va="top",
                        fontweight="bold")

            ax.set_title(f"Station {gid}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Date", fontsize=9)
            ax.set_ylabel("NO$_2$ (µg/m³)", fontsize=9)
            ax.xaxis.set_major_formatter(fmt)
            ax.xaxis.set_major_locator(ticks)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right",
                     fontsize=11, fontweight="bold")
            ax.tick_params(axis="y", labelsize=11)
            ax.set_xlim(date_grid[0], date_grid[-1])

            if not legend_done:
                ax.legend(fontsize=8, ncols=2, loc="upper right",
                          framealpha=0.85)
                legend_done = True

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            "Gaussian Process Surrogate — EPA Station UQ\n"
            "Dublin NO$_2$, June 1–29 2023",
            fontsize=13, fontweight="bold", y=1.01
        )
        fig.tight_layout()
        out = output_dir / "epa_station_temporal_uq.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"  Station UQ figure → {out}")


def save_station_table(gp_lur, epa_train, epa_test, output_dir: Path):
    all_grids = sorted(
        set(epa_train["grid_id"].unique()) | set(epa_test["grid_id"].unique())
    )
    rows = []
    for gid in all_grids:
        gp = gp_lur[gp_lur["grid_id"] == gid]
        tr = epa_train[epa_train["grid_id"] == gid]
        te = epa_test[epa_test["grid_id"] == gid]

        all_obs = np.concatenate([tr["value"].values, te["value"].values])
        gp_mean = gp["pred_mean"].mean() if len(gp) else np.nan
        gp_sig  = gp["pred_std"].mean() * Q_HAT_95 if len(gp) else np.nan

        picp = np.nan
        if len(te) > 0:
            gp_te = gp[gp["day"].isin(te["day"].values)].set_index("day")
            covered = 0
            for _, row in te.iterrows():
                if row["day"] in gp_te.index:
                    g    = gp_te.loc[row["day"]]
                    # Conformal PI: mu ± Q_HAT_95 · σ  (no extra 1.96 factor)
                    lo_i = g["pred_mean"] - Q_HAT_95 * g["pred_std"]
                    hi_i = g["pred_mean"] + Q_HAT_95 * g["pred_std"]
                    if lo_i <= row["value"] <= hi_i:
                        covered += 1
            picp = covered / len(te)

        rows.append({
            "station":        gid,
            "n_gp_days":      len(gp),
            "n_train_obs":    len(tr),
            "n_test_obs":     len(te),
            "obs_mean µg/m³": f"{all_obs.mean():.1f}",
            "obs_std µg/m³":  f"{all_obs.std():.1f}",
            "GP_mean µg/m³":  f"{gp_mean:.1f}" if not np.isnan(gp_mean) else "—",
            "GP_sigma_cal":   f"{gp_sig:.1f}"  if not np.isnan(gp_sig)  else "—",
            "bias µg/m³":     f"{gp_mean - all_obs.mean():+.1f}" if not np.isnan(gp_mean) else "—",
            "PICP_95":        f"{picp:.2f}" if not np.isnan(picp) else "—",
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(output_dir / "epa_station_summary.csv", index=False)
    print(f"\n  Station table → {output_dir}/epa_station_summary.csv")


# ---------------------------------------------------------------------------
# Callable entry point (used by run.py)
# ---------------------------------------------------------------------------

def run(tables_dir: Path, figures_dir: Path) -> dict:
    """
    Run EPA station UQ analysis.

    Args:
        tables_dir: Directory for CSV outputs.
        figures_dir: Directory for PNG outputs.

    Returns:
        dict of summary metrics.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("EPA Observations vs GP Predictions  —  Dublin NO₂  (Jun 2023)")
    print("=" * 65)

    print("\n[1] Loading data")
    gp_lur, epa_train, epa_test = load_data()
    all_grids = sorted(
        set(epa_train["grid_id"].unique()) | set(epa_test["grid_id"].unique())
    )
    n_stations = len(all_grids)
    print(f"  Stations: {n_stations}  |  "
          f"Train EPA: {len(epa_train)}  |  Test EPA: {len(epa_test)}  |  "
          f"GP LUR rows: {len(gp_lur):,}")

    print("\n[2] Per-station summary")
    print("-" * 65)
    save_station_table(gp_lur, epa_train, epa_test, tables_dir)

    print("\n[3] Plotting EPA vs GP predictions (Day 1 → Day 29)")
    print("-" * 65)
    plot_epa_vs_predictions(gp_lur, epa_train, epa_test, figures_dir)

    # Extract summary metrics from saved table
    tbl = pd.read_csv(tables_dir / "epa_station_summary.csv")
    picp_vals = pd.to_numeric(tbl["PICP_95"], errors="coerce").dropna()
    mean_picp = float(picp_vals.mean()) if len(picp_vals) else float("nan")
    bias_vals = pd.to_numeric(
        tbl["bias µg/m³"].astype(str).str.replace("+", "", regex=False),
        errors="coerce",
    ).dropna()
    mean_bias = float(bias_vals.mean()) if len(bias_vals) else float("nan")

    return {
        "n_stations":   n_stations,
        "mean_picp_95": f"{mean_picp:.3f}",
        "mean_bias":    f"{mean_bias:+.2f}",
    }


# ---------------------------------------------------------------------------
# Main (standalone)
# ---------------------------------------------------------------------------

def main():
    run_ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT_DIR / "output" / "epa_station_uq" / run_ts
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")
    run(tables_dir=output_dir, figures_dir=output_dir)


if __name__ == "__main__":
    main()
