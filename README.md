# Uncertainty Quantification for Multi-Source NO₂ Air Quality Fusion

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of the **Uncertainty Quantification** dissertation chapter for probabilistic fusion of multi-source NO₂ observations (EPA monitors, satellite retrievals, LUR model) over Dublin.

Builds on **[FusionGP](https://github.com/GabrielOduori/fusiongp)** — Sparse Variational GP fusion — and **[GAM-SSM-LUR](https://github.com/GabrielOduori/gam_ssm_lur)** — Land Use Regression baseline.

---

## Installation

```bash
git clone https://github.com/GabrielOduori/uncertainty_quantification.git
cd uncertainty_quantification

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Data

All scripts read from `data/uq/`:

| File | Description |
|------|-------------|
| `uq_train.csv` | Training observations — EPA, satellite, LUR sources |
| `uq_val.csv` | Validation set — satellite + LUR (used for conformal calibration) |
| `uq_test.csv` | Test set — EPA ground truth + satellite + LUR predictions |

Columns used: `source`, `value` (observed NO₂ µg/m³), `lur_no2` (LUR prediction), `latitude`, `longitude`, `timestamp`.

---

## Running the Pipeline

All analysis is driven by a single entry point. Run from the project root:

```bash
# Check that data and model checkpoint are present
python run.py --stage check

# Run individual stages
python run.py --stage predictions  # Generate GP predictions from checkpoint
python run.py --stage analytical   # GP surrogate UQ (temporal + spatial)
python run.py --stage stations     # EPA station comparison
python run.py --stage kriging      # Kriging/GPR demo figures
python run.py --stage mc           # Monte Carlo UQ

# Run everything
python run.py --stage all
```

Outputs go to a single timestamped folder:

```
results/<YYYYMMDD_HHMMSS>/
  tables/
    temporal_uq_daily.csv          # 29-day GP temporal UQ
    uq_results_table.csv           # calibration metrics (GP raw vs conformal)
    source_uncertainty_decomposition.csv
    epa_station_summary.csv        # per-station PICP, bias, GP mean
    mc_uq_results.csv              # Monte Carlo summary
    run_summary.json               # all stage metrics
  figures/
    temporal_uq_june2023.png       # temporal CI plot (June 2023)
    spatial_uq_panel.png           # spatial UQ grid
    epa_station_temporal_uq.png    # per-station timeseries
    gpr_kriging_panel.png          # Kriging demo panel
    gpr_temporal_uq.png            # temporal GPR
    mc_uq_panel.png                # MC 3×3 panel
  report.md                        # auto-generated summary
```

Key results (Dublin NO₂, EPA test set n=37):

| Metric | Value |
|--------|-------|
| RMSE | 19.34 µg/m³ |
| CRPS (post-conformal) | 11.40 µg/m³ |
| PICP₉₅ (post-conformal) | 100% (all 9 stations) |
| PI-95 width | 104.94 µg/m³ |
| Epistemic fraction | 61.5% |
| Aleatoric fraction | 38.5% |
| WHO daily (25 µg/m³) exceedance | 77.9% of grid cells |

---

## Project Structure

```
uncertainty_quantification/
├── run.py                        # Single entry point
├── config.py                     # Shared constants (paths, thresholds)
├── src/
│   ├── analytical_uq_dublin.py   # GP surrogate UQ + conformal prediction
│   ├── mc_uq_dublin.py           # Monte Carlo UQ (3 methods)
│   ├── epa_station_uq.py         # EPA station temporal comparison
│   ├── generate_predictions.py   # GP predictions from checkpoint
│   ├── kriging_gpr_demo.py       # Kriging/GPR illustration
│   └── legacy/                   # Archived UQ framework code
├── experiments/
│   ├── mc_uq_demo.py             # Synthetic MC demo (no data required)
│   ├── gp_uq_demo.py             # Synthetic GP demo (no data required)
│   ├── visualization_demo.py     # Visualization examples
│   └── reproduce_paper.py        # Reproduce paper results (RQ1–RQ4)
├── notebooks/
│   └── UQ.ipynb
├── results/                      # Pipeline outputs (timestamped, gitignored)
├── data/uq/                      # Input CSVs (not committed)
├── docs/                         # Dissertation chapter drafts
├── requirements.txt
└── README.md
```

---

## Research Questions

| RQ | Question | Key Finding |
|----|----------|-------------|
| RQ4 | How can UQ be integrated in probabilistic fusion to improve reliability? | Bootstrap MC confirms PICP₉₅ = 95%; epistemic uncertainty dominates (62%) due to sparse EPA coverage |

---

## Citation

```bibtex
@phdthesis{oduori2025uncertainty,
  title   = {Uncertainty Quantification in Probabilistic Air Quality Sensor Fusion},
  author  = {Oduori, Gabriel},
  year    = {2025},
  school  = {University College Dublin}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
