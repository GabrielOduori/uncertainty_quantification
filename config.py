"""
config.py — Shared constants for the UQ pipeline.

Import from any experiment script or run.py:
    from config import DATA_DIR, RESULTS_DIR, WHO_DAILY, Q_HAT_95
"""
from pathlib import Path

# ── Directories ──────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent
DATA_DIR    = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

# ── Model checkpoint ─────────────────────────────────────────────────────────
CHECKPOINT     = DATA_DIR / "checkpoints" / "best_model.pt"
CHECKPOINT_ALT = DATA_DIR / "checkpoints" / "fusiongp_model.pt"

# ── Derived prediction files (written by generate_predictions.py) ─────────────
PREDICTIONS_VAL  = DATA_DIR / "predictions_val.csv"
PREDICTIONS_TEST = DATA_DIR / "predictions_test.csv"

# ── Raw UQ data splits ────────────────────────────────────────────────────────
UQ_TRAIN = DATA_DIR / "uq" / "uq_train.csv"
UQ_VAL   = DATA_DIR / "uq" / "uq_val.csv"
UQ_TEST  = DATA_DIR / "uq" / "uq_test.csv"

# ── Domain constants ──────────────────────────────────────────────────────────
WHO_ANNUAL  = 10.0   # µg/m³ — WHO annual guideline (revised 2021)
WHO_DAILY   = 25.0   # µg/m³ — WHO 24-hour guideline
EU_ANNUAL   = 40.0   # µg/m³ — EU annual limit (Directive 2008/50/EC)

N_DAYS      = 29     # study period: June 1–29 2023
Q_HAT_95    = 2.8121 # conformal quantile (calibrated on satellite val set, n=946)
Q_HAT_90    = 2.7688
