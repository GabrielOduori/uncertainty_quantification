"""
run.py — Single entry point for the UQ pipeline.

Usage
-----
    python run.py --stage check           # Verify data + model exist
    python run.py --stage predictions     # Generate GP predictions from checkpoint
    python run.py --stage analytical      # GP surrogate UQ (temporal + spatial)
    python run.py --stage mc              # Monte Carlo UQ (three methods)
    python run.py --stage stations        # EPA station temporal UQ
    python run.py --stage kriging         # Kriging/GPR demo figures
    python run.py --stage report          # Aggregate last metrics into report only
    python run.py --stage all             # Full pipeline (check → report)

All outputs go to:
    results/<YYYYMMDD_HHMMSS>/tables/   — CSV tables
    results/<YYYYMMDD_HHMMSS>/figures/  — PNG figures
    results/<YYYYMMDD_HHMMSS>/report.md — auto-generated summary
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Prerequisite check
# ---------------------------------------------------------------------------

def check_prerequisites(verbose: bool = True) -> tuple[bool, bool]:
    """
    Returns (data_ok, predictions_ok).

    data_ok        — uq_train/val/test.csv AND model checkpoint present
    predictions_ok — predictions_val/test.csv present
    """
    from config import (
        UQ_TRAIN, UQ_VAL, UQ_TEST,
        CHECKPOINT, CHECKPOINT_ALT,
        PREDICTIONS_VAL, PREDICTIONS_TEST,
    )

    ok = True

    for path, label in [
        (UQ_TRAIN, "uq_train.csv"),
        (UQ_VAL,   "uq_val.csv"),
        (UQ_TEST,  "uq_test.csv"),
    ]:
        if path.exists():
            if verbose:
                print(f"  [OK]      {label}")
        else:
            print(f"  [MISSING] {label}: {path}")
            ok = False

    ckpt = CHECKPOINT if CHECKPOINT.exists() else (
        CHECKPOINT_ALT if CHECKPOINT_ALT.exists() else None
    )
    if ckpt:
        if verbose:
            print(f"  [OK]      checkpoint: {ckpt.name}")
    else:
        print(f"  [MISSING] model checkpoint")
        print(f"            checked: {CHECKPOINT}")
        print(f"            checked: {CHECKPOINT_ALT}")
        ok = False

    preds_ok = PREDICTIONS_VAL.exists() and PREDICTIONS_TEST.exists()
    if preds_ok:
        if verbose:
            print("  [OK]      predictions_val.csv  &  predictions_test.csv")
    else:
        if verbose:
            print("  [INFO]    predictions_val/test.csv not found "
                  "— run --stage predictions to generate them")

    return ok, preds_ok


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def generate_report(results_dir: Path, metrics: dict) -> None:
    lines = [
        "# UQ Pipeline Report",
        f"**Run:** `{results_dir.name}`",
        "",
    ]
    for stage, m in metrics.items():
        lines.append(f"## {stage}")
        for k, v in m.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    (results_dir / "report.md").write_text("\n".join(lines))

    tables_dir = results_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    with open(tables_dir / "run_summary.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)

    print(f"  Report → {results_dir / 'report.md'}")


# ---------------------------------------------------------------------------
# Stage runner helper
# ---------------------------------------------------------------------------

def run_stage(name: str, fn, tables_dir: Path, figures_dir: Path) -> dict:
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"STAGE: {name.upper()}")
    print(f"{'='*60}")
    result = fn(tables_dir, figures_dir)
    return result or {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STAGES = [
    "check",
    "predictions",
    "analytical",
    "mc",
    "stations",
    "kriging",
    "report",
    "all",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UQ Pipeline — Dublin NO₂ uncertainty quantification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="check",
        help="Pipeline stage to run (default: check)",
    )
    args = parser.parse_args()

    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    from config import RESULTS_DIR
    results_dir = RESULTS_DIR / ts
    tables_dir  = results_dir / "tables"
    figures_dir = results_dir / "figures"

    metrics: dict[str, dict] = {}

    # ── Always run check first ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STAGE: CHECK")
    print(f"{'='*60}")
    data_ok, preds_ok = check_prerequisites(verbose=True)
    if not data_ok:
        print("\n[ERROR] Missing required data or model checkpoint. Aborting.")
        sys.exit(1)

    # ── predictions ───────────────────────────────────────────────────────────
    if args.stage in ("predictions", "all"):
        if preds_ok and args.stage == "all":
            print("\n[SKIP] predictions_val/test.csv already present. "
                  "Delete them to force regeneration.")
        else:
            print(f"\n{'='*60}")
            print("STAGE: PREDICTIONS")
            print(f"{'='*60}")
            from src.generate_predictions import run as _run_preds
            _run_preds()
            metrics["predictions"] = {"status": "generated"}

    # ── analytical ────────────────────────────────────────────────────────────
    if args.stage in ("analytical", "all"):
        from src.analytical_uq_dublin import run as _run_analytical
        metrics["analytical"] = run_stage(
            "analytical", _run_analytical, tables_dir, figures_dir
        )

    # ── mc ────────────────────────────────────────────────────────────────────
    if args.stage in ("mc", "all"):
        from src.mc_uq_dublin import run as _run_mc
        metrics["mc"] = run_stage("mc", _run_mc, tables_dir, figures_dir)

    # ── stations ──────────────────────────────────────────────────────────────
    if args.stage in ("stations", "all"):
        from src.epa_station_uq import run as _run_stations
        metrics["stations"] = run_stage(
            "stations", _run_stations, tables_dir, figures_dir
        )

    # ── kriging ───────────────────────────────────────────────────────────────
    if args.stage in ("kriging", "all"):
        from src.kriging_gpr_demo import run as _run_kriging
        metrics["kriging"] = run_stage(
            "kriging", _run_kriging, tables_dir, figures_dir
        )

    # ── report ────────────────────────────────────────────────────────────────
    if args.stage in ("report", "all") and metrics:
        print(f"\n{'='*60}")
        print("STAGE: REPORT")
        print(f"{'='*60}")
        generate_report(results_dir, metrics)

    if metrics:
        print(f"\n{'='*60}")
        print(f"Done.  Results → {results_dir}")
        print(f"{'='*60}")
    elif args.stage == "check":
        print("\n[OK] All prerequisites satisfied.")


if __name__ == "__main__":
    main()
