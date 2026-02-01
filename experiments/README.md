# Experiments: Reproducing Paper Results

This directory contains scripts to reproduce all results for the dissertation chapter on **Rigorous Uncertainty Quantification for Air Quality Fusion Models**.

## Quick Start

### Option 1: Run Everything (Recommended)

```bash
# From project root
python experiments/reproduce_paper.py
```

Or use the convenience script:
```bash
./reproduce_paper.sh
```

### Option 2: Run Validation Only

```bash
python run_validation.py
```

---

## What Gets Generated

When you run `reproduce_paper.py`, it generates:

```
results/
├── paper_results.txt              # Summary of all findings
├── tables/
│   └── decision_report.csv        # Decision support table
└── raw_data/
    └── all_results.json           # Raw experimental results
```

---

## Experiments Overview

### Experiment 1: Uncertainty Decomposition (RQ1)
**Question**: How much of prediction uncertainty is reducible (epistemic) vs irreducible (aleatoric)?

**Methods**:
- Hierarchical variance tracking through fusion stages
- Epistemic/Aleatoric decomposition

**Expected Output**:
- Epistemic fraction: 60-80% far from sensors
- Aleatoric fraction: 60-70% near monitors

---

### Experiment 2: Hyperparameter Uncertainty (RQ2)
**Question**: How much do point estimates underestimate total uncertainty?

**Methods**:
- Bootstrap ensemble (n=10 models)
- Law of total variance: E[Var] + Var[E]

**Expected Output**:
- Underestimation: 10-30%
- Hyperparameter contribution: 15-25%

---

### Experiment 3: Model Calibration (RQ3)
**Question**: Is the model well-calibrated?

**Methods**:
- PICP, ECE, CRPS metrics
- Conformal prediction with guarantees

**Expected Output**:
- PICP(95%): ~0.95
- ECE: <0.05
- Conformal coverage: ≥0.95

---

### Experiment 4: OOD Detection (RQ4)
**Question**: Can OOD detection improve coverage?

**Methods**:
- Second-order uncertainty analysis
- Coefficient of variation (CV) thresholding

**Expected Output**:
- Identify 10-20% unreliable predictions
- CV > 0.3 flags high meta-uncertainty

---

### Experiment 5: Policy Outputs
**Question**: How to translate UQ to actionable decisions?

**Methods**:
- Health alert generation
- Sensor placement recommendations
- Decision support reports

**Expected Output**:
- Uncertainty-aware alerts
- Top-10 sensor locations
- Policy-relevant summaries

---

## Using Your Own Data

### Step 1: Modify `reproduce_paper.py`

Replace the `load_data()` method (around line 80):

```python
def load_data(self):
    """Load your actual data."""

    # Replace synthetic data with your loader
    from your_module import load_la_basin_data

    X_train, y_train, sources_train = load_la_basin_data('train')
    X_test, y_test, sources_test = load_la_basin_data('test')

    self.X_train = X_train
    self.y_train = y_train
    # ... etc
```

### Step 2: Modify `build_model()`

Replace the mock model (around line 120):

```python
def build_model(self):
    """Load your trained model."""

    from fusiongp import FusionGP
    self.model = FusionGP.load('path/to/trained_model.pkl')
```

### Step 3: Run

```bash
python experiments/reproduce_paper.py
```

---

## Runtime Expectations

**Synthetic Data** (default):
- Full pipeline: ~5-10 minutes
- Per experiment: 1-2 minutes each

**LA Basin Data** (1000+ samples):
- Full pipeline: ~20-30 minutes
- Bootstrap ensemble: ~15 minutes (dominant)

**To Speed Up**:
```python
# In reproduce_paper.py, reduce ensemble size
ensemble = BootstrapSVGPEnsemble(n_ensemble=5)  # Instead of 10
```

---

## Output Format

### `paper_results.txt` Structure

```
================================================================================
RIGOROUS UQ FOR AIR QUALITY FUSION: PAPER RESULTS
================================================================================

RESEARCH QUESTION 1: Uncertainty Decomposition
--------------------------------------------------------------------------------
Finding:
  - Epistemic (reducible): 67.3%
  - Aleatoric (irreducible): 32.7%
  - Total uncertainty: 8.45 μg/m³

RESEARCH QUESTION 2: Hyperparameter Uncertainty
--------------------------------------------------------------------------------
Finding:
  - Mean underestimation: 15.2%
  - Median underestimation: 14.1%
  - Hyperparameter contribution: 18.5%

... (continues for all RQs)
```

### `decision_report.csv` Structure

| location_id | mean_pm25 | std_pm25 | lower_95ci | upper_95ci | prob_exceed_moderate | certainty_moderate |
|-------------|-----------|----------|------------|------------|---------------------|-------------------|
| 0 | 45.3 | 5.2 | 35.1 | 55.5 | 0.73 | Likely |
| 1 | 52.1 | 6.8 | 38.8 | 65.4 | 0.89 | Certain |

---

## Customization Options

### Change Output Directory

```python
reproduction = PaperReproduction(output_dir="custom_results")
```

### Run Specific Experiments

```python
reproduction = PaperReproduction()
reproduction.load_data()
reproduction.build_model()

# Run only RQ2
reproduction.experiment_rq2_hyperparameter_uncertainty()
```

### Adjust Ensemble Size

```python
# In reproduce_paper.py
ensemble = BootstrapSVGPEnsemble(
    n_ensemble=20,  # More models = better UQ, slower
    parallel=True,   # Use parallel training
    n_workers=4      # Number of CPU cores
)
```

---

## Troubleshooting

### Error: "No module named scipy"

```bash
pip install scipy pandas numpy
```

### Error: "Model prediction failed"

Check that your model has a `predict_f()` method returning (mean, variance).

### Error: "Out of memory"

Reduce ensemble size or use smaller dataset:
```python
ensemble = BootstrapSVGPEnsemble(n_ensemble=5)
```

### Results Look Wrong

Make sure you're using your actual trained model, not the mock model.

---

## File Descriptions

| File | Purpose | Status |
|------|---------|--------|
| `reproduce_paper.py` | Main reproduction script | ✅ Production-ready |
| `comprehensive_validation.py` | Full validation pipeline | ✅ Production-ready |
| `README.md` | This file | ✅ Complete |

---

## Expected Runtime by Component

| Component | Time | Note |
|-----------|------|------|
| Data loading | <1s | Instant for synthetic |
| Model building | <1s | Mock model |
| RQ1: Decomposition | 5-10s | Fast |
| RQ2: Bootstrap ensemble | 5-15min | Dominant |
| RQ3: Calibration | 10-30s | Moderate |
| RQ4: OOD detection | 10-20s | Fast |
| RQ5: Policy outputs | 5-10s | Fast |
| **Total** | **6-16 min** | **Full pipeline** |

---

## Next Steps After Running

1. **Review** `results/paper_results.txt`
2. **Check** `results/tables/decision_report.csv`
3. **Use results** in dissertation chapter
4. **Generate figures** for publication
5. **Validate** with real LA Basin data

---

## Citation

If you use these experiments:

```bibtex
@phdthesis{oduori2025uq,
  title={Rigorous Uncertainty Quantification for Air Quality Fusion Models},
  author={Oduori, Gabriel},
  year={2025},
  note={Experiments reproduced via reproduce_paper.py}
}
```

---

**Ready to run!** Execute `python experiments/reproduce_paper.py` to start.
