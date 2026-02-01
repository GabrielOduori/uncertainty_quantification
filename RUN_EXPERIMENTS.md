# 🚀 How to Run Experiments - Simple Guide

## TL;DR - One Command

```bash
# Reproduce ALL paper results
python experiments/reproduce_paper.py
```

---

## What This Does

Runs **5 experiments** to answer your research questions:

1. ✅ **RQ1**: Uncertainty decomposition (epistemic vs aleatoric)
2. ✅ **RQ2**: Hyperparameter uncertainty quantification
3. ✅ **RQ3**: Model calibration evaluation
4. ✅ **RQ4**: OOD detection efficacy
5. ✅ **Policy outputs**: Actionable decisions

**Output**: `results/paper_results.txt` with all findings

---

## Prerequisites

Install scipy (one-time):
```bash
pip install scipy pandas numpy
```

---

## Run Options

### Option 1: Reproduce Paper (⭐ RECOMMENDED)
```bash
python experiments/reproduce_paper.py
```
- **Runtime**: 6-16 minutes
- **Outputs**:
  - `results/paper_results.txt` (summary)
  - `results/tables/decision_report.csv` (decisions)
  - `results/raw_data/all_results.json` (raw data)
- **Best for**: Dissertation chapter

---

### Option 2: Quick Validation
```bash
python run_validation.py
```
- **Runtime**: 2-5 minutes
- **Outputs**: Console output only
- **Best for**: Quick testing

---

### Option 3: Use Shell Script
```bash
./reproduce_paper.sh
```
- Checks dependencies automatically
- Runs reproduce_paper.py
- Shows results location

---

## Expected Output

After running, you'll see:

```
================================================================================
RIGOROUS UQ FOR AIR QUALITY FUSION: PAPER RESULTS
================================================================================

RESEARCH QUESTION 1: Uncertainty Decomposition
--------------------------------------------------------------------------------
Finding:
  - Epistemic (reducible): 67.3%
  - Aleatoric (irreducible): 32.7%

RESEARCH QUESTION 2: Hyperparameter Uncertainty
--------------------------------------------------------------------------------
Finding:
  - Mean underestimation: 15.2%
  - Hyperparameter contribution: 18.5%

... (continues for all RQs)

✅ ALL EXPERIMENTS COMPLETE
Results saved to: results/
```

---

## Customize for Your Data

Edit `experiments/reproduce_paper.py`:

```python
# Line 80: Replace load_data()
def load_data(self):
    from your_module import load_la_basin_data
    X_train, y_train, sources_train = load_la_basin_data('train')
    # ...

# Line 120: Replace build_model()
def build_model(self):
    from fusiongp import FusionGP
    self.model = FusionGP.load('your_model.pkl')
```

---

## Troubleshooting

**Error: No module named scipy**
```bash
pip install scipy
```

**Error: Permission denied**
```bash
chmod +x reproduce_paper.sh
```

**Too slow?** Reduce ensemble size in `reproduce_paper.py`:
```python
ensemble = BootstrapSVGPEnsemble(n_ensemble=5)  # Instead of 10
```

---

## What You Get

### Files Created:
```
results/
├── paper_results.txt              # Main findings (READ THIS FIRST)
├── tables/
│   └── decision_report.csv        # Decision support table
└── raw_data/
    └── all_results.json           # All experimental data
```

### Use in Your Dissertation:
1. Copy findings from `paper_results.txt`
2. Use decision_report.csv for policy section
3. Generate figures from raw_data/all_results.json

---

## Quick Reference

| Command | Purpose | Runtime | Output |
|---------|---------|---------|--------|
| `python experiments/reproduce_paper.py` | Full experiments | 6-16 min | Paper results |
| `python run_validation.py` | Quick test | 2-5 min | Console only |
| `./reproduce_paper.sh` | Automated | 6-16 min | Paper results |
| `python test_installation.py` | Check install | <1 min | Pass/fail |

---

**Ready?** Run: `python experiments/reproduce_paper.py`

See [experiments/README.md](experiments/README.md) for detailed documentation.
