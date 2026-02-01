# How to Run the UQ Framework Experiments

## Quick Start (3 Steps)

### Step 1: Install Dependencies

You need to install the required Python packages:

```bash
# Core dependencies (required)
pip install numpy scipy pandas

# Optional (for visualization)
pip install matplotlib seaborn

# For production with real GPflow models
pip install gpflow tensorflow
```

**Or install all at once:**
```bash
pip install numpy scipy pandas matplotlib seaborn gpflow tensorflow
```

---

### Step 2: Run the Validation

Once dependencies are installed, run the validation script:

```bash
# From the project root directory
cd /media/gabriel-oduori/SERVER/dev_space/uncertainty_quantification

# Run validation
python run_validation.py
```

This will:
- ✅ Test all 6 UQ components
- ✅ Answer research questions (RQ1-RQ4)
- ✅ Generate synthetic data validation
- ✅ Show example outputs

**Expected runtime**: 2-5 minutes

---

### Step 3: Use with Your FusionGP Model

After validation passes, integrate with your model:

```python
from uncertainty import HierarchicalUQTracker
from models import BootstrapSVGPEnsemble

# Load your trained FusionGP model
from fusiongp import FusionGP
model = FusionGP.load('your_model.pkl')

# Load your LA Basin data
X_train, y_train, sources_train = load_your_data()

# Run UQ analysis
tracker = HierarchicalUQTracker()
hierarchical_var = tracker.decompose_by_stage(model, X_test, sources_test)

ensemble = BootstrapSVGPEnsemble(n_ensemble=10)
ensemble.fit(X_train, y_train, sources_train)

# Get results
results = ensemble.predict_with_full_uncertainty(X_test)
```

---

## Alternative: Run Without Installing Dependencies

If you can't install scipy right now, you can test the module imports:

```bash
# Test basic imports (doesn't need scipy)
python -c "import sys; sys.path.insert(0, 'src'); print('Testing imports...')"
```

Then manually check each module:

```python
# Test hierarchical module
import sys
sys.path.insert(0, 'src')

from uncertainty.hierarchical import HierarchicalUQTracker
print("✅ Hierarchical module works!")

from models.ensemble import BootstrapSVGPEnsemble
print("✅ Ensemble module works!")

from decision.policy_translation import PolicyTranslator
print("✅ Decision module works!")
```

---

## Running with Your Own Data

### Option 1: Modify run_validation.py

Replace the synthetic data generation with your actual data:

```python
# In run_validation.py, replace lines 30-45 with:

# Load your data
from your_data_loader import load_la_basin_data

X_train, y_train, sources_train = load_la_basin_data('train')
X_test, y_test, sources_test = load_la_basin_data('test')

# Load your trained model
from fusiongp import FusionGP
model = FusionGP.load('trained_fusion_gp.pkl')
```

### Option 2: Use the Comprehensive Validator

```python
import sys
sys.path.insert(0, 'src')

from experiments.comprehensive_validation import ComprehensiveUQValidator

# Initialize validator
validator = ComprehensiveUQValidator(output_dir='results/la_basin')

# Run with your data and model
results = validator.run_full_validation(
    X_train=X_train_la_basin,
    y_train=y_train_la_basin,
    sources_train=sources_train_la_basin,
    X_test=X_test_la_basin,
    y_test=y_test_la_basin,
    sources_test=sources_test_la_basin,
    point_model=your_fusionGP_model  # Your trained model
)

# Results saved to: results/la_basin/validation_summary.txt
```

---

## Troubleshooting

### Issue: "No module named scipy"

**Solution**: Install scipy
```bash
pip install scipy
```

### Issue: "No module named gpflow"

**For testing**: The code uses mock models automatically
**For production**: Install GPflow
```bash
pip install gpflow tensorflow
```

### Issue: "Import error from uncertainty module"

**Solution**: Make sure you're in the project root:
```bash
cd /media/gabriel-oduori/SERVER/dev_space/uncertainty_quantification
python run_validation.py
```

### Issue: "Parallel training fails"

**Solution**: Use sequential training:
```python
ensemble = BootstrapSVGPEnsemble(n_ensemble=10, parallel=False)
```

---

## What Each Script Does

### `run_validation.py` (Simplified, Recommended)
- ✅ Quick validation without optional dependencies
- ✅ Tests all components with synthetic data
- ✅ Shows example outputs
- ✅ Answers research questions
- **Runtime**: 2-5 minutes

### `experiments/comprehensive_validation.py` (Full)
- ✅ Complete validation pipeline
- ✅ Generates publication-ready outputs
- ✅ Creates figures and reports
- Requires: matplotlib, seaborn
- **Runtime**: 10-20 minutes

### `test_installation.py` (Quick Check)
- ✅ Verifies installation
- ✅ Tests imports
- ✅ Runs basic workflow
- **Runtime**: < 1 minute

---

## Expected Output

When you run `python run_validation.py`, you should see:

```
================================================================================
RIGOROUS UQ FRAMEWORK - VALIDATION RUN
================================================================================

[STEP 1] Generating synthetic air quality data...
  Training data: 500 samples
  Test data: 100 samples
  Features: 3 dimensions

[STEP 2] Testing Hierarchical Variance Propagation...
  Variance by Stage:
    Stage 0 (Raw): EPA=2.10, LC=8.30, SAT=15.60
    Stage 1 (Epistemic): 5.23
    Stage 2 (Predictive): 12.45
  ✅ Hierarchical tracking successful

[STEP 3] Testing Bootstrap Ensemble (n=5 for speed)...
  Within-model σ: 4.52
  Between-model σ: 1.23
  Total σ: 4.68
  Hyperparameter fraction: 15.3%

  🎯 RQ2 Answer: Point estimates underestimate by 12.5%
     (Median: 11.8%)
  ✅ Bootstrap ensemble successful

[STEP 4] Testing Conformal Prediction...
  Calibrated quantile: 1.832
  Actual coverage: 0.950
  Target coverage: 0.950
  Achieves target: True
  Mean interval width: 15.23
  ✅ Conformal prediction successful

... (more output) ...

================================================================================
🚀 FRAMEWORK VALIDATION COMPLETE - ALL TESTS PASSED
================================================================================
```

---

## Quick Command Reference

```bash
# Install dependencies
pip install scipy pandas numpy

# Run validation (after install)
python run_validation.py

# Test installation only
python test_installation.py

# Use in Python
python
>>> import sys; sys.path.insert(0, 'src')
>>> from uncertainty import HierarchicalUQTracker
>>> tracker = HierarchicalUQTracker()
>>> # Now use with your model
```

---

## Next Steps After Validation Passes

1. **Integrate with FusionGP**: See `QUICKSTART_4DAYS.md`
2. **Run on LA Basin data**: Use your trained model
3. **Generate figures**: For dissertation chapter
4. **Answer research questions**: Using validation outputs

---

**Need Help?**
- Check `QUICKSTART_4DAYS.md` for detailed examples
- Read module docstrings for API documentation
- See `IMPLEMENTATION_COMPLETE.md` for technical details
