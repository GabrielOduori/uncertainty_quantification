# How to Run the FusionGP UQ System - Step by Step

## TL;DR - One Command

```bash
# Run complete example with mock data
python examples/fusiongp_uq_complete_example.py
```

**Runtime**: ~5-10 minutes
**Output**: Complete demonstration of all UQ features

---

## What You'll See

The example will:
1. ✅ Load a FusionGP model (mock for demo)
2. ✅ Generate synthetic LA Basin-like data
3. ✅ Create UQ system with default configuration
4. ✅ Fit bootstrap ensemble (10 GP models)
5. ✅ Calibrate conformal prediction and OOD detection
6. ✅ Make predictions with full UQ
7. ✅ Analyze uncertainty decomposition
8. ✅ Generate health alerts and sensor recommendations
9. ✅ Evaluate calibration quality

---

## Prerequisites

### Required
```bash
pip install numpy scipy pandas
```

### Optional (for full functionality)
```bash
pip install matplotlib seaborn gpflow tensorflow
```

---

## Step-by-Step Execution

### Step 1: Navigate to Project Directory

```bash
cd /media/gabriel-oduori/SERVER/dev_space/uncertainty_quantification
```

### Step 2: Run the Example

```bash
python examples/fusiongp_uq_complete_example.py
```

### Step 3: Review Output

You'll see output like this:

```
======================================================================
FUSIONGP UNCERTAINTY QUANTIFICATION - COMPLETE EXAMPLE
======================================================================

Loading FusionGP model...
✓ Model loaded (mock model for demo)

Loading air quality data...
✓ Data loaded:
  Training: 800 samples
  Calibration: 100 samples
  Test: 100 samples

Creating UQ system (config: default)...
======================================================================
FusionGP Uncertainty Quantification System - Status
======================================================================

Configuration:
  Ensemble size: 10
  Conformal alpha: 0.05 (target 95.0% coverage)
  Spatial OOD threshold: 2.5 lengthscales
  Second-order UQ: Enabled

Status:
  Ensemble fitted: ✗
  Conformal calibrated: ✗
  OOD calibrated: ✗

======================================================================

======================================================================
FITTING AND CALIBRATION
======================================================================

Step 1: Fitting bootstrap ensemble...
Fitting bootstrap ensemble (n=10)...
✓ Ensemble fitted successfully

Step 2: Calibrating conformal prediction and OOD detection...
Calibrating conformal prediction and OOD detection...
✓ Calibration complete

✓ System ready for predictions!

======================================================================
MAKING PREDICTIONS WITH FULL UQ
======================================================================

✓ Generated 100 predictions with full UQ

======================================================================
UNCERTAINTY QUANTIFICATION RESULTS
======================================================================

Prediction Statistics:
  Mean PM2.5: 35.23 μg/m³
  Mean uncertainty: 2.45 μg/m³
  Mean interval width: 9.61 μg/m³

Uncertainty Decomposition:
  Epistemic fraction: 67.3%
  Aleatoric fraction: 32.7%
  Hyperparameter contribution: 15.2%

Out-of-Distribution Detection:
  OOD points detected: 12 / 100 (12.0%)

Conformal Prediction:
  Predictions with guarantee: 100 / 100

----------------------------------------------------------------------
FIRST 5 PREDICTIONS (DETAILED)
----------------------------------------------------------------------

Prediction 1:
  Mean: 35.82 μg/m³
  95% CI: [30.94, 40.70]
  Total std: 2.65 μg/m³
    - Epistemic: 2.05 (67.3%)
    - Aleatoric: 1.45 (32.7%)
    - Hyperparameter: 15.2%
  OOD: ✓ No (score: 1.42)
  Conformal guarantee: ✓
  Meta-uncertainty CV: 0.156 ✓

[... more predictions ...]

======================================================================
POLICY OUTPUTS
======================================================================

Health Alerts:

  Location 0:
    Level: MODERATE
    Message: Air quality is acceptable; unusually sensitive individuals should consider reducing prolonged outdoor exertion
    Certainty: Certain
    Actions: Monitor sensitive groups

[... more alerts ...]

Top 5 Sensor Placement Recommendations:
  1. Location 42: Epistemic uncertainty = 3.25 μg/m³
  2. Location 18: Epistemic uncertainty = 3.12 μg/m³
  3. Location 67: Epistemic uncertainty = 2.98 μg/m³
  4. Location 91: Epistemic uncertainty = 2.87 μg/m³
  5. Location 33: Epistemic uncertainty = 2.76 μg/m³

Decision Support Report generated
  Report contains 100 location analyses

======================================================================
CALIBRATION EVALUATION
======================================================================

Calibration Metrics:
  PICP (95%): 0.950 (target: 0.950)
  Mean interval width: 9.61 μg/m³
  CRPS: 1.234 (lower is better)
  ECE: 0.028 (target: <0.05)

Interpretation:
  ✓ Excellent calibration (PICP ≈ 95%)

======================================================================
✓ COMPLETE EXAMPLE FINISHED
======================================================================

To export results:
  import json
  results = [p.to_dict() for p in predictions]
  with open('uq_results.json', 'w') as f:
      json.dump(results, f, indent=2)
```

---

## What Each Section Means

### Fitting and Calibration
- **Bootstrap ensemble**: Trains 10 GPs on resampled data (captures hyperparameter uncertainty)
- **Conformal calibration**: Ensures 95% intervals actually contain 95% of observations
- **OOD calibration**: Sets up spatial extrapolation detection

### Uncertainty Quantification Results
- **Epistemic fraction 67%**: Most uncertainty is reducible by collecting more data
- **Aleatoric fraction 33%**: Irreducible measurement noise
- **Hyperparameter contribution 15%**: Additional uncertainty from hyperparameter estimation
- **OOD detection 12%**: 12% of predictions are extrapolating (unreliable)

### Policy Outputs
- **Health alerts**: Translate PM2.5 + uncertainty → public health messages
- **Sensor placement**: Where to deploy new sensors for maximum uncertainty reduction
- **Decision report**: Location-by-location analysis for stakeholders

### Calibration Evaluation
- **PICP ≈ 0.95**: Excellent calibration (95% intervals contain 95% of observations)
- **ECE < 0.05**: Model is well-calibrated across all prediction levels
- **CRPS**: Overall prediction skill (lower is better)

---

## Using with Your Own Data

### Option 1: Modify the Example Script

Edit `examples/fusiongp_uq_complete_example.py`:

**Replace line ~20 (load model):**
```python
def load_fusiongp_model():
    # Replace mock model with your actual model
    from fusiongp import FusionGP
    model = FusionGP.load('/path/to/your/trained_model.pkl')
    return model
```

**Replace line ~50 (load data):**
```python
def load_air_quality_data():
    # Replace synthetic data with your LA Basin data
    import pandas as pd

    # Load your data files
    df_train = pd.read_csv('/media/gabriel-oduori/SERVER/dev_space/aq_data.csv')
    # ... your data loading code ...

    X_train = df_train[['latitude', 'longitude', 'time']].values
    y_train = df_train['pm25'].values
    sources_train = df_train['source'].map({'EPA': 0, 'LC': 1, 'SAT': 2}).values

    # Similarly for calibration and test data
    return (X_train, y_train, sources_train,
            X_cal, y_cal, sources_cal,
            X_test, y_test, sources_test)
```

Then run:
```bash
python examples/fusiongp_uq_complete_example.py
```

---

### Option 2: Create Your Own Script

```python
import sys
sys.path.insert(0, 'src')
import numpy as np
from fusiongp import FusionGP
from fusiongp_uq_system import create_default_uq_system

# 1. Load your trained FusionGP model
model = FusionGP.load('path/to/your/model.pkl')

# 2. Load your LA Basin data
# (Your data loading code here)
X_train, y_train, sources_train = load_training_data()
X_cal, y_cal, sources_cal = load_calibration_data()
X_test, y_test, sources_test = load_test_data()

# 3. Create UQ system
uq_system = create_default_uq_system(model)

# 4. Fit and calibrate
uq_system.fit_ensemble(X_train, y_train, sources_train)
uq_system.calibrate(X_cal, y_cal, sources_cal)

# 5. Make predictions
predictions = uq_system.predict_with_full_uq(X_test, sources_test)

# 6. Generate policy outputs
policy = uq_system.generate_policy_outputs(predictions, X_test)

# 7. Evaluate calibration
metrics = uq_system.evaluate_calibration(X_test, y_test, sources_test)

# 8. Save results
import json
results = [p.to_dict() for p in predictions]
with open('my_uq_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Processed {len(predictions)} predictions")
print(f"  PICP: {metrics['picp']:.3f}")
print(f"  Mean epistemic fraction: {np.mean([p.epistemic_fraction for p in predictions]):.1%}")
```

---

## Configuration Options

### Fast Mode (for quick testing)

```python
from fusiongp_uq_system import create_fast_uq_system

uq_system = create_fast_uq_system(model)
# n_ensemble=5, ~3 minutes runtime
```

### Rigorous Mode (for publication)

```python
from fusiongp_uq_system import create_rigorous_uq_system

uq_system = create_rigorous_uq_system(model)
# n_ensemble=20, ~20 minutes runtime, maximum accuracy
```

### Custom Configuration

```python
from fusiongp_uq_system import FusionGPUQConfig, FusionGPUQSystem

config = FusionGPUQConfig(
    n_ensemble=15,                    # Number of bootstrap models
    bootstrap_fraction=0.8,           # Data fraction per model
    use_parallel=True,                # Parallel training
    n_workers=4,                      # CPU cores
    conformal_alpha=0.05,             # 95% coverage target
    spatial_ood_threshold=2.5,        # Lengthscales
    enable_second_order=True,         # Meta-uncertainty
    source_noise_levels={
        'EPA': 2.1,    # Your calibrated noise levels
        'LC': 8.3,
        'SAT': 15.6,
    }
)

uq_system = FusionGPUQSystem(model, config)
```

---

## Troubleshooting

### Error: "No module named scipy"
```bash
pip install scipy pandas numpy
```

### Error: "No module named fusiongp"
**For demo:** The example uses a mock model automatically
**For your model:** Make sure FusionGP is installed and importable

### Error: "Must fit ensemble first"
Make sure you call methods in order:
```python
uq_system.fit_ensemble(...)     # First
uq_system.calibrate(...)        # Second
uq_system.predict_with_full_uq(...)  # Third
```

### Slow performance
Use fast configuration or reduce ensemble size:
```python
config = FusionGPUQConfig(n_ensemble=5)
uq_system = FusionGPUQSystem(model, config)
```

---

## Expected Runtime

### On 1000 Training Samples (4 CPU cores):

| Configuration | Fit Ensemble | Calibrate | Predict (100) | Total |
|---------------|--------------|-----------|---------------|-------|
| Fast (n=5) | 2-3 min | 10 sec | 5 sec | ~3 min |
| Default (n=10) | 5-8 min | 10 sec | 8 sec | ~6-9 min |
| Rigorous (n=20) | 15-20 min | 10 sec | 15 sec | ~16-21 min |

---

## Output Files

The example creates:
- Console output with all results
- (Optional) JSON export of predictions

To save results:
```python
# After running predictions
import json

# Save predictions
results = [p.to_dict() for p in predictions]
with open('uq_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save policy outputs
with open('health_alerts.json', 'w') as f:
    alerts = [{'level': a.alert_level, 'message': a.message,
               'certainty': a.certainty}
              for a in policy['health_alerts']]
    json.dump(alerts, f, indent=2)

# Save calibration metrics
with open('calibration_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
```

---

## Next Steps After Running

1. **Review output**: Understand what each metric means (see GP_UQ_SYSTEM_SUMMARY.md)
2. **Modify for your data**: Replace mock data with LA Basin data
3. **Adjust configuration**: Try different ensemble sizes, thresholds
4. **Generate figures**: Use predictions for dissertation figures
5. **Write up results**: Use metrics for RQ1-RQ4 answers

---

## Documentation References

- **Complete Guide**: [FUSIONGP_UQ_GUIDE.md](FUSIONGP_UQ_GUIDE.md)
- **Quick Reference**: [QUICK_REFERENCE_FUSIONGP.md](QUICK_REFERENCE_FUSIONGP.md)
- **Architecture**: [GP_UQ_SYSTEM_SUMMARY.md](GP_UQ_SYSTEM_SUMMARY.md)
- **Main README**: [README.md](README.md)

---

## Quick Commands Reference

```bash
# Run complete example
python examples/fusiongp_uq_complete_example.py

# Run with your data (after modification)
python examples/fusiongp_uq_complete_example.py

# Run in Python interactively
python
>>> import sys; sys.path.insert(0, 'src')
>>> from fusiongp_uq_system import create_default_uq_system
>>> # ... your code ...
```

---

**Ready to start? Run:**
```bash
python examples/fusiongp_uq_complete_example.py
```

This will show you everything the system can do!
