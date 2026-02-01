# FusionGP Uncertainty Quantification System - Complete Guide

## Overview

This guide shows you how to use the **FusionGP Uncertainty Quantification System** - a production-ready framework that provides rigorous uncertainty quantification for your FusionGP air quality models.

Since FusionGP is built on Gaussian Process regression, it naturally provides probabilistic predictions. This system extends those capabilities with:

1. **Epistemic/Aleatoric decomposition** - Separate reducible from irreducible uncertainty
2. **Hyperparameter uncertainty** - Capture uncertainty from GP hyperparameter estimation
3. **Multi-source tracking** - Handle EPA monitors, low-cost sensors, and satellite data
4. **Out-of-distribution detection** - Identify unreliable extrapolation
5. **Conformal prediction** - Distribution-free coverage guarantees
6. **Second-order UQ** - Quantify uncertainty about uncertainty
7. **Policy outputs** - Translate UQ to actionable decisions

---

## Quick Start (5 Steps)

### Step 1: Import the System

```python
import sys
sys.path.insert(0, 'src')

from fusiongp_uq_system import FusionGPUQSystem, create_default_uq_system
```

### Step 2: Load Your FusionGP Model

```python
from fusiongp import FusionGP

# Load your trained model
model = FusionGP.load('path/to/your/trained_model.pkl')
```

### Step 3: Create UQ System

```python
# Option A: Default configuration (recommended)
uq_system = create_default_uq_system(model)

# Option B: Custom configuration (see Configuration section)
from fusiongp_uq_system import FusionGPUQConfig
config = FusionGPUQConfig(n_ensemble=10, conformal_alpha=0.05)
uq_system = FusionGPUQSystem(model, config)
```

### Step 4: Fit and Calibrate

```python
# Fit bootstrap ensemble (required)
uq_system.fit_ensemble(X_train, y_train, sources_train)

# Calibrate conformal prediction (required)
uq_system.calibrate(X_cal, y_cal, sources_cal)
```

### Step 5: Make Predictions with Full UQ

```python
# Get predictions with complete uncertainty quantification
predictions = uq_system.predict_with_full_uq(X_test, sources_test)

# Each prediction contains:
for pred in predictions:
    print(f"Mean: {pred.mean:.2f} μg/m³")
    print(f"95% CI: [{pred.lower_95:.2f}, {pred.upper_95:.2f}]")
    print(f"Epistemic: {pred.epistemic_fraction:.1%}")
    print(f"OOD: {pred.spatial_ood}")
```

---

## Complete Example

Here's a full working example:

```python
import sys
sys.path.insert(0, 'src')
import numpy as np
from fusiongp import FusionGP
from fusiongp_uq_system import create_default_uq_system

# 1. Load your trained FusionGP model
model = FusionGP.load('trained_fusion_model.pkl')

# 2. Load your LA Basin data
# X: (n_samples, n_features) - typically [lat, lon, time, ...]
# y: (n_samples,) - PM2.5 concentrations
# sources: (n_samples,) - 0=EPA, 1=Low-cost, 2=Satellite
X_train, y_train, sources_train = load_training_data()
X_cal, y_cal, sources_cal = load_calibration_data()
X_test, y_test, sources_test = load_test_data()

# 3. Create UQ system
uq_system = create_default_uq_system(model)
print(uq_system.summary())

# 4. Fit ensemble (this trains n=10 bootstrap models)
print("Fitting ensemble...")
uq_system.fit_ensemble(X_train, y_train, sources_train)

# 5. Calibrate conformal prediction
print("Calibrating...")
uq_system.calibrate(X_cal, y_cal, sources_cal)

# 6. Make predictions with full UQ
print("Making predictions...")
predictions = uq_system.predict_with_full_uq(X_test, sources_test)

# 7. Analyze first prediction
pred = predictions[0]
print(f"\nPrediction at location 0:")
print(f"  Mean: {pred.mean:.2f} μg/m³")
print(f"  95% CI: [{pred.lower_95:.2f}, {pred.upper_95:.2f}]")
print(f"  Total uncertainty: {pred.std:.2f} μg/m³")
print(f"    - Epistemic (reducible): {pred.epistemic_std:.2f} ({pred.epistemic_fraction:.1%})")
print(f"    - Aleatoric (irreducible): {pred.aleatoric_std:.2f}")
print(f"    - Hyperparameter contribution: {pred.hyperparameter_contribution:.1%}")
print(f"  Out-of-distribution: {pred.spatial_ood}")
print(f"  Conformal guarantee: {pred.conformal_guaranteed}")

# 8. Generate policy outputs
policy_outputs = uq_system.generate_policy_outputs(predictions, X_test)

# Health alerts
for alert in policy_outputs['health_alerts'][:5]:
    print(f"\n{alert.message}")
    print(f"  Certainty: {alert.certainty}")

# Sensor placement recommendations
sensor_recs = policy_outputs['sensor_recommendations']
print("\nTop 5 sensor placement locations:")
for i, rec in enumerate(sensor_recs.priority_locations[:5], 1):
    print(f"  {i}. Location {rec['location_id']}: "
          f"Epistemic uncertainty = {rec['epistemic_uncertainty']:.2f}")

# 9. Evaluate calibration
metrics = uq_system.evaluate_calibration(X_test, y_test, sources_test)
print(f"\nCalibration metrics:")
print(f"  PICP (95%): {metrics['picp']:.3f} (target: 0.950)")
print(f"  CRPS: {metrics['crps']:.3f}")
```

---

## Configuration Options

### Default Configuration (Balanced)

```python
from fusiongp_uq_system import create_default_uq_system

uq_system = create_default_uq_system(model)
```

**Settings:**
- Ensemble size: 10 models
- Conformal alpha: 0.05 (95% coverage)
- Second-order UQ: Enabled
- Runtime: ~5-10 minutes on 1000 samples

**Best for:** Most applications, dissertation work

---

### Fast Configuration (Rapid Prototyping)

```python
from fusiongp_uq_system import create_fast_uq_system

uq_system = create_fast_uq_system(model)
```

**Settings:**
- Ensemble size: 5 models (faster)
- Second-order UQ: Disabled
- Runtime: ~2-3 minutes on 1000 samples

**Best for:** Quick testing, real-time applications

---

### Rigorous Configuration (Publication)

```python
from fusiongp_uq_system import create_rigorous_uq_system

uq_system = create_rigorous_uq_system(model)
```

**Settings:**
- Ensemble size: 20 models (more accurate)
- Spatial OOD threshold: 2.0 (more conservative)
- All features enabled
- Runtime: ~15-20 minutes on 1000 samples

**Best for:** Publication, high-stakes decisions

---

### Custom Configuration

```python
from fusiongp_uq_system import FusionGPUQSystem, FusionGPUQConfig

config = FusionGPUQConfig(
    # Ensemble settings
    n_ensemble=15,              # Number of bootstrap models
    bootstrap_fraction=0.8,     # Fraction of data per model
    use_parallel=True,          # Parallel training
    n_workers=4,                # CPU cores to use

    # Conformal prediction
    conformal_alpha=0.05,       # Target 95% coverage
    conformal_adaptive=True,    # Adaptive intervals

    # OOD detection
    spatial_ood_threshold=2.5,  # Lengthscales
    temporal_ood_window=30,     # Days

    # Second-order UQ
    enable_second_order=True,
    meta_uncertainty_threshold=0.3,

    # Source-specific noise levels (μg/m³)²
    source_noise_levels={
        'EPA': 2.1,      # Reference monitors
        'LC': 8.3,       # Low-cost sensors
        'SAT': 15.6,     # Satellite
    }
)

uq_system = FusionGPUQSystem(model, config)
```

---

## Understanding the Outputs

### UQPrediction Object

Each prediction returns a `UQPrediction` object with:

```python
pred = predictions[0]

# Point prediction
pred.mean              # Expected PM2.5 value
pred.std               # Total uncertainty (adjusted for OOD)

# Confidence intervals
pred.lower_95          # Lower bound of 95% CI
pred.upper_95          # Upper bound of 95% CI
pred.interval_width    # Width of interval

# Uncertainty decomposition
pred.epistemic_std            # Reducible uncertainty (GP posterior)
pred.aleatoric_std            # Irreducible uncertainty (noise)
pred.epistemic_fraction       # Fraction that is epistemic (0-1)

# Hyperparameter uncertainty
pred.within_model_std              # Within-model uncertainty
pred.between_model_std             # Between-model uncertainty
pred.hyperparameter_contribution   # Fraction from hyperparameters

# Meta-uncertainty (if enabled)
pred.meta_uncertainty_cv      # Coefficient of variation
pred.uncertainty_reliable     # Is uncertainty estimate reliable?

# Out-of-distribution flags
pred.spatial_ood       # Is prediction extrapolating spatially?
pred.temporal_ood      # Is prediction extrapolating temporally?
pred.ood_score         # OOD score (distance in lengthscales)

# Conformal guarantee
pred.conformal_guaranteed   # Does interval have coverage guarantee?
```

---

## How to Interpret Results

### Epistemic vs Aleatoric Uncertainty

**High Epistemic (>70%)**:
- Location is data-sparse
- **Action**: Deploy more sensors here to reduce uncertainty
- **Example**: Remote areas >10km from monitors

**High Aleatoric (>70%)**:
- Location is well-characterized but inherently noisy
- **Action**: Limited benefit from more data
- **Example**: Near EPA monitors with stable measurements

### Hyperparameter Contribution

**High contribution (>20%)**:
- Model hyperparameters are uncertain
- Single-model predictions underestimate uncertainty
- **Action**: Use ensemble predictions (which you are!)

**Low contribution (<10%)**:
- Hyperparameters are well-determined
- Single model would be sufficient

### Out-of-Distribution Warnings

**Spatial OOD = True**:
- Prediction is >2.5 lengthscales from training data
- GP is extrapolating (unreliable)
- **Action**: Treat prediction with caution, deploy sensors

**Temporal OOD = True**:
- Current conditions differ from historical patterns
- May indicate unusual event (wildfire, etc.)
- **Action**: Flag for manual review

### Meta-Uncertainty (Second-Order)

**High CV (>0.3)**:
- Uncertainty estimate itself is uncertain
- Different ensemble members disagree on uncertainty
- **Action**: Use conservative (upper bound) uncertainty

**Low CV (<0.1)**:
- Uncertainty estimate is reliable
- Ensemble members agree
- **Action**: Trust the reported uncertainty

---

## Policy Outputs

### Health Alerts

```python
policy_outputs = uq_system.generate_policy_outputs(predictions, X_test)

for alert in policy_outputs['health_alerts']:
    print(f"Level: {alert.alert_level}")        # Good, Moderate, USG, Unhealthy, etc.
    print(f"Message: {alert.message}")          # Human-readable description
    print(f"Certainty: {alert.certainty}")      # Certain, Likely, Uncertain
    print(f"Actions: {alert.recommended_actions}")  # Public health actions
```

**Certainty Levels:**
- **Certain**: >90% probability of exceeding threshold
- **Likely**: 70-90% probability
- **Possible**: 30-70% probability
- **Uncertain**: High uncertainty, threshold unclear

### Sensor Placement Recommendations

```python
sensor_recs = policy_outputs['sensor_recommendations']

for rec in sensor_recs.priority_locations[:10]:
    print(f"Location {rec['location_id']}")
    print(f"  Epistemic uncertainty: {rec['epistemic_uncertainty']:.2f}")
    print(f"  Coordinates: {rec['coordinates']}")
```

Prioritizes locations with highest epistemic uncertainty (where more data helps most).

### Decision Support Report

```python
report = policy_outputs['decision_report']

for entry in report:
    print(f"Location: {entry['location_name']}")
    print(f"  Mean PM2.5: {entry['mean_pm25']:.1f}")
    print(f"  95% CI: [{entry['lower_95ci']:.1f}, {entry['upper_95ci']:.1f}]")
    print(f"  Probability exceeding moderate (12.1): {entry['prob_exceed_moderate']:.2f}")
    print(f"  Certainty: {entry['certainty_moderate']}")
```

---

## Calibration Evaluation

Check that your UQ system is well-calibrated:

```python
metrics = uq_system.evaluate_calibration(X_test, y_test, sources_test)

print(f"PICP (95%): {metrics['picp']:.3f}")  # Should be ≈0.95
print(f"ECE: {metrics['ece']:.3f}")          # Should be <0.05
print(f"CRPS: {metrics['crps']:.3f}")        # Lower is better
```

**Interpretation:**

| PICP | Interpretation | Action |
|------|----------------|--------|
| 0.93-0.97 | Excellent calibration | None needed |
| 0.90-0.93 or 0.97-1.0 | Good calibration | Consider recalibration |
| <0.90 or >0.98 | Poor calibration | Check model, recalibrate |

**Expected Calibration Error (ECE):**
- ECE < 0.05: Excellent
- ECE 0.05-0.10: Good
- ECE > 0.10: Needs improvement

---

## Data Requirements

### Training Data

**Minimum:**
- 500+ samples
- Multi-source (EPA + low-cost + satellite)
- Spatial coverage across domain

**Recommended:**
- 1000+ samples
- Balanced source distribution (20% EPA, 50% LC, 30% SAT)
- Temporal coverage (multiple seasons)

### Calibration Data

**Minimum:**
- 100+ samples
- Independent from training set
- Representative of test distribution

**Recommended:**
- 200+ samples
- Recent data (for temporal OOD)
- Spatially distributed

### Data Format

```python
# Features (n_samples, n_features)
X = np.array([
    [lat, lon, time, ...],  # Sample 1
    [lat, lon, time, ...],  # Sample 2
    # ...
])

# Targets (n_samples,)
y = np.array([35.2, 42.1, ...])  # PM2.5 concentrations

# Sources (n_samples,)
sources = np.array([0, 1, 2, ...])  # 0=EPA, 1=LC, 2=SAT
```

---

## Integration with Existing FusionGP

### Your FusionGP Model Requirements

The UQ system works with any FusionGP model that has:

```python
# Required method
mean, var = model.predict_f(X_test)
```

**Returns:**
- `mean`: Predicted values (n_samples,)
- `var`: GP posterior variance (n_samples,)

### Optional (for better OOD detection):

```python
# Optional method
lengthscales = model.get_lengthscales()
```

**Returns:**
- `lengthscales`: List of kernel lengthscales

If this method is not available, the system uses default lengthscales.

---

## Runtime Expectations

### On 1000 Training Samples:

| Configuration | Ensemble Fit | Calibration | Prediction (100 pts) | Total |
|---------------|--------------|-------------|---------------------|-------|
| Fast (n=5) | 2-3 min | 10 sec | 5 sec | ~3 min |
| Default (n=10) | 5-8 min | 10 sec | 8 sec | ~6-9 min |
| Rigorous (n=20) | 15-20 min | 10 sec | 15 sec | ~16-21 min |

**Note:** Times assume parallel training on 4 CPU cores. Sequential training takes ~3x longer.

---

## Advanced Usage

### Exporting Results

```python
import json

# Convert predictions to dictionaries
results = [p.to_dict() for p in predictions]

# Save to JSON
with open('uq_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save policy outputs
with open('health_alerts.json', 'w') as f:
    json.dump([alert.__dict__ for alert in policy_outputs['health_alerts']], f)
```

### Using Only Specific Components

You can use individual components without the full system:

```python
# Just epistemic/aleatoric decomposition
from uncertainty.decomposition import UncertaintyDecomposer
decomposer = UncertaintyDecomposer(model_type='svgp')
components = decomposer.decompose_svgp(model, X_test, sources=sources_test)

# Just conformal prediction
from uncertainty.conformal import ConformalPredictionWrapper
conformal = ConformalPredictionWrapper(model, alpha=0.05)
conformal.calibrate(X_cal, y_cal)
intervals = conformal.predict_with_conformal_intervals(X_test)

# Just OOD detection
from uncertainty.ood_detection import SpatialOODDetector
detector = SpatialOODDetector(X_train, lengthscales=[0.05, 0.05])
ood_flags, scores = detector.detect(X_test)
```

### Custom Source Noise Levels

If you have different noise characteristics:

```python
config = FusionGPUQConfig(
    source_noise_levels={
        'EPA': 1.5,      # Your calibrated noise level
        'LC': 12.0,      # Your low-cost sensor noise
        'SAT': 20.0,     # Your satellite noise
        'CUSTOM': 5.0,   # Additional source type
    }
)
```

---

## Troubleshooting

### Error: "Must fit ensemble first"

**Solution:** Call `fit_ensemble()` before `predict_with_full_uq()`:

```python
uq_system.fit_ensemble(X_train, y_train, sources_train)
uq_system.predict_with_full_uq(X_test, sources_test)
```

### Error: "Could not extract lengthscales"

**Cause:** FusionGP model doesn't have standard lengthscale interface.

**Solution:** System uses default lengthscales automatically. To specify:

```python
# Manually set in spatial OOD detector
uq_system.spatial_ood_detector = SpatialOODDetector(
    X_train,
    lengthscales=[0.05, 0.05]  # Your model's lengthscales
)
```

### Poor Calibration (PICP far from 0.95)

**Possible causes:**
1. Insufficient calibration data (<100 samples)
2. Calibration data not representative of test data
3. Model misspecification

**Solutions:**
1. Increase calibration data size
2. Ensure calibration covers test distribution
3. Use conformal prediction (which adjusts automatically)

### Slow Ensemble Training

**Solutions:**
1. Reduce ensemble size: `n_ensemble=5`
2. Enable parallel training: `use_parallel=True`
3. Use smaller bootstrap fraction: `bootstrap_fraction=0.5`
4. Use fast configuration: `create_fast_uq_system(model)`

---

## Complete Workflow Checklist

- [ ] Load trained FusionGP model
- [ ] Prepare training data (X_train, y_train, sources_train)
- [ ] Prepare calibration data (X_cal, y_cal, sources_cal)
- [ ] Create UQ system with desired configuration
- [ ] Fit bootstrap ensemble
- [ ] Calibrate conformal prediction and OOD detection
- [ ] Make predictions on test data
- [ ] Analyze uncertainty decomposition
- [ ] Generate policy outputs
- [ ] Evaluate calibration metrics
- [ ] Export results for dissertation

---

## Example: Minimal Working Code

```python
# Minimal example (5 lines!)
import sys; sys.path.insert(0, 'src')
from fusiongp import FusionGP
from fusiongp_uq_system import create_default_uq_system

model = FusionGP.load('model.pkl')
uq = create_default_uq_system(model)
uq.fit_ensemble(X_train, y_train, sources_train)
uq.calibrate(X_cal, y_cal, sources_cal)
predictions = uq.predict_with_full_uq(X_test, sources_test)
```

---

## Full Working Example Script

See `examples/fusiongp_uq_complete_example.py` for a complete, runnable example with:
- Mock FusionGP model
- Synthetic LA Basin data
- All UQ features demonstrated
- Policy outputs
- Calibration evaluation

**Run it:**
```bash
python examples/fusiongp_uq_complete_example.py
```

---

## Next Steps

1. **Run the example**: `python examples/fusiongp_uq_complete_example.py`
2. **Replace with your data**: Modify `load_fusiongp_model()` and `load_air_quality_data()`
3. **Integrate into your workflow**: Use in your analysis scripts
4. **Generate results for dissertation**: Use policy outputs and calibration metrics

---

## References

**Key Papers:**
1. Shawe-Taylor & Cristianini (2004) - Kernel Methods
2. Quinonero-Candela & Rasmussen (2005) - GP approximations
3. Romano et al. (2019) - Conformal prediction
4. Gneiting & Raftery (2007) - Probabilistic forecasting

**Your Implementation:**
- Built on: GPflow, TensorFlow
- UQ framework: `/media/gabriel-oduori/SERVER/dev_space/uncertainty_quantification/src/`
- FusionGP: https://github.com/GabrielOduori/fusiongp

---

## Support

**Issues:** Check docstrings in source code
**Examples:** See `examples/` directory
**Module docs:** See `src/fusiongp_uq_system.py` for detailed API

---

**You now have a complete, production-ready UQ system for your FusionGP model!**
