# FusionGP UQ System - Quick Reference Card

## One-Minute Setup

```python
# 1. Import
import sys; sys.path.insert(0, 'src')
from fusiongp import FusionGP
from fusiongp_uq_system import create_default_uq_system

# 2. Create system
model = FusionGP.load('your_model.pkl')
uq_system = create_default_uq_system(model)

# 3. Fit and calibrate
uq_system.fit_ensemble(X_train, y_train, sources_train)
uq_system.calibrate(X_cal, y_cal, sources_cal)

# 4. Predict
predictions = uq_system.predict_with_full_uq(X_test, sources_test)
```

---

## Data Format

```python
X_train.shape    # (n_samples, n_features)  e.g., (1000, 3) for [lat, lon, time]
y_train.shape    # (n_samples,)             e.g., (1000,) PM2.5 values
sources_train    # (n_samples,)             0=EPA, 1=Low-cost, 2=Satellite
```

---

## Prediction Output

```python
pred = predictions[0]

pred.mean                      # 35.8 μg/m³
pred.std                       # 2.65 μg/m³
pred.lower_95, pred.upper_95   # [30.9, 40.7] μg/m³
pred.epistemic_fraction        # 0.67 (67% reducible)
pred.spatial_ood               # False (reliable)
pred.conformal_guaranteed      # True (95% coverage)
```

---

## Configuration Presets

```python
# Fast (n=5, ~3 min)
from fusiongp_uq_system import create_fast_uq_system
uq_system = create_fast_uq_system(model)

# Default (n=10, ~8 min) - RECOMMENDED
from fusiongp_uq_system import create_default_uq_system
uq_system = create_default_uq_system(model)

# Rigorous (n=20, ~20 min)
from fusiongp_uq_system import create_rigorous_uq_system
uq_system = create_rigorous_uq_system(model)
```

---

## Policy Outputs

```python
policy = uq_system.generate_policy_outputs(predictions, X_test)

# Health alerts
for alert in policy['health_alerts']:
    print(f"{alert.message} - {alert.certainty}")

# Sensor placement (top 10)
for rec in policy['sensor_recommendations'].priority_locations[:10]:
    print(f"Deploy sensor at {rec['coordinates']}")

# Decision report
report = policy['decision_report']  # DataFrame-like structure
```

---

## Calibration Check

```python
metrics = uq_system.evaluate_calibration(X_test, y_test, sources_test)

metrics['picp']  # Should be ≈ 0.95
metrics['ece']   # Should be < 0.05
metrics['crps']  # Lower is better
```

---

## Interpreting Uncertainty

| Epistemic Fraction | Meaning | Action |
|--------------------|---------|--------|
| >70% | Data-sparse region | Deploy more sensors |
| 30-70% | Mixed uncertainty | Moderate benefit from data |
| <30% | Well-characterized | Limited benefit from data |

| OOD Flag | Meaning | Action |
|----------|---------|--------|
| True | Extrapolating | Treat with caution |
| False | Interpolating | Trust prediction |

---

## Common Tasks

### Export Results
```python
import json
results = [p.to_dict() for p in predictions]
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### Get System Status
```python
print(uq_system.summary())
```

### Custom Configuration
```python
from fusiongp_uq_system import FusionGPUQConfig, FusionGPUQSystem

config = FusionGPUQConfig(
    n_ensemble=15,
    conformal_alpha=0.05,
    spatial_ood_threshold=2.5,
    enable_second_order=True,
    source_noise_levels={'EPA': 2.1, 'LC': 8.3, 'SAT': 15.6}
)
uq_system = FusionGPUQSystem(model, config)
```

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| "Must fit ensemble first" | Call `fit_ensemble()` before `predict_with_full_uq()` |
| "Could not extract lengthscales" | Ignore - system uses defaults automatically |
| PICP far from 0.95 | Increase calibration data or check data distribution |
| Too slow | Use `create_fast_uq_system()` or reduce `n_ensemble` |

---

## Runtime (1000 training samples, 4 CPU cores)

| Config | Time |
|--------|------|
| Fast (n=5) | ~3 min |
| Default (n=10) | ~8 min |
| Rigorous (n=20) | ~20 min |

---

## Minimum Requirements

- Training: 500+ samples (1000+ recommended)
- Calibration: 100+ samples (200+ recommended)
- Sources: Mix of EPA, low-cost, satellite
- FusionGP model with `predict_f(X)` method

---

## Visualization

```python
from visualization.gp_plots import GPUncertaintyVisualizer

viz = GPUncertaintyVisualizer()

# Classic GP plot with uncertainty bands
viz.plot_1d_with_uncertainty(X_test, predictions, y_test)

# Spatial uncertainty map
viz.plot_spatial_uncertainty_map(X_test, predictions, metric='epistemic')

# Complete 6-panel summary (perfect for dissertations!)
viz.plot_complete_summary(X_test, predictions, y_test, X_train,
                         save_path='results/summary.png')

# Or use quick convenience functions
from visualization.gp_plots import quick_plot, quick_spatial_plot, quick_summary

quick_summary(X_test, predictions, y_test, X_train,
              save_path='results/quick_summary.png')
```

**Available plots:**
- `plot_1d_with_uncertainty()` - Classic GP with shaded bands
- `plot_spatial_uncertainty_map()` - 2D heatmap
- `plot_uncertainty_decomposition()` - Epistemic vs aleatoric
- `plot_ood_detection()` - Highlight extrapolation
- `plot_calibration_curve()` - Reliability diagram
- `plot_complete_summary()` - 6-panel comprehensive figure ⭐

---

## Complete Examples

**UQ System:** `examples/fusiongp_uq_complete_example.py`
```bash
python examples/fusiongp_uq_complete_example.py
```

**Visualization:** `examples/visualization_demo.py`
```bash
python examples/visualization_demo.py
```

---

## Full Documentation

See: `FUSIONGP_UQ_GUIDE.md`

---

## What You Get

✓ Epistemic/Aleatoric decomposition
✓ Hyperparameter uncertainty quantification
✓ Out-of-distribution detection
✓ Conformal prediction guarantees (95% coverage)
✓ Second-order (meta) uncertainty
✓ Health alerts with certainty levels
✓ Sensor placement recommendations
✓ Decision support reports
✓ Calibration evaluation metrics

**All in 4 lines of code!**
