# FusionGP UQ System - Visualization Guide

## Overview

This guide shows you how to create beautiful, publication-quality visualizations of your Gaussian Process uncertainty quantification results. These are the classic "GP-style plots" with shaded uncertainty bands that you see in papers.

---

## Prerequisites

```bash
# Install required packages for visualization
pip install matplotlib seaborn scipy
```

---

## Quick Start

```bash
# Run the visualization demo
python examples/visualization_demo.py
```

This creates 10 different plots demonstrating all visualization capabilities.

**Output location:** `results/` directory

---

## The Plots You Get

### 1. Classic GP Plot with Uncertainty Bands ⭐

**What it shows:**
- Mean prediction (solid line)
- ±1σ uncertainty (darker shaded band)
- 95% confidence interval (lighter shaded band)
- True observations (dots)
- Training data (triangles)

**Best for:**
- Understanding prediction confidence
- Seeing how uncertainty increases away from training data
- Classic GP visualization that everyone recognizes

**Code:**
```python
from visualization.gp_plots import GPUncertaintyVisualizer

viz = GPUncertaintyVisualizer()
viz.plot_1d_with_uncertainty(X_test, predictions, y_test,
                             feature_idx=0,
                             feature_name='Latitude',
                             save_path='results/gp_1d.png')
```

**Use in dissertation for:**
- Illustrating GP predictions
- Showing calibration quality
- Demonstrating uncertainty propagation

---

### 2. Spatial Uncertainty Maps

**What it shows:**
- 2D heatmap of uncertainty across space
- Training locations (black dots)
- Can show: total, epistemic, or aleatoric uncertainty

**Best for:**
- Where are predictions most uncertain?
- Spatial patterns in uncertainty
- Identifying data gaps

**Code:**
```python
viz.plot_spatial_uncertainty_map(X_test, predictions,
                                 metric='epistemic',  # 'total', 'epistemic', 'aleatoric'
                                 plot_type='interpolated',  # 'scatter', 'contour', 'interpolated'
                                 title='Epistemic Uncertainty Map')
```

**Plot types:**
- `'scatter'`: Colored points (fast, shows exact locations)
- `'contour'`: Contour lines (shows gradients)
- `'interpolated'`: Smooth heatmap (publication-quality)

**Use in dissertation for:**
- Answering RQ1 (uncertainty decomposition)
- Sensor placement justification
- Showing data coverage

---

### 3. Uncertainty Decomposition Plot

**What it shows:**
- How much uncertainty is epistemic (reducible by collecting data)
- How much is aleatoric (irreducible measurement noise)
- Stacked shaded regions showing both components

**Best for:**
- Answering "can we reduce uncertainty by collecting more data?"
- Understanding where sensors would help most
- Separating model vs measurement uncertainty

**Code:**
```python
viz.plot_uncertainty_decomposition(X_test, predictions,
                                   feature_idx=0,
                                   feature_name='Latitude')
```

**Interpretation:**
- **Wide epistemic band**: More sensors would help
- **Wide aleatoric band**: Inherent measurement noise
- **Ratio changes spatially**: Near sensors → mostly aleatoric, far from sensors → mostly epistemic

**Use in dissertation for:**
- Research Question 1 (epistemic vs aleatoric)
- Justifying sensor deployment
- Understanding uncertainty sources

---

### 4. Out-of-Distribution (OOD) Detection Plot

**What it shows:**
- Which predictions are interpolating (trustworthy)
- Which are extrapolating (unreliable)
- OOD points marked with ✗
- Increased uncertainty in OOD regions

**Best for:**
- Identifying when model is guessing
- Understanding prediction reliability
- Spatial extrapolation detection

**Code:**
```python
viz.plot_ood_detection(X_test, predictions, X_train,
                       feature_idx=0,
                       feature_name='Latitude')
```

**Interpretation:**
- **Green points**: Interpolating → trust prediction
- **Red ✗**: Extrapolating → treat with caution
- **Training range shown**: Gray shaded region

**Use in dissertation for:**
- Research Question 4 (OOD detection)
- Model limitations discussion
- Reliability assessment

---

### 5. Calibration Curve (Reliability Diagram)

**What it shows:**
- Does 95% confidence interval contain 95% of observations?
- Are predictions well-calibrated?
- Diagonal line = perfect calibration

**Best for:**
- Model validation
- Checking if uncertainties are realistic
- Answering "can we trust these intervals?"

**Code:**
```python
viz.plot_calibration_curve(predictions, y_test,
                          title='Calibration Curve')
```

**Interpretation:**
- **On diagonal**: Well-calibrated
- **Above diagonal**: Over-confident (intervals too narrow)
- **Below diagonal**: Under-confident (intervals too wide)

**Metrics shown:**
- **PICP**: Should be ≈ 0.95 for 95% intervals
- **ECE**: Expected Calibration Error (lower is better, <0.05 is good)

**Use in dissertation for:**
- Research Question 3 (calibration)
- Model validation section
- Methods quality assessment

---

### 6. Complete Summary Figure ⭐⭐⭐

**What it shows:**
All 6 key visualizations in one figure:
1. 1D predictions with uncertainty
2. Spatial uncertainty map
3. Uncertainty decomposition
4. OOD detection
5. Calibration curve
6. Uncertainty statistics histogram

**Best for:**
- Dissertation figures
- Paper submissions
- Comprehensive overview
- Single-page summary

**Code:**
```python
viz.plot_complete_summary(X_test, predictions, y_test, X_train,
                         suptitle='FusionGP UQ System - Results',
                         save_path='results/complete_summary.png')
```

**Use in dissertation for:**
- Main results figure
- System validation
- Comprehensive evaluation

---

## Quick Convenience Functions

For rapid prototyping, use these one-liners:

```python
from visualization.gp_plots import quick_plot, quick_spatial_plot, quick_summary

# Quick 1D plot
quick_plot(X_test, predictions, y_test,
          save_path='results/quick.png')

# Quick spatial plot
quick_spatial_plot(X_test, predictions,
                  save_path='results/spatial.png')

# Quick complete summary
quick_summary(X_test, predictions, y_test, X_train,
             save_path='results/summary.png')
```

---

## Customization Options

### Colors and Style

The visualizer uses Seaborn's default palette and matplotlib styling. Customize:

```python
viz = GPUncertaintyVisualizer()

# Change DPI for higher resolution
fig, ax = viz.plot_1d_with_uncertainty(X_test, predictions, y_test)
plt.savefig('high_res.png', dpi=600)

# Change figure size
fig, ax = viz.plot_spatial_uncertainty_map(X_test, predictions,
                                           figsize=(12, 8))

# Change colormap
fig, ax = viz.plot_spatial_uncertainty_map(X_test, predictions,
                                           cmap='viridis')  # Default: 'YlOrRd'
```

### Feature Selection

For 1D plots, choose which feature to plot along:

```python
# Plot along latitude (feature 0)
viz.plot_1d_with_uncertainty(X_test, predictions, y_test,
                             feature_idx=0,
                             feature_name='Latitude')

# Plot along longitude (feature 1)
viz.plot_1d_with_uncertainty(X_test, predictions, y_test,
                             feature_idx=1,
                             feature_name='Longitude')

# Plot along time (feature 2)
viz.plot_1d_with_uncertainty(X_test, predictions, y_test,
                             feature_idx=2,
                             feature_name='Time (days)')
```

---

## Publication-Quality Figures

### For Papers

```python
viz = GPUncertaintyVisualizer()

# High-resolution complete summary
fig = viz.plot_complete_summary(X_test, predictions, y_test, X_train,
                               suptitle='FusionGP Uncertainty Quantification')
plt.savefig('paper_figure.png', dpi=600, bbox_inches='tight')
plt.savefig('paper_figure.pdf', bbox_inches='tight')  # Vector format
```

**Recommended settings:**
- DPI: 600 for raster (PNG)
- Format: PDF for vector graphics
- Size: Default (12×10) fits well in papers

### For Presentations

```python
# Larger text for visibility
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

viz = GPUncertaintyVisualizer()
fig = viz.plot_complete_summary(X_test, predictions, y_test, X_train)
plt.savefig('presentation_figure.png', dpi=300, bbox_inches='tight')
```

### For Dissertation

```python
# High-quality individual figures for dissertation chapters
viz = GPUncertaintyVisualizer()

# Chapter 4: Methods
viz.plot_1d_with_uncertainty(X_test, predictions, y_test,
                             title='GP Predictions with Uncertainty (Chapter 4)',
                             save_path='dissertation/ch4_gp_predictions.pdf')

# Chapter 5: Results - Uncertainty decomposition
viz.plot_uncertainty_decomposition(X_test, predictions,
                                   title='Epistemic vs Aleatoric Uncertainty (Chapter 5)',
                                   save_path='dissertation/ch5_decomposition.pdf')

# Chapter 5: Results - Spatial patterns
viz.plot_spatial_uncertainty_map(X_test, predictions,
                                 metric='epistemic',
                                 title='Epistemic Uncertainty Map (Chapter 5)',
                                 save_path='dissertation/ch5_spatial.pdf')

# Chapter 6: Validation
viz.plot_calibration_curve(predictions, y_test,
                          title='Model Calibration (Chapter 6)',
                          save_path='dissertation/ch6_calibration.pdf')

# Comprehensive summary
viz.plot_complete_summary(X_test, predictions, y_test, X_train,
                         suptitle='Complete UQ System Evaluation',
                         save_path='dissertation/complete_summary.pdf')
```

---

## Interpretation Guide

### What Each Plot Tells You

| Plot | Research Question | Key Insight |
|------|-------------------|-------------|
| 1D Uncertainty | RQ3: Calibration | Do intervals contain observations? |
| Spatial Map | RQ1: Decomposition | Where is uncertainty highest? |
| Decomposition | RQ1: Decomposition | How much is reducible? |
| OOD Detection | RQ4: OOD | When is model extrapolating? |
| Calibration Curve | RQ3: Calibration | Are uncertainties realistic? |
| Complete Summary | All RQs | Overall system performance |

### Reading the Uncertainty Bands

**1D Plot shading:**
- **Outer light band**: 95% confidence interval (conformal prediction)
- **Middle darker band**: ±1σ uncertainty (total)
- **Solid line**: Mean prediction
- **Dots**: True observations
- **Triangles**: Training data

**What "good" looks like:**
- ✓ Most observations fall within 95% band
- ✓ Uncertainty increases away from training data
- ✓ Bands widen in extrapolation regions
- ✓ Calibration curve follows diagonal

**Warning signs:**
- ⚠️ Many observations outside 95% band → under-confident
- ⚠️ No observations outside bands → over-confident
- ⚠️ Constant uncertainty everywhere → not capturing spatial variation
- ⚠️ Calibration curve far from diagonal → poorly calibrated

---

## Common Workflows

### Workflow 1: Quick Exploration

```python
from visualization.gp_plots import quick_summary

# After running UQ system
quick_summary(X_test, predictions, y_test, X_train,
             save_path='results/quick_summary.png')
```

**Use when:** You want to quickly see if everything looks reasonable

---

### Workflow 2: Detailed Analysis

```python
from visualization.gp_plots import GPUncertaintyVisualizer

viz = GPUncertaintyVisualizer()

# 1. Check predictions
viz.plot_1d_with_uncertainty(X_test, predictions, y_test,
                             save_path='1_predictions.png')

# 2. Check spatial patterns
viz.plot_spatial_uncertainty_map(X_test, predictions,
                                 metric='epistemic',
                                 save_path='2_spatial_epistemic.png')

# 3. Understand uncertainty sources
viz.plot_uncertainty_decomposition(X_test, predictions,
                                   save_path='3_decomposition.png')

# 4. Check reliability
viz.plot_ood_detection(X_test, predictions, X_train,
                       save_path='4_ood.png')
viz.plot_calibration_curve(predictions, y_test,
                          save_path='5_calibration.png')
```

**Use when:** Analyzing results for dissertation/paper

---

### Workflow 3: Dissertation Figure Generation

```python
import os
from visualization.gp_plots import GPUncertaintyVisualizer

# Create output directory
os.makedirs('dissertation/figures', exist_ok=True)

viz = GPUncertaintyVisualizer()

# Main results figure (goes in Chapter 5)
viz.plot_complete_summary(X_test, predictions, y_test, X_train,
                         suptitle='FusionGP UQ System - Complete Evaluation',
                         save_path='dissertation/figures/main_results.pdf')

# Individual component figures for detailed discussion
viz.plot_uncertainty_decomposition(X_test, predictions,
                                   save_path='dissertation/figures/decomposition.pdf')

viz.plot_spatial_uncertainty_map(X_test, predictions,
                                 metric='epistemic',
                                 plot_type='interpolated',
                                 save_path='dissertation/figures/spatial_epistemic.pdf')

viz.plot_calibration_curve(predictions, y_test,
                          save_path='dissertation/figures/calibration.pdf')
```

**Use when:** Generating final dissertation figures

---

## Technical Details

### Input Data Requirements

```python
X_test: np.ndarray
    Shape (n_test, n_features)
    Test locations [latitude, longitude, time, ...]

predictions: List[UQPrediction]
    Output from uq_system.predict_with_full_uq()
    Each prediction has: mean, std, lower_95, upper_95, etc.

y_test: np.ndarray
    Shape (n_test,)
    True PM2.5 values for comparison

X_train: np.ndarray
    Shape (n_train, n_features)
    Training locations (for OOD detection plots)
```

### Output Format

All plots return `(fig, ax)` tuple for further customization:

```python
fig, ax = viz.plot_1d_with_uncertainty(X_test, predictions, y_test)

# Further customize
ax.set_ylim([0, 100])
ax.axhline(35.5, color='red', linestyle='--', label='EPA threshold')
ax.legend()

# Save with custom settings
plt.savefig('custom.png', dpi=600, facecolor='white')
```

---

## Examples Output

Running `python examples/visualization_demo.py` creates:

```
results/
├── gp_1d_uncertainty.png              # Classic GP plot
├── gp_quick_1d.png                    # Quick version
├── spatial_total_uncertainty.png      # Total uncertainty map
├── spatial_epistemic_uncertainty.png  # Epistemic uncertainty map
├── spatial_quick.png                  # Quick spatial plot
├── uncertainty_decomposition.png      # Epistemic vs aleatoric
├── ood_detection.png                  # OOD detection
├── calibration_curve.png              # Reliability diagram
├── complete_summary.png               # 6-panel comprehensive ⭐
└── quick_summary.png                  # Quick comprehensive
```

**Runtime:** ~2 minutes total (including UQ system fitting)

---

## Troubleshooting

### Import Error

```python
# If import fails, check path
import sys
sys.path.insert(0, 'src')
from visualization.gp_plots import GPUncertaintyVisualizer
```

### Empty Plots

Check that predictions contain data:
```python
print(len(predictions))  # Should be > 0
print(predictions[0].mean)  # Should be a number
```

### Spatial Plots Look Wrong

For LA Basin data, ensure coordinates are in degrees:
```python
print(X_test[:5])  # Latitude should be ~34, longitude ~-118
```

### Low Resolution

Increase DPI when saving:
```python
plt.savefig('high_res.png', dpi=600)
```

---

## API Reference

### GPUncertaintyVisualizer Class

```python
class GPUncertaintyVisualizer:
    """Creates publication-quality GP uncertainty visualizations."""

    def plot_1d_with_uncertainty(X_test, predictions, y_test, feature_idx=0, ...)
    def plot_spatial_uncertainty_map(X_test, predictions, metric='total', ...)
    def plot_uncertainty_decomposition(X_test, predictions, feature_idx=0, ...)
    def plot_ood_detection(X_test, predictions, X_train, feature_idx=0, ...)
    def plot_calibration_curve(predictions, y_true, n_bins=10, ...)
    def plot_complete_summary(X_test, predictions, y_test, X_train, ...)
```

### Convenience Functions

```python
def quick_plot(X_test, predictions, y_test, feature_idx=0, save_path=None)
def quick_spatial_plot(X_test, predictions, metric='epistemic', save_path=None)
def quick_summary(X_test, predictions, y_test, X_train, save_path=None)
```

---

## Next Steps

1. **Run the demo:**
   ```bash
   python examples/visualization_demo.py
   ```

2. **Look at the outputs** in `results/` directory

3. **Modify for your data:**
   - Replace mock data with your LA Basin data
   - Generate figures for your dissertation

4. **Customize as needed:**
   - Adjust colors, sizes, labels
   - Create custom multi-panel figures
   - Add your own annotations

---

## Related Documentation

- [QUICK_REFERENCE_FUSIONGP.md](QUICK_REFERENCE_FUSIONGP.md) - Quick reference card
- [FUSIONGP_UQ_GUIDE.md](FUSIONGP_UQ_GUIDE.md) - Complete UQ system guide
- [RUN_FUSIONGP_UQ.md](RUN_FUSIONGP_UQ.md) - How to run the system
- [examples/visualization_demo.py](examples/visualization_demo.py) - Complete demo code

---

**Ready to create beautiful GP plots?**

```bash
python examples/visualization_demo.py
```

🎨 **Publication-quality figures in 2 minutes!**
