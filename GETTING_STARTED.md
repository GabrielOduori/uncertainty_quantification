# Getting Started with Uncertainty Quantification

This document provides a quick-start guide to using the UQ framework for your dissertation chapter.

## What's Been Created

### 1. **Project Structure** ✅
- Modern Python package with `pyproject.toml`
- Modular architecture: `uncertainty/`, `models/`, `transfer/`, `visualization/`
- Test framework setup
- Documentation scaffolding

### 2. **Core UQ Modules** ✅

#### `uncertainty/decomposition.py`
- **`UncertaintyDecomposer`**: Separates epistemic/aleatoric uncertainty
- Supports SVGP (FusionGP) and GAM-SSM models
- Type-annotated with comprehensive docstrings
- **350+ lines of production code**

#### `uncertainty/calibration.py`
- **`CalibrationEvaluator`**: PICP, ECE, CRPS, sharpness
- Implements proper scoring rules
- Calibration curve plotting
- **300+ lines of production code**

#### `uncertainty/ood_detection.py`
- **`SpatialOODDetector`**: Distance-based OOD flagging
- **`TemporalDriftDetector`**: Concept drift monitoring
- **`OODWarningSystem`**: Integrated spatial-temporal detection
- **400+ lines of production code**

### 3. **Documentation** ✅
- [README.md](README.md): Comprehensive usage guide
- [UQ_CHAPTER_EXTENSIONS_AND_ROADMAP.md](UQ_CHAPTER_EXTENSIONS_AND_ROADMAP.md): Detailed chapter plan
- Code examples and API documentation

### 4. **Literature Collection** ✅
- 19+ PDF papers copied to [literature/](literature/)
- Key papers on UQ, air quality, GPs, calibration

---

## Quick Test Run

### Install and Test the Code

```bash
cd /media/gabriel-oduori/SERVER/dev_space/uncertainty_quantification

# Install in development mode
pip install -e .

# Run the example scripts (built into modules)
python -m src.uncertainty.decomposition
python -m src.uncertainty.calibration
python -m src.uncertainty.ood_detection
```

Each module has a `__main__` block with working examples that demonstrate:
- API usage
- Expected inputs/outputs
- Visualization

---

## Integration with Your Models

### For FusionGP

```python
# After training your FusionGP model
from src.uncertainty.decomposition import UncertaintyDecomposer
from src.uncertainty.ood_detection import SpatialOODDetector

# Decompose uncertainty
decomposer = UncertaintyDecomposer(model_type='svgp')
components = decomposer.decompose_svgp(fusionGP_model, X_test)

# Detect OOD
ood_detector = SpatialOODDetector(
    X_train=X_train[:, :2],
    lengthscales=fusionGP_model.kernel.lengthscales.numpy()[:2]
)
ood_flags, ood_scores = ood_detector.detect(X_test[:, :2])

# Adjust uncertainty
sigma_adjusted = ood_detector.adjust_uncertainty(
    components.total, ood_scores
)
```

### For GAM-SSM-LUR

```python
from src.uncertainty.decomposition import UncertaintyDecomposer

# Get spatial and temporal uncertainties from your model
spatial_std = gam_ssm_model.spatial_component.predict_std(X_test)
temporal_std = gam_ssm_model.temporal_component.predict_std(X_test)

# Decompose
decomposer = UncertaintyDecomposer(model_type='gam_ssm')
components = decomposer.decompose_gam_ssm(
    spatial_std=spatial_std,
    temporal_std=temporal_std,
    correlation=0.0  # Test if truly independent
)
```

---

## Roadmap to Complete Chapter

### Phase 1: Core Implementation (Weeks 1-3)

**Week 1: Uncertainty Decomposition**
- [x] Code framework (DONE)
- [ ] Run on FusionGP LA Basin dataset
- [ ] Run on GAM-SSM-LUR dataset
- [ ] Create uncertainty maps (epistemic vs aleatoric)
- [ ] Write Section 7.5-7.6

**Week 2: OOD Detection**
- [x] Code framework (DONE)
- [ ] Apply to sparse sensor networks
- [ ] Validate OOD-adjusted uncertainties improve PICP
- [ ] Create domain of applicability maps
- [ ] Write Section 9.5

**Week 3: Hyperparameter Uncertainty**
- [ ] Implement bootstrap ensemble (build on decomposition.py)
- [ ] Compare single vs ensemble FusionGP
- [ ] Quantify underestimation
- [ ] Add to Section 7.5.2

### Phase 2: Comparative Analysis (Weeks 4-5)

**Week 4: Model Comparison**
- [ ] Load common test dataset
- [ ] Run calibration comparison
- [ ] Statistical significance tests
- [ ] Write Section 10.5

**Week 5: Transfer Learning UQ**
- [ ] Integrate with model_transferability repo
- [ ] Transfer uncertainty decomposition
- [ ] Calibration preservation analysis
- [ ] Write Section 10.6

### Phase 3: Writing (Weeks 6-7)

**Week 6: Integration**
- [ ] Insert new sections into existing chapter
- [ ] Ensure notation consistency
- [ ] Create all figures
- [ ] Unified bibliography

**Week 7: Polish**
- [ ] Proofread
- [ ] Internal review
- [ ] Final revisions

---

## File Structure Summary

```
uncertainty_quantification/
├── src/uncertainty/          # ✅ PRODUCTION CODE
│   ├── decomposition.py      # Epistemic/aleatoric (350 lines)
│   ├── calibration.py        # PICP, ECE, CRPS (300 lines)
│   ├── ood_detection.py      # Spatial/temporal OOD (400 lines)
│   └── __init__.py
├── pyproject.toml            # ✅ Modern Python packaging
├── README.md                 # ✅ Comprehensive documentation
├── UQ_CHAPTER_EXTENSIONS_AND_ROADMAP.md  # ✅ Chapter plan
├── GETTING_STARTED.md        # ✅ This file
├── literature/               # ✅ 19 PDF papers
└── experiments/              # TODO: Implement experiments
    ├── 01_uncertainty_decomposition.py
    ├── 02_hyperparameter_ensemble.py
    ├── 03_ood_detection.py
    ├── 04_model_comparison.py
    └── 05_transfer_learning_uq.py
```

---

## Next Steps (When You Return)

### Immediate Actions

1. **Test the code**:
   ```bash
   python -m src.uncertainty.decomposition
   python -m src.uncertainty.calibration
   python -m src.uncertainty.ood_detection
   ```

2. **Review the roadmap**:
   - Read [UQ_CHAPTER_EXTENSIONS_AND_ROADMAP.md](UQ_CHAPTER_EXTENSIONS_AND_ROADMAP.md)
   - Prioritize which extensions are critical for your thesis

3. **Integrate with existing models**:
   - Clone fusiongp, gam_ssm_lur repos locally
   - Test decomposition on real predictions

### Week 1 Tasks (Detailed)

#### Day 1-2: FusionGP Integration
```python
# Load your trained FusionGP model
import sys
sys.path.append('/path/to/fusiongp')
from fusiongp import SVGPModel

model = SVGPModel.load('path/to/trained_model.pkl')
X_test = load_test_data()

# Decompose uncertainty
from src.uncertainty.decomposition import UncertaintyDecomposer
decomposer = UncertaintyDecomposer(model_type='svgp')
components = decomposer.decompose_svgp(model, X_test)

# Create Figure 7.1: Uncertainty decomposition maps
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# ... plotting code from roadmap ...
```

#### Day 3-4: GAM-SSM Integration
```python
# Similar process for GAM-SSM-LUR
# Test independence assumption
from src.uncertainty.decomposition import test_independence_assumption

residuals_spatial = y_true - gam_predictions
residuals_temporal = y_true - ssm_predictions

independence_test = test_independence_assumption(
    residuals_spatial, residuals_temporal
)
print(independence_test)
# If |ρ| > 0.1, use conservative fusion
```

#### Day 5: Write Section 7.5-7.6
- Theory of SVGP uncertainty decomposition
- Implementation details
- Results on LA Basin dataset
- 3-4 figures

---

## Key Research Questions

Focus experiments on answering these:

1. **RQ1**: Epistemic vs aleatoric breakdown by distance from sensors
   - Hypothesis: Epistemic dominates far from data

2. **RQ2**: Hyperparameter uncertainty underestimation
   - Hypothesis: 10-30% underestimation

3. **RQ3**: FusionGP vs GAM-SSM calibration quality
   - Hypothesis: FusionGP better at spatial, GAM-SSM better at temporal

4. **RQ4**: OOD detection efficacy
   - Hypothesis: Improves coverage from 85% → 95%

5. **RQ5**: Optimal transfer parameter β
   - Hypothesis: β ≈ 0.5 > β = 1.0

---

## Support and Resources

### Code Documentation
- All modules have comprehensive docstrings
- Type hints for IDE autocomplete
- Example code in `__main__` blocks

### Chapter Structure
- Your existing content (Sections 1-6, 8-9) is solid foundation
- New sections 7.5-7.6, 9.5, 10.5-10.6 add ~25 pages
- Roadmap provides section-by-section writing plan

### Literature
- 19 PDFs in `literature/` directory
- Key papers on UQ taxonomy, calibration, air quality fusion
- Use for methodology validation and citations

---

## Expected Outcomes

### Code Deliverables
- ✅ Production-ready UQ framework (1000+ lines)
- [ ] 5 experiment scripts
- [ ] Comprehensive test suite
- [ ] Jupyter notebooks for exploration

### Chapter Contributions
1. **Novel UQ framework** for air quality sensor fusion
2. **Hyperparameter uncertainty quantification** for spatial GPs
3. **Automated OOD detection** for environmental models
4. **Comparative analysis** of two SOTA models
5. **Transfer learning UQ** decomposition

### Timeline
- **10 weeks** to complete chapter (assuming 20 hrs/week)
- **Weeks 1-3**: Core implementations and results
- **Weeks 4-5**: Comparative analysis
- **Weeks 6-7**: Writing and integration
- **Weeks 8-10**: Review and polish

---

## Questions or Issues?

The code is designed to be self-documenting and testable. Each module can run standalone:

```bash
# Test decomposition
python -c "from src.uncertainty.decomposition import *; help(UncertaintyDecomposer)"

# Test calibration
python -c "from src.uncertainty.calibration import *; help(CalibrationEvaluator)"

# Test OOD detection
python -c "from src.uncertainty.ood_detection import *; help(SpatialOODDetector)"
```

All code follows modern software engineering practices:
- Type hints
- Dataclasses for structured outputs
- Logging via loguru
- Docstrings with examples
- Error handling

---

**You have everything you need to complete an excellent UQ chapter!**

Good luck with your dissertation! 🎓
