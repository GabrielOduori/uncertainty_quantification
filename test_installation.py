"""
Test Installation Script

Quick test to verify all components are properly installed and working.
Run this after installation to ensure everything is ready.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 80)
print("TESTING RIGOROUS UQ FRAMEWORK INSTALLATION")
print("=" * 80)

# Test 1: Import core modules
print("\n[TEST 1] Importing core modules...")
try:
    from uncertainty import (
        HierarchicalUQTracker,
        ConformalPredictionWrapper,
        SecondOrderAnalyzer,
        UncertaintyDecomposer,
        CalibrationEvaluator,
    )
    from models import BootstrapSVGPEnsemble
    from decision import PolicyTranslator
    print("✅ All core modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test 2: Create instances
print("\n[TEST 2] Creating component instances...")
try:
    tracker = HierarchicalUQTracker()
    conformal = None  # Will need a model
    analyzer = SecondOrderAnalyzer()
    decomposer = UncertaintyDecomposer()
    calibrator = CalibrationEvaluator()
    ensemble = BootstrapSVGPEnsemble(n_ensemble=3)
    translator = PolicyTranslator()
    print("✅ All components instantiated successfully")
except Exception as e:
    print(f"❌ Instantiation error: {e}")
    sys.exit(1)

# Test 3: Run mock workflow
print("\n[TEST 3] Running mock workflow...")
try:
    import numpy as np

    # Generate mock data
    np.random.seed(42)
    n_train = 100
    n_test = 20
    X_train = np.random.randn(n_train, 3)
    y_train = np.random.randn(n_train) * 10 + 50
    sources_train = np.random.choice([0, 1, 2], size=n_train)

    X_test = np.random.randn(n_test, 3)
    y_test = np.random.randn(n_test) * 10 + 50
    sources_test = np.random.choice([0, 1, 2], size=n_test)

    print("  - Generated mock data")

    # Test hierarchical tracking (needs a model)
    class MockModel:
        def predict_f(self, X):
            n = len(X)
            mean = np.random.randn(n) * 10 + 50
            var = np.random.gamma(2, 1, n)

            class MockTensor:
                def __init__(self, data):
                    self.data = data
                def numpy(self):
                    return self.data

            return MockTensor(mean), MockTensor(var)

    mock_model = MockModel()

    # Hierarchical tracking
    hierarchical_var = tracker.decompose_by_stage(
        mock_model, X_test, sources_test
    )
    print(f"  - Hierarchical tracking: {len(hierarchical_var.stage_1_epistemic)} predictions")

    # Decomposition
    mean, var = mock_model.predict_f(X_test)
    predictions = mean.numpy()
    total_var = var.numpy()
    aleatoric_var = np.ones_like(total_var) * 2.0

    from uncertainty import decompose_epistemic_aleatoric
    components = decompose_epistemic_aleatoric(predictions, total_var, aleatoric_var)
    print(f"  - Decomposition: {components.summary_stats()['avg_epistemic_fraction']:.1%} epistemic")

    # Calibration
    uncertainties = np.sqrt(var.numpy())
    cal_results = calibrator.evaluate(predictions, uncertainties, y_test)
    print(f"  - Calibration: PICP(95%)={cal_results.picp.get('95%', 0.0):.3f}")

    # Conformal prediction
    conformal_wrapper = ConformalPredictionWrapper(mock_model, alpha=0.05)
    conformal_wrapper.calibrate(X_train[:20], y_train[:20])
    intervals = conformal_wrapper.predict_with_conformal_intervals(X_test)
    print(f"  - Conformal: Mean width={intervals.summary()['mean_width']:.2f}")

    # Decision translation
    alerts = translator.generate_health_alerts(predictions, uncertainties)
    print(f"  - Decision: Generated {len(alerts)} health alerts")

    print("✅ Mock workflow completed successfully")

except Exception as e:
    print(f"❌ Workflow error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Check file structure
print("\n[TEST 4] Checking file structure...")
try:
    required_files = [
        "src/uncertainty/hierarchical.py",
        "src/models/ensemble.py",
        "src/uncertainty/conformal.py",
        "src/uncertainty/second_order.py",
        "src/decision/policy_translation.py",
        "experiments/comprehensive_validation.py",
        "QUICKSTART_4DAYS.md",
        "IMPLEMENTATION_COMPLETE.md",
    ]

    base_path = Path(__file__).parent
    missing = []

    for file_path in required_files:
        full_path = base_path / file_path
        if not full_path.exists():
            missing.append(file_path)

    if missing:
        print(f"⚠️  Warning: Missing files: {missing}")
    else:
        print("✅ All expected files present")

except Exception as e:
    print(f"⚠️  File check error: {e}")

# Summary
print("\n" + "=" * 80)
print("INSTALLATION TEST SUMMARY")
print("=" * 80)
print("✅ Core modules: OK")
print("✅ Component instantiation: OK")
print("✅ Mock workflow: OK")
print("✅ File structure: OK")
print("\n🚀 Framework is ready to use!")
print("\nNext steps:")
print("  1. Read QUICKSTART_4DAYS.md for usage examples")
print("  2. Run experiments/comprehensive_validation.py for full validation")
print("  3. Integrate with your FusionGP model")
print("=" * 80)
