"""
Complete Example: FusionGP Uncertainty Quantification System

This example demonstrates the full pipeline for uncertainty quantification
with FusionGP models, from data loading to policy outputs.

Usage:
    python examples/fusiongp_uq_complete_example.py
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from typing import Tuple

# Import the FusionGP UQ system
from fusiongp_uq_system import (
    FusionGPUQSystem,
    FusionGPUQConfig,
    create_default_uq_system,
    create_fast_uq_system,
    create_rigorous_uq_system,
)


# ============================================================================
# PART 1: Load Your FusionGP Model
# ============================================================================

def load_fusiongp_model():
    """
    Load your trained FusionGP model.

    Replace this with your actual model loading code.
    """
    print("Loading FusionGP model...")

    # Option 1: Load from file
    # from fusiongp import FusionGP
    # model = FusionGP.load('path/to/your/trained_model.pkl')

    # Option 2: Use existing model instance
    # model = your_trained_fusiongp_model

    # For this example, we'll create a mock model
    class MockFusionGP:
        """Mock FusionGP model for demonstration."""

        def predict_f(self, X):
            """Simulate GP predictions."""
            n = X.shape[0]
            mean = 35.0 + 10.0 * np.sin(X[:, 0] * 10) + np.random.randn(n) * 2
            var = 4.0 + 2.0 * np.random.rand(n)
            return mean, var

        def get_lengthscales(self):
            return [0.05, 0.05]  # ~5km in degrees

    model = MockFusionGP()
    print("✓ Model loaded (mock model for demo)")
    return model


# ============================================================================
# PART 2: Load Your Air Quality Data
# ============================================================================

def load_air_quality_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                      np.ndarray, np.ndarray, np.ndarray,
                                      np.ndarray, np.ndarray, np.ndarray]:
    """
    Load air quality data for training, calibration, and testing.

    Replace this with your actual data loading code.

    Returns
    -------
    X_train, y_train, sources_train : Training data
    X_cal, y_cal, sources_cal : Calibration data
    X_test, y_test, sources_test : Test data
    """
    print("\nLoading air quality data...")

    # Option 1: Load from your data files
    # import pandas as pd
    # df = pd.read_csv('/path/to/aq_data.csv')
    # X_train = df[['latitude', 'longitude', 'time']].values
    # y_train = df['pm25'].values
    # sources_train = df['source'].values  # 0=EPA, 1=LC, 2=SAT

    # For this example, generate synthetic LA Basin-like data
    np.random.seed(42)

    # Training data (800 samples)
    n_train = 800
    X_train = np.random.randn(n_train, 3) * 0.1 + [34.05, -118.25, 0]
    y_train = 35.0 + 10.0 * np.sin(X_train[:, 0] * 10) + np.random.randn(n_train) * 5
    sources_train = np.random.choice([0, 1, 2], size=n_train, p=[0.2, 0.5, 0.3])

    # Calibration data (100 samples)
    n_cal = 100
    X_cal = np.random.randn(n_cal, 3) * 0.1 + [34.05, -118.25, 0]
    y_cal = 35.0 + 10.0 * np.sin(X_cal[:, 0] * 10) + np.random.randn(n_cal) * 5
    sources_cal = np.random.choice([0, 1, 2], size=n_cal, p=[0.2, 0.5, 0.3])

    # Test data (100 samples)
    n_test = 100
    X_test = np.random.randn(n_test, 3) * 0.1 + [34.05, -118.25, 0]
    y_test = 35.0 + 10.0 * np.sin(X_test[:, 0] * 10) + np.random.randn(n_test) * 5
    sources_test = np.random.choice([0, 1, 2], size=n_test, p=[0.2, 0.5, 0.3])

    print(f"✓ Data loaded:")
    print(f"  Training: {n_train} samples")
    print(f"  Calibration: {n_cal} samples")
    print(f"  Test: {n_test} samples")

    return (X_train, y_train, sources_train,
            X_cal, y_cal, sources_cal,
            X_test, y_test, sources_test)


# ============================================================================
# PART 3: Create and Configure UQ System
# ============================================================================

def create_uq_system(model, config_type='default'):
    """
    Create UQ system with specified configuration.

    Parameters
    ----------
    model : FusionGP model
        Your trained model
    config_type : str, one of {'default', 'fast', 'rigorous', 'custom'}
        Configuration preset

    Returns
    -------
    uq_system : FusionGPUQSystem
    """
    print(f"\nCreating UQ system (config: {config_type})...")

    if config_type == 'default':
        uq_system = create_default_uq_system(model)

    elif config_type == 'fast':
        # For rapid prototyping (n=5 ensemble)
        uq_system = create_fast_uq_system(model)

    elif config_type == 'rigorous':
        # For publication/high-stakes decisions (n=20 ensemble)
        uq_system = create_rigorous_uq_system(model)

    elif config_type == 'custom':
        # Custom configuration
        config = FusionGPUQConfig(
            n_ensemble=10,
            bootstrap_fraction=0.8,
            use_parallel=True,
            n_workers=4,
            conformal_alpha=0.05,  # 95% coverage
            spatial_ood_threshold=2.5,
            enable_second_order=True,
            source_noise_levels={
                'EPA': 2.1,   # (μg/m³)² - Reference monitors
                'LC': 8.3,    # (μg/m³)² - Low-cost sensors
                'SAT': 15.6,  # (μg/m³)² - Satellite
            }
        )
        uq_system = FusionGPUQSystem(model, config)

    else:
        raise ValueError(f"Unknown config_type: {config_type}")

    print(uq_system.summary())
    return uq_system


# ============================================================================
# PART 4: Fit Ensemble and Calibrate
# ============================================================================

def fit_and_calibrate(uq_system, X_train, y_train, sources_train,
                      X_cal, y_cal, sources_cal):
    """
    Fit bootstrap ensemble and calibrate conformal prediction.

    This is a required step before making predictions.
    """
    print("\n" + "=" * 70)
    print("FITTING AND CALIBRATION")
    print("=" * 70)

    # Step 1: Fit bootstrap ensemble
    print("\nStep 1: Fitting bootstrap ensemble...")
    uq_system.fit_ensemble(
        X_train, y_train, sources_train,
        verbose=True
    )

    # Step 2: Calibrate conformal prediction and OOD detection
    print("\nStep 2: Calibrating conformal prediction and OOD detection...")
    uq_system.calibrate(
        X_cal, y_cal, sources_cal,
        timestamps_cal=None,  # Optional: add timestamps for temporal OOD
        verbose=True
    )

    print("\n✓ System ready for predictions!")


# ============================================================================
# PART 5: Make Predictions with Full UQ
# ============================================================================

def make_predictions(uq_system, X_test, sources_test):
    """
    Generate predictions with complete uncertainty quantification.

    Returns
    -------
    predictions : List[UQPrediction]
        Complete UQ for each test point
    """
    print("\n" + "=" * 70)
    print("MAKING PREDICTIONS WITH FULL UQ")
    print("=" * 70)

    predictions = uq_system.predict_with_full_uq(
        X_test,
        sources_test,
        timestamps_test=None  # Optional: add for temporal OOD detection
    )

    print(f"\n✓ Generated {len(predictions)} predictions with full UQ")
    return predictions


# ============================================================================
# PART 6: Analyze Results
# ============================================================================

def analyze_results(predictions):
    """
    Analyze and display UQ results.
    """
    print("\n" + "=" * 70)
    print("UNCERTAINTY QUANTIFICATION RESULTS")
    print("=" * 70)

    # Extract statistics
    means = np.array([p.mean for p in predictions])
    stds = np.array([p.std for p in predictions])
    epistemic_fractions = np.array([p.epistemic_fraction for p in predictions])
    hyperparameter_contributions = np.array([p.hyperparameter_contribution for p in predictions])
    ood_flags = np.array([p.spatial_ood for p in predictions])

    print("\nPrediction Statistics:")
    print(f"  Mean PM2.5: {np.mean(means):.2f} μg/m³")
    print(f"  Mean uncertainty: {np.mean(stds):.2f} μg/m³")
    print(f"  Mean interval width: {np.mean([p.interval_width for p in predictions]):.2f} μg/m³")

    print("\nUncertainty Decomposition:")
    print(f"  Epistemic fraction: {np.mean(epistemic_fractions):.1%}")
    print(f"  Aleatoric fraction: {1 - np.mean(epistemic_fractions):.1%}")
    print(f"  Hyperparameter contribution: {np.mean(hyperparameter_contributions):.1%}")

    print("\nOut-of-Distribution Detection:")
    print(f"  OOD points detected: {np.sum(ood_flags)} / {len(predictions)} ({np.mean(ood_flags):.1%})")

    print("\nConformal Prediction:")
    conformal_guaranteed = np.sum([p.conformal_guaranteed for p in predictions])
    print(f"  Predictions with guarantee: {conformal_guaranteed} / {len(predictions)}")

    # Show first 5 predictions in detail
    print("\n" + "-" * 70)
    print("FIRST 5 PREDICTIONS (DETAILED)")
    print("-" * 70)

    for i in range(min(5, len(predictions))):
        p = predictions[i]
        print(f"\nPrediction {i + 1}:")
        print(f"  Mean: {p.mean:.2f} μg/m³")
        print(f"  95% CI: [{p.lower_95:.2f}, {p.upper_95:.2f}]")
        print(f"  Total std: {p.std:.2f} μg/m³")
        print(f"    - Epistemic: {p.epistemic_std:.2f} ({p.epistemic_fraction:.1%})")
        print(f"    - Aleatoric: {p.aleatoric_std:.2f} ({1-p.epistemic_fraction:.1%})")
        print(f"    - Hyperparameter: {p.hyperparameter_contribution:.1%}")
        print(f"  OOD: {'⚠️ YES' if p.spatial_ood else '✓ No'} (score: {p.ood_score:.2f})")
        print(f"  Conformal guarantee: {'✓' if p.conformal_guaranteed else '✗'}")
        if p.meta_uncertainty_cv is not None:
            print(f"  Meta-uncertainty CV: {p.meta_uncertainty_cv:.3f} {'⚠️' if not p.uncertainty_reliable else '✓'}")


# ============================================================================
# PART 7: Generate Policy Outputs
# ============================================================================

def generate_policy_outputs(uq_system, predictions, X_test):
    """
    Generate actionable policy outputs from UQ predictions.
    """
    print("\n" + "=" * 70)
    print("POLICY OUTPUTS")
    print("=" * 70)

    # Generate policy outputs
    policy_outputs = uq_system.generate_policy_outputs(
        predictions,
        X_test=X_test,
        location_names=[f"Location_{i}" for i in range(len(predictions))]
    )

    # Display health alerts
    print("\nHealth Alerts:")
    health_alerts = policy_outputs['health_alerts']
    for i, alert in enumerate(health_alerts[:5]):  # Show first 5
        print(f"\n  Location {i}:")
        print(f"    Level: {alert.alert_level}")
        print(f"    Message: {alert.message}")
        print(f"    Certainty: {alert.certainty}")
        if alert.recommended_actions:
            print(f"    Actions: {', '.join(alert.recommended_actions)}")

    # Display sensor placement recommendations
    if policy_outputs['sensor_recommendations'] is not None:
        print("\nTop 5 Sensor Placement Recommendations:")
        sensor_recs = policy_outputs['sensor_recommendations']
        # Handle both dict and object formats
        if isinstance(sensor_recs, dict) and 'priority_locations' in sensor_recs:
            locs = sensor_recs['priority_locations']
        elif hasattr(sensor_recs, 'priority_locations'):
            locs = sensor_recs.priority_locations
        else:
            locs = []

        for i, rec in enumerate(locs[:5], 1):
            print(f"  {i}. Location {rec['location_id']}: "
                  f"Epistemic uncertainty = {rec['epistemic_uncertainty']:.2f} μg/m³")

    # Display decision report summary
    print("\nDecision Support Report generated")
    print(f"  Report contains {len(policy_outputs['decision_report'])} location analyses")


# ============================================================================
# PART 8: Evaluate Calibration
# ============================================================================

def evaluate_calibration(uq_system, X_test, y_test, sources_test):
    """
    Evaluate calibration quality on test data.
    """
    print("\n" + "=" * 70)
    print("CALIBRATION EVALUATION")
    print("=" * 70)

    metrics = uq_system.evaluate_calibration(X_test, y_test, sources_test)

    print("\nCalibration Metrics:")
    print(f"  PICP (95%): {metrics['picp']:.3f} (target: 0.950)")
    print(f"  Mean interval width: {metrics['mean_interval_width']:.2f} μg/m³")
    print(f"  CRPS: {metrics['crps']:.3f} (lower is better)")

    if 'ece' in metrics:
        print(f"  ECE: {metrics['ece']:.3f} (target: <0.05)")

    # Interpretation
    print("\nInterpretation:")
    if abs(metrics['picp'] - 0.95) < 0.02:
        print("  ✓ Excellent calibration (PICP ≈ 95%)")
    elif abs(metrics['picp'] - 0.95) < 0.05:
        print("  ✓ Good calibration (PICP close to 95%)")
    else:
        print(f"  ⚠️ Calibration issue: PICP = {metrics['picp']:.1%} (target 95%)")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete FusionGP UQ example."""

    print("=" * 70)
    print("FUSIONGP UNCERTAINTY QUANTIFICATION - COMPLETE EXAMPLE")
    print("=" * 70)

    # 1. Load model and data
    model = load_fusiongp_model()
    (X_train, y_train, sources_train,
     X_cal, y_cal, sources_cal,
     X_test, y_test, sources_test) = load_air_quality_data()

    # 2. Create UQ system
    # Options: 'default', 'fast', 'rigorous', 'custom'
    uq_system = create_uq_system(model, config_type='default')

    # 3. Fit and calibrate
    fit_and_calibrate(
        uq_system,
        X_train, y_train, sources_train,
        X_cal, y_cal, sources_cal
    )

    # 4. Make predictions
    predictions = make_predictions(uq_system, X_test, sources_test)

    # 5. Analyze results
    analyze_results(predictions)

    # 6. Generate policy outputs
    generate_policy_outputs(uq_system, predictions, X_test)

    # 7. Evaluate calibration
    evaluate_calibration(uq_system, X_test, y_test, sources_test)

    print("\n" + "=" * 70)
    print("✓ COMPLETE EXAMPLE FINISHED")
    print("=" * 70)

    # Export results example
    print("\nTo export results:")
    print("  import json")
    print("  results = [p.to_dict() for p in predictions]")
    print("  with open('uq_results.json', 'w') as f:")
    print("      json.dump(results, f, indent=2)")


if __name__ == '__main__':
    main()
