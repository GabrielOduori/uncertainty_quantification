"""
Visualization Demo: Creating Beautiful GP-Style Plots

This example demonstrates how to create publication-quality visualizations
of the FusionGP UQ system results using the GP visualization tools.

Usage:
    python examples/visualization_demo.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# Import the UQ system
from fusiongp_uq_system import create_default_uq_system

# Import visualization tools
from visualization.gp_plots import (
    GPUncertaintyVisualizer,
    quick_plot,
    quick_spatial_plot,
    quick_summary
)


# ============================================================================
# PART 1: Load Mock Model and Data (Same as complete example)
# ============================================================================

def create_mock_model():
    """Create mock FusionGP model for demonstration."""

    class MockFusionGP:
        """Mock FusionGP model for demonstration."""

        def predict_f(self, X):
            """Simulate GP predictions with spatial structure."""
            n = X.shape[0]
            # Create spatial pattern based on latitude/longitude
            spatial_pattern = 35.0 + 10.0 * np.sin(X[:, 0] * 10) + 5.0 * np.cos(X[:, 1] * 10)
            mean = spatial_pattern + np.random.randn(n) * 0.5

            # Variance increases with distance from center
            center = np.array([34.05, -118.25])
            distances = np.sqrt((X[:, 0] - center[0])**2 + (X[:, 1] - center[1])**2)
            var = 2.0 + 3.0 * distances + np.random.rand(n) * 0.5

            return mean, var

        def get_lengthscales(self):
            return [0.05, 0.05]  # ~5km in degrees

    return MockFusionGP()


def create_demo_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                 np.ndarray, np.ndarray, np.ndarray,
                                 np.ndarray, np.ndarray, np.ndarray]:
    """
    Create demonstration data with clear spatial structure.

    Returns
    -------
    X_train, y_train, sources_train : Training data
    X_cal, y_cal, sources_cal : Calibration data
    X_test, y_test, sources_test : Test data
    """
    print("Creating demonstration data with spatial structure...")

    np.random.seed(42)

    # LA Basin approximate bounds
    lat_min, lat_max = 33.7, 34.3
    lon_min, lon_max = -118.7, -117.6

    # Training data (500 samples) - clustered near monitors
    n_train = 500
    X_train = np.random.randn(n_train, 3) * 0.08 + [34.05, -118.25, 0]
    X_train[:, 0] = np.clip(X_train[:, 0], lat_min, lat_max)
    X_train[:, 1] = np.clip(X_train[:, 1], lon_min, lon_max)

    # Generate true function values
    y_train = 35.0 + 10.0 * np.sin(X_train[:, 0] * 10) + \
              5.0 * np.cos(X_train[:, 1] * 10) + np.random.randn(n_train) * 3
    sources_train = np.random.choice([0, 1, 2], size=n_train, p=[0.3, 0.5, 0.2])

    # Calibration data (100 samples)
    n_cal = 100
    X_cal = np.random.randn(n_cal, 3) * 0.08 + [34.05, -118.25, 0]
    X_cal[:, 0] = np.clip(X_cal[:, 0], lat_min, lat_max)
    X_cal[:, 1] = np.clip(X_cal[:, 1], lon_min, lon_max)
    y_cal = 35.0 + 10.0 * np.sin(X_cal[:, 0] * 10) + \
            5.0 * np.cos(X_cal[:, 1] * 10) + np.random.randn(n_cal) * 3
    sources_cal = np.random.choice([0, 1, 2], size=n_cal, p=[0.3, 0.5, 0.2])

    # Test data (200 samples) - grid covering full domain including OOD
    n_test = 200
    # Create a grid with some points far from training data
    lat_grid = np.linspace(lat_min, lat_max, 14)
    lon_grid = np.linspace(lon_min, lon_max, 14)
    lat_test, lon_test = np.meshgrid(lat_grid, lon_grid)
    X_test = np.column_stack([lat_test.flatten()[:n_test],
                              lon_test.flatten()[:n_test],
                              np.zeros(n_test)])

    y_test = 35.0 + 10.0 * np.sin(X_test[:, 0] * 10) + \
             5.0 * np.cos(X_test[:, 1] * 10) + np.random.randn(n_test) * 3
    sources_test = np.random.choice([0, 1, 2], size=n_test, p=[0.3, 0.5, 0.2])

    print(f"✓ Data created:")
    print(f"  Training: {n_train} samples")
    print(f"  Calibration: {n_cal} samples")
    print(f"  Test: {n_test} samples")

    return (X_train, y_train, sources_train,
            X_cal, y_cal, sources_cal,
            X_test, y_test, sources_test)


# ============================================================================
# PART 2: Run UQ System
# ============================================================================

def run_uq_system():
    """Run the UQ system and return predictions."""
    print("\n" + "=" * 70)
    print("RUNNING UQ SYSTEM")
    print("=" * 70)

    # Create model and data
    model = create_mock_model()
    (X_train, y_train, sources_train,
     X_cal, y_cal, sources_cal,
     X_test, y_test, sources_test) = create_demo_data()

    # Create UQ system (fast mode for demo)
    print("\nCreating UQ system (fast mode for demo)...")
    from fusiongp_uq_system import create_fast_uq_system
    uq_system = create_fast_uq_system(model)

    # Fit and calibrate
    print("\nFitting ensemble...")
    uq_system.fit_ensemble(X_train, y_train, sources_train, verbose=False)

    print("Calibrating...")
    uq_system.calibrate(X_cal, y_cal, sources_cal, verbose=False)

    # Make predictions
    print("Making predictions with full UQ...")
    predictions = uq_system.predict_with_full_uq(X_test, sources_test)

    print(f"\n✓ Generated {len(predictions)} predictions")

    return X_train, y_train, X_test, y_test, predictions


# ============================================================================
# PART 3: Create Visualizations
# ============================================================================

def demo_basic_plots(X_train, y_train, X_test, y_test, predictions):
    """Demonstrate basic 1D and spatial plots."""
    print("\n" + "=" * 70)
    print("DEMO 1: BASIC GP PLOTS")
    print("=" * 70)

    viz = GPUncertaintyVisualizer()

    # 1D plot along latitude
    print("\n1. Creating 1D uncertainty plot along latitude...")
    fig1, ax1 = viz.plot_1d_with_uncertainty(
        X_test, predictions, y_test,
        feature_idx=0,
        feature_name='Latitude',
        y_name='PM2.5 (μg/m³)',
        title='GP Predictions with Uncertainty Bands'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'gp_1d_uncertainty.png'),
                dpi=300, bbox_inches='tight')
    print("   ✓ Saved to results/gp_1d_uncertainty.png")
    plt.close()

    # Quick plot version
    print("\n2. Creating quick 1D plot (convenience function)...")
    fig2 = quick_plot(X_test, predictions, y_test,
                      save_path=os.path.join(OUT_DIR, 'gp_quick_1d.png'))
    printprint("   ✓ Saved to results/gp_quick_1d.png")
    plt.close()


def demo_spatial_plots(X_train, y_train, X_test, y_test, predictions):
    """Demonstrate spatial uncertainty maps."""
    print("\n" + "=" * 70)
    print("DEMO 2: SPATIAL UNCERTAINTY MAPS")
    print("=" * 70)

    viz = GPUncertaintyVisualizer()

    # Total uncertainty map
    print("\n1. Creating total uncertainty map...")
    fig1, ax1 = viz.plot_spatial_uncertainty_map(
        X_test, predictions,
        metric='total',
        title='Total Uncertainty (σ)',
        plot_type='scatter'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'spatial_total_uncertainty.png'),
                dpi=300, bbox_inches='tight')
    print("   ✓ Saved to results/spatial_total_uncertainty.png")
    plt.close()

    # Epistemic uncertainty map
    print("\n2. Creating epistemic uncertainty map...")
    fig2, ax2 = viz.plot_spatial_uncertainty_map(
        X_test, predictions,
        metric='epistemic',
        title='Epistemic Uncertainty (Reducible)',
        plot_type='interpolated'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'spatial_epistemic_uncertainty.png'),
                dpi=300, bbox_inches='tight')
    print("   ✓ Saved to results/spatial_epistemic_uncertainty.png")
    plt.close()

    # Quick spatial plot
    print("\n3. Creating quick spatial plot (convenience function)...")
    fig3 = quick_spatial_plot(X_test, predictions,
                              save_path=os.path.join(OUT_DIR, 'spatial_quick.png'))
    printprint("   ✓ Saved to results/spatial_quick.png")
    plt.close()


def demo_decomposition_plot(X_train, y_train, X_test, y_test, predictions):
    """Demonstrate uncertainty decomposition."""
    print("\n" + "=" * 70)
    print("DEMO 3: UNCERTAINTY DECOMPOSITION")
    print("=" * 70)

    viz = GPUncertaintyVisualizer()

    print("\nCreating uncertainty decomposition plot...")
    fig, ax = viz.plot_uncertainty_decomposition(
        X_test, predictions,
        feature_idx=0,
        feature_name='Latitude',
        title='Epistemic vs Aleatoric Uncertainty'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'uncertainty_decomposition.png'),
                dpi=300, bbox_inches='tight')
    print("✓ Saved to results/uncertainty_decomposition.png")
    plt.close()


def demo_ood_detection(X_train, y_train, X_test, y_test, predictions):
    """Demonstrate OOD detection visualization."""
    print("\n" + "=" * 70)
    print("DEMO 4: OUT-OF-DISTRIBUTION DETECTION")
    print("=" * 70)

    viz = GPUncertaintyVisualizer()

    print("\nCreating OOD detection plot...")
    fig, ax = viz.plot_ood_detection(
        X_test, predictions, X_train,
        feature_idx=0,
        feature_name='Latitude',
        title='OOD Detection (Spatial Extrapolation)'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'ood_detection.png'),
                dpi=300, bbox_inches='tight')
    print("✓ Saved to results/ood_detection.png")
    plt.close()


def demo_calibration_curve(X_train, y_train, X_test, y_test, predictions):
    """Demonstrate calibration curve."""
    print("\n" + "=" * 70)
    print("DEMO 5: CALIBRATION CURVE")
    print("=" * 70)

    viz = GPUncertaintyVisualizer()

    print("\nCreating calibration curve...")
    fig, ax = viz.plot_calibration_curve(
        predictions, y_test,
        title='Calibration Curve (Reliability Diagram)'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'calibration_curve.png'),
                dpi=300, bbox_inches='tight')
    print("✓ Saved to results/calibration_curve.png")
    plt.close()


def demo_complete_summary(X_train, y_train, X_test, y_test, predictions):
    """Demonstrate complete summary figure."""
    print("\n" + "=" * 70)
    print("DEMO 6: COMPLETE SUMMARY FIGURE (PUBLICATION-QUALITY)")
    print("=" * 70)

    viz = GPUncertaintyVisualizer()

    print("\nCreating complete 6-panel summary figure...")
    print("This creates a comprehensive figure perfect for dissertations/papers!")

    fig = viz.plot_complete_summary(
        X_test, predictions, y_test, X_train,
        suptitle='FusionGP Uncertainty Quantification - Complete Summary'
    )
    plt.savefig(os.path.join(OUT_DIR, 'complete_summary.png'),
                dpi=300, bbox_inches='tight')
    print("✓ Saved to results/complete_summary.png")
    plt.close()

    # Quick summary version
    print("\nCreating quick summary (convenience function)...")
    fig2 = quick_summary(X_test, predictions, y_test, X_train,
                        save_path=os.path.join(OUT_DIR, 'quick_summary.png'))
    printprint("✓ Saved to results/quick_summary.png")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete visualization demo."""
    global OUT_DIR
    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR = str(Path(__file__).resolve().parent.parent / "output" / "visualization_demo" / run_ts)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("FUSIONGP UQ SYSTEM - VISUALIZATION DEMO")
    print("=" * 70)
    print(f"\nOutput → {OUT_DIR}")
    print("\nThis demo creates beautiful GP-style plots of UQ results.")
    print("Perfect for dissertations, papers, and presentations!")

    # Run UQ system
    X_train, y_train, X_test, y_test, predictions = run_uq_system()

    # Create all visualizations
    demo_basic_plots(X_train, y_train, X_test, y_test, predictions)
    demo_spatial_plots(X_train, y_train, X_test, y_test, predictions)
    demo_decomposition_plot(X_train, y_train, X_test, y_test, predictions)
    demo_ood_detection(X_train, y_train, X_test, y_test, predictions)
    demo_calibration_curve(X_train, y_train, X_test, y_test, predictions)
    demo_complete_summary(X_train, y_train, X_test, y_test, predictions)

    # Summary
    print("\n" + "=" * 70)
    print("✓ VISUALIZATION DEMO COMPLETE")
    print("=" * 70)
    print("\nGenerated Plots:")
    print("  1. gp_1d_uncertainty.png - Classic GP plot with uncertainty bands")
    print("  2. gp_quick_1d.png - Quick 1D plot (convenience function)")
    print("  3. spatial_total_uncertainty.png - Total uncertainty map")
    print("  4. spatial_epistemic_uncertainty.png - Epistemic uncertainty map")
    print("  5. spatial_quick.png - Quick spatial plot")
    print("  6. uncertainty_decomposition.png - Epistemic vs aleatoric")
    print("  7. ood_detection.png - Out-of-distribution detection")
    print("  8. calibration_curve.png - Reliability diagram")
    print("  9. complete_summary.png - 6-panel comprehensive figure ⭐")
    print("  10. quick_summary.png - Quick summary version")
    print("\nAll plots saved to: results/")
    print("\n🎨 Beautiful GP-style plots created successfully!")


if __name__ == '__main__':
    main()
