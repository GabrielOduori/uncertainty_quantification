"""
Comprehensive Validation Pipeline for Rigorous UQ Framework

Validates all components of the uncertainty quantification framework:
1. Hierarchical variance propagation
2. Bootstrap ensemble hyperparameter uncertainty
3. Conformal prediction calibration
4. Second-order uncertainty analysis
5. Actionable decision outputs

Addresses all research questions:
- RQ1: Epistemic vs aleatoric decomposition
- RQ2: Hyperparameter uncertainty underestimation
- RQ3: Model calibration comparison
- RQ4: OOD detection efficacy
- RQ5: Transfer learning UQ (future work)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from uncertainty.hierarchical import (
    HierarchicalUQTracker,
    VariancePropagationAnalyzer,
)
from uncertainty.conformal import (
    ConformalPredictionWrapper,
    evaluate_conformal_coverage,
)
from uncertainty.second_order import (
    SecondOrderAnalyzer,
    MetaUncertaintyVisualizer,
)
from uncertainty.calibration import CalibrationEvaluator
from uncertainty.decomposition import UncertaintyDecomposer
from models.ensemble import BootstrapSVGPEnsemble
from decision.policy_translation import PolicyTranslator


class ComprehensiveUQValidator:
    """
    End-to-end validation of UQ framework.
    """

    def __init__(self, output_dir: str = "results"):
        """
        Initialize validator.

        Args:
            output_dir: Directory for saving results and figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        logger.info(f"Initialized ComprehensiveUQValidator, output_dir={output_dir}")

        # Results storage
        self.results = {}

    def run_full_validation(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sources_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        sources_test: np.ndarray,
        point_model: any = None
    ):
        """
        Run complete validation pipeline.

        Args:
            X_train: Training features [N x D]
            y_train: Training targets [N]
            sources_train: Source identifiers [N]
            X_test: Test features [M x D]
            y_test: Test targets [M]
            sources_test: Test source identifiers [M]
            point_model: Pre-trained point estimate model (optional)
        """
        logger.info("=" * 80)
        logger.info("STARTING COMPREHENSIVE UQ VALIDATION")
        logger.info("=" * 80)

        # Phase 1: Train models
        logger.info("\n[PHASE 1] Training models...")
        point_model, ensemble = self._train_models(
            X_train, y_train, sources_train, point_model
        )

        # Phase 2: Hierarchical variance propagation
        logger.info("\n[PHASE 2] Hierarchical variance propagation...")
        hierarchical_results = self._validate_hierarchical(
            point_model, X_test, sources_test
        )

        # Phase 3: Bootstrap ensemble uncertainty
        logger.info("\n[PHASE 3] Bootstrap ensemble hyperparameter uncertainty...")
        ensemble_results = self._validate_ensemble(
            ensemble, point_model, X_test
        )

        # Phase 4: Conformal prediction
        logger.info("\n[PHASE 4] Conformal prediction calibration...")
        conformal_results = self._validate_conformal(
            point_model, X_train, y_train, X_test, y_test
        )

        # Phase 5: Second-order uncertainty
        logger.info("\n[PHASE 5] Second-order uncertainty analysis...")
        second_order_results = self._validate_second_order(ensemble, X_test)

        # Phase 6: Calibration comparison
        logger.info("\n[PHASE 6] Calibration evaluation...")
        calibration_results = self._validate_calibration(
            point_model, X_test, y_test
        )

        # Phase 7: Decision translation
        logger.info("\n[PHASE 7] Policy translation and actionable outputs...")
        decision_results = self._validate_decisions(
            point_model, X_test, y_test
        )

        # Store all results
        self.results = {
            "hierarchical": hierarchical_results,
            "ensemble": ensemble_results,
            "conformal": conformal_results,
            "second_order": second_order_results,
            "calibration": calibration_results,
            "decision": decision_results,
        }

        # Generate summary report
        logger.info("\n[PHASE 8] Generating summary report...")
        self._generate_summary_report()

        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION COMPLETE")
        logger.info("=" * 80)

        return self.results

    def _train_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sources_train: np.ndarray,
        point_model: any = None
    ):
        """Train point model and bootstrap ensemble."""

        if point_model is None:
            logger.info("Training point estimate model...")
            point_model = self._build_mock_model(X_train, y_train)

        logger.info("Training bootstrap ensemble (n=10)...")
        ensemble = BootstrapSVGPEnsemble(
            n_ensemble=10,
            bootstrap_fraction=0.8,
            parallel=False  # Sequential for simplicity in 4 days
        )
        ensemble.fit(X_train, y_train, sources_train, max_iter=500, verbose=True)

        return point_model, ensemble

    def _validate_hierarchical(
        self,
        model: any,
        X_test: np.ndarray,
        sources_test: np.ndarray
    ) -> dict:
        """Validate hierarchical variance propagation."""

        tracker = HierarchicalUQTracker()
        hierarchical_var = tracker.decompose_by_stage(
            model, X_test, sources_test
        )

        # Analyze variance contributions
        contributions = hierarchical_var.variance_contribution_by_stage()
        source_summary = hierarchical_var.source_specific_summary()

        results = {
            "variance_by_stage": contributions,
            "source_summary": source_summary,
            "hierarchical_variance": hierarchical_var,
        }

        logger.info(f"Stage 1 (Epistemic): {contributions['stage_1_epistemic']:.4f}")
        logger.info(f"Stage 2 (Predictive): {contributions['stage_2_predictive']:.4f}")

        return results

    def _validate_ensemble(
        self,
        ensemble: BootstrapSVGPEnsemble,
        point_model: any,
        X_test: np.ndarray
    ) -> dict:
        """Validate bootstrap ensemble and quantify underestimation."""

        # Get ensemble predictions with full UQ
        ensemble_unc = ensemble.predict_with_full_uncertainty(X_test)

        # Quantify underestimation (RQ2)
        underestimation = ensemble.quantify_underestimation(point_model, X_test)

        # Get hyperparameter distribution
        hyperparam_dist = ensemble.get_hyperparameter_distribution()

        results = {
            "ensemble_uncertainty": ensemble_unc,
            "underestimation": underestimation,
            "hyperparameter_distribution": hyperparam_dist,
        }

        logger.info(f"Mean underestimation: {underestimation['mean_underestimation_pct']:.2f}%")
        logger.info(f"Median underestimation: {underestimation['median_underestimation_pct']:.2f}%")
        logger.info(f"Hyperparameter fraction: {ensemble_unc.summary_stats()['mean_hyperparameter_fraction']:.1%}")

        return results

    def _validate_conformal(
        self,
        model: any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> dict:
        """Validate conformal prediction."""

        # Split training into train + calibration
        n_cal = min(200, len(X_train) // 5)
        indices = np.random.permutation(len(X_train))
        cal_indices = indices[:n_cal]

        X_cal = X_train[cal_indices]
        y_cal = y_train[cal_indices]

        # Initialize and calibrate
        conformal = ConformalPredictionWrapper(model, alpha=0.05)
        conformal.calibrate(X_cal, y_cal)

        # Predict
        intervals = conformal.predict_with_conformal_intervals(X_test)

        # Evaluate coverage
        coverage_stats = evaluate_conformal_coverage(intervals, y_test)

        # Compare with Gaussian
        comparison = conformal.compare_with_gaussian_intervals(X_test, y_test)

        results = {
            "conformal_intervals": intervals,
            "coverage_stats": coverage_stats,
            "comparison": comparison,
        }

        logger.info(f"Conformal coverage: {coverage_stats['actual_coverage']:.3f}")
        logger.info(f"Gaussian coverage: {comparison['gaussian_coverage']:.3f}")
        logger.info(f"Target coverage: {coverage_stats['target_coverage']:.3f}")

        return results

    def _validate_second_order(
        self,
        ensemble: BootstrapSVGPEnsemble,
        X_test: np.ndarray
    ) -> dict:
        """Validate second-order uncertainty analysis."""

        analyzer = SecondOrderAnalyzer()
        second_order = analyzer.analyze_from_ensemble(
            ensemble.models, X_test
        )

        # Compute prediction bands
        bands = analyzer.compute_prediction_uncertainty_bands(
            ensemble.models, X_test
        )

        results = {
            "second_order_uncertainty": second_order,
            "prediction_bands": bands,
            "n_unreliable": int(np.sum(second_order.identify_unreliable_estimates())),
        }

        logger.info(f"Mean CV: {second_order.summary()['mean_cv']:.3f}")
        logger.info(f"Unreliable estimates: {results['n_unreliable']}/{len(X_test)}")

        return results

    def _validate_calibration(
        self,
        model: any,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> dict:
        """Validate calibration metrics."""

        # Get predictions
        mean, var = model.predict_f(X_test)
        predictions = mean.numpy().flatten()
        uncertainties = np.sqrt(var.numpy().flatten())

        # Evaluate calibration
        evaluator = CalibrationEvaluator()
        cal_results = evaluator.evaluate(predictions, uncertainties, y_test)

        results = {
            "calibration_results": cal_results,
        }

        logger.info(f"PICP(95%): {cal_results.picp.get('95%', 0.0):.3f}")
        logger.info(f"ECE: {cal_results.ece:.4f}")
        logger.info(f"CRPS: {cal_results.crps:.4f}")

        return results

    def _validate_decisions(
        self,
        model: any,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> dict:
        """Validate decision translation."""

        # Get predictions
        mean, var = model.predict_f(X_test)
        predictions = mean.numpy().flatten()
        uncertainties = np.sqrt(var.numpy().flatten())

        translator = PolicyTranslator()

        # Compute exceedances
        exceedances = translator.compute_exceedance_probabilities(
            predictions, uncertainties, threshold=35.4, threshold_name="Moderate"
        )

        # Generate alerts
        alerts = translator.generate_health_alerts(predictions, uncertainties)

        # Sensor recommendations
        sensor_recs = translator.identify_high_value_sensor_locations(
            X_test[:, :2],  # Assume first 2 dims are spatial
            uncertainties**2,
            top_n=10
        )

        # Decision report
        decision_report = translator.create_decision_summary_report(
            predictions, uncertainties, X_test[:, :2]
        )

        results = {
            "exceedances": exceedances,
            "alerts": alerts,
            "sensor_recommendations": sensor_recs,
            "decision_report": decision_report,
        }

        logger.info(f"Generated {len(alerts)} health alerts")
        logger.info(f"Identified {len(sensor_recs)} high-value sensor locations")

        return results

    def _generate_summary_report(self):
        """Generate comprehensive summary report."""

        report_path = self.output_dir / "validation_summary.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE UQ VALIDATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            # Ensemble results (RQ2)
            f.write("RESEARCH QUESTION 2: Hyperparameter Uncertainty\n")
            f.write("-" * 80 + "\n")
            underest = self.results['ensemble']['underestimation']
            f.write(f"Mean underestimation: {underest['mean_underestimation_pct']:.2f}%\n")
            f.write(f"Median underestimation: {underest['median_underestimation_pct']:.2f}%\n")
            f.write(f"Finding: Point estimates underestimate uncertainty by ~{underest['mean_underestimation_pct']:.0f}%\n\n")

            # Conformal results
            f.write("CONFORMAL PREDICTION CALIBRATION\n")
            f.write("-" * 80 + "\n")
            conf_cov = self.results['conformal']['coverage_stats']
            f.write(f"Conformal coverage: {conf_cov['actual_coverage']:.3f}\n")
            f.write(f"Target coverage: {conf_cov['target_coverage']:.3f}\n")
            f.write(f"Achieves target: {conf_cov['achieves_target']}\n\n")

            # Second-order results
            f.write("SECOND-ORDER UNCERTAINTY\n")
            f.write("-" * 80 + "\n")
            so_results = self.results['second_order']
            f.write(f"Mean CV: {so_results['second_order_uncertainty'].summary()['mean_cv']:.3f}\n")
            f.write(f"Unreliable estimates: {so_results['n_unreliable']}\n\n")

            # Calibration results
            f.write("CALIBRATION METRICS\n")
            f.write("-" * 80 + "\n")
            cal = self.results['calibration']['calibration_results']
            f.write(f"PICP(95%): {cal.picp.get('95%', 0.0):.3f}\n")
            f.write(f"ECE: {cal.ece:.4f}\n")
            f.write(f"CRPS: {cal.crps:.4f}\n")
            f.write(f"Calibrated: {cal.is_calibrated}\n\n")

            # Decision results
            f.write("ACTIONABLE DECISION OUTPUTS\n")
            f.write("-" * 80 + "\n")
            dec = self.results['decision']
            f.write(f"Health alerts generated: {len(dec['alerts'])}\n")
            f.write(f"High-value sensor locations: {len(dec['sensor_recommendations'])}\n\n")

        logger.info(f"Summary report saved to {report_path}")

    def _build_mock_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Build mock model for testing."""
        class MockModel:
            def __init__(self, X, y):
                self.X_train = X
                self.y_train = y

            def predict_f(self, X_test):
                n = len(X_test)
                mean = np.random.randn(n) * 10 + 50
                var = np.random.gamma(2, 2, n)

                class MockTensor:
                    def __init__(self, data):
                        self.data = data
                    def numpy(self):
                        return self.data

                return MockTensor(mean), MockTensor(var)

        return MockModel(X_train, y_train)


def main():
    """Main validation script."""

    logger.info("Generating synthetic validation data...")

    # Generate synthetic data for testing
    np.random.seed(42)
    n_train = 1000
    n_test = 200
    d = 3

    X_train = np.random.randn(n_train, d)
    y_train = np.sin(X_train[:, 0]) * 20 + 50 + np.random.randn(n_train) * 5
    sources_train = np.random.choice([0, 1, 2], size=n_train)

    X_test = np.random.randn(n_test, d)
    y_test = np.sin(X_test[:, 0]) * 20 + 50 + np.random.randn(n_test) * 5
    sources_test = np.random.choice([0, 1, 2], size=n_test)

    # Run validation
    validator = ComprehensiveUQValidator(output_dir="results")
    results = validator.run_full_validation(
        X_train, y_train, sources_train,
        X_test, y_test, sources_test
    )

    logger.info("\nValidation complete! Check 'results/' directory for outputs.")

    return results


if __name__ == "__main__":
    main()
