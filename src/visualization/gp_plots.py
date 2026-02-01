"""
Gaussian Process Visualization for Uncertainty Quantification

Creates publication-quality plots showing:
- GP predictions with uncertainty bands
- Spatial uncertainty maps
- Calibration curves
- Epistemic/aleatoric decomposition
- OOD detection visualization

These are the "classic GP plots" showing predictions with beautiful uncertainty bands.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import List, Optional, Tuple, Dict, Any
import warnings

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


class GPUncertaintyVisualizer:
    """
    Create beautiful GP-style uncertainty visualizations.

    Examples
    --------
    >>> from visualization.gp_plots import GPUncertaintyVisualizer
    >>>
    >>> # After getting predictions from UQ system
    >>> viz = GPUncertaintyVisualizer()
    >>>
    >>> # 1D plot with uncertainty bands
    >>> viz.plot_1d_with_uncertainty(X_test, predictions, y_test)
    >>>
    >>> # 2D spatial map
    >>> viz.plot_spatial_uncertainty_map(X_test, predictions)
    >>>
    >>> # Epistemic/Aleatoric decomposition
    >>> viz.plot_uncertainty_decomposition(X_test, predictions)
    """

    def __init__(self, style: str = 'seaborn'):
        """
        Initialize visualizer.

        Parameters
        ----------
        style : str, default='seaborn'
            Plot style ('seaborn', 'classic', 'ggplot')
        """
        if style == 'seaborn' and not SEABORN_AVAILABLE:
            warnings.warn("seaborn not available, using matplotlib defaults")
            style = 'classic'

        if style != 'seaborn':
            plt.style.use(style)

        self.colors = {
            'mean': '#2E86AB',
            'uncertainty': '#A23B72',
            'epistemic': '#F18F01',
            'aleatoric': '#C73E1D',
            'ood': '#E63946',
            'data': '#06A77D',
        }

    def plot_1d_with_uncertainty(
        self,
        X_test: np.ndarray,
        predictions: List[Any],
        y_test: Optional[np.ndarray] = None,
        x_dim: int = 0,
        title: str = "GP Predictions with Uncertainty",
        xlabel: str = "Input",
        ylabel: str = "PM2.5 (μg/m³)",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ):
        """
        Plot 1D predictions with classic GP uncertainty bands.

        This creates the iconic GP plot with shaded uncertainty regions.

        Parameters
        ----------
        X_test : np.ndarray
            Test inputs (n_samples, n_features)
        predictions : List[UQPrediction]
            Predictions from UQ system
        y_test : np.ndarray, optional
            True values (for comparison)
        x_dim : int, default=0
            Which dimension to plot (for multi-dimensional inputs)
        title : str
            Plot title
        xlabel, ylabel : str
            Axis labels
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        """
        # Extract data
        x = X_test[:, x_dim]
        means = np.array([p.mean for p in predictions])
        lower_95 = np.array([p.lower_95 for p in predictions])
        upper_95 = np.array([p.upper_95 for p in predictions])
        stds = np.array([p.std for p in predictions])

        # Sort by x for nice plotting
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        means = means[sort_idx]
        lower_95 = lower_95[sort_idx]
        upper_95 = upper_95[sort_idx]
        stds = stds[sort_idx]
        if y_test is not None:
            y_test = y_test[sort_idx]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot uncertainty bands (95% CI)
        ax.fill_between(
            x, lower_95, upper_95,
            alpha=0.3, color=self.colors['uncertainty'],
            label='95% Confidence Interval'
        )

        # Plot 1-sigma bands
        ax.fill_between(
            x, means - stds, means + stds,
            alpha=0.5, color=self.colors['uncertainty'],
            label='±1σ (68% interval)'
        )

        # Plot mean prediction
        ax.plot(x, means, color=self.colors['mean'], linewidth=2, label='Mean Prediction')

        # Plot true values if available
        if y_test is not None:
            ax.scatter(x, y_test, color=self.colors['data'], s=20, alpha=0.6,
                      label='True Values', zorder=5)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        return fig, ax

    def plot_spatial_uncertainty_map(
        self,
        X_test: np.ndarray,
        predictions: List[Any],
        metric: str = 'total',
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 6),
        save_path: Optional[str] = None
    ):
        """
        Plot spatial map of uncertainty (2D).

        Creates a heatmap showing how uncertainty varies across space.

        Parameters
        ----------
        X_test : np.ndarray
            Test locations (n_samples, n_features) - assumes [lat, lon, ...]
        predictions : List[UQPrediction]
            Predictions from UQ system
        metric : str, default='total'
            What to plot: 'total', 'epistemic', 'aleatoric', 'mean'
        title : str, optional
            Plot title
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        """
        # Extract spatial coordinates
        lats = X_test[:, 0]
        lons = X_test[:, 1]

        # Extract metric
        if metric == 'total':
            values = np.array([p.std for p in predictions])
            label = 'Total Uncertainty (μg/m³)'
            if title is None:
                title = 'Spatial Distribution of Total Uncertainty'
        elif metric == 'epistemic':
            values = np.array([p.epistemic_std for p in predictions])
            label = 'Epistemic Uncertainty (μg/m³)'
            if title is None:
                title = 'Spatial Distribution of Epistemic Uncertainty'
        elif metric == 'aleatoric':
            values = np.array([p.aleatoric_std for p in predictions])
            label = 'Aleatoric Uncertainty (μg/m³)'
            if title is None:
                title = 'Spatial Distribution of Aleatoric Uncertainty'
        elif metric == 'mean':
            values = np.array([p.mean for p in predictions])
            label = 'Mean PM2.5 (μg/m³)'
            if title is None:
                title = 'Spatial Distribution of PM2.5 Predictions'
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Create figure with two subplots
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 2, width_ratios=[1, 1])

        # Scatter plot
        ax1 = fig.add_subplot(gs[0])
        scatter = ax1.scatter(lons, lats, c=values, s=100, cmap='YlOrRd',
                            edgecolors='black', linewidth=0.5, alpha=0.8)
        ax1.set_xlabel('Longitude', fontsize=12)
        ax1.set_ylabel('Latitude', fontsize=12)
        ax1.set_title(f'{title}\n(Scatter View)', fontsize=12, fontweight='bold')
        cbar1 = plt.colorbar(scatter, ax=ax1)
        cbar1.set_label(label, fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Contour plot (interpolated)
        ax2 = fig.add_subplot(gs[1])

        # Create grid for interpolation
        lat_grid = np.linspace(lats.min(), lats.max(), 50)
        lon_grid = np.linspace(lons.min(), lons.max(), 50)
        LON_GRID, LAT_GRID = np.meshgrid(lon_grid, lat_grid)

        # Interpolate values onto grid
        from scipy.interpolate import griddata
        Z = griddata((lons, lats), values, (LON_GRID, LAT_GRID), method='cubic')

        contour = ax2.contourf(LON_GRID, LAT_GRID, Z, levels=15, cmap='YlOrRd', alpha=0.8)
        ax2.scatter(lons, lats, c='black', s=20, alpha=0.3, label='Measurement locations')
        ax2.set_xlabel('Longitude', fontsize=12)
        ax2.set_ylabel('Latitude', fontsize=12)
        ax2.set_title(f'{title}\n(Interpolated View)', fontsize=12, fontweight='bold')
        cbar2 = plt.colorbar(contour, ax=ax2)
        cbar2.set_label(label, fontsize=10)
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        return fig, (ax1, ax2)

    def plot_uncertainty_decomposition(
        self,
        X_test: np.ndarray,
        predictions: List[Any],
        x_dim: int = 0,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ):
        """
        Plot epistemic vs aleatoric uncertainty decomposition.

        Shows how different types of uncertainty vary across input space.

        Parameters
        ----------
        X_test : np.ndarray
            Test inputs
        predictions : List[UQPrediction]
            Predictions from UQ system
        x_dim : int, default=0
            Which dimension to plot
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        """
        # Extract data
        x = X_test[:, x_dim]
        means = np.array([p.mean for p in predictions])
        epistemic_std = np.array([p.epistemic_std for p in predictions])
        aleatoric_std = np.array([p.aleatoric_std for p in predictions])
        total_std = np.array([p.std for p in predictions])
        epistemic_frac = np.array([p.epistemic_fraction for p in predictions])

        # Sort by x
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        means = means[sort_idx]
        epistemic_std = epistemic_std[sort_idx]
        aleatoric_std = aleatoric_std[sort_idx]
        total_std = total_std[sort_idx]
        epistemic_frac = epistemic_frac[sort_idx]

        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

        # Subplot 1: Mean with decomposed uncertainty
        ax1 = axes[0]
        ax1.fill_between(x, means - total_std, means + total_std,
                        alpha=0.3, color='gray', label='Total Uncertainty')
        ax1.fill_between(x, means - epistemic_std, means + epistemic_std,
                        alpha=0.5, color=self.colors['epistemic'], label='Epistemic')
        ax1.fill_between(x, means - aleatoric_std, means + aleatoric_std,
                        alpha=0.5, color=self.colors['aleatoric'], label='Aleatoric')
        ax1.plot(x, means, color=self.colors['mean'], linewidth=2, label='Mean')
        ax1.set_ylabel('PM2.5 (μg/m³)', fontsize=11)
        ax1.set_title('GP Predictions with Decomposed Uncertainty', fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Uncertainty magnitudes
        ax2 = axes[1]
        ax2.fill_between(x, 0, total_std, alpha=0.3, color='gray', label='Total')
        ax2.plot(x, epistemic_std, color=self.colors['epistemic'], linewidth=2,
                label='Epistemic (reducible)')
        ax2.plot(x, aleatoric_std, color=self.colors['aleatoric'], linewidth=2,
                label='Aleatoric (irreducible)')
        ax2.set_ylabel('Uncertainty (μg/m³)', fontsize=11)
        ax2.set_title('Uncertainty Components', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Subplot 3: Epistemic fraction
        ax3 = axes[2]
        ax3.fill_between(x, 0, epistemic_frac, alpha=0.5, color=self.colors['epistemic'],
                        label='Epistemic Fraction')
        ax3.axhline(y=0.5, color='black', linestyle='--', linewidth=1, alpha=0.5,
                   label='50% threshold')
        ax3.set_xlabel('Input', fontsize=11)
        ax3.set_ylabel('Epistemic Fraction', fontsize=11)
        ax3.set_title('Fraction of Uncertainty that is Reducible', fontsize=13, fontweight='bold')
        ax3.set_ylim([0, 1])
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        return fig, axes

    def plot_ood_detection(
        self,
        X_test: np.ndarray,
        predictions: List[Any],
        X_train: Optional[np.ndarray] = None,
        x_dim: int = 0,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ):
        """
        Visualize out-of-distribution detection.

        Shows where the GP is extrapolating vs interpolating.

        Parameters
        ----------
        X_test : np.ndarray
            Test inputs
        predictions : List[UQPrediction]
            Predictions from UQ system
        X_train : np.ndarray, optional
            Training inputs (to show where data exists)
        x_dim : int, default=0
            Which dimension to plot
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        """
        # Extract data
        x = X_test[:, x_dim]
        means = np.array([p.mean for p in predictions])
        stds = np.array([p.std for p in predictions])
        ood_flags = np.array([p.spatial_ood for p in predictions])
        ood_scores = np.array([p.ood_score for p in predictions])

        # Sort by x
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        means = means[sort_idx]
        stds = stds[sort_idx]
        ood_flags = ood_flags[sort_idx]
        ood_scores = ood_scores[sort_idx]

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Subplot 1: Predictions with OOD regions highlighted
        ax1.fill_between(x, means - 1.96*stds, means + 1.96*stds,
                        alpha=0.3, color=self.colors['uncertainty'])
        ax1.plot(x, means, color=self.colors['mean'], linewidth=2, label='Mean Prediction')

        # Highlight OOD regions
        ood_x = x[ood_flags]
        ood_means = means[ood_flags]
        if len(ood_x) > 0:
            ax1.scatter(ood_x, ood_means, color=self.colors['ood'], s=100,
                       marker='x', linewidths=3, label='OOD Points', zorder=5)

        # Show training data locations
        if X_train is not None:
            x_train = X_train[:, x_dim]
            ax1.scatter(x_train, np.ones_like(x_train) * ax1.get_ylim()[0],
                       color=self.colors['data'], s=20, alpha=0.5,
                       marker='|', label='Training Data')

        ax1.set_ylabel('PM2.5 (μg/m³)', fontsize=11)
        ax1.set_title('GP Predictions with OOD Detection', fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Subplot 2: OOD scores
        ax2.fill_between(x, 0, ood_scores, alpha=0.5, color=self.colors['ood'])
        ax2.axhline(y=2.5, color='red', linestyle='--', linewidth=2,
                   label='OOD Threshold (2.5 lengthscales)')
        ax2.set_xlabel('Input', fontsize=11)
        ax2.set_ylabel('OOD Score\n(distance in lengthscales)', fontsize=11)
        ax2.set_title('Out-of-Distribution Scores', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        return fig, (ax1, ax2)

    def plot_calibration_curve(
        self,
        predictions: List[Any],
        y_true: np.ndarray,
        n_bins: int = 10,
        figsize: Tuple[int, int] = (10, 5),
        save_path: Optional[str] = None
    ):
        """
        Plot calibration curve showing reliability of uncertainty estimates.

        Parameters
        ----------
        predictions : List[UQPrediction]
            Predictions from UQ system
        y_true : np.ndarray
            True values
        n_bins : int, default=10
            Number of bins for calibration curve
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        """
        means = np.array([p.mean for p in predictions])
        stds = np.array([p.std for p in predictions])
        lower = np.array([p.lower_95 for p in predictions])
        upper = np.array([p.upper_95 for p in predictions])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Left: Calibration curve
        confidence_levels = np.linspace(0, 1, n_bins+1)[1:]
        empirical_coverage = []

        for conf in confidence_levels:
            z = np.abs(np.percentile(np.random.randn(10000), [(1-conf)/2*100, (1+conf)/2*100]))
            lower_bound = means - z[1] * stds
            upper_bound = means + z[1] * stds
            coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
            empirical_coverage.append(coverage)

        ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
        ax1.plot(confidence_levels, empirical_coverage, 'o-', linewidth=2,
                color=self.colors['mean'], markersize=8, label='Observed')
        ax1.set_xlabel('Expected Coverage', fontsize=11)
        ax1.set_ylabel('Empirical Coverage', fontsize=11)
        ax1.set_title('Calibration Curve', fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])

        # Right: Prediction intervals
        within_bounds = (y_true >= lower) & (y_true <= upper)
        picp = np.mean(within_bounds)

        # Sort by prediction for nice visualization
        sort_idx = np.argsort(means)
        x_plot = np.arange(len(predictions))

        ax2.fill_between(x_plot, lower[sort_idx], upper[sort_idx],
                        alpha=0.3, color=self.colors['uncertainty'],
                        label='95% Prediction Interval')
        ax2.scatter(x_plot[within_bounds[sort_idx]], y_true[sort_idx][within_bounds[sort_idx]],
                   color=self.colors['data'], s=20, alpha=0.6, label='Within Interval')
        ax2.scatter(x_plot[~within_bounds[sort_idx]], y_true[sort_idx][~within_bounds[sort_idx]],
                   color=self.colors['ood'], s=20, alpha=0.8, label='Outside Interval')
        ax2.plot(x_plot, means[sort_idx], color=self.colors['mean'], linewidth=1,
                label='Mean Prediction')

        ax2.set_xlabel('Prediction Index (sorted)', fontsize=11)
        ax2.set_ylabel('PM2.5 (μg/m³)', fontsize=11)
        ax2.set_title(f'Prediction Intervals (PICP = {picp:.3f})', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        return fig, (ax1, ax2)

    def plot_complete_summary(
        self,
        X_test: np.ndarray,
        predictions: List[Any],
        y_test: Optional[np.ndarray] = None,
        X_train: Optional[np.ndarray] = None,
        x_dim: int = 0,
        figsize: Tuple[int, int] = (16, 12),
        save_path: Optional[str] = None
    ):
        """
        Create comprehensive summary figure with all plots.

        Perfect for dissertation or publication.

        Parameters
        ----------
        X_test : np.ndarray
            Test inputs
        predictions : List[UQPrediction]
            Predictions from UQ system
        y_test : np.ndarray, optional
            True values
        X_train : np.ndarray, optional
            Training inputs
        x_dim : int, default=0
            Which dimension to plot
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 2, hspace=0.3, wspace=0.3)

        # Extract data
        x = X_test[:, x_dim]
        sort_idx = np.argsort(x)
        x = x[sort_idx]

        means = np.array([p.mean for p in predictions])[sort_idx]
        stds = np.array([p.std for p in predictions])[sort_idx]
        lower = np.array([p.lower_95 for p in predictions])[sort_idx]
        upper = np.array([p.upper_95 for p in predictions])[sort_idx]
        epistemic_std = np.array([p.epistemic_std for p in predictions])[sort_idx]
        aleatoric_std = np.array([p.aleatoric_std for p in predictions])[sort_idx]
        epistemic_frac = np.array([p.epistemic_fraction for p in predictions])[sort_idx]
        ood_flags = np.array([p.spatial_ood for p in predictions])[sort_idx]
        ood_scores = np.array([p.ood_score for p in predictions])[sort_idx]

        # Plot 1: Main prediction with uncertainty
        ax1 = fig.add_subplot(gs[0, :])
        ax1.fill_between(x, lower, upper, alpha=0.3, color=self.colors['uncertainty'],
                        label='95% CI')
        ax1.fill_between(x, means - stds, means + stds, alpha=0.5,
                        color=self.colors['uncertainty'], label='±1σ')
        ax1.plot(x, means, color=self.colors['mean'], linewidth=2, label='Mean')
        if y_test is not None:
            y_sorted = y_test[sort_idx]
            ax1.scatter(x, y_sorted, color=self.colors['data'], s=20, alpha=0.6,
                       label='True', zorder=5)
        ax1.set_ylabel('PM2.5 (μg/m³)', fontsize=10)
        ax1.set_title('GP Predictions with Uncertainty Bands', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Uncertainty decomposition
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(x, epistemic_std, color=self.colors['epistemic'], linewidth=2,
                label='Epistemic')
        ax2.plot(x, aleatoric_std, color=self.colors['aleatoric'], linewidth=2,
                label='Aleatoric')
        ax2.plot(x, stds, color='gray', linewidth=1, linestyle='--', label='Total')
        ax2.set_ylabel('Uncertainty (μg/m³)', fontsize=10)
        ax2.set_title('Uncertainty Decomposition', fontsize=11, fontweight='bold')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Epistemic fraction
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.fill_between(x, 0, epistemic_frac, alpha=0.5, color=self.colors['epistemic'])
        ax3.axhline(y=0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax3.set_ylabel('Epistemic Fraction', fontsize=10)
        ax3.set_title('Reducible vs Irreducible Uncertainty', fontsize=11, fontweight='bold')
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3)

        # Plot 4: OOD detection
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.fill_between(x, 0, ood_scores, alpha=0.5, color=self.colors['ood'])
        ax4.axhline(y=2.5, color='red', linestyle='--', linewidth=2, label='Threshold')
        if X_train is not None:
            x_train_sorted = np.sort(X_train[:, x_dim])
            ax4.scatter(x_train_sorted, np.zeros_like(x_train_sorted),
                       color=self.colors['data'], s=10, alpha=0.5, marker='|',
                       label='Training Data')
        ax4.set_xlabel('Input', fontsize=10)
        ax4.set_ylabel('OOD Score', fontsize=10)
        ax4.set_title('Out-of-Distribution Detection', fontsize=11, fontweight='bold')
        ax4.legend(loc='best', fontsize=8)
        ax4.grid(True, alpha=0.3)

        # Plot 5: Statistics summary
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')

        # Compute statistics
        mean_unc = np.mean(stds)
        mean_epistemic_frac = np.mean(epistemic_frac)
        n_ood = np.sum(ood_flags)
        ood_frac = n_ood / len(predictions)

        if y_test is not None:
            within = np.sum((y_test >= lower[np.argsort(sort_idx)]) &
                           (y_test <= upper[np.argsort(sort_idx)]))
            picp = within / len(y_test)
        else:
            picp = None

        stats_text = f"""
        UNCERTAINTY QUANTIFICATION SUMMARY
        {'='*40}

        Total Predictions: {len(predictions)}

        Uncertainty Statistics:
        • Mean uncertainty: {mean_unc:.2f} μg/m³
        • Epistemic fraction: {mean_epistemic_frac:.1%}
        • Aleatoric fraction: {1-mean_epistemic_frac:.1%}

        Out-of-Distribution:
        • OOD points: {n_ood} ({ood_frac:.1%})
        • Reliable predictions: {len(predictions)-n_ood}
        """

        if picp is not None:
            stats_text += f"\n        Calibration:\n        • PICP (95%): {picp:.3f}\n        • Target: 0.950"

        ax5.text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
                verticalalignment='center')

        plt.suptitle('Complete GP Uncertainty Quantification Summary',
                    fontsize=14, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        return fig


# Convenience functions
def quick_plot(X_test, predictions, y_test=None, save_path=None):
    """Quick 1D plot with uncertainty bands."""
    viz = GPUncertaintyVisualizer()
    return viz.plot_1d_with_uncertainty(X_test, predictions, y_test, save_path=save_path)


def quick_spatial_plot(X_test, predictions, metric='total', save_path=None):
    """Quick spatial uncertainty map."""
    viz = GPUncertaintyVisualizer()
    return viz.plot_spatial_uncertainty_map(X_test, predictions, metric=metric, save_path=save_path)


def quick_summary(X_test, predictions, y_test=None, X_train=None, save_path=None):
    """Quick comprehensive summary plot."""
    viz = GPUncertaintyVisualizer()
    return viz.plot_complete_summary(X_test, predictions, y_test, X_train, save_path=save_path)
