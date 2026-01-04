"""Multi-replica plotter with error bars."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from src.analysis.base import BasePlotter, OKABE_ITO, STABILITY_COLORS, REPLICA_COLORS
from src.analysis.data_structures import (
    AggregatedMetrics,
    StabilityGroupStats,
    MetricStats,
)

# Set numpy seed for reproducible jitter
np.random.seed(42)


class MultiReplicaPlotter(BasePlotter):
    """Creates visualizations for multi-replica analysis with error bars."""
    
    # Publication-quality figure settings
    FIGURE_SIZE = (14, 10)
    DPI = 300
    
    # Metric colors mapped to Okabe-Ito
    METRIC_COLORS = {
        'recall': OKABE_ITO['blue'],
        'precision': OKABE_ITO['orange'],
        'f1': OKABE_ITO['vermilion'],
        'mcc': OKABE_ITO['sky_blue'],
        'mean_pairwise_f1': OKABE_ITO['bluish_green'],
        'baseline_f1': '#7A9E7E',
    }

    def generate_plots(
        self,
        aggregated_metrics: AggregatedMetrics,
    ) -> List[Path]:
        """
        Generate all multi-replica plots.

        Args:
            aggregated_metrics: Aggregated metrics across replicas

        Returns:
            List of generated plot paths
        """
        generated_plots = []
        
        self.logger.info("Generating multi-replica plots")

        # Overall metrics comparison
        path = self._plot_overall_metrics(aggregated_metrics)
        if path:
            generated_plots.append(path)

        # Stability metrics (training frequency)
        path = self._plot_stability_metrics(
            aggregated_metrics,
            frequency_type="training",
            title="Performance by Interaction Stability (Training Frequency)",
            filename="multi_replica_stability_training",
        )
        if path:
            generated_plots.append(path)

        # Stability metrics (test frequency)
        path = self._plot_stability_metrics(
            aggregated_metrics,
            frequency_type="test",
            title="Performance by Interaction Stability (Test Frequency)",
            filename="multi_replica_stability_test",
        )
        if path:
            generated_plots.append(path)
            
        self.logger.info(f"Generated {len(generated_plots)} multi-replica plots")

        return generated_plots

    def _format_std(self, value: float) -> str:
        """Format standard deviation to up to 2 significant digits."""
        if value == 0:
            return "0.00"
        s = f"{value:.2g}"
        if 'e' in s:
            if abs(value) < 0.001:
                return "0.00"
            return f"{value:.3f}"
        return s

    def _plot_overall_metrics(
        self,
        aggregated: AggregatedMetrics,
    ) -> Optional[Path]:
        """
        Plot overall metrics comparison with error bars and data points.

        Shows Recall, Precision, F1, MCC, Mean Pairwise F1 with mean ± std.
        """
        fig, ax = plt.subplots(figsize=self.FIGURE_SIZE, dpi=self.DPI)

        metrics = ["Recall", "Precision", "F1", "MCC"]
        stats = [
            aggregated.overall_stats.recall,
            aggregated.overall_stats.precision,
            aggregated.overall_stats.f1,
            aggregated.overall_stats.mcc,
        ]
        
        metric_colors = [
            self.METRIC_COLORS['recall'],
            self.METRIC_COLORS['precision'],
            self.METRIC_COLORS['f1'],
            self.METRIC_COLORS['mcc'],
        ]

        x = np.arange(len(metrics))
        means = [s.mean for s in stats]
        stds = [s.std for s in stats]

        bars = ax.bar(
            x,
            means,
            yerr=stds,
            capsize=5,
            color=metric_colors,
            alpha=0.85,
            edgecolor="white",
            linewidth=1,
            zorder=3,
            error_kw={"linewidth": 2, "capthick": 2, "ecolor": "black"},
        )

        # Add individual data points as scatter overlay
        for i, stat in enumerate(stats):
            if hasattr(stat, 'values') and stat.values and len(stat.values) > 1:
                jitter = np.random.normal(0, 0.05, len(stat.values))
                x_jittered = np.full(len(stat.values), x[i]) + jitter
                ax.scatter(
                    x_jittered, 
                    stat.values, 
                    color='white',
                    s=40, 
                    alpha=0.9, 
                    edgecolors=metric_colors[i], 
                    linewidth=1.5, 
                    zorder=4
                )

        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + std + 0.02,
                f"{mean:.2f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=12,
            )

        ax.set_ylabel("Score", fontsize=26, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=20)
        ax.set_ylim(0, 1.1)

        # Grid - horizontal only
        ax.grid(visible=True, axis="y", alpha=0.2)
        ax.grid(visible=False, axis="x")

        # Add replica count annotation
        stats_text = f"N = {aggregated.n_replicas} replicas"
        props = dict(boxstyle='round,pad=0.25', facecolor='white', 
                    alpha=0.85, edgecolor='gray', linewidth=0.5)
        ax.text(0.98, 0.98, stats_text, 
               transform=ax.transAxes, fontsize=18,
               bbox=props,
               verticalalignment='top', horizontalalignment='right')

        plt.tight_layout()

        paths = self._save_figure(fig, "multi_replica_overall_metrics", formats=("png", "svg"))
        plt.close(fig)

        return paths[0] if paths else None

    def _plot_stability_metrics(
        self,
        aggregated_metrics: AggregatedMetrics,
        frequency_type: str = "training",
        title: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Plot stability-stratified metrics with error bars and data points.

        Shows all metrics for each stability group with error bars and scatter overlay.
        """
        # Get the appropriate stability stats
        if frequency_type == "training":
            stability_stats = aggregated_metrics.training_frequency_stats
        else:
            stability_stats = aggregated_metrics.test_frequency_stats
            
        if not stability_stats:
            self.logger.warning(f"No stability metrics found for {frequency_type} frequency")
            return None

        fig, ax = plt.subplots(figsize=self.FIGURE_SIZE, dpi=self.DPI)

        # Required categories in order
        required_categories = ["Rare", "Transient", "Stable"]
        existing_groups = [g for g in required_categories if g in stability_stats]
        
        if not existing_groups:
            # Map RNN-MD category names if needed
            name_map = {
                "Rare (<5%)": "Rare",
                "Moderate (5-50%)": "Transient", 
                "Stable (>50%)": "Stable",
            }
            existing_groups = []
            for rnn_name, rapid_name in name_map.items():
                if rnn_name in stability_stats:
                    existing_groups.append(rnn_name)
            if not existing_groups:
                return None

        # Metrics to plot
        metric_names = ['recall', 'precision', 'f1', 'mcc', 'mean_pairwise_f1']
        metric_labels = ['Recall', 'Precision', 'F1', 'MCC', 'Mean Pairwise F1']

        # Check if any stability group has baseline F1 data
        has_baseline_f1 = any(
            hasattr(stability_stats.get(g), 'baseline_f1') and 
            getattr(stability_stats.get(g), 'baseline_f1', None) is not None
            for g in existing_groups
        )
        
        if has_baseline_f1:
            metric_names.append('baseline_f1')
            metric_labels.append('Baseline F1')

        # Set up bar positions
        n_groups = len(existing_groups)
        n_metrics = len(metric_names)
        bar_width = 0.15
        x_pos = np.arange(n_groups)

        # Plot each metric
        for i, (metric, label) in enumerate(zip(metric_names, metric_labels)):
            means = []
            stds = []
            
            for group_name in existing_groups:
                group_stats = stability_stats.get(group_name)
                if group_stats is None:
                    means.append(0.0)
                    stds.append(0.0)
                    continue
                    
                if metric == 'baseline_f1':
                    baseline_f1_stats = getattr(group_stats, 'baseline_f1', None)
                    if baseline_f1_stats is not None:
                        means.append(baseline_f1_stats.mean)
                        stds.append(baseline_f1_stats.std)
                    else:
                        means.append(0.0)
                        stds.append(0.0)
                else:
                    metric_stats = getattr(group_stats, metric, None)
                    if metric_stats is not None:
                        means.append(metric_stats.mean)
                        stds.append(metric_stats.std)
                    else:
                        means.append(0.0)
                        stds.append(0.0)

            # Create bars with error bars
            x_offset = x_pos + (i - n_metrics / 2) * bar_width
            
            if metric == 'baseline_f1':
                bars = ax.bar(
                    x_offset, means, bar_width, yerr=stds, capsize=3,
                    label=label, color='#7A9E7E',
                    alpha=0.55, edgecolor='white', linewidth=1
                )
                for rect in bars:
                    rect.set_hatch('//')
            else:
                bars = ax.bar(
                    x_offset, means, bar_width, yerr=stds, capsize=3,
                    label=label, color=self.METRIC_COLORS.get(metric, f'C{i}'),
                    alpha=0.8, edgecolor='white', linewidth=0.5
                )

            # Add individual data points as scatter overlay
            for j, group_name in enumerate(existing_groups):
                group_stats = stability_stats.get(group_name)
                if group_stats is None:
                    continue
                    
                if metric == 'baseline_f1':
                    baseline_f1_stats = getattr(group_stats, 'baseline_f1', None)
                    if baseline_f1_stats is not None and hasattr(baseline_f1_stats, 'values'):
                        if baseline_f1_stats.values and len(baseline_f1_stats.values) > 1:
                            jitter = np.random.normal(0, 0.02, len(baseline_f1_stats.values))
                            x_jittered = np.full(len(baseline_f1_stats.values), x_offset[j]) + jitter
                            ax.scatter(
                                x_jittered, baseline_f1_stats.values,
                                color='#7A9E7E', s=30, alpha=0.9,
                                edgecolors='white', linewidth=1, zorder=3
                            )
                else:
                    metric_stats = getattr(group_stats, metric, None)
                    if metric_stats is not None and hasattr(metric_stats, 'values'):
                        if metric_stats.values and len(metric_stats.values) > 1:
                            jitter = np.random.normal(0, 0.02, len(metric_stats.values))
                            x_jittered = np.full(len(metric_stats.values), x_offset[j]) + jitter
                            ax.scatter(
                                x_jittered, metric_stats.values,
                                color=self.METRIC_COLORS.get(metric, f'C{i}'),
                                s=30, alpha=0.9, edgecolors='white', 
                                linewidth=1, zorder=3
                            )

            # Add value labels
            for bar, mean, std in zip(bars, means, stds):
                if mean > 0:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + std + 0.02,
                        f'{mean:.2f}',
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold', alpha=0.9
                    )

        # Styling
        ax.set_xlabel('Stability Groups', fontsize=26, fontweight='bold')
        ax.set_ylabel('Score', fontsize=26, fontweight='bold')

        # X-axis labels with pair counts (± std)
        group_labels_with_counts = []
        for group_name in existing_groups:
            group_stats = stability_stats.get(group_name)
            if group_stats is not None:
                pair_count_mean = group_stats.pair_count.mean
                pair_count_std = group_stats.pair_count.std
                pair_count = int(round(pair_count_mean))
                
                if pair_count_std >= 0.5 or aggregated_metrics.n_replicas > 1:
                    std_rounded = max(1, int(round(pair_count_std))) if pair_count_std > 0 else 0
                    if std_rounded > 0:
                        count_str = f'(N={pair_count}±{std_rounded})'
                    else:
                        count_str = f'(N={pair_count})'
                else:
                    count_str = f'(N={pair_count})'
            else:
                count_str = '(N=0)'

            # Clean up group name for display
            clean_name = group_name.replace('Moderate', 'Transient')
            clean_name = clean_name.replace(' (<5%)', '').replace(' (5-50%)', '').replace(' (>50%)', '')
            group_labels_with_counts.append(f'{clean_name}\n{count_str}')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(group_labels_with_counts, fontsize=13)
        ax.set_ylim(0, 1.1)

        # Legend at bottom
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(
                handles, labels, loc='lower center',
                bbox_to_anchor=(0.5, -0.02),
                ncol=min(len(handles), 4),
                frameon=False, fontsize=20
            )

        # Grid
        ax.grid(True, alpha=0.3, axis='y')

        # Add replica count annotation
        ax.text(
            0.02, 0.98, f'N = {aggregated_metrics.n_replicas} replicas',
            transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
            verticalalignment='top'
        )

        plt.tight_layout()

        if filename is None:
            filename = f"multi_replica_stability_{frequency_type}"
            
        paths = self._save_figure(fig, filename, formats=("png", "svg"))
        plt.close(fig)

        self.logger.info(f"Generated stability metrics plot: {filename}")
        return paths[0] if paths else None
