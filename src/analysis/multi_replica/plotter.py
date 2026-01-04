"""Multi-replica plotter for creating visualizations with error bars."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import os
import math

# Set numpy seed for reproducible jitter
np.random.seed(42)

from src.core.utils import get_logger
from .data_structures import AggregatedMetrics, StabilityGroupStats, MetricStats


class MultiReplicaPlotter:
    """Creates visualizations for multi-replica analysis with error bars."""
    
    def __init__(self, output_directory: str):
        """Initialize multi-replica plotter.
        
        Args:
            output_directory: Directory to save plots
        """
        self.output_directory = output_directory
        self.logger = get_logger(__name__)
        
        # Create output directory
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        
        # Plot styling - Publication Ready (Okabe-Ito & Guide)
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
            'font.size': 20,
            'axes.titlesize': 26,
            'axes.labelsize': 26,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'legend.fontsize': 22,
            'figure.titlesize': 32,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.linewidth': 1.5,
            'axes.edgecolor': '#2C3E50',
            'axes.grid': True,
            'grid.alpha': 0.2,
            'grid.linewidth': 0.6,
            'grid.color': '#BDC3C7',
            'axes.axisbelow': True,
            'legend.frameon': False,
            'legend.fancybox': True,
            'legend.borderpad': 0.5,
        })
        
        self.figure_size = (14, 10)
        self.dpi = 300
        
        # Okabe-Ito Colorblind-Safe Palette
        self.okabe_ito = {
            'vermilion': '#D55E00',
            'blue': '#0072B2',
            'bluish_green': '#009E73',
            'orange': '#E69F00',
            'sky_blue': '#56B4E9',
            'reddish_purple': '#CC79A7',
            'yellow': '#F0E442',
            'black': '#000000'
        }

        # Stability Group Colors (Okabe-Ito)
        self.stability_palette = {
            'Rare': self.okabe_ito['vermilion'],      # #D55E00
            'Transient': self.okabe_ito['orange'],    # #E69F00
            'Stable': self.okabe_ito['bluish_green']  # #009E73
        }
        
        # Publication utility colors
        self.publication_colors = {
            'primary': '#2C3E50',
            'secondary': '#34495E',
            'background': '#ECF0F1',
            'grid': '#BDC3C7',
            'highlight': '#3498DB'
        }
        
        # Replica colors (Okabe-Ito sequence)
        self.replica_colors = [
            self.okabe_ito['blue'], 
            self.okabe_ito['vermilion'], 
            self.okabe_ito['bluish_green'], 
            self.okabe_ito['orange'], 
            self.okabe_ito['sky_blue'], 
            self.okabe_ito['reddish_purple'], 
            self.okabe_ito['yellow'], 
            self.okabe_ito['black']
        ]
        
        # Metric Colors mapped to Okabe-Ito
        self.colors = {
            'recall': self.okabe_ito['blue'],            # #0072B2
            'precision': self.okabe_ito['orange'],       # #E69F00
            'f1': self.okabe_ito['vermilion'],           # #D55E00
            'mcc': self.okabe_ito['sky_blue'],           # #56B4E9
            'mean_pairwise_f1': self.okabe_ito['bluish_green'], # #009E73
            'baseline_f1': '#7A9E7E',
            'baseline_mean_pairwise_f1': '#7A9E7E'
        }
    
    def _format_std(self, value: float) -> str:
        """Format standard deviation to up to 2 significant digits."""
        if value == 0:
            return "0.00"
        
        # Use %.2g for 2 significant digits
        s = f"{value:.2g}"
        
        # Ensure it doesn't look like scientific notation if it's just a small number like 0.001
        if 'e' in s:
            # If scientific notation, check if we can represent it reasonably in fixed point
            # For plot labels, we probably don't want 1e-5. 
            # If it's that small, it's effectively 0.00 for the scale of 0-1 metrics.
            if abs(value) < 0.001:
                return "0.00"
            return f"{value:.3f}" # Fallback
            
        return s

    def generate_all_plots(self, aggregated_metrics: AggregatedMetrics) -> List[str]:
        """Generate all multi-replica plots.
        
        Args:
            aggregated_metrics: Aggregated metrics across replicas
            
        Returns:
            List of generated plot file paths
        """
        self.logger.info("Generating multi-replica plots")
        
        generated_plots = []
        
        try:
            # Overall metrics comparison plot
            overall_plot = self.plot_overall_metrics(aggregated_metrics)
            if overall_plot:
                generated_plots.append(overall_plot)
            
            # Stability group metrics (training frequency)
            training_plot = self.plot_stability_metrics(
                aggregated_metrics, 
                frequency_type="training",
                title="Performance by Interaction Stability (Training Frequency)",
                filename="multi_replica_stability_training.png"
            )
            if training_plot:
                generated_plots.append(training_plot)
            
            # Stability group metrics (test frequency)
            test_plot = self.plot_stability_metrics(
                aggregated_metrics,
                frequency_type="test", 
                title="Performance by Interaction Stability (Test Frequency)",
                filename="multi_replica_stability_test.png"
            )
            if test_plot:
                generated_plots.append(test_plot)
            
            self.logger.info(f"Generated {len(generated_plots)} multi-replica plots")
            
        except Exception as e:
            self.logger.error(f"Failed to generate multi-replica plots: {e}")
        
        return generated_plots
    
    def plot_overall_metrics(self, aggregated_metrics: AggregatedMetrics) -> Optional[str]:
        """Plot overall model performance across replicas.
        
        Args:
            aggregated_metrics: Aggregated metrics data
            
        Returns:
            Path to generated plot file or None
        """
        try:
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            overall = aggregated_metrics.overall_stats
            
            # Metrics to plot (include baseline if available)
            metrics = ['model_recall', 'model_precision', 'model_f1', 'model_mcc', 'model_mean_pairwise_f1']
            metric_labels = ['Recall', 'Precision', 'F1', 'MCC', 'Mean Pairwise F1']

            # Determine if baseline metrics are available
            baseline_available = any([
                overall.baseline_recall is not None,
                overall.baseline_precision is not None,
                overall.baseline_f1 is not None,
                overall.baseline_mcc is not None,
                overall.baseline_mean_pairwise_f1 is not None
            ])
            
            # Get values and error bars for model (and baseline if available)
            model_means = []
            model_stds = []
            model_colors = []

            for metric in metrics:
                metric_stats: MetricStats = getattr(overall, metric)
                model_means.append(metric_stats.mean)
                model_stds.append(metric_stats.std)
                model_colors.append(self.colors.get(metric.replace('model_', ''), '#333333'))

            # Baseline series (optional)
            baseline_means = []
            baseline_stds = []
            baseline_colors = []
            if baseline_available:
                for metric in metrics:
                    base_attr = metric.replace('model_', 'baseline_')
                    base_stats: Optional[MetricStats] = getattr(overall, base_attr, None)
                    if base_stats is not None:
                        baseline_means.append(base_stats.mean)
                        baseline_stds.append(base_stats.std)
                    else:
                        baseline_means.append(0.0)
                        baseline_stds.append(0.0)
                    baseline_colors.append(self.colors.get(base_attr.replace('baseline_', 'baseline_'), '#7A9E7E'))

            # Create grouped bar plot with error bars
            x_pos = np.arange(len(metric_labels))
            bar_width = 0.35 if baseline_available else 0.7
            bars_model = ax.bar(
                x_pos - (bar_width/2 if baseline_available else 0),
                model_means,
                bar_width,
                yerr=model_stds,
                capsize=5,
                color=model_colors,
                alpha=0.85,
                edgecolor='white',
                linewidth=1,
                label='Model'
            )

            bars_baseline = None
            if baseline_available:
                bars_baseline = ax.bar(
                    x_pos + bar_width/2,
                    baseline_means,
                    bar_width,
                    yerr=baseline_stds,
                    capsize=5,
                    color='#7A9E7E',
                    alpha=0.55,
                    edgecolor='white',
                    linewidth=1,
                    label='Baseline'
                )
                # Apply hatch pattern to baseline bars to avoid relying on colors
                for rect in bars_baseline:
                    rect.set_hatch('//')
            
            # Add individual datapoints using values from MetricStats
            for i, metric in enumerate(metrics):
                metric_stats: MetricStats = getattr(overall, metric)
                if metric_stats.values and len(metric_stats.values) > 1:
                    jitter = np.random.normal(0, 0.05, len(metric_stats.values))
                    x_center = x_pos[i] - (bar_width/2 if baseline_available else 0)
                    x_jittered = np.full(len(metric_stats.values), x_center) + jitter
                    ax.scatter(x_jittered, metric_stats.values, color='white', 
                               s=40, alpha=0.9, edgecolors=model_colors[i], linewidth=1.5, zorder=3)
                if baseline_available:
                    base_attr = metric.replace('model_', 'baseline_')
                    base_stats: Optional[MetricStats] = getattr(overall, base_attr, None)
                    if base_stats is not None and base_stats.values and len(base_stats.values) > 1:
                        jitter = np.random.normal(0, 0.05, len(base_stats.values))
                        x_center = x_pos[i] + bar_width/2
                        x_jittered = np.full(len(base_stats.values), x_center) + jitter
                        ax.scatter(
                            x_jittered,
                            base_stats.values,
                            color='white',
                            s=40,
                            alpha=0.9,
                            edgecolors='#7A9E7E',
                            linewidth=1.5,
                            zorder=3,
                            marker='s'  # square markers for baseline for pattern-based distinction
                        )
            
            # Add value labels on bars
            for i, (bar, mean, std) in enumerate(zip(bars_model, model_means, model_stds)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                        f'{mean:.2f}', ha='center', va='bottom', 
                        fontweight='bold', fontsize=12)
            if baseline_available and bars_baseline is not None:
                for i, (bar, mean, std) in enumerate(zip(bars_baseline, baseline_means, baseline_stds)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                            f'{mean:.2f}', ha='center', va='bottom', 
                            fontweight='bold', fontsize=12, color='#2C3E50')
            
            # Styling
            ax.set_ylabel('Score', fontsize=26, fontweight='bold')
            # Remove redundant X label if metrics are clear
            # ax.set_xlabel('Metrics', fontweight='bold') 
            
            # Removed title as requested
            # title_suffix = f'Across {overall.n_replicas} Replicas'
            # ax.set_title(f'Overall Performance {title_suffix}', fontweight='bold', pad=20)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(metric_labels)
            ax.set_ylim(0, 1.1) # Increased ylim to accommodate taller labels
            
            # Grid - horizontal only
            ax.grid(visible=True, axis='y', alpha=0.2)
            ax.grid(visible=False, axis='x')
            
            # Add replica count annotation
            stats_text = f"N = {overall.n_replicas} replicas"
            props = dict(boxstyle='round,pad=0.25', facecolor='white', 
                        alpha=0.85, edgecolor='gray', linewidth=0.5)
            ax.text(0.98, 0.98, stats_text, 
                   transform=ax.transAxes, fontsize=18,
                   bbox=props,
                   verticalalignment='top', horizontalalignment='right')
            
            if baseline_available:
                # Unified Legend
                handles = [
                    Patch(facecolor='gray', edgecolor='none', alpha=0.55, label='Model'),
                    Patch(facecolor='#7A9E7E', edgecolor='white', hatch='//', alpha=0.55, label='Baseline')
                ]
                fig.legend(handles=handles, loc='lower center', 
                          bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)
                plt.subplots_adjust(bottom=0.15)
            
            plt.tight_layout()
            # Add space for bottom legend if present
            if handles:
                plt.subplots_adjust(bottom=0.18)
            if baseline_available:
                 plt.subplots_adjust(bottom=0.15) # Adjust again after tight_layout
            
            # Save plot
            output_path = os.path.join(self.output_directory, "multi_replica_overall_metrics.png")
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            
            # Also save SVG
            svg_path = os.path.join(self.output_directory, "multi_replica_overall_metrics.svg")
            plt.savefig(svg_path, format='svg', bbox_inches='tight')
            
            plt.close()
            
            self.logger.info(f"Generated overall metrics plot: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate overall metrics plot: {e}")
            return None
    
    def plot_stability_metrics(
        self, 
        aggregated_metrics: AggregatedMetrics,
        frequency_type: str = "training",
        title: Optional[str] = None,
        filename: Optional[str] = None
    ) -> Optional[str]:
        """Plot stability group metrics with error bars.
        
        Args:
            aggregated_metrics: Aggregated metrics data
            frequency_type: Either "training" or "test"
            title: Custom plot title
            filename: Custom filename
            
        Returns:
            Path to generated plot file or None
        """
        try:
            stability_stats = aggregated_metrics.get_stability_groups(frequency_type)
            
            if not stability_stats:
                self.logger.warning(f"No stability metrics found for {frequency_type} frequency")
                return None
            
            # Always include the three main stability categories
            required_categories = ['Rare (<5%)', 'Moderate (5-50%)', 'Stable (>50%)']
            
            # Use only the required categories to ensure consistency
            existing_groups = required_categories
            
            # Metrics to plot
            metric_names = ['recall', 'precision', 'f1', 'mcc', 'mean_pairwise_f1']
            metric_labels = ['Recall', 'Precision', 'F1', 'MCC', 'Mean Pairwise F1']
            
            # Check if any stability group has baseline F1 data, and if so, add it to metrics
            has_baseline_f1 = any(
                hasattr(group_stats, 'baseline_f1') and group_stats.baseline_f1 is not None
                for group_stats in stability_stats.values()
            )
            
            if has_baseline_f1:
                metric_names.append('baseline_f1')
                metric_labels.append('Baseline F1')
            
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            # Set up bar positions
            n_groups = len(existing_groups)
            n_metrics = len(metric_names)
            bar_width = 0.15
            x_pos = np.arange(n_groups)
            
            # Plot each metric
            for i, (metric, label) in enumerate(zip(metric_names, metric_labels)):
                means = []
                stds = []
                pair_counts = []
                
                for group_name in existing_groups:
                    if group_name in stability_stats:
                        group_stats = stability_stats[group_name]
                        if metric == 'baseline_f1':
                            # Handle baseline_f1 specially since it might be None
                            baseline_f1_stats = getattr(group_stats, 'baseline_f1', None)
                            if baseline_f1_stats is not None:
                                means.append(baseline_f1_stats.mean)
                                stds.append(baseline_f1_stats.std)
                            else:
                                means.append(0.0)
                                stds.append(0.0)
                        else:
                            metric_stats: MetricStats = getattr(group_stats, metric)
                            means.append(metric_stats.mean)
                            stds.append(metric_stats.std)
                        pair_counts.append(int(group_stats.pair_count.mean))
                    else:
                        # Group doesn't exist, use zero values
                        means.append(0.0)
                        stds.append(0.0)
                        pair_counts.append(0)
                
                # Create bars with error bars
                x_offset = x_pos + (i - n_metrics/2) * bar_width
                # Use baseline styling for baseline_f1 to match overall plot
                if metric == 'baseline_f1':
                    bars = ax.bar(x_offset, means, bar_width, yerr=stds, capsize=3,
                                 label=label, color='#7A9E7E', 
                                 alpha=0.55, edgecolor='white', linewidth=1)
                    # Apply hatch pattern to match overall plot baseline style
                    for rect in bars:
                        rect.set_hatch('//')
                else:
                    bars = ax.bar(x_offset, means, bar_width, yerr=stds, capsize=3,
                             label=label, color=self.colors.get(metric, f'C{i}'), 
                             alpha=0.8, edgecolor='white', linewidth=0.5)
                
                # Add individual datapoints using values from MetricStats
                for j, group_name in enumerate(existing_groups):
                    if group_name in stability_stats:
                        group_stats = stability_stats[group_name]
                        if metric == 'baseline_f1':
                            # Handle baseline_f1 specially since it might be None
                            baseline_f1_stats = getattr(group_stats, 'baseline_f1', None)
                            if baseline_f1_stats is not None and baseline_f1_stats.values and len(baseline_f1_stats.values) > 1:
                                jitter = np.random.normal(0, 0.02, len(baseline_f1_stats.values))
                                x_jittered = np.full(len(baseline_f1_stats.values), x_offset[j]) + jitter
                                ax.scatter(x_jittered, baseline_f1_stats.values, 
                                         color='#7A9E7E', 
                                         s=30, alpha=0.9, edgecolors='white', linewidth=1, zorder=3)
                        else:
                            metric_stats: MetricStats = getattr(group_stats, metric)
                            if metric_stats.values and len(metric_stats.values) > 1:  # Only show points if we have multiple replicas
                                # Add small random jitter to x-position for visibility
                                jitter = np.random.normal(0, 0.02, len(metric_stats.values))
                                x_jittered = np.full(len(metric_stats.values), x_offset[j]) + jitter
                                
                                # Plot individual points
                                ax.scatter(x_jittered, metric_stats.values, 
                                         color=self.colors.get(metric, f'C{i}'), 
                                         s=30, alpha=0.9, edgecolors='white', linewidth=1, zorder=3)
                
                # Add value labels to all bars
                for j, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                    if mean > 0:  # Only show labels for non-zero values
                        height = bar.get_height()
                        # Adjust font size based on bar height and metric type
                        fontsize = 10 if metric in ['recall', 'precision', 'mcc', 'mean_pairwise_f1'] else 11
                        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                               f'{mean:.2f}', ha='center', va='bottom', 
                               fontsize=fontsize, fontweight='bold', alpha=0.9)
            
            # Styling
            ax.set_xlabel('Stability Groups', fontsize=26, fontweight='bold')
            ax.set_ylabel('Score', fontsize=26, fontweight='bold')
            
            # Removed title as requested
            # if title is None:
            #     title = f'Performance by Interaction Stability ({frequency_type.title()} Frequency)'
            # ax.set_title(f'{title}\nAcross {aggregated_metrics.n_replicas} Replicas', 
            #             fontsize=16, fontweight='bold', pad=20)
            
            # X-axis labels with pair counts
            group_labels_with_counts = []
            for group_name in existing_groups:
                if group_name in stability_stats:
                    group_stats = stability_stats[group_name] 
                    pair_count_mean = group_stats.pair_count.mean
                    pair_count_std = group_stats.pair_count.std
                    
                    # Round mean to integer, but be more careful with std
                    pair_count = int(round(pair_count_mean))
                    
                    # Show std if it's meaningful (> 0.5) or if there are multiple replicas
                    if pair_count_std >= 0.5 or aggregated_metrics.n_replicas > 1:
                        # Round std but ensure it shows at least 1 if there's any variation
                        std_rounded = max(1, int(round(pair_count_std))) if pair_count_std > 0 else 0
                        if std_rounded > 0:
                            count_str = f'(N={pair_count}±{std_rounded})'
                        else:
                            count_str = f'(N={pair_count})'
                    else:
                        count_str = f'(N={pair_count})'
                else:
                    # Group doesn't exist, show zero count
                    count_str = '(N=0)'
                
                # Clean up group name
                clean_name = group_name.replace('Moderate', 'Transient')
                if group_name == 'Undefined':
                    clean_name = 'Not in Train' if frequency_type == 'training' else 'Not in Test'
                
                group_labels_with_counts.append(f'{clean_name}\n{count_str}')
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(group_labels_with_counts, fontsize=13)
            ax.set_ylim(0, 1.1)
            
            # Legend (figure-level, bottom)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc='lower center',
                           bbox_to_anchor=(0.5, -0.02),
                           ncol=min(len(handles), 4),
                           frameon=False, fontsize=20)
            
            # Grid
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add replica count annotation
            ax.text(0.02, 0.98, f'N = {aggregated_metrics.n_replicas} replicas', 
                   transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                   verticalalignment='top')
            
            plt.tight_layout()
            
            # Save plot
            if filename is None:
                filename = f"multi_replica_stability_{frequency_type}.png"
            output_path = os.path.join(self.output_directory, filename)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Generated stability metrics plot: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate stability metrics plot: {e}")
            return None
