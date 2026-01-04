#!/usr/bin/env python3
"""Combined multi-replica analysis for multiple complexes and temporal splits.

This script generates combined figures for multi-replica analysis across:
- 2 complexes: 1EAW, 1JPS
- 3 temporal splits: 25-375-375, 50-25-25, 80-10-10
- 4 replicas per combination

Input structure:
    results_all/
    ├── 1EAW/
    │   ├── result_25-375-375/   (contains 4 replica analysis dirs)
    │   ├── result_50-25-25/
    │   └── result_80-10-10/
    └── 1JPS/
        ├── result_25-375-375/
        ├── result_50-25-25/
        └── result_80-10-10/

Output:
- combined_overall_metrics.png/svg: 2×3 grid of overall metrics plots
- combined_stability_training.png/svg: 2×3 grid of stability (training freq) plots
- combined_stability_test.png/svg: 2×3 grid of stability (test freq) plots
"""

import sys
from pathlib import Path

import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any

from src.core.utils import setup_logging, get_logger
from src.analysis.multi_replica.analyzer import MultiReplicaAnalyzer
from src.analysis.multi_replica.data_structures import AggregatedMetrics, MetricStats

# Set numpy seed for reproducible jitter
np.random.seed(42)


class CombinedMultiReplicaPlotter:
    """Creates combined visualizations for multi-replica analysis across complexes and splits."""
    
    # Configuration
    COMPLEXES = ['1EAW', '1JPS']
    SPLITS = ['80-10-10', '50-25-25', '25-375-375']
    SPLIT_LABELS = ['80/10/10', '50/25/25', '25/37.5/37.5']
    
    def __init__(self, output_directory: str):
        """Initialize combined multi-replica plotter.
        
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
            'font.size': 18,
            'axes.titlesize': 24,
            'axes.labelsize': 22,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            'legend.fontsize': 20,
            'figure.titlesize': 28,
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
        
        # Metric Colors mapped to Okabe-Ito
        self.colors = {
            'recall': self.okabe_ito['blue'],
            'precision': self.okabe_ito['orange'],
            'f1': self.okabe_ito['vermilion'],
            'mcc': self.okabe_ito['sky_blue'],
            'mean_pairwise_f1': self.okabe_ito['bluish_green'],
            'baseline_f1': '#7A9E7E',
            'baseline_mean_pairwise_f1': '#7A9E7E'
        }
        
        # Publication utility colors
        self.publication_colors = {
            'primary': '#2C3E50',
            'secondary': '#34495E',
            'background': '#ECF0F1',
            'grid': '#BDC3C7',
            'highlight': '#3498DB'
        }
    
    def discover_and_load_all_metrics(
        self, 
        base_path: str, 
        pattern: str = "*/analysis"
    ) -> Dict[str, Dict[str, AggregatedMetrics]]:
        """Discover and load metrics for all complexes and splits.
        
        Args:
            base_path: Base directory containing complex folders
            pattern: Pattern to find replica analysis directories within each split
            
        Returns:
            Nested dict: {complex: {split: AggregatedMetrics}}
        """
        all_metrics = {}
        
        base_dir = Path(base_path)
        if not base_dir.exists():
            self.logger.error(f"Base path does not exist: {base_path}")
            return all_metrics
        
        for complex_name in self.COMPLEXES:
            complex_dir = base_dir / complex_name
            if not complex_dir.exists():
                self.logger.warning(f"Complex directory not found: {complex_dir}")
                continue
            
            all_metrics[complex_name] = {}
            
            for split in self.SPLITS:
                split_dir = complex_dir / f"result_{split}"
                if not split_dir.exists():
                    self.logger.warning(f"Split directory not found: {split_dir}")
                    continue
                
                # Load replicas for this complex/split combination
                analyzer = MultiReplicaAnalyzer()
                replica_paths = analyzer.discover_replica_directories(str(split_dir), pattern)
                
                if not replica_paths:
                    self.logger.warning(f"No replica directories found in {split_dir}")
                    continue
                
                success = analyzer.load_replica_metrics(replica_paths)
                if not success:
                    self.logger.warning(f"Failed to load metrics for {complex_name}/{split}")
                    continue
                
                aggregated = analyzer.aggregate_metrics()
                if aggregated:
                    all_metrics[complex_name][split] = aggregated
                    self.logger.info(
                        f"Loaded {aggregated.n_replicas} replicas for {complex_name}/{split}"
                    )
        
        return all_metrics
    
    def generate_all_combined_plots(
        self, 
        all_metrics: Dict[str, Dict[str, AggregatedMetrics]]
    ) -> List[str]:
        """Generate all combined plots.
        
        Args:
            all_metrics: Nested dict of aggregated metrics
            
        Returns:
            List of generated plot file paths
        """
        generated_plots = []
        
        # Combined overall metrics plot (2×3 grid)
        overall_plot = self.plot_combined_overall_metrics(all_metrics)
        if overall_plot:
            generated_plots.append(overall_plot)
        
        # Combined stability plots (training frequency)
        training_plot = self.plot_combined_stability_metrics(
            all_metrics, frequency_type="training"
        )
        if training_plot:
            generated_plots.append(training_plot)
        
        # Combined stability plots (test frequency)
        test_plot = self.plot_combined_stability_metrics(
            all_metrics, frequency_type="test"
        )
        if test_plot:
            generated_plots.append(test_plot)
        
        return generated_plots
    
    def plot_combined_overall_metrics(
        self, 
        all_metrics: Dict[str, Dict[str, AggregatedMetrics]]
    ) -> Optional[str]:
        """Plot combined overall metrics in a 3×2 grid.
        
        Rows: Splits (25/37.5/37.5, 50/25/25, 80/10/10)
        Columns: Complexes (1EAW, 1JPS)
        
        Args:
            all_metrics: Nested dict of aggregated metrics
            
        Returns:
            Path to generated plot file or None
        """
        try:
            fig, axes = plt.subplots(3, 2, figsize=(16, 20), dpi=self.dpi)
            
            metrics = ['model_recall', 'model_precision', 'model_f1', 'model_mcc', 'model_mean_pairwise_f1']
            metric_labels = ['Recall', 'Precision', 'F1', 'MCC', 'Mean\nPairwise F1']
            
            # Track if any subplot has baseline for unified legend
            any_baseline = False
            
            for row_idx, (split, split_label) in enumerate(zip(self.SPLITS, self.SPLIT_LABELS)):
                for col_idx, complex_name in enumerate(self.COMPLEXES):
                    ax = axes[row_idx, col_idx]
                    
                    # Check if data exists for this combination
                    if complex_name not in all_metrics or split not in all_metrics[complex_name]:
                        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                               fontsize=18, color='gray', transform=ax.transAxes)
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                        continue
                    
                    aggregated = all_metrics[complex_name][split]
                    overall = aggregated.overall_stats
                    
                    # Determine if baseline is available
                    baseline_available = any([
                        overall.baseline_recall is not None,
                        overall.baseline_precision is not None,
                        overall.baseline_f1 is not None,
                        overall.baseline_mcc is not None,
                        overall.baseline_mean_pairwise_f1 is not None
                    ])
                    if baseline_available:
                        any_baseline = True
                    
                    # Get model values
                    model_means = []
                    model_stds = []
                    model_colors = []
                    for metric in metrics:
                        metric_stats: MetricStats = getattr(overall, metric)
                        model_means.append(metric_stats.mean)
                        model_stds.append(metric_stats.std)
                        model_colors.append(self.colors.get(metric.replace('model_', ''), '#333333'))
                    
                    # Get baseline values if available
                    baseline_means = []
                    baseline_stds = []
                    if baseline_available:
                        for metric in metrics:
                            base_attr = metric.replace('model_', 'baseline_')
                            base_stats = getattr(overall, base_attr, None)
                            if base_stats is not None:
                                baseline_means.append(base_stats.mean)
                                baseline_stds.append(base_stats.std)
                            else:
                                baseline_means.append(0.0)
                                baseline_stds.append(0.0)
                    
                    # Create bar plot
                    x_pos = np.arange(len(metric_labels))
                    bar_width = 0.35 if baseline_available else 0.7
                    
                    bars_model = ax.bar(
                        x_pos - (bar_width/2 if baseline_available else 0),
                        model_means, bar_width, yerr=model_stds, capsize=3,
                        color=model_colors, alpha=0.85, edgecolor='white', linewidth=1
                    )
                    
                    if baseline_available:
                        bars_baseline = ax.bar(
                            x_pos + bar_width/2, baseline_means, bar_width,
                            yerr=baseline_stds, capsize=3, color='#7A9E7E',
                            alpha=0.55, edgecolor='white', linewidth=1
                        )
                        for rect in bars_baseline:
                            rect.set_hatch('//')
                    
                    # Add scatter points for individual replicas
                    for i, metric in enumerate(metrics):
                        metric_stats: MetricStats = getattr(overall, metric)
                        if metric_stats.values and len(metric_stats.values) > 1:
                            jitter = np.random.normal(0, 0.04, len(metric_stats.values))
                            x_center = x_pos[i] - (bar_width/2 if baseline_available else 0)
                            x_jittered = np.full(len(metric_stats.values), x_center) + jitter
                            ax.scatter(x_jittered, metric_stats.values, color='white',
                                      s=25, alpha=0.9, edgecolors=model_colors[i], 
                                      linewidth=1, zorder=3)
                        
                        if baseline_available:
                            base_attr = metric.replace('model_', 'baseline_')
                            base_stats = getattr(overall, base_attr, None)
                            if base_stats is not None and base_stats.values and len(base_stats.values) > 1:
                                jitter = np.random.normal(0, 0.04, len(base_stats.values))
                                x_center = x_pos[i] + bar_width/2
                                x_jittered = np.full(len(base_stats.values), x_center) + jitter
                                ax.scatter(x_jittered, base_stats.values, color='white',
                                          s=25, alpha=0.9, edgecolors='#7A9E7E',
                                          linewidth=1, zorder=3, marker='s')
                    
                    # Styling
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(metric_labels, fontsize=16)
                    ax.set_ylim(0, 1.15)
                    ax.grid(visible=True, axis='y', alpha=0.2)
                    ax.grid(visible=False, axis='x')
                    
                    # Column headers (top row only) - Complex names
                    if row_idx == 0:
                        ax.set_title(complex_name, fontsize=22, fontweight='bold', pad=10)
                    
                    # Y-axis label (left column only)
                    if col_idx == 0:
                        ax.set_ylabel('Score', fontsize=20)
                    
                    # Row labels (right side) - Split names
                    if col_idx == 1:
                        ax.annotate(f'{split_label} Split', xy=(1.05, 0.5), xycoords='axes fraction',
                                   fontsize=22, fontweight='bold', ha='left', va='center', rotation=-90)
            
            # Panel labels (3x2 grid: A-F)
            for idx, (row_idx, col_idx) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]):
                label = chr(ord('A') + idx)
                ax = axes[row_idx, col_idx]
                ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=26,
                       fontweight='bold', va='top', ha='left')
            
            # Unified legend at bottom
            if any_baseline:
                handles = [
                    Patch(facecolor='gray', edgecolor='none', alpha=0.55, label='Model'),
                    Patch(facecolor='#7A9E7E', edgecolor='white', hatch='//', alpha=0.55, label='Baseline')
                ]
                fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.02),
                          ncol=2, frameon=False, fontsize=20)
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.08, right=0.88, hspace=0.25, wspace=0.15)
            
            # Save
            output_path = os.path.join(self.output_directory, "combined_overall_metrics.png")
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            svg_path = os.path.join(self.output_directory, "combined_overall_metrics.svg")
            plt.savefig(svg_path, format='svg', bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Generated combined overall metrics plot: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate combined overall metrics plot: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def plot_combined_stability_metrics(
        self, 
        all_metrics: Dict[str, Dict[str, AggregatedMetrics]],
        frequency_type: str = "training"
    ) -> Optional[str]:
        """Plot combined stability metrics in a 3×2 grid.
        
        Rows: Splits (25/37.5/37.5, 50/25/25, 80/10/10)
        Columns: Complexes (1EAW, 1JPS)
        
        Args:
            all_metrics: Nested dict of aggregated metrics
            frequency_type: Either "training" or "test"
            
        Returns:
            Path to generated plot file or None
        """
        try:
            fig, axes = plt.subplots(3, 2, figsize=(16, 20), dpi=self.dpi)
            
            # Metrics to plot
            metric_names = ['recall', 'precision', 'f1', 'mcc', 'mean_pairwise_f1']
            metric_labels = ['Recall', 'Precision', 'F1', 'MCC', 'Mean Pairwise F1']
            
            # Stability groups
            required_categories = ['Rare (<5%)', 'Moderate (5-50%)', 'Stable (>50%)']
            group_display_names = ['Rare\n(<5%)', 'Transient\n(5-50%)', 'Stable\n(>50%)']
            
            # Check if any subplot has baseline
            any_baseline = False
            for complex_name in self.COMPLEXES:
                for split in self.SPLITS:
                    if complex_name in all_metrics and split in all_metrics[complex_name]:
                        stability_stats = all_metrics[complex_name][split].get_stability_groups(frequency_type)
                        if stability_stats:
                            for group_stats in stability_stats.values():
                                if hasattr(group_stats, 'baseline_f1') and group_stats.baseline_f1 is not None:
                                    any_baseline = True
                                    break
            
            # Add baseline to metrics if present
            plot_metric_names = metric_names.copy()
            plot_metric_labels = metric_labels.copy()
            if any_baseline:
                plot_metric_names.append('baseline_f1')
                plot_metric_labels.append('Baseline F1')
            
            # Create legend handles (will be used for unified legend)
            legend_handles = []
            legend_labels = []
            
            for row_idx, (split, split_label) in enumerate(zip(self.SPLITS, self.SPLIT_LABELS)):
                for col_idx, complex_name in enumerate(self.COMPLEXES):
                    ax = axes[row_idx, col_idx]
                    
                    # Check if data exists
                    if complex_name not in all_metrics or split not in all_metrics[complex_name]:
                        ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                               fontsize=18, color='gray', transform=ax.transAxes)
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                        continue
                    
                    aggregated = all_metrics[complex_name][split]
                    stability_stats = aggregated.get_stability_groups(frequency_type)
                    
                    if not stability_stats:
                        ax.text(0.5, 0.5, 'No Stability Data', ha='center', va='center',
                               fontsize=18, color='gray', transform=ax.transAxes)
                        continue
                    
                    # Set up bar positions
                    n_groups = len(required_categories)
                    n_metrics = len(plot_metric_names)
                    bar_width = 0.12
                    x_pos = np.arange(n_groups)
                    
                    # Plot each metric
                    for i, (metric, label) in enumerate(zip(plot_metric_names, plot_metric_labels)):
                        means = []
                        stds = []
                        
                        for group_name in required_categories:
                            if group_name in stability_stats:
                                group_stats = stability_stats[group_name]
                                if metric == 'baseline_f1':
                                    baseline_f1_stats = getattr(group_stats, 'baseline_f1', None)
                                    if baseline_f1_stats is not None:
                                        means.append(baseline_f1_stats.mean)
                                        stds.append(baseline_f1_stats.std)
                                    else:
                                        means.append(0.0)
                                        stds.append(0.0)
                                else:
                                    metric_stats = getattr(group_stats, metric)
                                    means.append(metric_stats.mean)
                                    stds.append(metric_stats.std)
                            else:
                                means.append(0.0)
                                stds.append(0.0)
                        
                        # Create bars
                        x_offset = x_pos + (i - n_metrics/2) * bar_width
                        
                        if metric == 'baseline_f1':
                            bars = ax.bar(x_offset, means, bar_width, yerr=stds, capsize=2,
                                         color='#7A9E7E', alpha=0.55, edgecolor='white', linewidth=1)
                            for rect in bars:
                                rect.set_hatch('//')
                        else:
                            bars = ax.bar(x_offset, means, bar_width, yerr=stds, capsize=2,
                                         color=self.colors.get(metric, f'C{i}'),
                                         alpha=0.8, edgecolor='white', linewidth=0.5)
                        
                        # Store handles for legend (only from first valid subplot)
                        if row_idx == 0 and col_idx == 0:
                            legend_handles.append(bars[0])
                            legend_labels.append(label)
                        
                        # Add scatter points
                        for j, group_name in enumerate(required_categories):
                            if group_name in stability_stats:
                                group_stats = stability_stats[group_name]
                                if metric == 'baseline_f1':
                                    baseline_f1_stats = getattr(group_stats, 'baseline_f1', None)
                                    if baseline_f1_stats is not None and baseline_f1_stats.values and len(baseline_f1_stats.values) > 1:
                                        jitter = np.random.normal(0, 0.015, len(baseline_f1_stats.values))
                                        x_jittered = np.full(len(baseline_f1_stats.values), x_offset[j]) + jitter
                                        ax.scatter(x_jittered, baseline_f1_stats.values, color='#7A9E7E',
                                                  s=15, alpha=0.9, edgecolors='white', linewidth=0.5, zorder=3)
                                else:
                                    metric_stats = getattr(group_stats, metric)
                                    if metric_stats.values and len(metric_stats.values) > 1:
                                        jitter = np.random.normal(0, 0.015, len(metric_stats.values))
                                        x_jittered = np.full(len(metric_stats.values), x_offset[j]) + jitter
                                        ax.scatter(x_jittered, metric_stats.values,
                                                  color=self.colors.get(metric, f'C{i}'),
                                                  s=15, alpha=0.9, edgecolors='white', linewidth=0.5, zorder=3)
                        
                    # Styling
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(group_display_names, fontsize=16)
                    ax.set_ylim(0, 1.15)
                    ax.grid(True, alpha=0.3, axis='y')
                    ax.grid(False, axis='x')
                    
                    # Column headers (top row only) - Complex names
                    if row_idx == 0:
                        ax.set_title(complex_name, fontsize=22, fontweight='bold', pad=10)
                    
                    # Y-axis label (left column only)
                    if col_idx == 0:
                        ax.set_ylabel('Score', fontsize=20)
                    
                    # Row labels (right side) - Split names
                    if col_idx == 1:
                        ax.annotate(f'{split_label} Split', xy=(1.05, 0.5), xycoords='axes fraction',
                                   fontsize=22, fontweight='bold', ha='left', va='center', rotation=-90)
            
            # Panel labels (3x2 grid: A-F)
            for idx, (row_idx, col_idx) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]):
                label = chr(ord('A') + idx)
                ax = axes[row_idx, col_idx]
                ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=26,
                       fontweight='bold', va='top', ha='left')
            
            # Unified legend at bottom
            if legend_handles:
                fig.legend(legend_handles, legend_labels, loc='lower center',
                          bbox_to_anchor=(0.5, 0.01), ncol=min(len(legend_handles), 6),
                          frameon=False, fontsize=20)
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.1, right=0.88, hspace=0.25, wspace=0.15)
            
            # Save
            filename = f"combined_stability_{frequency_type}"
            output_path = os.path.join(self.output_directory, f"{filename}.png")
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            svg_path = os.path.join(self.output_directory, f"{filename}.svg")
            plt.savefig(svg_path, format='svg', bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Generated combined stability ({frequency_type}) plot: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate combined stability plot: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main CLI function for combined multi-replica analysis."""
    
    parser = argparse.ArgumentParser(
        description="Combined multi-replica analysis across complexes and temporal splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.analysis.multi_replica.combined_analysis --results_dir ./results_all
  python -m src.analysis.multi_replica.combined_analysis --results_dir ./results_all --output combined_plots
        """
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Base directory containing complex folders (e.g., ./results_all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='combined_multi_replica_analysis',
        help='Output directory for combined plots (default: combined_multi_replica_analysis)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*/analysis',
        help='Pattern to find analysis directories within each split (default: */analysis)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level=log_level)
    logger = get_logger(__name__)
    
    try:
        logger.info("Starting combined multi-replica analysis")
        logger.info(f"Results directory: {args.results_dir}")
        logger.info(f"Output directory: {args.output}")
        
        # Initialize plotter
        plotter = CombinedMultiReplicaPlotter(args.output)
        
        # Discover and load all metrics
        all_metrics = plotter.discover_and_load_all_metrics(args.results_dir, args.pattern)
        
        if not all_metrics:
            logger.error("No metrics found. Check your results directory structure.")
            return 1
        
        # Print summary
        logger.info("=" * 60)
        logger.info("LOADED DATA SUMMARY")
        logger.info("=" * 60)
        for complex_name, splits in all_metrics.items():
            for split, aggregated in splits.items():
                logger.info(f"  {complex_name}/{split}: {aggregated.n_replicas} replicas")
        logger.info("=" * 60)
        
        # Generate combined plots
        generated_plots = plotter.generate_all_combined_plots(all_metrics)
        
        if generated_plots:
            logger.info(f"Generated {len(generated_plots)} combined plots in {args.output}/:")
            for plot_path in generated_plots:
                logger.info(f"  {Path(plot_path).name}")
        else:
            logger.warning("No plots were generated")
        
        logger.info("Combined multi-replica analysis completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Combined multi-replica analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
