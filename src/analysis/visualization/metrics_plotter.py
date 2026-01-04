"""Metrics visualization components for performance analysis."""

import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any

from .base_plotter import BasePlotter
from ..data.data_processor import ProcessedData
from ..analytics.metrics_calculator import MetricsReport


class MetricsPlotter(BasePlotter):
    """Handles metrics visualizations for performance analysis."""
    
    def generate_plots(
        self,
        processed_data: ProcessedData,
        metrics_report: MetricsReport,
        scores_file_path: str
    ) -> List[str]:
        """Generate all metrics plots.
        
        Args:
            processed_data: Processed analysis data
            metrics_report: Calculated metrics report
            scores_file_path: Path to scores file for appending stability metrics
            
        Returns:
            List of generated plot file paths
        """
        generated_plots = []
        
        try:
            # Plot stability-based metrics if available
            if processed_data.stability_bins is not None and metrics_report.metrics_by_stability:
                stability_metrics_path = self._plot_metrics_by_stability_group(
                    metrics_mapping=metrics_report.metrics_by_stability,
                    scores_file_path=scores_file_path,
                    header_title="Performance by Interaction Stability (TRAINING Set Frequency)",
                    plot_title='Performance Metrics by Interaction Stability (based on Training Freq.)',
                    xlabel='Stability Bin (Training Set Frequency)',
                    filename='metrics_by_stability_bar_trainfreq.png',
                    undefined_label_replacement='Not in Train'
                )
                generated_plots.append(stability_metrics_path)
            
            # Plot test-frequency-based stability metrics if available
            if (hasattr(processed_data, 'stability_bins_test') and processed_data.stability_bins_test is not None 
                and hasattr(metrics_report, 'metrics_by_test_stability') and metrics_report.metrics_by_test_stability):
                stability_metrics_test_path = self._plot_metrics_by_stability_group(
                    metrics_mapping=metrics_report.metrics_by_test_stability,
                    scores_file_path=scores_file_path,
                    header_title="Performance by Interaction Stability (TEST Set Frequency)",
                    plot_title='Performance Metrics by Interaction Stability (based on Test Freq.)',
                    xlabel='Stability Bin (Test Set Frequency)',
                    filename='metrics_by_stability_bar_testfreq.png',
                    undefined_label_replacement='Not in Test'
                )
                generated_plots.append(stability_metrics_test_path)
            
            # Plot Mean Pairwise F1 Score comparison
            if hasattr(metrics_report, 'mean_pairwise_f1'):
                mean_f1_path = self._plot_mean_pairwise_f1_comparison(metrics_report)
                generated_plots.append(mean_f1_path)
            
            self.logger.info(f"Generated {len(generated_plots)} metrics plots")
            
        except Exception as e:
            self.logger.error(f"Failed to generate metrics plots: {e}")
        
        return generated_plots

    
    def _plot_metrics_by_stability_group(
        self,
        metrics_mapping: Dict[str, Any],
        scores_file_path: str,
        header_title: str,
        plot_title: str,
        xlabel: str,
        filename: str,
        undefined_label_replacement: str
    ) -> str:
        """Generic plotter for metrics grouped by stability (train/test).
        
        Args:
            metrics_mapping: Mapping of stability bin -> metrics detail
            scores_file_path: Path to append textual metrics
            header_title: Section header to write to scores file
            plot_title: Title used on the plot
            xlabel: X-axis label
            filename: Output filename for the plot
            undefined_label_replacement: Label to use for 'Undefined' bin in text output
        """
        if not metrics_mapping:
            self.logger.warning("No metrics by stability data provided, skipping plot.")
            return ""

        # Assemble DataFrame from mapping
        plot_rows = []
        for bin_label, stability_detail in metrics_mapping.items():
            plot_rows.append({
                'Bin Label': bin_label,
                'Recall': stability_detail.Recall,
                'Precision': stability_detail.Precision,
                'F1': stability_detail.F1,
                'MCC': stability_detail.MCC,
                'Mean Pairwise F1': stability_detail.mean_pairwise_f1,
                'Baseline F1': stability_detail.baseline_f1 if getattr(stability_detail, 'baseline_f1', None) is not None else None,
                'Pair Count': stability_detail.pair_count
            })
        metrics_df_from_report = pd.DataFrame(plot_rows).set_index('Bin Label')

        # Append detailed metrics to scores file
        with open(scores_file_path, "a") as scores_output_file:
            print(f"\n--- {header_title} ---", file=scores_output_file)
            for bin_label, stability_detail in metrics_mapping.items():
                pair_count = stability_detail.pair_count
                output_label = bin_label
                if bin_label == "Moderate (5-50%)":
                    output_label = "Transient (5-50%)"
                if bin_label == "Undefined":
                    output_label = undefined_label_replacement
                print(f"\nMetrics for {output_label} interactions ({pair_count} pairs):", file=scores_output_file)
                if pair_count == 0 and getattr(stability_detail, 'TP', 0) == 0:
                    print("No interactions found or processed in this bin for metric calculation.", file=scores_output_file)
                else:
                    print(
                        f"Recall: {stability_detail.Recall:.4f}, Precision: {stability_detail.Precision:.4f}, "
                        f"TPR: {stability_detail.TPR:.4f}, FPR: {stability_detail.FPR:.4f}, "
                        f"F1: {stability_detail.F1:.4f}, MCC: {stability_detail.MCC:.4f}, "
                        f"Mean Pairwise F1: {stability_detail.mean_pairwise_f1:.4f}",
                        file=scores_output_file
                    )
                    if getattr(stability_detail, 'baseline_f1', None) is not None:
                        print(
                            f"Baseline F1: {stability_detail.baseline_f1:.4f}",
                            file=scores_output_file
                        )

        # Prepare DataFrame for plotting (excluding 'Undefined' bin)
        metrics_df_plot = metrics_df_from_report.drop('Undefined', errors='ignore')

        # Select columns to plot, add baseline if present
        columns_to_plot = ['Recall', 'Precision', 'F1', 'MCC', 'Mean Pairwise F1']
        if 'Baseline F1' in metrics_df_plot.columns:
            if metrics_df_plot['Baseline F1'].notna().any():
                columns_to_plot.append('Baseline F1')

        metrics_df_plot = metrics_df_plot[columns_to_plot]

        # Always include the three main stability categories, even if they have no data
        required_categories = ['Rare (<5%)', 'Moderate (5-50%)', 'Stable (>50%)']
        
        # Create zero-filled rows for missing categories
        for category in required_categories:
            if category not in metrics_df_plot.index:
                # Create a row of zeros for missing category
                zero_row = pd.Series({col: 0.0 for col in columns_to_plot}, name=category)
                metrics_df_plot = pd.concat([metrics_df_plot, zero_row.to_frame().T])
        
        # Reindex to ensure consistent order across all plots
        metrics_df_plot = metrics_df_plot.reindex(required_categories)

        # Render bar chart
        if metrics_df_plot.empty:
            self.logger.warning("No data to plot for metrics by stability")
            return ""

        metric_colors = {
            'Recall': '#2E86AB',
            'Precision': '#A23B72',
            'F1': '#F18F01',
            'MCC': '#C73E1D',
            'Mean Pairwise F1': '#36213E',
            'Baseline F1': '#7A9E7E'
        }

        fig, ax = plt.subplots(figsize=(12, 8))

        cols_to_plot_in_bar = [col for col in columns_to_plot if col in metrics_df_plot.columns]
        colors_for_bars = [metric_colors[col] for col in cols_to_plot_in_bar]
        metrics_df_plot[cols_to_plot_in_bar].plot(
            kind='bar',
            ax=ax,
            color=colors_for_bars,
            width=0.8,
            edgecolor='white',
            linewidth=0.7
        )

        # Value labels (only for non-zero bars)
        for container in ax.containers:
            # Get the values for this container
            labels = []
            for bar in container:
                height = bar.get_height()
                if height > 0:
                    labels.append(f'{height:.3f}')
                else:
                    labels.append('')  # Empty label for zero bars
            ax.bar_label(container, labels=labels, label_type='edge', padding=3, fontsize=9, fontweight='bold')

        # X labels with counts
        x_labels_with_counts = []
        for bin_label_for_plot in required_categories:
            count = metrics_df_from_report.loc[bin_label_for_plot, 'Pair Count'] if bin_label_for_plot in metrics_df_from_report.index else 0
            display_label = "Transient (5-50%)" if bin_label_for_plot == "Moderate (5-50%)" else bin_label_for_plot
            x_labels_with_counts.append(f'{display_label}\n(N={int(count)})')
        ax.set_xticklabels(x_labels_with_counts, rotation=0, ha='center')

        # Styling
        ax.set_title(plot_title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.legend(
            title='Metric',
            title_fontsize=11,
            fontsize=10,
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            frameon=True,
            fancybox=True,
            shadow=True
        )
        ax.grid(True, axis='y', linestyle='--', alpha=0.4, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0, top=min(1.1, max(1.05, ax.get_ylim()[1] * 1.08)))
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='x', which='major', pad=5)
        ax.set_facecolor('#f8f9fa')
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        plot_path = self.save_plot(filename, fig)
        self.close_plot(fig)
        return plot_path

    # Removed duplicated test-specific method in favor of the generic implementation above
    

    def _plot_mean_pairwise_f1_comparison(self, metrics_report: MetricsReport) -> str:
        """Plot Mean Pairwise F1 Score compared with other metrics.
        
        Args:
            metrics_report: Complete metrics report
            
        Returns:
            Path to saved plot
        """
        self.logger.info("Generating Mean Pairwise F1 Score comparison plot")
        
        try:
            metrics_rows = [
                ("Overall F1", metrics_report.model_metrics.F1, "#2E86AB"),
                ("Mean Pairwise F1", metrics_report.mean_pairwise_f1, "#A23B72"),
                ("Precision", metrics_report.model_metrics.Precision, "#F18F01"),
                ("Recall", metrics_report.model_metrics.Recall, "#C73E1D"),
                ("MCC", metrics_report.model_metrics.MCC, "#36213E"),
            ]

            if metrics_report.baseline_metrics and metrics_report.baseline_mean_pairwise_f1 is not None:
                metrics_rows.extend(
                    [
                        ("Baseline Overall F1", metrics_report.baseline_metrics.F1, "#7A9E7E"),
                        ("Baseline Mean Pairwise F1", metrics_report.baseline_mean_pairwise_f1, "#D4A574"),
                        ("Baseline Precision", metrics_report.baseline_metrics.Precision, "#E07A5F"),
                        ("Baseline Recall", metrics_report.baseline_metrics.Recall, "#81B29A"),
                        ("Baseline MCC", metrics_report.baseline_metrics.MCC, "#F2CC8F"),
                    ]
                )

            labels = [row[0] for row in metrics_rows]
            values = [row[1] for row in metrics_rows]
            colors = [row[2] for row in metrics_rows]

            fig_height = max(5.5, 0.45 * len(labels) + 3)
            fig, ax = plt.subplots(figsize=(10, fig_height))
            fig.subplots_adjust(left=0.35)

            y_positions = list(range(len(labels)))
            ax.barh(
                y_positions,
                values,
                color=colors,
                alpha=0.85,
                edgecolor="white",
                linewidth=1,
            )

            ax.set_yticks(y_positions)
            ax.set_yticklabels(labels, fontsize=10)
            ax.invert_yaxis()
            ax.set_xlabel("Score", fontsize=11)
            ax.set_title("Mean Pairwise F1 Comparison", fontsize=12, fontweight="bold")

            x_max = min(1.1, max(values) * 1.1) if values else 1.0
            ax.set_xlim(0, x_max)
            ax.grid(True, axis="x", linestyle="--", alpha=0.3)

            for idx, value in enumerate(values):
                ax.text(
                    min(value + 0.01, x_max - 0.02),
                    idx,
                    f"{value:.3f}",
                    va="center",
                    ha="left",
                    fontsize=9,
                    fontweight="bold",
                )

            explanation = (
                "Mean Pairwise F1: Average of per-pair F1 scores across time."
            )
            fig.text(0.5, 0.01, explanation, ha="center", fontsize=9, style="italic")

            plt.tight_layout(rect=[0, 0.04, 1, 1])

            filename = "mean_pairwise_f1_comparison.png"
            plot_path = self.save_plot(filename, fig)
            self.close_plot(fig)

            return plot_path

        except Exception as e:
            self.logger.error(
                f"Failed to create Mean Pairwise F1 comparison plot: {e}"
            )
            return ""
