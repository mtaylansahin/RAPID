"""Metrics visualization plots for stability-stratified analysis."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.base import BasePlotter, OKABE_ITO
from src.analysis.data_structures import StabilityMetrics


class MetricsPlotter(BasePlotter):
    """Creates bar charts of metrics grouped by interaction stability."""

    def generate_plots(
        self,
        stability_metrics_train: Dict[str, StabilityMetrics],
        stability_metrics_test: Dict[str, StabilityMetrics],
        overall_metrics: Dict[str, float],
    ) -> List[Path]:
        """Generate all metrics plots."""
        generated_plots = []

        try:
            # Plot stability-based metrics (Training freq)
            if stability_metrics_train:
                path = self._plot_metrics_by_stability_group(
                    metrics_mapping=stability_metrics_train,
                    plot_title="Performance Metrics by Interaction Stability (based on Training Freq.)",
                    xlabel="Stability Bin (Training Set Frequency)",
                    filename="metrics_by_training_freq",
                )
                if path:
                    generated_plots.append(path)

            # Plot stability-based metrics (Test freq)
            if stability_metrics_test:
                path = self._plot_metrics_by_stability_group(
                    metrics_mapping=stability_metrics_test,
                    plot_title="Performance Metrics by Interaction Stability (based on Test Freq.)",
                    xlabel="Stability Bin (Test Set Frequency)",
                    filename="metrics_by_test_freq",
                )
                if path:
                    generated_plots.append(path)

            # Plot Mean Pairwise F1 Comparison
            path_f1 = self._plot_mean_pairwise_f1_comparison(overall_metrics)
            if path_f1:
                generated_plots.append(path_f1)

        except Exception as e:
            print(f"Failed to generate metrics plots: {e}")

        return generated_plots

    def _plot_metrics_by_stability_group(
        self,
        metrics_mapping: Dict[str, StabilityMetrics],
        plot_title: str,
        xlabel: str,
        filename: str,
    ) -> Optional[Path]:
        """Generic plotter for metrics grouped by stability."""
        if not metrics_mapping:
            return None

        # Assemble DataFrame
        rows = []
        for bin_label, m in metrics_mapping.items():
            rows.append(
                {
                    "Bin Label": bin_label,
                    "Recall": m.recall,
                    "Precision": m.precision,
                    "F1": m.f1,
                    "MCC": m.mcc,
                    "Mean Pairwise F1": m.mean_pairwise_f1,
                    "Baseline F1": m.baseline_f1 if m.baseline_f1 is not None else None,
                    "Pair Count": m.pair_count,
                }
            )

        df = pd.DataFrame(rows).set_index("Bin Label")

        # Map RAPID group names to RNN-MD display names
        name_map = {
            "Rare": "Rare (<5%)",
            "Transient": "Moderate (5-50%)",
            "Stable": "Stable (>50%)",
        }

        # Rename index where possible
        df = df.rename(index=name_map)

        required_categories = ["Rare (<5%)", "Moderate (5-50%)", "Stable (>50%)"]
        
        # Build columns to plot - only include Baseline F1 if present and has valid values
        columns_to_plot = ["Recall", "Precision", "F1", "MCC", "Mean Pairwise F1"]
        if "Baseline F1" in df.columns and df["Baseline F1"].notna().any():
            columns_to_plot.append("Baseline F1")

        # Fill missing categories with zero rows
        for cat in required_categories:
            if cat not in df.index:
                zero_row = pd.Series({col: 0.0 for col in columns_to_plot}, name=cat)
                zero_row["Pair Count"] = 0
                df = pd.concat([df, zero_row.to_frame().T])

        # Reindex to force order
        df = df.reindex(required_categories)

        # Plot setup
        fig, ax = plt.subplots(figsize=(12, 8))

        metric_colors = {
            "Recall": "#2E86AB",
            "Precision": "#A23B72",
            "F1": "#F18F01",
            "MCC": "#C73E1D",
            "Mean Pairwise F1": "#36213E",
            "Baseline F1": "#7A9E7E",
        }

        cols_to_plot_in_bar = [col for col in columns_to_plot if col in df.columns]
        colors = [metric_colors[col] for col in cols_to_plot_in_bar]

        df[cols_to_plot_in_bar].plot(
            kind="bar", ax=ax, color=colors, width=0.8, edgecolor="white", linewidth=0.7
        )

        # Value labels (only for non-zero bars)
        for container in ax.containers:
            labels = []
            for bar in container:
                height = bar.get_height()
                if height > 0:
                    labels.append(f"{height:.3f}")
                else:
                    labels.append("")
            ax.bar_label(
                container,
                labels=labels,
                label_type="edge",
                padding=3,
                fontsize=9,
                fontweight="bold",
            )

        # X labels with counts - use Transient for display
        x_labels = []
        for cat in required_categories:
            count = int(df.loc[cat, "Pair Count"]) if "Pair Count" in df.columns else 0
            display_label = "Transient (5-50%)" if cat == "Moderate (5-50%)" else cat
            x_labels.append(f"{display_label}\n(N={count})")

        ax.set_xticklabels(x_labels, rotation=0, ha="center")

        # Styling
        ax.set_title(plot_title, fontsize=14, fontweight="bold", pad=20)
        ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")

        ax.legend(
            title="Metric",
            title_fontsize=11,
            fontsize=10,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=True,
            fancybox=True,
            shadow=True,
        )

        ax.grid(True, axis="y", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Dynamic Y-lim matching RNN-MD calculation
        ax.set_ylim(bottom=0, top=min(1.1, max(1.05, ax.get_ylim()[1] * 1.08)))
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='x', which='major', pad=5)
        ax.set_facecolor("#f8f9fa")

        plt.tight_layout(rect=[0, 0, 0.85, 1])

        paths = self._save_figure(fig, filename)
        plt.close(fig)
        return paths[0] if paths else None

    def _plot_mean_pairwise_f1_comparison(
        self, overall_metrics: Dict[str, float]
    ) -> Optional[Path]:
        """Plot Mean Pairwise F1 Score compared with other metrics.
        
        Single, clean bar chart comparing all key metrics including baseline.
        """
        try:
            # Core metrics with consistent order
            metric_config = [
                ("Recall", "recall", OKABE_ITO["sky_blue"]),
                ("Precision", "precision", OKABE_ITO["vermilion"]),
                ("Overall F1", "f1", OKABE_ITO["orange"]),
                ("MCC", "mcc", OKABE_ITO["reddish_purple"]),
                ("Mean Pairwise F1", "mean_pairwise_f1", OKABE_ITO["bluish_green"]),
            ]
            
            # Build data lists
            names = []
            values = []
            colors = []
            
            for display_name, key, color in metric_config:
                val = overall_metrics.get(key, 0.0)
                names.append(display_name)
                values.append(val)
                colors.append(color)
            
            # Add baseline placeholder with TODO
            # TODO: Implement actual baseline metrics when available
            baseline_f1 = overall_metrics.get("baseline_f1")
            if baseline_f1 is not None:
                names.append("Baseline F1")
                values.append(baseline_f1)
                colors.append("#999999")  # Gray for baseline placeholder
            
            # Simple single-panel plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x_pos = np.arange(len(names))
            bar_width = 0.6
            
            bars = ax.bar(x_pos, values, width=bar_width, color=colors, 
                         alpha=0.85, edgecolor="white", linewidth=1.5)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.015,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=11,
                )
            
            # Style the plot
            ax.set_title("Metric Comparison", fontsize=16, fontweight="bold", pad=15)
            ax.set_ylabel("Score", fontsize=13, fontweight="bold")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(names, fontsize=11, fontweight="medium")
            ax.set_ylim(0, min(1.15, max(values) * 1.2 if values else 1.0))
            ax.set_xlim(-0.5, len(names) - 0.5)
            
            # Grid and styling
            ax.grid(True, axis="y", linestyle="--", alpha=0.3, zorder=0)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", which="major", labelsize=11)
            
            plt.tight_layout()
            
            paths = self._save_figure(fig, "mean_pairwise_f1_comparison")
            plt.close(fig)
            return paths[0] if paths else None

        except Exception as e:
            print(f"Failed to plot F1 comparison: {e}")
            return None
