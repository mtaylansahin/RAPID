"""Interaction stability distribution analysis for raw simulation data."""

import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from src.analysis.base import BasePlotter, STABILITY_COLORS


class StabilityAnalyzer(BasePlotter):
    """Analyzes interaction stability distributions from raw simulation data.

    Generates histograms of interaction frequency categorized as
    Rare (<5%), Transient (5-50%), or Stable (>50%).
    """

    def __init__(self, output_dir: Path):
        super().__init__(output_dir)
        self.all_data: Dict[str, Dict[str, pd.DataFrame]] = {}

    def set_data(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
        """
        Set pre-loaded data from TemporalAnalyzer.

        Args:
            data: Dictionary of complex -> replica -> DataFrame
        """
        self.all_data = data

    def generate_plots(self, **kwargs) -> List[Path]:
        """Generate all stability distribution plots."""
        generated_plots = []

        if not self.all_data:
            print("No data available. Set data first.")
            return generated_plots

        # Generate 2x4 stability grid
        path, stats = self._plot_stability_grid()
        if path:
            generated_plots.append(path)

        return generated_plots

    def _plot_stability_grid(self) -> tuple[Optional[Path], Dict]:
        """
        Create 2x4 grid of stability histograms.

        Rows: Different complexes (max 2)
        Columns: Different replicas (max 4)
        """
        complexes = self._prioritized_complexes()[:2]

        if not complexes:
            return None, {}

        fig = plt.figure(figsize=(20, 5 * len(complexes)))

        all_stats = {}

        for row_idx, complex_name in enumerate(complexes):
            replicas = self.all_data[complex_name]
            if not replicas:
                continue

            replica_list = sorted(replicas.items())[:4]
            clean_name = self._format_complex_label(complex_name)

            complex_stats = {
                "rare_counts": [],
                "transient_counts": [],
                "stable_counts": [],
                "total_pairs": [],
            }

            row_axes = []

            for col_idx, (replica_name, data) in enumerate(replica_list):
                ax = fig.add_subplot(len(complexes), 4, row_idx * 4 + col_idx + 1)
                row_axes.append(ax)

                if data.empty:
                    ax.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=12)
                    continue

                # Calculate interaction frequencies
                frequencies = self._calculate_frequencies(data)

                if frequencies.empty:
                    continue

                # Create histogram
                bins = np.arange(0, 101, 5)
                counts, bin_edges, patches = ax.hist(
                    frequencies,
                    bins=bins,
                    alpha=0.9,
                    edgecolor="white",
                    linewidth=0.5,
                )

                # Color by stability category
                for patch, bin_start, bin_end in zip(
                    patches, bin_edges[:-1], bin_edges[1:]
                ):
                    bin_center = (bin_start + bin_end) / 2
                    if bin_center < 5:
                        patch.set_facecolor(STABILITY_COLORS["Rare"])
                    elif bin_center < 50:
                        patch.set_facecolor(STABILITY_COLORS["Transient"])
                    else:
                        patch.set_facecolor(STABILITY_COLORS["Stable"])

                # Count categories
                rare = (frequencies < 5).sum()
                transient = ((frequencies >= 5) & (frequencies <= 50)).sum()
                stable = (frequencies > 50).sum()
                total = len(frequencies)

                complex_stats["rare_counts"].append(rare)
                complex_stats["transient_counts"].append(transient)
                complex_stats["stable_counts"].append(stable)
                complex_stats["total_pairs"].append(total)

                # Labels
                if row_idx == 0:
                    label = self._format_replica_label(replica_name)
                    ax.set_title(label, fontsize=14, fontweight="bold")

                if col_idx == 3:
                    ax.annotate(
                        clean_name,
                        xy=(1.05, 0.5),
                        xycoords="axes fraction",
                        ha="left",
                        va="center",
                        rotation=-90,
                        fontsize=14,
                        fontweight="bold",
                    )

                # Hide x labels except bottom row
                if row_idx != len(complexes) - 1:
                    ax.set_xticklabels([])

                # Hide y labels except first column
                if col_idx != 0:
                    ax.set_yticklabels([])

                # Stats annotation
                ax.text(
                    0.95,
                    0.95,
                    f"n={total:,}",
                    transform=ax.transAxes,
                    fontsize=10,
                    va="top",
                    ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

                ax.set_xlim(0, 100)
                ax.grid(True, alpha=0.2)

            # Uniform y-axis for row
            if row_axes:
                max_y = max(ax.get_ylim()[1] for ax in row_axes if ax.get_ylim()[1] > 0)
                for ax in row_axes:
                    ax.set_ylim(0, max_y)
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            all_stats[complex_name] = complex_stats

        # Legend
        legend_elements = [
            Patch(facecolor=STABILITY_COLORS["Rare"], label="Rare (<5%)"),
            Patch(facecolor=STABILITY_COLORS["Transient"], label="Transient (5-50%)"),
            Patch(facecolor=STABILITY_COLORS["Stable"], label="Stable (>50%)"),
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=3,
            frameon=False,
            fontsize=12,
        )

        # Axis labels
        fig.text(
            0.04,
            0.5,
            "Interaction Count",
            rotation=90,
            ha="center",
            va="center",
            fontsize=14,
        )
        fig.text(
            0.5,
            0.02,
            "Interaction Frequency (%)",
            ha="center",
            va="center",
            fontsize=14,
        )

        plt.tight_layout(rect=[0.05, 0.08, 0.95, 1.0])

        paths = self._save_figure(
            fig, "interaction_stability_grid", formats=("png", "svg")
        )
        plt.close(fig)

        # Print summary
        self._print_summary(all_stats)

        return paths[0] if paths else None, all_stats

    def _calculate_frequencies(self, data: pd.DataFrame) -> pd.Series:
        """Calculate per-pair interaction frequencies as percentage."""
        total_timepoints = data["time_stamp"].nunique()

        if total_timepoints == 0:
            return pd.Series()

        # Deduplicate
        data_clean = data.drop_duplicates()

        # Count timepoints per pair
        pair_counts = data_clean.groupby(["resid_a", "resid_b"])["time_stamp"].nunique()

        return (pair_counts / total_timepoints) * 100

    def _print_summary(self, stats: Dict) -> None:
        """Print summary statistics."""
        print("\n" + "=" * 50)
        print("STABILITY ANALYSIS SUMMARY")
        print("=" * 50)

        for complex_name, complex_stats in stats.items():
            if complex_stats["total_pairs"]:
                clean_name = self._format_complex_label(complex_name)
                total = sum(complex_stats["total_pairs"])
                avg_rare = np.mean(complex_stats["rare_counts"])
                avg_trans = np.mean(complex_stats["transient_counts"])
                avg_stable = np.mean(complex_stats["stable_counts"])

                print(f"\n{clean_name}:")
                print(f"  Total pairs: {total:,}")
                print(
                    f"  Avg per replica - Rare: {avg_rare:.1f}, Transient: {avg_trans:.1f}, Stable: {avg_stable:.1f}"
                )

    @staticmethod
    def _format_replica_label(replica_name: str) -> str:
        if "replica" in replica_name.lower():
            nums = re.findall(r"\d+", replica_name)
            if nums:
                return f"Replica {nums[0]}"
        return replica_name.capitalize()

    @staticmethod
    def _format_complex_label(complex_name: str) -> str:
        if not complex_name:
            return complex_name
        return complex_name.replace("_interchain", "")

    def _prioritized_complexes(self) -> List[str]:
        keys = sorted(self.all_data.keys())
        ordered = []
        for target in ("1JPS", "1EAW"):
            ordered.extend(
                [k for k in keys if target in k.upper() and k not in ordered]
            )
        ordered.extend([k for k in keys if k not in ordered])
        return ordered
