"""Heatmap visualizations for interaction dynamics."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter, MultipleLocator
import seaborn as sns
import numpy as np
import pandas as pd

from src.analysis.base import BasePlotter, OKABE_ITO


class HeatmapPlotter(BasePlotter):
    """Handles heatmap visualizations for interaction analysis."""

    OKABE_ITO = {
        'vermilion': '#D55E00',
        'blue': '#0072B2',
        'bluish_green': '#009E73',
        'orange': '#E69F00',
        'sky_blue': '#56B4E9',
        'white': '#FFFFFF',
        'light_gray': '#F0F0F0',
        'dark_gray': '#2C3E50',
        'neutral_gray': '#999999'
    }

    CELL_HEIGHT = 0.4
    MIN_FIG_HEIGHT = 12
    MIN_FIG_WIDTH = 20
    NS_PER_TIMESTEP = 0.5

    def generate_plots(
        self,
        predictions: List[Tuple[int, int, int, int, float, int]],
        ground_truth: List[Tuple[int, int, int, int]],
        train_data: np.ndarray,
        valid_data: np.ndarray,
        test_timesteps: List[int],
        num_pairs_to_show: int = 50,
    ) -> List[Path]:
        """Generate all heatmap plots."""
        generated_plots = []

        try:
            # Prepare DataFrames
            gt_df = pd.DataFrame(
                ground_truth, columns=["e1", "rel", "e2", "time_stamp"]
            )
            gt_df["present"] = 1
            gt_df["pair"] = gt_df.apply(
                lambda r: f"{min(r.e1, r.e2)}_{max(r.e1, r.e2)}", axis=1
            )

            pred_df = pd.DataFrame(
                predictions,
                columns=["e1", "rel", "e2", "time_stamp", "score", "present"],
            )
            pred_df["pair"] = pred_df.apply(
                lambda r: f"{min(r.e1, r.e2)}_{max(r.e1, r.e2)}", axis=1
            )

            # History Data
            train_df = pd.DataFrame()
            if len(train_data) > 0:
                train_df = pd.DataFrame(
                    train_data, columns=["e1", "rel", "e2", "time_stamp"]
                )
                train_df["pair"] = train_df.apply(
                    lambda r: f"{min(int(r.e1), int(r.e2))}_{max(int(r.e1), int(r.e2))}",
                    axis=1,
                )

            valid_df = pd.DataFrame()
            if len(valid_data) > 0:
                valid_df = pd.DataFrame(
                    valid_data, columns=["e1", "rel", "e2", "time_stamp"]
                )
                valid_df["pair"] = valid_df.apply(
                    lambda r: f"{min(int(r.e1), int(r.e2))}_{max(int(r.e1), int(r.e2))}",
                    axis=1,
                )

            # Generate Plot
            heatmap_path = self.plot_time_vs_pair_heatmaps(
                gt_df, pred_df, train_df, valid_df, num_pairs_to_show
            )
            if heatmap_path:
                generated_plots.append(heatmap_path)

        except Exception as e:
            print(f"Failed to generate heatmap plots: {e}")
            import traceback

            traceback.print_exc()

        return generated_plots

    def _apply_style(self):
        """Apply publication-ready style settings."""
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
            'font.size': 40,
            'axes.titlesize': 54,
            'axes.labelsize': 50,
            'xtick.labelsize': 40,
            'ytick.labelsize': 27,
            'legend.fontsize': 40,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.linewidth': 1.5,
            'axes.edgecolor': self.OKABE_ITO['dark_gray'],
        })

    def _calculate_y_fontsize(self, fig_height: float, num_labels: int) -> int:
        """Calculate appropriate font size for Y-axis labels."""
        if num_labels == 0:
            return 21
        return max(15, min(27, int((fig_height / num_labels) * 72 * 0.45)))

    def _create_colormaps(self):
        """Create color maps for the heatmap."""
        cmap_overlay = mcolors.ListedColormap([
            self.OKABE_ITO['light_gray'],
            self.OKABE_ITO['sky_blue'],
            self.OKABE_ITO['vermilion'],
            self.OKABE_ITO['bluish_green'],
            self.OKABE_ITO['neutral_gray']
        ])
        return cmap_overlay

    def _create_time_formatter(self, timestamps):
        """Create formatter function for x-axis time labels."""
        def formatter(x, pos):
            try:
                idx = int(x)
                if 0 <= idx < len(timestamps):
                    return f"{timestamps[idx] * self.NS_PER_TIMESTEP:.0f}"
                return ""
            except:
                return ""
        return formatter

    def custom_sort_residues(self, value: str) -> Tuple[float, str]:
        """Custom sorting function for residue names."""
        try:
            parts = value.split('-')
            numeric_part = int(parts[0])
            string_part = '-'.join(parts[1:])
            return numeric_part, string_part
        except (ValueError, IndexError):
            return float('inf'), value

    def plot_time_vs_pair_heatmaps(
        self,
        gt_full: pd.DataFrame,
        pred_full: pd.DataFrame,
        train_set: pd.DataFrame,
        valid_set: pd.DataFrame,
        num_pairs_to_show: int = 50,
    ) -> Optional[Path]:
        """Plot interaction dynamics heatmap with history and test regions."""
        self._apply_style()

        # Process Test Data
        # Use pivot_table with 'max' aggregation - if ANY prediction is positive, count as 1
        gt_pivot = gt_full.pivot_table(
            index="pair", columns="time_stamp", values="present", aggfunc="max", fill_value=0
        )
        pred_pivot = pred_full.pivot_table(
            index="pair", columns="time_stamp", values="present", aggfunc="max", fill_value=0
        )

        # Ensure pairs are aligned
        relevant_pairs = gt_pivot.index.union(pred_pivot.index)
        gt_pivot = gt_pivot.reindex(relevant_pairs, fill_value=0).fillna(0).astype(int)
        pred_pivot = (
            pred_pivot.reindex(relevant_pairs, fill_value=0).fillna(0).astype(int)
        )

        # Select top pairs by activity frequency
        if len(relevant_pairs) > num_pairs_to_show:
            pair_freq = gt_pivot.sum(axis=1).sort_values(ascending=False)
            selected_pairs = pair_freq.head(num_pairs_to_show).index
            # Sort numerically
            try:
                selected_pairs = sorted(
                    selected_pairs, key=lambda x: [int(p) for p in x.split("_")]
                )
            except:
                selected_pairs = sorted(selected_pairs)
        else:
            selected_pairs = relevant_pairs
            try:
                selected_pairs = sorted(
                    selected_pairs, key=lambda x: [int(p) for p in x.split("_")]
                )
            except:
                selected_pairs = sorted(selected_pairs)

        gt_pivot = gt_pivot.loc[selected_pairs]
        pred_pivot = pred_pivot.loc[selected_pairs]

        # Process History Data
        history_dfs = []
        if not train_set.empty:
            history_dfs.append(train_set)
        if not valid_set.empty:
            history_dfs.append(valid_set)

        if history_dfs:
            combined_history = pd.concat(history_dfs)
            combined_history = combined_history[
                combined_history["pair"].isin(selected_pairs)
            ].copy()
            combined_history["present"] = 1
            if not combined_history.empty:
                history_pivot = combined_history.pivot_table(
                    index="pair", columns="time_stamp", values="present", fill_value=0
                )
                history_pivot = history_pivot.reindex(selected_pairs, fill_value=0)
                history_pivot = history_pivot.reindex(
                    sorted(history_pivot.columns), axis=1
                )
            else:
                history_pivot = pd.DataFrame(index=selected_pairs)
        else:
            history_pivot = pd.DataFrame(index=selected_pairs)

        # Create overlay matrix: 0=TN, 1=FN, 2=FP, 3=TP
        overlay_test = pd.DataFrame(0, index=gt_pivot.index, columns=gt_pivot.columns)
        common_cols = gt_pivot.columns.intersection(pred_pivot.columns)
        overlay_test.loc[:, common_cols] = 0

        g = gt_pivot[common_cols]
        p = pred_pivot[common_cols]

        # Apply classification masks
        mask_fn = (g == 1) & (p == 0)
        overlay_test[mask_fn] = 1

        mask_fp = (g == 0) & (p == 1)
        overlay_test[mask_fp] = 2

        mask_tp = (g == 1) & (p == 1)
        overlay_test[mask_tp] = 3

        # Combine with history (value 4)
        if not history_pivot.empty:
            history_overlay = history_pivot.replace({0: 0, 1: 4})
            overlay_combined = pd.concat([history_overlay, overlay_test], axis=1)
            history_offset = history_pivot.shape[1]
        else:
            overlay_combined = overlay_test
            history_offset = 0

        # Handle NaN values
        overlay_combined = overlay_combined.fillna(0).astype(int)

        # Calculate figure size to maintain consistent cell aspect ratio
        num_cols = overlay_combined.shape[1]
        num_rows = len(overlay_combined.index)
        
        target_cell_aspect = 1.0
        fig_height = max(self.MIN_FIG_HEIGHT, num_rows * self.CELL_HEIGHT)
        cell_height = fig_height / num_rows
        cell_width = cell_height * target_cell_aspect
        fig_width = max(self.MIN_FIG_WIDTH, num_cols * cell_width)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Create colormap using the helper method
        cmap_overlay = self._create_colormaps()
        bounds = [0, 1, 2, 3, 4, 5]
        norm = mcolors.BoundaryNorm(bounds, cmap_overlay.N)

        # Plot using seaborn heatmap for publication-quality output
        cax = sns.heatmap(
            overlay_combined.fillna(0).astype(int),
            ax=ax,
            cmap=cmap_overlay,
            norm=norm,
            cbar=True,
            yticklabels=overlay_combined.index,
            linewidths=0.05,
            linecolor='white',
            square=False,
            cbar_kws={"ticks": [0.5, 1.5, 2.5, 3.5, 4.5], "pad": 0.02}
        )

        # Labels with publication-quality font sizes
        ax.set_title('')
        ax.set_xlabel('Simulation time (ns)', fontsize=50, labelpad=60)
        ax.set_ylabel('Residue Pair', fontsize=50)

        # Configure colorbar with proper labels
        colorbar = cax.collections[0].colorbar
        colorbar.set_ticklabels(['TN', 'FN', 'FP', 'TP', 'History'])
        colorbar.ax.tick_params(labelsize=45)
        colorbar.set_label("Result Type", fontsize=50)

        # Calculate and apply dynamic Y-axis font size
        y_fontsize = self._calculate_y_fontsize(fig_height, num_rows)
        ax.tick_params(axis='y', labelsize=y_fontsize)

        # X-axis formatting with time formatter
        timestamps = overlay_combined.columns
        ax.xaxis.set_major_formatter(FuncFormatter(self._create_time_formatter(timestamps)))
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.tick_params(axis='x', labelsize=45, rotation=0)

        # History/Test separator line
        if history_offset > 0:
            ax.axvline(
                x=history_offset, 
                color=self.OKABE_ITO['dark_gray'],
                linestyle='--', 
                linewidth=3
            )

            # Add History/Test labels above the plot
            label_y_pos = ax.get_ylim()[0] * 1.05

            ax.text(
                history_offset / 2., 
                label_y_pos, 
                'History',
                ha='center', 
                va='top', 
                color=self.OKABE_ITO['dark_gray'], 
                fontsize=45
            )
            ax.text(
                history_offset + (num_cols - history_offset) / 2., 
                label_y_pos, 
                'Test',
                ha='center', 
                va='top', 
                color=self.OKABE_ITO['dark_gray'], 
                fontsize=45
            )

        plt.tight_layout()

        # Save in multiple formats
        paths = self._save_figure(fig, "heatmap_time_vs_pairs_VERTICAL_full_history", formats=("png", "svg"))
        plt.close(fig)

        return paths[0] if paths else None
