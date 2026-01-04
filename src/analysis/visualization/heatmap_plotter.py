"""Heatmap visualization components for interaction analysis."""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter, MultipleLocator
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List

from .base_plotter import BasePlotter
from ..data.data_processor import ProcessedData


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
        processed_data: ProcessedData,
        num_pairs_to_show: int = 50,
        valid_steps_to_show: int = 20
    ) -> List[str]:
        """Generate all heatmap plots."""
        generated_plots = []
        
        try:
            heatmap_path = self.plot_time_vs_pair_heatmaps(
                processed_data.ground_truth_full,
                processed_data.predictions_full,
                processed_data.train_set_processed,
                processed_data.valid_set_processed,
                num_pairs_to_show
            )
            generated_plots.append(heatmap_path)
            self.logger.info(f"Generated {len(generated_plots)} heatmap plots")
        except Exception as e:
            self.logger.error(f"Failed to generate heatmap plots: {e}")
        
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
    
    def _calculate_figure_size(self, num_cols: int, num_rows: int) -> tuple:
        """Calculate figure size to maintain consistent cell aspect ratio."""
        target_cell_aspect = 1.0
        
        fig_height = max(self.MIN_FIG_HEIGHT, num_rows * self.CELL_HEIGHT)
        cell_height = fig_height / num_rows
        cell_width = cell_height * target_cell_aspect
        fig_width = max(self.MIN_FIG_WIDTH, num_cols * cell_width)
        
        return fig_width, fig_height
    
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
    
    def _process_test_data(
        self, 
        gt_full: pd.DataFrame, 
        pred_full: pd.DataFrame, 
        num_pairs_to_show: int
    ) -> tuple:
        """Process and align test data."""
        gt_pivot = gt_full.pivot(index='pair', columns='time_stamp', values='present')
        pred_pivot = pred_full.pivot(index='pair', columns='time_stamp', values='present')
        
        common_pairs = gt_pivot.index.intersection(pred_pivot.index)
        gt_pivot = gt_pivot.loc[common_pairs].fillna(0).astype(int)
        pred_pivot = pred_pivot.loc[common_pairs].fillna(0).astype(int)
        
        if len(common_pairs) > num_pairs_to_show:
            pair_freq = gt_pivot.sum(axis=1).sort_values(ascending=False)
            selected_pairs = pair_freq.head(num_pairs_to_show).index
            gt_pivot = gt_pivot.loc[selected_pairs]
            pred_pivot = pred_pivot.loc[selected_pairs]
        else:
            selected_pairs = common_pairs
        
        try:
            sort_keys = gt_pivot.index.map(
                lambda x: tuple(self.custom_sort_residues(p) for p in x.split("_"))
            )
            sort_order = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])
            gt_pivot = gt_pivot.iloc[sort_order]
            pred_pivot = pred_pivot.iloc[sort_order]
        except Exception as e:
            self.logger.warning(f"Custom pair sorting failed ({e}). Using string sort.")
            gt_pivot = gt_pivot.sort_index()
            pred_pivot = pred_pivot.loc[gt_pivot.index]
        
        return gt_pivot, pred_pivot, selected_pairs
    
    def _process_history_data(
        self,
        train_set: pd.DataFrame,
        valid_set: pd.DataFrame,
        selected_pairs: pd.Index,
        gt_pivot: pd.DataFrame
    ) -> pd.DataFrame:
        """Process and combine history (train + valid) data."""
        history_dfs = []
        if not train_set.empty:
            history_dfs.append(train_set)
        if not valid_set.empty:
            history_dfs.append(valid_set)
        
        if not history_dfs:
            return pd.DataFrame()
        
        combined_history = pd.concat(history_dfs)
        combined_history = combined_history[combined_history['pair'].isin(selected_pairs)].copy()
        
        if combined_history.empty:
            return pd.DataFrame()
        
        combined_history['present'] = 1
        history_pivot = combined_history.pivot_table(
            index='pair', columns='time_stamp', values='present', fill_value=0
        )
        history_pivot = history_pivot.reindex(gt_pivot.index, fill_value=0)
        history_pivot = history_pivot.reindex(sorted(history_pivot.columns), axis=1)
        
        return history_pivot
    
    def _create_overlay_matrix(
        self,
        gt_pivot: pd.DataFrame,
        pred_pivot: pd.DataFrame,
        history_pivot: pd.DataFrame
    ) -> tuple:
        """Create the overlay matrix combining history and test data."""
        overlay_test = pd.DataFrame(0, index=gt_pivot.index, columns=gt_pivot.columns)
        overlay_test[(gt_pivot == 1) & (pred_pivot == 0)] = 1  # FN
        overlay_test[(gt_pivot == 0) & (pred_pivot == 1)] = 2  # FP
        overlay_test[(gt_pivot == 1) & (pred_pivot == 1)] = 3  # TP
        
        if not history_pivot.empty:
            history_overlay = history_pivot.replace({0: 0, 1: 4})
            overlay_combined = pd.concat([history_overlay, overlay_test], axis=1)
            history_offset = history_pivot.shape[1]
        else:
            overlay_combined = overlay_test
            history_offset = 0
        
        return overlay_combined, history_offset
    
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
    
    def plot_time_vs_pair_heatmaps(
        self,
        gt_full: pd.DataFrame,
        pred_full: pd.DataFrame,
        train_set: pd.DataFrame,
        valid_set: pd.DataFrame,
        num_pairs_to_show: int = 50
    ) -> str:
        """Plot interaction dynamics heatmap with history and test regions."""
        self.logger.info("Generating interaction dynamics heatmap")
        self._apply_style()
        
        gt_pivot, pred_pivot, selected_pairs = self._process_test_data(
            gt_full, pred_full, num_pairs_to_show
        )
        
        history_pivot = self._process_history_data(
            train_set, valid_set, selected_pairs, gt_pivot
        )
        
        overlay_combined, history_offset = self._create_overlay_matrix(
            gt_pivot, pred_pivot, history_pivot
        )
        
        num_cols = overlay_combined.shape[1]
        num_rows = len(overlay_combined.index)
        fig_width, fig_height = self._calculate_figure_size(num_cols, num_rows)
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        cmap_overlay = self._create_colormaps()
        bounds = [0, 1, 2, 3, 4, 5]
        norm = mcolors.BoundaryNorm(bounds, cmap_overlay.N)
        
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
        
        ax.set_title('')
        ax.set_xlabel('Simulation time (ns)', fontsize=50, labelpad=60)
        ax.set_ylabel('Residue Pair', fontsize=50)
        
        colorbar = cax.collections[0].colorbar
        colorbar.set_ticklabels(['TN', 'FN', 'FP', 'TP', 'History'])
        colorbar.ax.tick_params(labelsize=45)
        colorbar.set_label("Result Type", fontsize=50)
        
        y_fontsize = self._calculate_y_fontsize(fig_height, num_rows)
        ax.tick_params(axis='y', labelsize=y_fontsize)
        
        timestamps = overlay_combined.columns
        ax.xaxis.set_major_formatter(FuncFormatter(self._create_time_formatter(timestamps)))
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.tick_params(axis='x', labelsize=45, rotation=0)
        
        if history_offset > 0:
            ax.axvline(x=history_offset, color=self.OKABE_ITO['dark_gray'], 
                      linestyle='--', linewidth=3)
            
            label_y_pos = ax.get_ylim()[0] * 1.05
            
            ax.text(
                history_offset / 2., label_y_pos, 'History',
                ha='center', va='top', color=self.OKABE_ITO['dark_gray'], fontsize=45
            )
            ax.text(
                history_offset + (num_cols - history_offset) / 2., label_y_pos, 'Test',
                ha='center', va='top', color=self.OKABE_ITO['dark_gray'], fontsize=45
            )
        
        plt.tight_layout()
        
        filename = 'heatmap_time_vs_pairs_VERTICAL_full_history.png'
        plot_path = self.save_plot(filename, fig)
        self.save_plot(filename.replace('.png', '.svg'), fig)
        self.close_plot(fig)
        
        return plot_path
