#!/usr/bin/env python3
"""Combined heatmap visualization for multiple complexes and temporal splits.

Generates separate heatmap figures for each complex:
- One file per complex (heatmap_1EAW.png, heatmap_1JPS.png)
- 3 rows per file: one for each split (80/10/10, 50/25/25, 25/37.5/37.5)

Input structure:
    results_all/
    ├── 1EAW/
    │   ├── result_25-375-375/
    │   │   ├── interfacea-interchain_results_XXXXX/
    │   │   │   ├── preprocess/  (labels.txt, train.txt, valid.txt, test.txt)
    │   │   │   ├── outputs/     (prediction file)
    │   │   │   └── interfacea-interchain_metadata_XXXXX.txt
    │   │   └── ...
    │   ├── result_50-25-25/
    │   └── result_80-10-10/
    └── 1JPS/
        └── ...
"""

from pathlib import Path
import argparse
import re

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Any

from src.core.utils import setup_logging, get_logger
from src.analysis.data.data_loader import DataLoader
from src.analysis.data.data_processor import DataProcessor, ProcessedData


class CombinedHeatmapPlotter:
    """Creates combined heatmap visualizations across complexes and splits."""
    
    COMPLEXES = ['1EAW', '1JPS']
    SPLITS = ['80-10-10', '50-25-25', '25-375-375']
    SPLIT_LABELS = ['80/10/10', '50/25/25', '25/37.5/37.5']
    TARGET_REPLICA = 'replica1'
    
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
    
    NS_PER_TIMESTEP = 0.5
    
    def __init__(self, output_directory: str):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)
        self.data_loader = DataLoader()
        self.data_processor = DataProcessor()
        self._apply_style()
    
    def _apply_style(self):
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
            'font.size': 14,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 10,
            'legend.fontsize': 14,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.linewidth': 1.5,
            'axes.edgecolor': self.OKABE_ITO['dark_gray'],
        })
    
    def discover_replica_folders(self, base_path: str) -> Dict[str, Dict[str, Path]]:
        """Discover replica 1 folders for all complexes and splits."""
        base_dir = Path(base_path)
        discovered = {}
        
        for complex_name in self.COMPLEXES:
            complex_dir = base_dir / complex_name
            if not complex_dir.exists():
                self.logger.warning(f"Complex directory not found: {complex_dir}")
                continue
            
            discovered[complex_name] = {}
            
            for split in self.SPLITS:
                split_dir = complex_dir / f"result_{split}"
                if not split_dir.exists():
                    self.logger.warning(f"Split directory not found: {split_dir}")
                    continue
                
                replica_path = self._find_target_replica(split_dir)
                if replica_path:
                    discovered[complex_name][split] = replica_path
                    self.logger.info(f"Found {self.TARGET_REPLICA} for {complex_name}/{split}")
                else:
                    self.logger.warning(f"Could not find {self.TARGET_REPLICA} for {complex_name}/{split}")
        
        return discovered
    
    def _find_target_replica(self, split_dir: Path) -> Optional[Path]:
        """Find the target replica folder by reading metadata files."""
        for results_dir in split_dir.glob("interfacea-interchain_results_*"):
            if not results_dir.is_dir():
                continue
            
            metadata_files = list(results_dir.glob("interfacea-interchain_metadata_*.txt"))
            if not metadata_files:
                continue
            
            metadata_file = metadata_files[0]
            try:
                with open(metadata_file, 'r') as f:
                    content = f.read()
                
                match = re.search(r'replica:\s*(\w+)', content)
                if match and match.group(1) == self.TARGET_REPLICA:
                    return results_dir
            except Exception as e:
                self.logger.warning(f"Error reading metadata {metadata_file}: {e}")
        
        return None
    
    def _find_output_file(self, results_dir: Path) -> Optional[Path]:
        """Find the prediction output file in the outputs directory."""
        outputs_dir = results_dir / "outputs"
        if not outputs_dir.exists():
            return None
        
        prediction_files = list(outputs_dir.glob("*_prediction_set_*.txt"))
        if prediction_files:
            return prediction_files[0]
        
        prediction_files = list(outputs_dir.glob("*.txt"))
        if prediction_files:
            return prediction_files[0]
        
        return None
    
    def load_and_process_data(self, results_dir: Path) -> Optional[ProcessedData]:
        """Load and process data from a results directory."""
        preprocess_dir = results_dir / "preprocess"
        
        if not preprocess_dir.exists():
            self.logger.warning(f"Preprocess directory not found: {preprocess_dir}")
            return None
        
        output_file = self._find_output_file(results_dir)
        if output_file is None:
            self.logger.warning(f"Output file not found in {results_dir / 'outputs'}")
            return None
        
        try:
            loaded_data = self.data_loader.load_all_data(
                input_directory=str(preprocess_dir),
                output_file_path=str(output_file)
            )
            processed_data = self.data_processor.process_all_data(loaded_data)
            return processed_data
        except Exception as e:
            self.logger.error(f"Error processing data from {results_dir}: {e}")
            return None
    
    def _process_data_for_heatmap(
        self, 
        processed_data: ProcessedData, 
        num_pairs: int = 30
    ) -> Tuple[pd.DataFrame, int]:
        """Process data into overlay matrix for heatmap."""
        gt_full = processed_data.ground_truth_full
        pred_full = processed_data.predictions_full
        train_set = processed_data.train_set_processed
        valid_set = processed_data.valid_set_processed
        
        gt_pivot = gt_full.pivot(index='pair', columns='time_stamp', values='present')
        pred_pivot = pred_full.pivot(index='pair', columns='time_stamp', values='present')
        
        common_pairs = gt_pivot.index.intersection(pred_pivot.index)
        gt_pivot = gt_pivot.loc[common_pairs].fillna(0).astype(int)
        pred_pivot = pred_pivot.loc[common_pairs].fillna(0).astype(int)
        
        if len(common_pairs) > num_pairs:
            pair_freq = gt_pivot.sum(axis=1).sort_values(ascending=False)
            selected_pairs = pair_freq.head(num_pairs).index
            gt_pivot = gt_pivot.loc[selected_pairs]
            pred_pivot = pred_pivot.loc[selected_pairs]
        else:
            selected_pairs = common_pairs
        
        gt_pivot = gt_pivot.sort_index()
        pred_pivot = pred_pivot.loc[gt_pivot.index]
        
        overlay_test = pd.DataFrame(0, index=gt_pivot.index, columns=gt_pivot.columns)
        overlay_test[(gt_pivot == 1) & (pred_pivot == 0)] = 1  # FN
        overlay_test[(gt_pivot == 0) & (pred_pivot == 1)] = 2  # FP
        overlay_test[(gt_pivot == 1) & (pred_pivot == 1)] = 3  # TP
        
        history_offset = 0
        if not train_set.empty or not valid_set.empty:
            history_dfs = []
            if not train_set.empty:
                history_dfs.append(train_set)
            if not valid_set.empty:
                history_dfs.append(valid_set)
            
            combined_history = pd.concat(history_dfs)
            combined_history = combined_history[combined_history['pair'].isin(selected_pairs)].copy()
            
            if not combined_history.empty:
                combined_history['present'] = 1
                history_pivot = combined_history.pivot_table(
                    index='pair', columns='time_stamp', values='present', fill_value=0
                )
                history_pivot = history_pivot.reindex(gt_pivot.index, fill_value=0)
                history_pivot = history_pivot.reindex(sorted(history_pivot.columns), axis=1)
                
                history_overlay = history_pivot.replace({0: 0, 1: 4})
                overlay_combined = pd.concat([history_overlay, overlay_test], axis=1)
                history_offset = history_pivot.shape[1]
                return overlay_combined, history_offset
        
        return overlay_test, history_offset
    
    def _create_time_formatter(self, timestamps):
        def formatter(x, pos):
            try:
                idx = int(x)
                if 0 <= idx < len(timestamps):
                    return f"{timestamps[idx] * self.NS_PER_TIMESTEP:.0f}"
                return ""
            except:
                return ""
        return formatter
    
    def plot_heatmaps_by_complex(
        self, 
        discovered_paths: Dict[str, Dict[str, Path]],
        num_pairs: int = 30
    ) -> List[str]:
        """Generate separate heatmap figures for each complex (one file per complex)."""
        cmap_overlay = mcolors.ListedColormap([
            self.OKABE_ITO['light_gray'],
            self.OKABE_ITO['sky_blue'],
            self.OKABE_ITO['vermilion'],
            self.OKABE_ITO['bluish_green'],
            self.OKABE_ITO['neutral_gray']
        ])
        bounds = [0, 1, 2, 3, 4, 5]
        norm = mcolors.BoundaryNorm(bounds, cmap_overlay.N)
        
        output_paths = []
        
        for complex_name in self.COMPLEXES:
            if complex_name not in discovered_paths:
                self.logger.warning(f"No data for complex {complex_name}, skipping")
                continue
            
            plot_data: List[Dict[str, Any]] = []
            for split, split_label in zip(self.SPLITS, self.SPLIT_LABELS):
                entry = {
                    'split': split,
                    'split_label': split_label,
                    'overlay_matrix': None,
                    'history_offset': 0,
                    'n_rows': 1
                }
                
                if split in discovered_paths[complex_name]:
                    results_dir = discovered_paths[complex_name][split]
                    self.logger.info(f"Loading {complex_name}/{split}...")
                    processed_data = self.load_and_process_data(results_dir)
                    
                    if processed_data is not None:
                        overlay_matrix, history_offset = self._process_data_for_heatmap(processed_data, num_pairs)
                        entry['overlay_matrix'] = overlay_matrix
                        entry['history_offset'] = history_offset
                        entry['n_rows'] = len(overlay_matrix.index)
                        self.logger.info(f"  {complex_name}/{split}: {entry['n_rows']} pairs")
                
                plot_data.append(entry)
            
            height_ratios = [max(d['n_rows'], 5) for d in plot_data]
            
            cell_height = 0.08
            total_data_height = sum(height_ratios) * cell_height
            title_space = len(plot_data) * 0.4
            legend_space = 1.2
            fig_height = total_data_height + title_space + legend_space
            
            fig = plt.figure(figsize=(16, fig_height), dpi=300)
            gs = gridspec.GridSpec(len(plot_data), 1, height_ratios=height_ratios, hspace=0.25)
            
            for row_idx, entry in enumerate(plot_data):
                ax = fig.add_subplot(gs[row_idx])
                is_last_row = (row_idx == len(plot_data) - 1)
                split_label = entry['split_label']
                overlay_matrix = entry['overlay_matrix']
                history_offset = entry['history_offset']
                
                if overlay_matrix is None:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                           fontsize=14, color='gray', transform=ax.transAxes)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_title(f'{split_label} Split', fontsize=16, pad=8)
                    continue
                
                sns.heatmap(
                    overlay_matrix.fillna(0).astype(int),
                    ax=ax,
                    cmap=cmap_overlay,
                    norm=norm,
                    cbar=False,
                    yticklabels=overlay_matrix.index,
                    xticklabels=False,
                    linewidths=0.02,
                    linecolor='white',
                    square=False
                )
                
                timestamps = overlay_matrix.columns
                ax.xaxis.set_major_formatter(FuncFormatter(self._create_time_formatter(timestamps)))
                ax.xaxis.set_major_locator(MultipleLocator(10))
                ax.tick_params(axis='x', labelsize=10, rotation=0)
                
                num_labels = len(overlay_matrix.index)
                y_fontsize = max(5, min(8, int(120 / num_labels)))
                ax.tick_params(axis='y', labelsize=y_fontsize)
                
                if history_offset > 0:
                    ax.axvline(x=history_offset, color=self.OKABE_ITO['dark_gray'],
                              linestyle='--', linewidth=1.5, alpha=0.7)
                
                ax.set_title(f'{split_label} Split', fontsize=16, pad=8)
                ax.set_ylabel('Residue Pair', fontsize=14)
                
                if is_last_row:
                    ax.set_xlabel('Simulation time (ns)', fontsize=14)
                else:
                    ax.set_xlabel('')
                
                label = chr(ord('A') + row_idx)
                ax.text(-0.08, 1.15, label, transform=ax.transAxes, fontsize=18,
                       fontweight='bold', va='top', ha='left')
            
            legend_elements = [
                Patch(facecolor=self.OKABE_ITO['light_gray'], label='TN'),
                Patch(facecolor=self.OKABE_ITO['sky_blue'], label='FN'),
                Patch(facecolor=self.OKABE_ITO['vermilion'], label='FP'),
                Patch(facecolor=self.OKABE_ITO['bluish_green'], label='TP'),
                Patch(facecolor=self.OKABE_ITO['neutral_gray'], label='History'),
            ]
            fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02),
                      ncol=5, frameon=False, fontsize=14)
            
            plt.subplots_adjust(bottom=0.06, top=0.97, left=0.15, right=0.95)
            
            output_path = self.output_directory / f"heatmap_{complex_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            svg_path = self.output_directory / f"heatmap_{complex_name}.svg"
            plt.savefig(svg_path, format='svg', bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Generated heatmap for {complex_name}: {output_path}")
            output_paths.append(str(output_path))
        
        return output_paths


def main():
    parser = argparse.ArgumentParser(
        description="Generate combined heatmap visualization across complexes and splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.analysis.visualization.combined_heatmap --results_dir ./results_all
  python -m src.analysis.visualization.combined_heatmap --results_dir ./results_all --output heatmap_plots --pairs 40
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
        default='combined_heatmap_analysis',
        help='Output directory for plots (default: combined_heatmap_analysis)'
    )
    parser.add_argument(
        '--pairs',
        type=int,
        default=30,
        help='Number of pairs to show per heatmap (default: 30)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level=log_level)
    logger = get_logger(__name__)
    
    try:
        logger.info("Starting combined heatmap generation")
        logger.info(f"Results directory: {args.results_dir}")
        logger.info(f"Output directory: {args.output}")
        
        plotter = CombinedHeatmapPlotter(args.output)
        
        discovered = plotter.discover_replica_folders(args.results_dir)
        
        if not discovered:
            logger.error("No data found. Check your results directory structure.")
            return 1
        
        logger.info("=" * 60)
        logger.info("DISCOVERED DATA SUMMARY")
        logger.info("=" * 60)
        for complex_name, splits in discovered.items():
            for split, path in splits.items():
                logger.info(f"  {complex_name}/{split}: {path}")
        logger.info("=" * 60)
        
        output_paths = plotter.plot_heatmaps_by_complex(discovered, num_pairs=args.pairs)
        
        if output_paths:
            for path in output_paths:
                logger.info(f"Successfully generated: {path}")
        else:
            logger.warning("No plots were generated")
        
        logger.info("Combined heatmap generation completed")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
