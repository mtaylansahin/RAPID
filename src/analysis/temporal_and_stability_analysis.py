#!/usr/bin/env python3
"""
Temporal Dynamics and Interaction Stability Analysis

This script provides focused analysis on:
1. Temporal dynamics with separate plots for each complex system
2. Interaction stability distribution analysis with 2x2 grid layout

Combines the best visualizations from comprehensive_protein_analysis.py and improved_analysis_plots.py
"""

import argparse
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
import warnings
from matplotlib.patches import Patch, Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
    'font.size': 20,
    'axes.titlesize': 26,
    'axes.labelsize': 22,
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
    
    # Grid and background
    'axes.grid': True,
    'grid.alpha': 0.2,     # Lighter grid
    'grid.linewidth': 0.6,
    'grid.color': '#BDC3C7',
    'axes.axisbelow': True,
    
    # Lines and patches
    'lines.linewidth': 2.0,
    'lines.solid_capstyle': 'round',
    'patch.linewidth': 1.0,
    'patch.edgecolor': 'white',
    
    # Figure settings
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    
    # Text and annotations
    'text.color': '#2C3E50',
    'axes.labelcolor': '#2C3E50',
    'xtick.color': '#2C3E50',
    'ytick.color': '#2C3E50',
    
    # Legend styling
    'legend.frameon': False,
    'legend.fancybox': True,
    'legend.shadow': False,
    'legend.borderpad': 0.5
})

class TemporalStabilityAnalysis:
    """
    Focused analysis class for temporal dynamics and interaction stability
    """

    TIME_STEP_NS = 0.5
    
    def __init__(self, data_dir="data", output_dir="temporal_stability_results"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.all_data = {}
        
        self.stability_palette = {
            'Rare': '#D55E00',      # Vermilion (Okabe-Ito)
            'Transient': '#E69F00',  # Orange (Okabe-Ito)
            'Stable': '#009E73'     # Bluish Green (Okabe-Ito)
        }
        
        self.publication_colors = {
            'primary': '#2C3E50',    # Dark blue-gray for text/lines
            'secondary': '#34495E',  # Lighter blue-gray for accents
            'background': '#ECF0F1', # Light gray for backgrounds
            'grid': '#BDC3C7',       # Medium gray for grids
            'highlight': '#3498DB'   # Blue for highlights
        }
        
        self.replica_colors = [
            '#D55E00',  # Vermilion
            '#0072B2',  # Blue
            '#009E73',  # Bluish Green
            '#E69F00',  # Orange
        ]
        
        print(f"Initialized Temporal & Stability Analysis for data in: {self.data_dir}")
        print(f"Results will be saved to: {self.output_dir}")

    @staticmethod
    def _smooth_series(series: pd.Series, window_size: int) -> pd.Series:
        """Apply centered rolling mean when enough points are available."""
        if len(series) > window_size:
            return series.rolling(window=window_size, center=True).mean()
        return series

    @staticmethod
    def _format_replica_label(replica_name: str) -> str:
        """Return human-friendly replica label."""
        if 'replica' in replica_name.lower():
            nums = re.findall(r'\d+', replica_name)
            if nums:
                return f"Replica {nums[0]}"
        return replica_name.capitalize() if replica_name else replica_name

    @staticmethod
    def _format_complex_label(complex_name: str) -> str:
        """Return publication-ready complex label."""
        if not complex_name:
            return complex_name

        base = complex_name.replace("_interchain", "")
        base_upper = base.upper()

        if "1EAW" in base_upper:
            return "Enzyme-inhibitor complex"
        if "1JPS" in base_upper:
            return "Antibody-antigen complex"
        return base

    def _prioritized_complexes(self) -> List[str]:
        """Order complexes with 1JPS first, then 1EAW, then any others."""
        keys = sorted(self.all_data.keys())
        ordered = []
        for target in ("1JPS", "1EAW"):
            ordered.extend([k for k in keys if target in k.upper() and k not in ordered])
        ordered.extend([k for k in keys if k not in ordered])
        return ordered

    def load_all_data(self):
        """Load all interaction data from all complexes and replicas"""
        print("Loading all interaction data...")
        
        for complex_dir in self.data_dir.iterdir():
            if complex_dir.is_dir() and not complex_dir.name.startswith('.'):
                complex_name = complex_dir.name
                print(f"Processing complex: {complex_name}")
                
                self.all_data[complex_name] = {}
                
                for replica_dir in complex_dir.iterdir():
                    if replica_dir.is_dir() and replica_dir.name.startswith('replica'):
                        replica_name = replica_dir.name
                        print(f"  Processing {replica_name}")
                        
                        # Find the interfacea directory
                        interfacea_dirs = list(replica_dir.glob("*interfacea*"))
                        if interfacea_dirs:
                            interfacea_dir = interfacea_dirs[0]
                            data = self._load_replica_data(interfacea_dir)
                            self.all_data[complex_name][replica_name] = data
                        
        print(f"Loaded data for {len(self.all_data)} complexes")
        return self.all_data
    
    def _load_replica_data(self, interfacea_dir):
        """Load data from a single replica"""
        replica_data = []
        
        interfacea_files = list(interfacea_dir.glob("*.interfacea"))
        interfacea_files.sort(key=lambda x: int(re.findall(r'\d+', x.name)[0]))
        
        for i, file_path in enumerate(interfacea_files):
            try:
                # Extract time stamp from filename
                time_stamp = int(re.findall(r'\d+', file_path.name)[0])
                
                # Read the file
                df = pd.read_csv(file_path, sep=r'\s+', header=0,
                               names=['itype', 'chain_a', 'chain_b', 'resname_a', 'resname_b',
                                     'resid_a', 'resid_b', 'atom_a', 'atom_b'])
                
                if not df.empty:
                    df['time_stamp'] = time_stamp
                    df['snapshot'] = i
                    replica_data.append(df)
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if replica_data:
            return pd.concat(replica_data, ignore_index=True)
        else:
            return pd.DataFrame()

    def analyze_temporal_dynamics(self):
        """
        Analyze temporal dynamics with smoothed visualizations - merged 2x2 plot for 1JPS and 1EAW
        """
        print("=" * 80)
        print("TEMPORAL DYNAMICS ANALYSIS (MERGED)")
        print("=" * 80)
        
        window_size = 10
        
        plot_complexes = self._prioritized_complexes()[:2]
        
        print(f"Generating merged plot for: {', '.join(plot_complexes)}")
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14), sharex=True)
        
        legend_handles = []
        legend_labels = []
        
        for row_idx, complex_name in enumerate(plot_complexes):
            complex_data = self.all_data[complex_name]
            clean_name = self._format_complex_label(complex_name)
            
            axes[row_idx, 1].annotate(clean_name, 
                                    xy=(1.03, 0.5), xytext=(0, 0),
                                    xycoords='axes fraction', textcoords='offset points',
                                    size=22, ha='left', va='center', rotation=-90, fontweight='bold')
            
            for replica_idx, (replica_name, replica_data) in enumerate(complex_data.items()):
                if replica_data.empty:
                    continue
                
                color = self.replica_colors[replica_idx % len(self.replica_colors)]
                
                dedup_df = replica_data.copy()
                chains_a = dedup_df['chain_a'].astype(str)
                chains_b = dedup_df['chain_b'].astype(str)
                resid_a_int = dedup_df['resid_a'].astype(int)
                resid_b_int = dedup_df['resid_b'].astype(int)
                a_first = (chains_a < chains_b) | ((chains_a == chains_b) & (resid_a_int <= resid_b_int))
                dedup_df['n_chain1'] = np.where(a_first, chains_a, chains_b)
                dedup_df['n_resid1'] = np.where(a_first, resid_a_int, resid_b_int)
                dedup_df['n_chain2'] = np.where(a_first, chains_b, chains_a)
                dedup_df['n_resid2'] = np.where(a_first, resid_b_int, resid_a_int)

                unique_per_ts = dedup_df.drop_duplicates(
                    subset=['time_stamp', 'itype', 'n_chain1', 'n_resid1', 'n_chain2', 'n_resid2']
                )
                time_counts = unique_per_ts.groupby('time_stamp').size()
                
                time_counts.index = time_counts.index * self.TIME_STEP_NS
                time_counts_smooth = self._smooth_series(time_counts, window_size)
                
                ax_total = axes[row_idx, 0]
                ax_total.plot(time_counts.index, time_counts.values, 
                              alpha=0.18, linewidth=0.8, color=color)
                
                label_clean = self._format_replica_label(replica_name)
                
                line, = ax_total.plot(time_counts.index, time_counts_smooth.values, 
                              label=label_clean, linewidth=2.0, alpha=0.9, color=color)
                
                # Collect legend handles from first complex only
                if row_idx == 0:
                    legend_handles.append(line)
                    legend_labels.append(label_clean)

                # --- Plotting Interface Size (Column 1) ---
                interface_residues = unique_per_ts.groupby('time_stamp').apply(
                    lambda x: len(set(x['n_resid1']).union(set(x['n_resid2']))))
                
                interface_residues.index = interface_residues.index * self.TIME_STEP_NS
                
                interface_smooth = self._smooth_series(interface_residues, window_size)
                
                ax_size = axes[row_idx, 1]
                ax_size.yaxis.set_major_locator(MaxNLocator(integer=True))
                
                ax_size.plot(interface_residues.index, interface_residues.values, 
                              alpha=0.18, linewidth=0.8, color=color)
                ax_size.plot(interface_residues.index, interface_smooth.values, 
                              label=replica_name, linewidth=2.0, alpha=0.9, color=color)

        axes[0, 0].set_title("Total Interactions", fontweight='bold', pad=15)
        axes[0, 1].set_title("Interface Size", fontweight='bold', pad=15)
        
        for ax in axes.flatten():
            ax.grid(visible=True, axis='y', alpha=0.3, linestyle='-')
            ax.grid(visible=False, axis='x')
        
        axes[0, 0].set_ylabel('Interaction Count')
        axes[1, 0].set_ylabel('Interaction Count')
        axes[0, 1].set_ylabel('Residue Count')
        axes[1, 1].set_ylabel('Residue Count')

        axes[1, 0].set_xlabel('Simulation time (ns)')
        axes[1, 1].set_xlabel('Simulation time (ns)')
        
        panel_labels = ['A', 'B', 'C', 'D']
        for i, ax in enumerate(axes.flatten()):
            ax.text(-0.15, 1.05, panel_labels[i], transform=ax.transAxes, 
                   fontsize=24, fontweight='bold', va='top', ha='right')

        # Unified Legend
        if legend_handles:
            fig.legend(handles=legend_handles, labels=legend_labels, 
                      loc='lower center', bbox_to_anchor=(0.5, 0.02),
                      ncol=min(len(legend_handles), 4), frameon=False,
                      fontsize=22)

        plt.subplots_adjust(bottom=0.12, left=0.08, right=0.92, top=0.92, wspace=0.2, hspace=0.1)
        
        # Save as SVG (Vector)
        svg_path = self.output_dir / 'temporal_dynamics_merged.svg'
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"✓ Saved merged temporal dynamics plot to {svg_path}")
        
        # Also save PNG for quick preview/compatibility
        png_path = self.output_dir / 'temporal_dynamics_merged.png'
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
        
        # Close figure to free memory
        plt.close(fig)

    def plot_interaction_stability(self):
        """
        Generate enhanced interaction stability analysis with grouped layout for publication.
        Creates a single figure with 2x4 grid (2 complexes × 4 replicas) for efficient space usage.
        """
        print("\n🎯 Generating publication-ready interaction stability plot with grouped layout...")
        
        complex_names = self._prioritized_complexes()
            
        print(f"Creating grouped stability plot for {len(complex_names)} complexes...")
        
        fig = plt.figure(figsize=(24, 14))
        gs = GridSpec(2, 4, hspace=0.1, wspace=0.1, 
                     left=0.08, right=0.92, top=0.90, bottom=0.10)

        all_complex_stats = {}
        
        for complex_idx, complex_name in enumerate(complex_names):
            replicas = self.all_data[complex_name]
            if not replicas:
                continue
            
            complex_display_name = self._format_complex_label(complex_name)
            replica_list = sorted(list(replicas.items()))
            
            complex_stats = {
                'rare_counts': [],
                'transient_counts': [],
                'stable_counts': [],
                'total_pairs': [],
                'mean_frequencies': [],
                'median_frequencies': []
            }
            
            row_axes = []
            
            for replica_idx, (replica_name, data) in enumerate(replica_list):
                if replica_idx >= 4:
                    break
                    
                ax = fig.add_subplot(gs[complex_idx, replica_idx])
                row_axes.append(ax)
                
                if data.empty:
                    ax.text(0.5, 0.5, "No Data", 
                           ha="center", va="center", fontsize=16, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                    continue

                # Calculate interaction frequencies for this replica
                total_timepoints = data["time_stamp"].nunique()
                
                data_clean = data.drop_duplicates()
                
                pair_timepoint_counts = data_clean.groupby(["resid_a", "resid_b"])["time_stamp"].nunique()
                frequencies = (pair_timepoint_counts / total_timepoints) * 100

                bins = np.arange(0, 101, 5)
                counts, bin_edges, patches = ax.hist(
                    frequencies, bins=bins, alpha=0.9, 
                    edgecolor='white', linewidth=0.5, density=False
                )
                
                for patch, bin_start, bin_end in zip(patches, bin_edges[:-1], bin_edges[1:]):
                    bin_center = (bin_start + bin_end) / 2
                    if bin_center < 5:
                        patch.set_facecolor(self.stability_palette['Rare'])
                    elif bin_center < 50:
                        patch.set_facecolor(self.stability_palette['Transient'])
                    else:
                        patch.set_facecolor(self.stability_palette['Stable'])
                
                rare_count = (frequencies < 5).sum()
                transient_count = ((frequencies >= 5) & (frequencies <= 50)).sum()
                stable_count = (frequencies > 50).sum()
                total_pairs = len(frequencies)

                complex_stats['rare_counts'].append(rare_count)
                complex_stats['transient_counts'].append(transient_count)
                complex_stats['stable_counts'].append(stable_count)
                complex_stats['total_pairs'].append(total_pairs)
                complex_stats['mean_frequencies'].append(frequencies.mean())
                complex_stats['median_frequencies'].append(frequencies.median())

                if complex_idx == 0:
                    label_clean = self._format_replica_label(replica_name)
                    ax.set_title(label_clean, fontsize=26, fontweight='bold', pad=15)
                
                if replica_idx == 3:
                    ax.annotate(complex_display_name, 
                               xy=(1.05, 0.5), xytext=(0, 0),
                               xycoords='axes fraction', textcoords='offset points',
                               size=24, ha='left', va='center', rotation=-90, fontweight='bold')

                if complex_idx != len(complex_names) - 1:
                    ax.set_xticklabels([])
                    
                if replica_idx != 0:
                    ax.set_yticklabels([])
                
                stats_text = f"n={total_pairs:,}"
                props = dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=0.5)
                ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=18,
                       verticalalignment='top', horizontalalignment='right', bbox=props)
                
                ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
                ax.set_xlim(0, 100)
                
            uniform_ylim_max = 30
            for row_ax in row_axes:
                row_ax.set_ylim(0, uniform_ylim_max)
                row_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            
            all_complex_stats[complex_name] = complex_stats
        
        legend_elements = [
            Patch(facecolor=self.stability_palette['Rare'], 
                  label="Rare Interactions (<5%)", alpha=0.9),
            Patch(facecolor=self.stability_palette['Transient'], 
                  label="Transient Interactions (5-50%)", alpha=0.9),
            Patch(facecolor=self.stability_palette['Stable'], 
                  label="Stable Interactions (>50%)", alpha=0.9),
        ]
        
        fig.legend(handles=legend_elements, 
                  loc='lower center', 
                  bbox_to_anchor=(0.5, -0.035),
                  ncol=3, 
                  frameon=False, 
                  fontsize=22)
        
        fig.text(0.05, 0.5, 'Interaction Count', rotation=90, ha='center', va='center', fontsize=22)
        fig.text(0.5, 0.045, 'Interaction Frequency (%)', ha='center', va='center', fontsize=22)
        
        fig.text(0.02, 0.90, 'A', fontsize=28, fontweight='bold', ha='left', va='top')
        fig.text(0.02, 0.50, 'B', fontsize=28, fontweight='bold', ha='left', va='top')

        plt.tight_layout(rect=[0.08, 0.15, 0.92, 0.95])
        
        plt.savefig(self.output_dir / 'interaction_stability_2x4_grid.png', 
                   dpi=300, bbox_inches='tight', facecolor='white', 
                   format='png')
        
        plt.savefig(self.output_dir / 'interaction_stability_2x4_grid.svg', 
                   bbox_inches='tight', facecolor='white', format='svg')
        
        plt.close(fig)
        
        # Print summary statistics
        print(f"\n📊 PUBLICATION SUMMARY:")
        print("=" * 60)
        for complex_name, stats in all_complex_stats.items():
            if stats['total_pairs']:
                display_name = complex_name.replace('_interchain', '')
                total_pairs = sum(stats['total_pairs'])
                avg_rare = np.mean(stats['rare_counts'])
                avg_transient = np.mean(stats['transient_counts'])
                avg_stable = np.mean(stats['stable_counts'])
                
                print(f"{display_name}:")
                print(f"  Total interaction pairs: {total_pairs:,}")
                print(f"  Average per replica - Rare: {avg_rare:.1f}, Transient: {avg_transient:.1f}, Stable: {avg_stable:.1f}")
                print(f"  Mean frequency: {np.mean(stats['mean_frequencies']):.1f}% ± {np.std(stats['mean_frequencies']):.1f}%")
        
        print(f"\n✅ Generated publication-ready grouped stability plot (2×4 grid)")
        print(f"📁 Saved as PNG (raster) and SVG (vector) formats")
        
        return all_complex_stats  # Return stats for potential cross-complex analysis

    def plot_cross_complex_comparison(self, complex_stats=None):
        if complex_stats is None:
            print("No statistics provided. Run plot_interaction_stability first.")
            return
            
        print("\n📊 Generating cross-complex comparison plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Cross-Complex Interaction Stability Comparison", 
                    fontsize=18, fontweight='bold', y=0.95)
        
        complex_names = list(complex_stats.keys())
        display_names = [self._format_complex_label(name) for name in complex_names]
        
        ax1 = axes[0, 0]
        rare_props = []
        transient_props = []
        stable_props = []
        
        for complex_name in complex_names:
            stats = complex_stats[complex_name]
            if stats['total_pairs']:
                total_pairs = sum(stats['total_pairs'])
                rare_prop = (sum(stats['rare_counts']) / total_pairs) * 100
                transient_prop = (sum(stats['transient_counts']) / total_pairs) * 100
                stable_prop = (sum(stats['stable_counts']) / total_pairs) * 100
                
                rare_props.append(rare_prop)
                transient_props.append(transient_prop)
                stable_props.append(stable_prop)
        
        width = 0.6
        x_pos = np.arange(len(display_names))
        
        bars1 = ax1.bar(x_pos, rare_props, width, 
                       label='Rare (<10%)', color=self.stability_palette['Rare'], alpha=0.8)
        bars2 = ax1.bar(x_pos, transient_props, width, bottom=rare_props,
                       label='Transient (10-50%)', color=self.stability_palette['Transient'], alpha=0.8)
        bars3 = ax1.bar(x_pos, stable_props, width, 
                       bottom=np.array(rare_props) + np.array(transient_props),
                       label='Stable (>50%)', color=self.stability_palette['Stable'], alpha=0.8)
        
        ax1.set_xlabel('Protein Complex', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Proportion of Interactions (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Interaction Stability Distribution', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(display_names)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        for i, (rare, trans, stable) in enumerate(zip(rare_props, transient_props, stable_props)):
            if rare > 5:
                ax1.text(i, rare/2, f'{rare:.1f}%', ha='center', va='center', 
                        fontsize=10, fontweight='bold', color='white')
            if trans > 5:
                ax1.text(i, rare + trans/2, f'{trans:.1f}%', ha='center', va='center', 
                        fontsize=10, fontweight='bold', color='white')
            if stable > 5:
                ax1.text(i, rare + trans + stable/2, f'{stable:.1f}%', ha='center', va='center', 
                        fontsize=10, fontweight='bold', color='white')
        
        ax2 = axes[0, 1]
        mean_freqs = []
        std_freqs = []
        
        for complex_name in complex_names:
            stats = complex_stats[complex_name]
            if stats['mean_frequencies']:
                mean_freqs.append(np.mean(stats['mean_frequencies']))
                std_freqs.append(np.std(stats['mean_frequencies']))
        
        bars = ax2.bar(x_pos, mean_freqs, width, yerr=std_freqs, capsize=5,
                      color=self.publication_colors['highlight'], alpha=0.7,
                      error_kw={'linewidth': 2, 'capthick': 2})
        
        ax2.set_xlabel('Protein Complex', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Mean Interaction Frequency (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Average Interaction Frequency', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(display_names)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for i, (mean, std) in enumerate(zip(mean_freqs, std_freqs)):
            ax2.text(i, mean + std + 1, f'{mean:.1f}±{std:.1f}%', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax3 = axes[1, 0]
        total_pairs_per_complex = []
        
        for complex_name in complex_names:
            stats = complex_stats[complex_name]
            if stats['total_pairs']:
                total_pairs_per_complex.append(sum(stats['total_pairs']))
        
        bars = ax3.bar(x_pos, total_pairs_per_complex, width,
                      color=self.publication_colors['secondary'], alpha=0.7)
        
        ax3.set_xlabel('Protein Complex', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Total Interaction Pairs', fontsize=12, fontweight='bold')
        ax3.set_title('Data Volume Comparison', fontsize=14, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(display_names)
        ax3.grid(True, alpha=0.3, axis='y')
        
        for i, total in enumerate(total_pairs_per_complex):
            ax3.text(i, total + max(total_pairs_per_complex) * 0.01, f'{total:,}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = []
        summary_text.append("CROSS-COMPLEX STATISTICAL SUMMARY")
        summary_text.append("=" * 40)
        summary_text.append("")
        
        for i, complex_name in enumerate(complex_names):
            stats = complex_stats[complex_name]
            display_name = display_names[i]
            
            if stats['total_pairs']:
                total_pairs = sum(stats['total_pairs'])
                mean_freq = np.mean(stats['mean_frequencies'])
                std_freq = np.std(stats['mean_frequencies'])
                
                # Stability proportions
                rare_prop = (sum(stats['rare_counts']) / total_pairs) * 100
                transient_prop = (sum(stats['transient_counts']) / total_pairs) * 100  
                stable_prop = (sum(stats['stable_counts']) / total_pairs) * 100
                
                summary_text.append(f"{display_name}:")
                summary_text.append(f"  Total pairs: {total_pairs:,}")
                summary_text.append(f"  Mean frequency: {mean_freq:.1f}% ± {std_freq:.1f}%")
                summary_text.append(f"  Stability: {rare_prop:.1f}% rare, {transient_prop:.1f}% transient, {stable_prop:.1f}% stable")
                summary_text.append("")
        
        if len(mean_freqs) == 2:
            from scipy import stats
            try:
                t_stat, p_value = stats.ttest_ind(
                    complex_stats[complex_names[0]]['mean_frequencies'],
                    complex_stats[complex_names[1]]['mean_frequencies']
                )
                summary_text.append("STATISTICAL COMPARISON:")
                summary_text.append(f"t-statistic: {t_stat:.3f}")
                summary_text.append(f"p-value: {p_value:.3f}")
                summary_text.append(f"Significance: {'**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'}")
            except ImportError:
                summary_text.append("STATISTICAL COMPARISON:")
                summary_text.append("(scipy not available for statistical tests)")
        
        # Display summary text
        ax4.text(0.05, 0.95, '\n'.join(summary_text), 
                transform=ax4.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=self.publication_colors['background'], alpha=0.8))
        
        # Save publication-quality comparison plot
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(self.output_dir / 'cross_complex_comparison.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'cross_complex_comparison.pdf', 
                   bbox_inches='tight', facecolor='white', format='pdf')
        plt.show()
        
        print("✅ Generated cross-complex comparison plot for publication")

    def generate_summary_report(self):
        """Generate a summary report of the analysis"""
        report = []
        report.append("=" * 80)
        report.append("TEMPORAL DYNAMICS AND STABILITY ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Data overview
        report.append("1. DATA OVERVIEW")
        report.append("-" * 40)
        total_complexes = len(self.all_data)
        total_replicas = sum(len(complex_data) for complex_data in self.all_data.values())
        
        report.append(f"Total Complexes Analyzed: {total_complexes}")
        report.append(f"Total Replicas: {total_replicas}")
        report.append("")
        
        total_interactions = 0
        for complex_name, complex_data in self.all_data.items():
            report.append(f"{complex_name}:")
            complex_total = 0
            for replica_name, replica_data in complex_data.items():
                if not replica_data.empty:
                    n_interactions = len(replica_data)
                    n_timepoints = len(replica_data['time_stamp'].unique())
                    n_pairs = len(replica_data[['resid_a', 'resid_b']].drop_duplicates())
                    
                    report.append(f"  {replica_name}: {n_interactions:,} interactions, "
                                f"{n_timepoints} timepoints, {n_pairs} unique pairs")
                    complex_total += n_interactions
            report.append(f"  Total: {complex_total:,} interactions")
            total_interactions += complex_total
            report.append("")
        
        report.append(f"OVERALL TOTAL: {total_interactions:,} interactions")
        report.append("")
        
        # Analysis summary
        report.append("2. ANALYSIS RESULTS")
        report.append("-" * 40)
        report.append("")
        
        report.append("Temporal Dynamics Analysis:")
        report.append("✓ Individual plots for each protein complex system")
        report.append("✓ Smoothed interaction counts over time")
        report.append("✓ Interface stability and size tracking")
        report.append("✓ Interaction diversity patterns")
        report.append("✓ System-specific stability assessments")
        report.append("")
        
        report.append("Interaction Stability Analysis:")
        report.append("✓ Separate 2x2 grid layouts for each complex system")
        report.append("✓ Stability categories: Rare (<10%), Transient (10-50%), Stable (>50%)")
        report.append("✓ Statistical summaries for each replica")
        report.append("✓ Enhanced visualizations with color coding")
        report.append("")
        
        # Files generated
        report.append("3. OUTPUT FILES GENERATED")
        report.append("-" * 40)
        output_files = list(self.output_dir.glob("*.png"))
        for file_path in sorted(output_files):
            report.append(f"• {file_path.name}")
        report.append("")
        
        report.append("=" * 80)
        
        # Save and display report
        report_text = "\n".join(report)
        with open(self.output_dir / "temporal_stability_report.txt", "w") as f:
            f.write(report_text)
        
        print(report_text)
        return report_text

    def run_complete_analysis(self):
        """Run the complete focused analysis pipeline"""
        print("=" * 80)
        print("STARTING TEMPORAL DYNAMICS & STABILITY ANALYSIS")
        print("=" * 80)
        
        # Load all data
        self.load_all_data()
        
        if not self.all_data:
            print("❌ No data found! Please check your data directory.")
            return
        
        print(f"✓ Successfully loaded data for {len(self.all_data)} complexes")
        
        # Run analyses
        try:
            # Temporal dynamics analysis
            print("\n" + "📊 " + "="*60)
            self.analyze_temporal_dynamics()
            
            # Interaction stability analysis
            print("\n" + "🎯 " + "="*60)
            complex_stats = self.plot_interaction_stability()
            
            # Cross-complex comparison (publication-ready)
            if complex_stats and len(complex_stats) > 1:
                print("\n" + "📈 " + "="*60)
                self.plot_cross_complex_comparison(complex_stats)
            
            # Generate summary report
            print("\n" + "📋 " + "="*60)
            self.generate_summary_report()
            
            print("\n" + "✅ " + "="*60)
            print("PUBLICATION-READY ANALYSIS COMPLETE!")
            print(f"📁 All results saved to: {self.output_dir}")
            print("🔬 Generated publication-quality figures in PNG and PDF formats")
            print("📊 Includes grouped stability plots and cross-complex comparisons")
            print("="*80)
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Temporal dynamics and interaction stability analysis"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing complex/replica/interfacea folders",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="temporal_stability_results",
        help="Directory to write plots and reports",
    )
    args = parser.parse_args()

    print("🧬 Temporal Dynamics & Interaction Stability Analysis")
    print("=" * 80)

    analyzer = TemporalStabilityAnalysis(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
