"""Temporal dynamics analysis for raw simulation data."""

import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from src.analysis.base import BasePlotter, REPLICA_COLORS


class TemporalAnalyzer(BasePlotter):
    """Analyzes temporal dynamics from raw .interfacea trajectory files.

    Generates temporal dynamics plots showing interaction counts and
    interface sizes over simulation time.
    """

    TIME_STEP_NS = 0.5  # Timestep in nanoseconds

    def __init__(self, output_dir: Path):
        super().__init__(output_dir)
        self.all_data: Dict[str, Dict[str, pd.DataFrame]] = {}

    def load_data(self, data_dir: Path) -> bool:
        """
        Load all interaction data from complexes and replicas.

        Expects directory structure:
        data_dir/
          complex_name/
            replica1/
              *interfacea*/
                *.interfacea files

        Args:
            data_dir: Root directory containing complex subdirectories

        Returns:
            True if data was loaded successfully
        """
        data_dir = Path(data_dir)
        print(f"Loading data from: {data_dir}")

        for complex_dir in data_dir.iterdir():
            if not complex_dir.is_dir() or complex_dir.name.startswith("."):
                continue

            complex_name = complex_dir.name
            print(f"  Processing complex: {complex_name}")
            self.all_data[complex_name] = {}

            for replica_dir in complex_dir.iterdir():
                if not replica_dir.is_dir():
                    continue
                if not replica_dir.name.startswith("replica"):
                    continue

                replica_name = replica_dir.name
                print(f"    Processing {replica_name}")

                # Find interfacea directory
                interfacea_dirs = list(replica_dir.glob("*interfacea*"))
                if interfacea_dirs:
                    data = self._load_replica_data(interfacea_dirs[0])
                    if not data.empty:
                        self.all_data[complex_name][replica_name] = data

        n_complexes = len(self.all_data)
        n_replicas = sum(len(r) for r in self.all_data.values())
        print(f"Loaded data for {n_complexes} complexes, {n_replicas} replicas")

        return n_replicas > 0

    def _load_replica_data(self, interfacea_dir: Path) -> pd.DataFrame:
        """Load .interfacea files from a single replica directory."""
        replica_data = []

        files = sorted(
            interfacea_dir.glob("*.interfacea"),
            key=lambda x: int(re.findall(r"\d+", x.name)[0])
            if re.findall(r"\d+", x.name)
            else 0,
        )

        for i, file_path in enumerate(files):
            try:
                # Extract timestamp from filename
                time_stamp = int(re.findall(r"\d+", file_path.name)[0])

                df = pd.read_csv(
                    file_path,
                    sep=r"\s+",
                    header=0,
                    names=[
                        "itype",
                        "chain_a",
                        "chain_b",
                        "resname_a",
                        "resname_b",
                        "resid_a",
                        "resid_b",
                        "atom_a",
                        "atom_b",
                    ],
                )

                if not df.empty:
                    df["time_stamp"] = time_stamp
                    df["snapshot"] = i
                    replica_data.append(df)

            except Exception as e:
                print(f"      Error loading {file_path.name}: {e}")

        if replica_data:
            return pd.concat(replica_data, ignore_index=True)
        return pd.DataFrame()

    def generate_plots(self, **kwargs) -> List[Path]:
        """Generate all temporal dynamics plots."""
        generated_plots = []

        if not self.all_data:
            print("No data loaded. Call load_data() first.")
            return generated_plots

        # Generate merged 2x2 temporal dynamics plot
        path = self._plot_temporal_dynamics_merged()
        if path:
            generated_plots.append(path)

        return generated_plots

    def _plot_temporal_dynamics_merged(self, window_size: int = 10) -> Optional[Path]:
        """
        Create merged 2x2 temporal dynamics plot.

        Rows: Different complexes
        Columns: Total Interactions | Interface Size
        """
        # Get up to 2 complexes
        complexes = self._prioritized_complexes()[:2]

        if not complexes:
            return None

        fig, axes = plt.subplots(
            len(complexes), 2, figsize=(14, 5 * len(complexes)), sharex=True
        )

        if len(complexes) == 1:
            axes = axes.reshape(1, -1)

        legend_handles = []
        legend_labels = []

        for row_idx, complex_name in enumerate(complexes):
            complex_data = self.all_data[complex_name]
            clean_name = self._format_complex_label(complex_name)

            # Annotate complex name on right side
            axes[row_idx, 1].annotate(
                clean_name,
                xy=(1.03, 0.5),
                xycoords="axes fraction",
                ha="left",
                va="center",
                rotation=-90,
                fontsize=14,
                fontweight="bold",
            )

            for replica_idx, (replica_name, replica_data) in enumerate(
                sorted(complex_data.items())
            ):
                if replica_data.empty:
                    continue

                color = REPLICA_COLORS[replica_idx % len(REPLICA_COLORS)]

                # Deduplicate interactions per timestep
                dedup_df = self._deduplicate_interactions(replica_data)

                # Unique interactions per timestep
                time_counts = dedup_df.groupby("time_stamp").size()
                time_counts.index = time_counts.index * self.TIME_STEP_NS
                time_counts_smooth = self._smooth_series(time_counts, window_size)

                # Plot Total Interactions
                ax_total = axes[row_idx, 0]
                ax_total.plot(
                    time_counts.index,
                    time_counts.values,
                    alpha=0.2,
                    linewidth=0.8,
                    color=color,
                )
                label_clean = self._format_replica_label(replica_name)
                (line,) = ax_total.plot(
                    time_counts.index,
                    time_counts_smooth.values,
                    label=label_clean,
                    linewidth=2.0,
                    alpha=0.9,
                    color=color,
                )

                if row_idx == 0:
                    legend_handles.append(line)
                    legend_labels.append(label_clean)

                # Interface size (unique residues)
                interface_sizes = dedup_df.groupby("time_stamp").apply(
                    lambda x: len(set(x["n_resid1"]).union(set(x["n_resid2"])))
                )
                interface_sizes.index = interface_sizes.index * self.TIME_STEP_NS
                interface_smooth = self._smooth_series(interface_sizes, window_size)

                ax_size = axes[row_idx, 1]
                ax_size.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax_size.plot(
                    interface_sizes.index,
                    interface_sizes.values,
                    alpha=0.2,
                    linewidth=0.8,
                    color=color,
                )
                ax_size.plot(
                    interface_sizes.index,
                    interface_smooth.values,
                    linewidth=2.0,
                    alpha=0.9,
                    color=color,
                )

        # Titles and labels
        axes[0, 0].set_title("Total Interactions", fontweight="bold", fontsize=14)
        axes[0, 1].set_title("Interface Size", fontweight="bold", fontsize=14)

        for ax in axes.flatten():
            ax.grid(visible=True, axis="y", alpha=0.3)
            ax.grid(visible=False, axis="x")

        for row_idx in range(len(complexes)):
            axes[row_idx, 0].set_ylabel("Interaction Count")
            axes[row_idx, 1].set_ylabel("Residue Count")

        axes[-1, 0].set_xlabel("Simulation time (ns)")
        axes[-1, 1].set_xlabel("Simulation time (ns)")

        # Legend
        if legend_handles:
            fig.legend(
                handles=legend_handles,
                labels=legend_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=min(len(legend_handles), 4),
                frameon=False,
                fontsize=12,
            )

        plt.tight_layout(rect=[0, 0.05, 0.95, 1])

        paths = self._save_figure(
            fig, "temporal_dynamics_merged", formats=("png", "svg")
        )
        plt.close(fig)

        return paths[0] if paths else None

    def _deduplicate_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate interactions to canonical form (chain1 < chain2)."""
        dedup = df.copy()

        chains_a = dedup["chain_a"].astype(str)
        chains_b = dedup["chain_b"].astype(str)
        resid_a = dedup["resid_a"].astype(int)
        resid_b = dedup["resid_b"].astype(int)

        a_first = (chains_a < chains_b) | (
            (chains_a == chains_b) & (resid_a <= resid_b)
        )

        dedup["n_chain1"] = np.where(a_first, chains_a, chains_b)
        dedup["n_resid1"] = np.where(a_first, resid_a, resid_b)
        dedup["n_chain2"] = np.where(a_first, chains_b, chains_a)
        dedup["n_resid2"] = np.where(a_first, resid_b, resid_a)

        return dedup.drop_duplicates(
            subset=[
                "time_stamp",
                "itype",
                "n_chain1",
                "n_resid1",
                "n_chain2",
                "n_resid2",
            ]
        )

    @staticmethod
    def _smooth_series(series: pd.Series, window_size: int) -> pd.Series:
        """Apply centered rolling mean smoothing."""
        if len(series) > window_size:
            return series.rolling(window=window_size, center=True).mean()
        return series

    @staticmethod
    def _format_replica_label(replica_name: str) -> str:
        """Format replica name for display."""
        if "replica" in replica_name.lower():
            nums = re.findall(r"\d+", replica_name)
            if nums:
                return f"Replica {nums[0]}"
        return replica_name.capitalize()

    @staticmethod
    def _format_complex_label(complex_name: str) -> str:
        """Format complex name for display."""
        if not complex_name:
            return complex_name
        base = complex_name.replace("_interchain", "")
        return base

    def _prioritized_complexes(self) -> List[str]:
        """Return complexes in priority order."""
        keys = sorted(self.all_data.keys())
        # Prioritize certain known complex types
        ordered = []
        for target in ("1JPS", "1EAW"):
            ordered.extend(
                [k for k in keys if target in k.upper() and k not in ordered]
            )
        ordered.extend([k for k in keys if k not in ordered])
        return ordered
