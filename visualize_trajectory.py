#!/usr/bin/env python
"""
Generate heatmap visualization for trajectory predictions.

Uses the exact same style as the existing HeatmapPlotter.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter, MultipleLocator
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data.trajectory_dataset import TrajectoryDataModule
from src.models.trajectory import TrajectoryModel


# Okabe-Ito colorblind-friendly palette (from HeatmapPlotter)
OKABE_ITO = {
    "vermilion": "#D55E00",
    "blue": "#0072B2",
    "bluish_green": "#009E73",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "white": "#FFFFFF",
    "light_gray": "#F0F0F0",
    "dark_gray": "#2C3E50",
    "neutral_gray": "#999999",
}

CELL_HEIGHT = 0.4
MIN_FIG_HEIGHT = 12
MIN_FIG_WIDTH = 20
NS_PER_TIMESTEP = 0.5


def apply_style():
    """Apply publication-ready style settings."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "DejaVu Sans",
                "Liberation Sans",
                "sans-serif",
            ],
            "font.size": 40,
            "axes.titlesize": 54,
            "axes.labelsize": 50,
            "xtick.labelsize": 40,
            "ytick.labelsize": 27,
            "legend.fontsize": 40,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.linewidth": 1.5,
            "axes.edgecolor": OKABE_ITO["dark_gray"],
        }
    )


def calculate_figure_size(num_cols: int, num_rows: int) -> tuple:
    """Calculate figure size to maintain consistent cell aspect ratio."""
    target_cell_aspect = 1.0

    fig_height = max(MIN_FIG_HEIGHT, num_rows * CELL_HEIGHT)
    cell_height = fig_height / num_rows
    cell_width = cell_height * target_cell_aspect
    fig_width = max(MIN_FIG_WIDTH, num_cols * cell_width)

    return fig_width, fig_height


def calculate_y_fontsize(fig_height: float, num_labels: int) -> int:
    """Calculate appropriate font size for Y-axis labels."""
    if num_labels == 0:
        return 21
    return max(15, min(27, int((fig_height / num_labels) * 72 * 0.45)))


def create_colormaps():
    """Create color maps for the heatmap (same as HeatmapPlotter)."""
    # Order: TN (0), FN (1), FP (2), TP (3), History (4)
    cmap_overlay = mcolors.ListedColormap(
        [
            OKABE_ITO["light_gray"],  # 0: TN - light gray
            OKABE_ITO["sky_blue"],  # 1: FN - blue (missed)
            OKABE_ITO["vermilion"],  # 2: FP - red (false alarm)
            OKABE_ITO["bluish_green"],  # 3: TP - green (correct)
            OKABE_ITO["neutral_gray"],  # 4: History ON
        ]
    )
    return cmap_overlay


def create_time_formatter(timestamps):
    """Create formatter function for x-axis time labels."""

    def formatter(x, pos):
        try:
            idx = int(x)
            if 0 <= idx < len(timestamps):
                return f"{timestamps[idx] * NS_PER_TIMESTEP:.0f}"
            return ""
        except:
            return ""

    return formatter


def generate_trajectory_heatmap(
    model: TrajectoryModel,
    data_module: TrajectoryDataModule,
    device: torch.device,
    output_dir: Path,
    threshold: float = 0.5,
    num_pairs: int = 50,
):
    """Generate heatmap comparing predicted and ground truth trajectories."""
    model.eval()
    model.to(device)

    test_loader = data_module.get_test_dataloader()

    # Collect predictions
    all_probs = []
    all_targets = []
    all_entity1 = []
    all_entity2 = []
    all_history = []

    print("Collecting trajectory predictions...")
    for batch in tqdm(test_loader):
        entity1 = batch["entity1"].to(device)
        entity2 = batch["entity2"].to(device)
        history = batch["history"].to(device)
        target = batch["target"].to(device)
        neighbor_e1 = batch["neighbor_entity1"].to(device)
        neighbor_e2 = batch["neighbor_entity2"].to(device)
        neighbor_hist = batch["neighbor_history"].to(device)
        neighbor_mask = batch["neighbor_mask"].to(device)

        traj_len = target.size(1)

        with torch.no_grad():
            logits = model(
                entity1_ids=entity1,
                entity2_ids=entity2,
                history=history,
                neighbor_entity1=neighbor_e1,
                neighbor_entity2=neighbor_e2,
                neighbor_history=neighbor_hist,
                traj_len=traj_len,
                neighbor_mask=neighbor_mask,
            )
            probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu().numpy())
        all_targets.append(target.cpu().numpy())
        all_entity1.append(entity1.cpu().numpy())
        all_entity2.append(entity2.cpu().numpy())
        all_history.append(history.cpu().numpy())

    # Stack results
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_entity1 = np.concatenate(all_entity1, axis=0)
    all_entity2 = np.concatenate(all_entity2, axis=0)
    all_history = np.concatenate(all_history, axis=0)

    n_edges, traj_len = all_probs.shape
    hist_len = all_history.shape[1]

    # Select pairs to show (prioritize those with more activity)
    activity_count = all_targets.sum(axis=1)
    pair_order = np.argsort(-activity_count)[:num_pairs]

    n_show = min(num_pairs, n_edges)
    full_len = hist_len + traj_len

    pred_binary = (all_probs > threshold).astype(int)

    # Build pair names
    pair_names = [f"{all_entity1[idx]}_{all_entity2[idx]}" for idx in pair_order]

    # Build DataFrames for history and test regions
    # History columns: 0 to hist_len-1
    # Test columns: hist_len to full_len-1
    all_columns = list(range(full_len))

    # Create overlay matrix: same logic as HeatmapPlotter
    # 0 = TN, 1 = FN, 2 = FP, 3 = TP, 4 = History ON
    overlay_data = np.zeros((n_show, full_len), dtype=int)

    for i, idx in enumerate(pair_order):
        # History region: 4 = ON (neutral gray), 0 = OFF (light gray)
        for t in range(hist_len):
            if all_history[idx, t] > 0.5:
                overlay_data[i, t] = 4  # History ON
            else:
                overlay_data[i, t] = 0  # TN (same as light gray)

        # Test region: TP/TN/FP/FN
        for t in range(traj_len):
            pred = pred_binary[idx, t]
            actual = int(all_targets[idx, t])

            if actual == 0 and pred == 0:
                val = 0  # TN
            elif actual == 1 and pred == 0:
                val = 1  # FN
            elif actual == 0 and pred == 1:
                val = 2  # FP
            else:  # actual == 1 and pred == 1
                val = 3  # TP

            overlay_data[i, hist_len + t] = val

    # Create DataFrame
    overlay_df = pd.DataFrame(overlay_data, index=pair_names, columns=all_columns)

    # Apply style
    apply_style()

    # Calculate figure size
    num_cols = overlay_df.shape[1]
    num_rows = len(overlay_df.index)
    fig_width, fig_height = calculate_figure_size(num_cols, num_rows)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Create colormap
    cmap_overlay = create_colormaps()
    bounds = [0, 1, 2, 3, 4, 5]
    norm = mcolors.BoundaryNorm(bounds, cmap_overlay.N)

    # Plot heatmap using seaborn
    cax = sns.heatmap(
        overlay_df,
        ax=ax,
        cmap=cmap_overlay,
        norm=norm,
        cbar=True,
        yticklabels=overlay_df.index,
        linewidths=0.05,
        linecolor="white",
        square=False,
        cbar_kws={"ticks": [0.5, 1.5, 2.5, 3.5, 4.5], "pad": 0.02},
    )

    ax.set_title("")
    ax.set_xlabel("Simulation time (ns)", fontsize=50, labelpad=60)
    ax.set_ylabel("Residue Pair", fontsize=50)

    # Configure colorbar
    colorbar = cax.collections[0].colorbar
    colorbar.set_ticklabels(["TN", "FN", "FP", "TP", "History"])
    colorbar.ax.tick_params(labelsize=45)
    colorbar.set_label("Result Type", fontsize=50)

    # Y-axis font size
    y_fontsize = calculate_y_fontsize(fig_height, num_rows)
    ax.tick_params(axis="y", labelsize=y_fontsize)

    # X-axis formatting
    timestamps = np.array(all_columns)
    ax.xaxis.set_major_formatter(FuncFormatter(create_time_formatter(timestamps)))
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.tick_params(axis="x", labelsize=45, rotation=0)

    # Add vertical line at history/test boundary
    if hist_len > 0:
        ax.axvline(
            x=hist_len, color=OKABE_ITO["dark_gray"], linestyle="--", linewidth=3
        )

        label_y_pos = ax.get_ylim()[0] * 1.05

        ax.text(
            hist_len / 2.0,
            label_y_pos,
            "History",
            ha="center",
            va="top",
            color=OKABE_ITO["dark_gray"],
            fontsize=45,
        )
        ax.text(
            hist_len + (num_cols - hist_len) / 2.0,
            label_y_pos,
            "Test",
            ha="center",
            va="top",
            color=OKABE_ITO["dark_gray"],
            fontsize=45,
        )

    plt.tight_layout()

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / "heatmap_time_vs_pairs_VERTICAL_full_history.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"PNG saved to: {png_path}")

    svg_path = output_dir / "heatmap_time_vs_pairs_VERTICAL_full_history.svg"
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    print(f"SVG saved to: {svg_path}")

    plt.close(fig)

    return png_path


def main():
    parser = argparse.ArgumentParser(description="Generate trajectory heatmaps")
    parser.add_argument("--dataset", type=str, default="1JPS")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/trajectory_full/best.pth"
    )
    parser.add_argument(
        "--output_dir", type=str, default="analysis_outputs/trajectory_full"
    )
    parser.add_argument("--test_ratio", type=float, default=0.25)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument(
        "--num_pairs", type=int, default=50, help="Number of pairs to show"
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n_hops", type=int, default=1, help="Number of neighbor hops")
    parser.add_argument(
        "--use_full_neighbors",
        action="store_true",
        help="Include intrachain edges as neighbors",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data_path = Path(args.data_dir) / args.dataset
    print(f"\nLoading dataset: {args.dataset}")

    data_module = TrajectoryDataModule(
        data_path=data_path,
        test_ratio=args.test_ratio,
        batch_size=64,
        use_full_neighbors=args.use_full_neighbors,
        n_hops=args.n_hops,
    )

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model = TrajectoryModel(
        num_entities=data_module.num_entities,
        hidden_dim=args.hidden_dim,
        max_traj_len=data_module.traj_len + 10,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Generate heatmap
    output_dir = Path(args.output_dir)
    generate_trajectory_heatmap(
        model=model,
        data_module=data_module,
        device=device,
        output_dir=output_dir,
        threshold=args.threshold,
        num_pairs=args.num_pairs,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
