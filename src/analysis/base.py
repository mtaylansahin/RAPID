"""Base plotting utilities with publication-quality defaults."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np


# Okabe-Ito colorblind-friendly palette
OKABE_ITO = {
    "vermilion": "#D55E00",
    "blue": "#0072B2",
    "bluish_green": "#009E73",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "reddish_purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#000000",
    "white": "#FFFFFF",
    "light_gray": "#F0F0F0",
    "dark_gray": "#2C3E50",
    "neutral_gray": "#999999",
}

# Stability category colors
STABILITY_COLORS = {
    "Rare": OKABE_ITO["vermilion"],
    "Transient": OKABE_ITO["orange"],
    "Stable": OKABE_ITO["bluish_green"],
}

# Default replica colors for multi-replica plots
REPLICA_COLORS = [
    OKABE_ITO["blue"],
    OKABE_ITO["vermilion"],
    OKABE_ITO["bluish_green"],
    OKABE_ITO["orange"],
    OKABE_ITO["sky_blue"],
    OKABE_ITO["reddish_purple"],
    OKABE_ITO["yellow"],
    OKABE_ITO["black"],
]


def apply_publication_style() -> None:
    """Apply publication-ready matplotlib style settings."""
    plt.style.use('default')
    sns.set_palette("husl")
    
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "DejaVu Sans",
                "Liberation Sans",
                "sans-serif",
            ],
            "font.size": 20,
            "axes.titlesize": 26,
            "axes.labelsize": 26,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 22,
            "figure.titlesize": 32,
            # Spines
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.linewidth": 1.5,
            "axes.edgecolor": "#2C3E50",
            # Grid
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.linewidth": 0.6,
            "grid.color": "#BDC3C7",
            "axes.axisbelow": True,
            # Lines
            "lines.linewidth": 1.5,
            "lines.solid_capstyle": "round",
            # Figure
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            # Text
            "text.color": "#2C3E50",
            "axes.labelcolor": "#2C3E50",
            "xtick.color": "#2C3E50",
            "ytick.color": "#2C3E50",
            # Legend
            "legend.frameon": False,
            "legend.fancybox": True,
            "legend.borderpad": 0.5,
        }
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class BasePlotter(ABC):
    """Abstract base class for plotters with publication-quality defaults."""

    # Class-level access to color palette
    OKABE_ITO = OKABE_ITO
    STABILITY_COLORS = STABILITY_COLORS

    def __init__(self, output_dir: Path, publication_mode: bool = False):
        """
        Initialize plotter.

        Args:
            output_dir: Directory to save generated plots
            publication_mode: If True, add figure panel labels (A, B, C) and use SVG output
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(self.__class__.__name__)
        self.publication_mode = publication_mode
        apply_publication_style()

    def _add_panel_label(
        self,
        fig: plt.Figure,
        label: str,
        x: float = 0.02,
        y: float = 0.98,
    ) -> None:
        """Add a panel label (A, B, C, etc.) to a figure for publication.
        
        Args:
            fig: Matplotlib figure
            label: Panel label (e.g., 'A', 'B', 'C')
            x: X position in figure coordinates (0-1)
            y: Y position in figure coordinates (0-1)
        """
        fig.text(
            x, y, label,
            fontsize=28,
            fontweight='bold',
            ha='left',
            va='top',
            transform=fig.transFigure,
        )

    def _save_figure(
        self,
        fig: plt.Figure,
        filename: str,
        formats: Tuple[str, ...] = ("png",),
        panel_label: Optional[str] = None,
    ) -> list[Path]:
        """
        Save figure to file(s).

        Args:
            fig: Matplotlib figure to save
            filename: Base filename (without extension)
            formats: Tuple of formats to save (e.g., ('png', 'svg'))
            panel_label: Optional panel label (A, B, C) for publication mode

        Returns:
            List of saved file paths
        """
        # Add panel label if in publication mode
        if self.publication_mode and panel_label:
            self._add_panel_label(fig, panel_label)
        
        # In publication mode, always include SVG
        if self.publication_mode and "svg" not in formats:
            formats = formats + ("svg",)
        
        saved_paths = []
        for fmt in formats:
            path = self.output_dir / f"{filename}.{fmt}"
            fig.savefig(path, format=fmt, bbox_inches="tight", dpi=300)
            saved_paths.append(path)
        return saved_paths
        return saved_paths

    @staticmethod
    def _calculate_figure_size(
        num_cols: int,
        num_rows: int,
        cell_width: float = 0.15,
        cell_height: float = 0.25,
        min_width: float = 8.0,
        min_height: float = 6.0,
    ) -> Tuple[float, float]:
        """Calculate figure size to maintain consistent cell aspect ratio."""
        width = max(min_width, num_cols * cell_width + 2)
        height = max(min_height, num_rows * cell_height + 2)
        return width, height

    @abstractmethod
    def generate_plots(self, *args, **kwargs) -> list[Path]:
        """Generate all plots. Implemented by subclasses."""
        pass
