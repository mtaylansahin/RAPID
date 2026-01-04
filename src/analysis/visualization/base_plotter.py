"""Base plotter class providing common visualization functionality."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple

from src.core.utils import get_logger


class BasePlotter(ABC):
    """Base class for all visualization components."""
    
    def __init__(self, output_directory: str):
        """Initialize base plotter.
        
        Args:
            output_directory: Directory to save plots
        """
        self.output_directory = Path(output_directory)
        self.logger = get_logger(__name__)
        
        # Ensure output directory exists
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Set default style
        self._setup_plot_style()
    
    def _setup_plot_style(self) -> None:
        """Setup consistent plot styling."""
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Set default figure parameters
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9
        })
    
    def save_plot(self, filename: str, fig: Optional[plt.Figure] = None) -> str:
        """Save plot to output directory.
        
        Args:
            filename: Name of the file to save
            fig: Figure to save (uses current figure if None)
            
        Returns:
            Full path to saved file
        """
        output_path = self.output_directory / filename
        
        if fig is not None:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        self.logger.info(f"Saved plot: {output_path}")
        return str(output_path)
    
    def close_plot(self, fig: Optional[plt.Figure] = None) -> None:
        """Close plot to free memory.
        
        Args:
            fig: Figure to close (closes current figure if None)
        """
        if fig is not None:
            plt.close(fig)
        else:
            plt.close()
    
    def custom_sort_residues(self, value: str) -> Tuple[float, str]:
        """Custom sorting function for residue names.
        
        Args:
            value: Residue name in format "Number-String"
            
        Returns:
            Tuple for sorting (numeric_part, string_part)
        """
        try:
            parts = value.split('-')
            numeric_part = int(parts[0])
            string_part = '-'.join(parts[1:])
            return numeric_part, string_part
        except (ValueError, IndexError):
            return float('inf'), value
    
    def get_color_palette(self, palette_type: str = "default") -> Dict[str, str]:
        """Get consistent color palette for plots.
        
        Args:
            palette_type: Type of palette to return
            
        Returns:
            Dictionary mapping categories to colors
        """
        palettes = {
            "default": {
                "ground_truth": "#0074D9",
                "prediction": "#FFA500", 
                "validation": "#2ECC71",
                "baseline": "#E74C3C"
            },
            "overlay": {
                "TN": "#f0f0f0",  # True Negative (Grey)
                "FN": "#0074D9",  # False Negative (Blue) 
                "FP": "#FFA500",  # False Positive (Orange)
                "TP": "#2ecc71"   # True Positive (Green)
            },
            "stability": {
                "Rare (<5%)": "#E74C3C",
                            "Moderate (5-50%)": "#F39C12",
            "Stable (>50%)": "#27AE60"
            }
        }
        
        return palettes.get(palette_type, palettes["default"])
    
    @abstractmethod
    def generate_plots(self, *args, **kwargs) -> List[str]:
        """Generate all plots for this plotter.
        
        Returns:
            List of generated plot file paths
        """
        pass 
