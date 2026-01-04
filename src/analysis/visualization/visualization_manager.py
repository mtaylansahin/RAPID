"""Visualization manager for coordinating all visualization components."""

from pathlib import Path

from typing import List, Dict, Any

from src.core.utils import get_logger
from ..data.data_processor import ProcessedData
from ..analytics.metrics_calculator import MetricsReport
from .heatmap_plotter import HeatmapPlotter
from .metrics_plotter import MetricsPlotter


class VisualizationManager:
    """Manages and coordinates all visualization components."""
    
    def __init__(self, output_directory: str):
        """Initialize visualization manager.
        
        Args:
            output_directory: Directory to save all plots
        """
        self.output_directory = output_directory
        self.logger = get_logger(__name__)
        
        # Initialize plotters
        self.heatmap_plotter = HeatmapPlotter(output_directory)
        self.metrics_plotter = MetricsPlotter(output_directory)
    
    def generate_all_visualizations(
        self,
        processed_data: ProcessedData,
        metrics_report: MetricsReport,
        scores_file_path: str,
        num_pairs_to_show: int = 50,
        valid_steps_to_show: int = 20
    ) -> Dict[str, List[str]]:
        """Generate all visualizations.
        
        Args:
            processed_data: Processed analysis data
            metrics_report: Calculated metrics report
            scores_file_path: Path to scores file for appending metrics
            num_pairs_to_show: Number of pairs to show in heatmaps
            valid_steps_to_show: Number of validation steps to show
        Returns:
            Dictionary mapping visualization types to lists of generated file paths
        """
        self.logger.info("Starting comprehensive visualization generation")
        
        generated_plots = {
            'heatmaps': [],
            'metrics': [],
            'all': []
        }
        
        try:
            # Generate heatmap visualizations
            self.logger.info("Generating heatmap visualizations")
            heatmap_plots = self.heatmap_plotter.generate_plots(
                processed_data, num_pairs_to_show, valid_steps_to_show
            )
            generated_plots['heatmaps'] = heatmap_plots
            generated_plots['all'].extend(heatmap_plots)
            
            # Generate metrics visualizations
            self.logger.info("Generating metrics visualizations")
            metrics_plots = self.metrics_plotter.generate_plots(
                processed_data, metrics_report, scores_file_path
            )
            generated_plots['metrics'] = metrics_plots
            generated_plots['all'].extend(metrics_plots)
            
            # Log summary
            total_plots = len(generated_plots['all'])
            self.logger.info(f"Generated {total_plots} visualization plots:")
            self.logger.info(f"  - Heatmaps: {len(generated_plots['heatmaps'])}")
            self.logger.info(f"  - Metrics: {len(generated_plots['metrics'])}")
            
            # List all generated files
            for plot_path in generated_plots['all']:
                if plot_path:  # Only log non-empty paths
                    self.logger.info(f"  Generated: {Path(plot_path).name}")
            
        except Exception as e:
            self.logger.error(f"Error during visualization generation: {e}")
        
        return generated_plots
    
    def generate_heatmaps_only(
        self,
        processed_data: ProcessedData,
        num_pairs_to_show: int = 50,
        valid_steps_to_show: int = 20
    ) -> List[str]:
        """Generate only heatmap visualizations.
        
        Args:
            processed_data: Processed analysis data
            num_pairs_to_show: Number of pairs to show in heatmaps
            valid_steps_to_show: Number of validation steps to show
            
        Returns:
            List of generated heatmap file paths
        """
        self.logger.info("Generating heatmap visualizations only")
        return self.heatmap_plotter.generate_plots(
            processed_data, num_pairs_to_show, valid_steps_to_show
        )
    
    
    def generate_metrics_only(
        self,
        processed_data: ProcessedData,
        metrics_report: MetricsReport,
        scores_file_path: str
    ) -> List[str]:
        """Generate only metrics visualizations.
        
        Args:
            processed_data: Processed analysis data
            metrics_report: Calculated metrics report
            scores_file_path: Path to scores file for appending metrics
            
        Returns:
            List of generated metrics file paths
        """
        self.logger.info("Generating metrics visualizations only")
        return self.metrics_plotter.generate_plots(
            processed_data, metrics_report, scores_file_path
        ) 
