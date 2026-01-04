"""Configuration for analysis and visualization parameters."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import os


@dataclass
class AnalysisConfig:
    """Configuration for results analysis and visualization."""
    
    # Input/Output paths
    input_directory: str
    output_directory: str
    output_file_path: str
    
    # Analysis parameters
    window_size: int = 10
    keep_latest_backups: int = 3
    
    # Plotting parameters
    figure_dpi: int = 300
    heatmap_linewidth: float = 0.1
    font_size_min: int = 4
    font_size_max: int = 10
    
    # File patterns
    required_input_files: List[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.required_input_files is None:
            self.required_input_files = [
                "labels.txt", "test.txt", "train.txt", "valid.txt"
            ]
        
        self._validate_directories()
        self._validate_parameters()
    
    def _validate_directories(self) -> None:
        """Validate input/output directory paths."""
        if not os.path.exists(self.input_directory):
            raise FileNotFoundError(f"Input directory does not exist: {self.input_directory}")
        
        Path(self.output_directory).mkdir(parents=True, exist_ok=True)
    
    def _validate_parameters(self) -> None:
        """Validate analysis parameters."""
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        
        if self.figure_dpi <= 0:
            raise ValueError("figure_dpi must be positive")
    
    def validate_required_files(self) -> List[str]:
        """Validate that all required input files exist.
        
        Returns:
            List of missing files (empty if all present)
        """
        missing_files = []
        input_path = Path(self.input_directory)
        
        for filename in self.required_input_files:
            if not (input_path / filename).exists():
                missing_files.append(filename)
        
        return missing_files
    
    @property
    def labels_file_path(self) -> str:
        """Get path to labels.txt file."""
        return os.path.join(self.input_directory, "labels.txt")
    
    @property
    def test_file_path(self) -> str:
        """Get path to test.txt file."""
        return os.path.join(self.input_directory, "test.txt")
    
    @property
    def train_file_path(self) -> str:
        """Get path to train.txt file."""
        return os.path.join(self.input_directory, "train.txt")
    
    @property
    def valid_file_path(self) -> str:
        """Get path to valid.txt file."""
        return os.path.join(self.input_directory, "valid.txt") 