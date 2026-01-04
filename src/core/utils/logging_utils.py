"""Logging utilities for structured logging throughout the application."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        experiment_name: Optional[str] = None
) -> None:
    """Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If None, logs only to console
        experiment_name: Optional experiment name for log file naming
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if log_file is specified
    if log_file:
        log_path = _create_log_file_path(log_file, experiment_name)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Set root logger level
    root_logger.setLevel(numeric_level)


def _create_log_file_path(log_file: str, experiment_name: Optional[str]) -> str:
    """Create log file path with optional experiment name and timestamp.
    
    Args:
        log_file: Base log file path
        experiment_name: Optional experiment name to include in filename
        
    Returns:
        Complete log file path
    """
    if experiment_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(log_file).stem
        extension = Path(log_file).suffix
        parent_dir = Path(log_file).parent

        log_filename = f"{base_name}_{experiment_name}_{timestamp}{extension}"
        return str(parent_dir / log_filename)

    return log_file


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__ from calling module)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class ExperimentLogger:
    """Enhanced logger for experiment tracking with structured information."""

    def __init__(self, experiment_name: str, log_directory: str = "logs"):
        """Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            log_directory: Directory to store log files
        """
        self.experiment_name = experiment_name
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)

        # Attach a file handler to a child logger without reconfiguring root
        self.logger = get_logger(f"experiment.{experiment_name}")
        log_file = self.log_directory / f"{self.experiment_name}.log"
        if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == str(log_file) for h in
                   self.logger.handlers):
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            self.logger.addHandler(file_handler)

    def log_experiment_start(self, config_dict: dict) -> None:
        """Log the start of an experiment with configuration.
        
        Args:
            config_dict: Configuration dictionary to log
        """
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        self.logger.info("Experiment configuration:")
        for key, value in config_dict.items():
            self.logger.info(f"  {key}: {value}")

    def log_phase_start(self, phase_name: str, phase_config: dict = None) -> None:
        """Log the start of an experiment phase.
        
        Args:
            phase_name: Name of the phase (e.g., 'pretraining', 'training', 'testing')
            phase_config: Optional phase-specific configuration
        """
        self.logger.info(f"Starting phase: {phase_name}")
        if phase_config:
            for key, value in phase_config.items():
                self.logger.info(f"  {key}: {value}")

    def log_phase_completion(self, phase_name: str, results: dict = None) -> None:
        """Log the completion of an experiment phase.
        
        Args:
            phase_name: Name of the completed phase
            results: Optional results dictionary
        """
        self.logger.info(f"Completed phase: {phase_name}")
        if results:
            for key, value in results.items():
                self.logger.info(f"  {key}: {value}")

    def log_metrics(self, phase: str, metrics: dict, epoch: Optional[int] = None) -> None:
        """Log performance metrics.
        
        Args:
            phase: Phase name (training, validation, testing)
            metrics: Dictionary of metrics (loss, accuracy, etc.)
            epoch: Optional epoch number
        """
        epoch_str = f" (epoch {epoch})" if epoch is not None else ""
        self.logger.info(f"Metrics for {phase}{epoch_str}:")
        for metric_name, value in metrics.items():
            self.logger.info(f"  {metric_name}: {value}")

    def log_error(self, error_message: str, exception: Optional[Exception] = None) -> None:
        """Log an error with optional exception details.
        
        Args:
            error_message: Human-readable error description
            exception: Optional exception object for stack trace
        """
        self.logger.error(error_message)
        if exception:
            self.logger.exception("Exception details:", exc_info=exception)
