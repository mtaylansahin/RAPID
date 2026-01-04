"""Configuration classes for experiment and hyperparameter management."""

from dataclasses import dataclass, field
from typing import List
import os
from pathlib import Path


@dataclass
class HyperparameterConfig:
    """Configuration for model hyperparameters with validation."""

    dropout_rates: List[float] = field(default_factory=lambda: [0.5])
    learning_rates: List[float] = field(default_factory=lambda: [0.001])
    batch_sizes: List[int] = field(default_factory=lambda: [128])
    pretrain_epochs: List[int] = field(default_factory=lambda: [30])
    train_epochs: List[int] = field(default_factory=lambda: [10])
    hidden_units: List[int] = field(default_factory=lambda: [100])

    def __post_init__(self):
        """Validate hyperparameter ranges after initialization."""
        self._validate_dropout_rates()
        self._validate_learning_rates()
        self._validate_batch_sizes()
        self._validate_epochs()
        self._validate_hidden_units()

    def _validate_dropout_rates(self) -> None:
        """Ensure dropout rates are within valid range [0, 1]."""
        for rate in self.dropout_rates:
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"Dropout rate {rate} must be between 0.0 and 1.0")

    def _validate_learning_rates(self) -> None:
        """Ensure learning rates are positive."""
        for rate in self.learning_rates:
            if rate <= 0:
                raise ValueError(f"Learning rate {rate} must be positive")

    def _validate_batch_sizes(self) -> None:
        """Ensure batch sizes are positive integers."""
        for size in self.batch_sizes:
            if size <= 0:
                raise ValueError(f"Batch size {size} must be positive")

    def _validate_epochs(self) -> None:
        """Ensure epoch counts are positive integers."""
        for epochs in self.pretrain_epochs + self.train_epochs:
            if epochs <= 0:
                raise ValueError(f"Epoch count {epochs} must be positive")

    def _validate_hidden_units(self) -> None:
        """Ensure hidden unit counts are positive integers."""
        for units in self.hidden_units:
            if units <= 0:
                raise ValueError(f"Hidden units {units} must be positive")


@dataclass
class DataConfig:
    """Configuration for data processing and file management."""

    data_directory: str
    replica: str
    chain1: str
    chain2: str
    train_ratio: float
    validation_ratio: float
    interaction_type: str = "residue"

    def __post_init__(self):
        """Validate data configuration after initialization."""
        self._validate_ratios()
        self._validate_directories()
        self._validate_interaction_type()

    def _validate_ratios(self) -> None:
        """Ensure train and validation ratios are valid."""
        if not 0.0 < self.train_ratio < 1.0:
            raise ValueError(f"Train ratio {self.train_ratio} must be between 0.0 and 1.0")

        if not 0.0 < self.validation_ratio < 1.0:
            raise ValueError(f"Validation ratio {self.validation_ratio} must be between 0.0 and 1.0")

        if self.train_ratio + self.validation_ratio >= 1.0:
            raise ValueError("Sum of train and validation ratios must be less than 1.0")

    def _validate_directories(self) -> None:
        """Check if data directory exists."""
        if not os.path.exists(self.data_directory):
            raise FileNotFoundError(f"Data directory {self.data_directory} does not exist")

    def _validate_interaction_type(self) -> None:
        """Validate interaction type is supported."""
        valid_types = ["residue", "atomic"]
        if self.interaction_type not in valid_types:
            raise ValueError(f"Interaction type {self.interaction_type} must be one of {valid_types}")


@dataclass
class ExperimentConfig:
    """Main configuration class for RAPID experiments."""

    # Core configuration
    experiment_name: str
    data_config: DataConfig
    hyperparameters: HyperparameterConfig

    # System configuration
    gpu_device: int = 0
    use_cuda: bool = True
    random_seed: int = 999

    # Directory configuration
    renet_directory: str = "RE-Net"
    results_base_directory: str = "results"
    models_directory: str = "models"

    # RE-Net specific configuration
    num_k_parameter: int = 5
    maxpool: int = 1

    def __post_init__(self):
        """Validate experiment configuration after initialization."""
        self._validate_system_config()
        self._create_directories()

    def _validate_system_config(self) -> None:
        """Validate system-level configuration."""
        if self.gpu_device < 0:
            raise ValueError(f"GPU device {self.gpu_device} must be non-negative")

        if self.random_seed < 0:
            raise ValueError(f"Random seed {self.random_seed} must be non-negative")

    def _create_directories(self) -> None:
        """Create necessary directories for the experiment."""
        directories = [
            self.results_base_directory,
            self.models_directory,
            os.path.join(self.renet_directory, "data", self.experiment_name)
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def get_results_directory(self, run_id: str) -> str:
        """Get the results directory for a specific run."""
        return os.path.join(self.results_base_directory, f"{self.experiment_name}_results_{run_id}")
