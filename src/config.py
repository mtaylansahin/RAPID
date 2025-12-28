"""Configuration dataclasses for RAPID - Protein Interaction Dynamics prediction."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    dataset: str = "RAPID"
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    
    # Negative sampling
    neg_ratio: float = 1.0  # Ratio of negative to positive samples
    temporal_neg_ratio: float = 0.5  # Fraction of negatives that are temporal
    
    # Batch settings
    batch_size: int = 128
    num_workers: int = 4
    
    @property
    def dataset_path(self) -> Path:
        return self.data_dir / self.dataset
    
    def __post_init__(self):
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    
    # Dimensions
    hidden_dim: int = 200
    
    # RGCN settings
    num_rgcn_layers: int = 2
    num_bases: int = 100  # Number of basis functions for RGCN
    
    # Temporal encoder
    seq_len: int = 10  # History sequence length
    
    # Classifier head
    classifier_hidden_dim: int = 128
    classifier_dropout: float = 0.2
    
    # General
    dropout: float = 0.2
    
    # Aggregator type
    aggregator: Literal["rgcn", "mean", "attention"] = "rgcn"


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    
    # Training schedule
    max_epochs: int = 100
    patience: int = 10  # Early stopping patience
    
    # Focal loss
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None  # If None, no class weighting
    
    # Logging
    log_interval: int = 10  # Log every N batches
    eval_interval: int = 1  # Evaluate every N epochs
    
    # Device
    gpu: int = -1  # -1 for CPU, >= 0 for GPU ID
    seed: int = 42
    
    @property
    def use_cuda(self) -> bool:
        import torch
        return self.gpu >= 0 and torch.cuda.is_available()


@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Experiment settings
    experiment_name: str = "ppi_dynamics"
    checkpoint_dir: Path = field(default_factory=lambda: Path("./checkpoints"))
    log_dir: Path = field(default_factory=lambda: Path("./logs"))
    
    def __post_init__(self):
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        if isinstance(self.log_dir, str):
            self.log_dir = Path(self.log_dir)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create config from nested dictionary."""
        data_cfg = DataConfig(**config_dict.get("data", {}))
        model_cfg = ModelConfig(**config_dict.get("model", {}))
        training_cfg = TrainingConfig(**config_dict.get("training", {}))
        
        return cls(
            data=data_cfg,
            model=model_cfg,
            training=training_cfg,
            experiment_name=config_dict.get("experiment_name", "ppi_dynamics"),
            checkpoint_dir=Path(config_dict.get("checkpoint_dir", "./checkpoints")),
            log_dir=Path(config_dict.get("log_dir", "./logs")),
        )
    
    def to_dict(self) -> dict:
        """Convert config to nested dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    def save(self, path: Path) -> None:
        """Save config to YAML file."""
        import yaml
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: Path) -> "Config":
        """Load config from YAML file."""
        import yaml
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
