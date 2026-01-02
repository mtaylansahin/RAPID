"""Configuration dataclasses for RAPID - Protein Interaction Dynamics prediction."""

from dataclasses import dataclass
from typing import Optional


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
    eval_interval: int = 1  # Evaluate every N epochs
