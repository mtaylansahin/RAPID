"""Data structures for analysis and visualization."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class StabilityGroup:
    """Defines a stability category based on frequency thresholds."""

    name: str
    min_freq: float  # Inclusive
    max_freq: float  # Exclusive (except for last bin)
    color: str


# Default stability groups matching RNN-MD
DEFAULT_STABILITY_GROUPS = [
    StabilityGroup("Rare", 0.0, 5.0, "#D55E00"),
    StabilityGroup("Transient", 5.0, 50.0, "#E69F00"),
    StabilityGroup("Stable", 50.0, 100.1, "#009E73"),  # 100.1 to include 100%
]


@dataclass
class StabilityMetrics:
    """Metrics for a single stability group."""

    group_name: str
    pair_count: int
    recall: float
    precision: float
    f1: float
    mcc: float = 0.0
    tpr: float = 0.0
    fpr: float = 0.0
    mean_pairwise_f1: float = 0.0
    baseline_f1: Optional[float] = None
    baseline_mean_pairwise_f1: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "group_name": self.group_name,
            "pair_count": self.pair_count,
            "recall": self.recall,
            "precision": self.precision,
            "f1": self.f1,
            "mcc": self.mcc,
            "tpr": self.tpr,
            "fpr": self.fpr,
            "mean_pairwise_f1": self.mean_pairwise_f1,
        }


@dataclass
class EvaluationResult:
    """Container for single-run evaluation outputs.

    Stores predictions, ground truth, and computed metrics in a format
    suitable for visualization and reporting.
    """

    # Core data: predictions list of (e1, rel, e2, t, score, pred)
    predictions: List[Tuple[int, int, int, int, float, int]]

    # Ground truth: list of (e1, rel, e2, t) tuples
    ground_truth: List[Tuple[int, int, int, int]]

    # Dataset info
    num_entities: int
    num_relations: int
    train_timesteps: List[int]
    valid_timesteps: List[int]
    test_timesteps: List[int]

    # Overall metrics
    auroc: float = 0.0
    auprc: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    mcc: float = 0.0
    mean_pairwise_f1: float = 0.0

    # Baseline metrics (optional)
    baseline_metrics: Optional[Dict] = None
    baseline_mean_pairwise_f1: Optional[float] = None

    # Per-timestep metrics
    per_timestep_f1: Dict[int, float] = field(default_factory=dict)
    per_timestep_auprc: Dict[int, float] = field(default_factory=dict)

    # Stability-stratified metrics
    stability_metrics_train_freq: Dict[str, StabilityMetrics] = field(
        default_factory=dict
    )
    stability_metrics_test_freq: Dict[str, StabilityMetrics] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall": {
                "auroc": self.auroc,
                "auprc": self.auprc,
                "precision": self.precision,
                "recall": self.recall,
                "f1": self.f1,
                "mcc": self.mcc,
            },
            "per_timestep": {
                "f1": self.per_timestep_f1,
                "auprc": self.per_timestep_auprc,
            },
            "stability_by_train_freq": {
                k: v.to_dict() for k, v in self.stability_metrics_train_freq.items()
            },
            "stability_by_test_freq": {
                k: v.to_dict() for k, v in self.stability_metrics_test_freq.items()
            },
            "dataset": {
                "num_entities": self.num_entities,
                "num_relations": self.num_relations,
                "train_timesteps": len(self.train_timesteps),
                "valid_timesteps": len(self.valid_timesteps),
                "test_timesteps": len(self.test_timesteps),
            },
        }


@dataclass
class MetricStats:
    """Statistical summary for a metric across replicas."""

    mean: float
    std: float
    min: float
    max: float
    n_replicas: int

    @classmethod
    def from_values(cls, values: List[float]) -> "MetricStats":
        if not values:
            return cls(0.0, 0.0, 0.0, 0.0, 0)
        arr = np.array(values)
        return cls(
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            n_replicas=len(values),
        )

    def to_dict(self) -> Dict:
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "n_replicas": self.n_replicas,
        }


@dataclass
class StabilityGroupStats:
    """Aggregated statistics for a stability group across replicas."""

    group_name: str
    n_replicas: int
    recall: MetricStats
    precision: MetricStats
    f1: MetricStats
    mcc: MetricStats
    pair_count: MetricStats

    def to_dict(self) -> Dict:
        return {
            "group_name": self.group_name,
            "n_replicas": self.n_replicas,
            "recall": self.recall.to_dict(),
            "precision": self.precision.to_dict(),
            "f1": self.f1.to_dict(),
            "mcc": self.mcc.to_dict(),
            "pair_count": self.pair_count.to_dict(),
        }


@dataclass
class OverallStats:
    """Aggregated overall metrics across replicas."""

    n_replicas: int
    auroc: MetricStats
    auprc: MetricStats
    precision: MetricStats
    recall: MetricStats
    f1: MetricStats
    mcc: MetricStats

    def to_dict(self) -> Dict:
        return {
            "n_replicas": self.n_replicas,
            "auroc": self.auroc.to_dict(),
            "auprc": self.auprc.to_dict(),
            "precision": self.precision.to_dict(),
            "recall": self.recall.to_dict(),
            "f1": self.f1.to_dict(),
            "mcc": self.mcc.to_dict(),
        }


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple replica runs."""

    experiment_name: str
    n_replicas: int
    replica_paths: List[str]
    overall_stats: OverallStats
    training_frequency_stats: Dict[str, StabilityGroupStats] = field(
        default_factory=dict
    )
    test_frequency_stats: Dict[str, StabilityGroupStats] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "experiment_name": self.experiment_name,
            "n_replicas": self.n_replicas,
            "replica_paths": self.replica_paths,
            "overall_stats": self.overall_stats.to_dict(),
            "training_frequency_stats": {
                k: v.to_dict() for k, v in self.training_frequency_stats.items()
            },
            "test_frequency_stats": {
                k: v.to_dict() for k, v in self.test_frequency_stats.items()
            },
        }
