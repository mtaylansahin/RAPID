"""Data structures for multi-replica analysis."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class MetricStats:
    """Statistics for a single metric across replicas."""
    
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    values: List[float] = field(default_factory=list)
    n_replicas: int = 0
    
    def __post_init__(self):
        """Calculate statistics if values are provided."""
        if self.values and self.n_replicas == 0:
            self.n_replicas = len(self.values)
            if self.n_replicas > 0:
                self.mean = float(np.mean(self.values))
                self.std = float(np.std(self.values, ddof=1)) if self.n_replicas > 1 else 0.0
                self.min = float(np.min(self.values))
                self.max = float(np.max(self.values))


@dataclass
class StabilityGroupStats:
    """Aggregated statistics for a stability group across replicas."""
    
    # Performance metrics
    recall: MetricStats
    precision: MetricStats
    f1: MetricStats
    mcc: MetricStats
    tpr: MetricStats
    fpr: MetricStats
    
    # Additional metrics
    mean_pairwise_f1: MetricStats
    pair_count: MetricStats
    
    # Baseline metrics (optional)
    baseline_f1: Optional[MetricStats] = None
    baseline_mean_pairwise_f1: Optional[MetricStats] = None
    
    # Metadata
    group_name: str = ""
    n_replicas: int = 0


@dataclass
class OverallStats:
    """Overall model performance stats across replicas."""
    
    # Model metrics
    model_recall: MetricStats
    model_precision: MetricStats
    model_f1: MetricStats
    model_mcc: MetricStats
    model_mean_pairwise_f1: MetricStats
    
    # Baseline metrics (optional)
    baseline_recall: Optional[MetricStats] = None
    baseline_precision: Optional[MetricStats] = None
    baseline_f1: Optional[MetricStats] = None
    baseline_mcc: Optional[MetricStats] = None
    baseline_mean_pairwise_f1: Optional[MetricStats] = None
    
    # Metadata
    n_replicas: int = 0


@dataclass
class AggregatedMetrics:
    """Container for all aggregated metrics across replicas."""
    
    # Overall performance
    overall_stats: OverallStats
    
    # Stability group statistics
    training_frequency_stats: Dict[str, StabilityGroupStats] = field(default_factory=dict)
    test_frequency_stats: Dict[str, StabilityGroupStats] = field(default_factory=dict)
    
    # Time series data (optional)
    time_series_stats: Optional[Dict[str, List[MetricStats]]] = None
    
    # Metadata
    replica_paths: List[str] = field(default_factory=list)
    n_replicas: int = 0
    experiment_name: str = ""
    
    def get_stability_groups(self, frequency_type: str = "training") -> Dict[str, StabilityGroupStats]:
        """Get stability group stats by frequency type.
        
        Args:
            frequency_type: Either "training" or "test"
            
        Returns:
            Dictionary of stability group statistics
        """
        if frequency_type == "training":
            return self.training_frequency_stats
        elif frequency_type == "test":
            return self.test_frequency_stats
        else:
            raise ValueError(f"Invalid frequency_type: {frequency_type}. Must be 'training' or 'test'")
    
    def get_group_names(self, frequency_type: str = "training") -> List[str]:
        """Get list of stability group names.
        
        Args:
            frequency_type: Either "training" or "test"
            
        Returns:
            List of group names in standard order
        """
        groups = self.get_stability_groups(frequency_type)
        
        # Standard order for stability groups
        preferred_order = ['Rare (<5%)', 'Moderate (5-50%)', 'Stable (>50%)', 'Undefined']
        
        # Return groups in preferred order if they exist
        ordered_groups = []
        for group in preferred_order:
            if group in groups:
                ordered_groups.append(group)
        
        # Add any additional groups not in preferred order
        for group in groups:
            if group not in ordered_groups:
                ordered_groups.append(group)
        
        return ordered_groups


@dataclass 
class ReplicaInfo:
    """Information about a single replica."""
    
    replica_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    load_success: bool = True
    error_message: str = ""
    
    def __post_init__(self):
        """Extract replica identifier from path."""
        self.replica_id = Path(self.replica_path).parent.name
