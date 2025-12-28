"""Classification metrics for PPI dynamics prediction."""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
    precision_recall_curve,
)


@dataclass
class ClassificationMetrics:
    """Container for binary classification metrics."""
    
    # Primary metrics
    auroc: float = 0.0
    auprc: float = 0.0  # Average Precision - primary metric for imbalanced data
    
    # Threshold-dependent metrics
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    
    # Confusion matrix
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    
    # Loss
    loss: float = 0.0
    
    # Optimal threshold found on validation
    threshold: float = 0.5
    
    # Sample counts
    n_positive: int = 0
    n_negative: int = 0
    
    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        if total == 0:
            return 0.0
        return (self.tp + self.tn) / total
    
    @property
    def specificity(self) -> float:
        """True negative rate."""
        if self.tn + self.fp == 0:
            return 0.0
        return self.tn / (self.tn + self.fp)
    
    @property
    def balanced_accuracy(self) -> float:
        """Average of sensitivity (recall) and specificity."""
        return (self.recall + self.specificity) / 2
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "auroc": self.auroc,
            "auprc": self.auprc,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "specificity": self.specificity,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "loss": self.loss,
            "threshold": self.threshold,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
        }
    
    def __str__(self) -> str:
        return (
            f"AUROC: {self.auroc:.4f} | AUPRC: {self.auprc:.4f} | "
            f"F1: {self.f1:.4f} (P: {self.precision:.4f}, R: {self.recall:.4f}) | "
            f"Acc: {self.accuracy:.4f} | "
            f"TP: {self.tp}, FP: {self.fp}, TN: {self.tn}, FN: {self.fn}"
        )
    
    def short_str(self) -> str:
        """Short summary for logging."""
        return f"AUPRC: {self.auprc:.4f} | F1: {self.f1:.4f} | AUROC: {self.auroc:.4f}"


class MetricsComputer:
    """
    Computes classification metrics from predictions and targets.
    
    Accumulates predictions across batches and computes metrics at the end.
    """
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self) -> None:
        """Reset accumulated predictions."""
        self._logits: List[np.ndarray] = []
        self._targets: List[np.ndarray] = []
        self._losses: List[float] = []
        self._batch_sizes: List[int] = []
    
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss: Optional[float] = None,
    ) -> None:
        """
        Add batch predictions to accumulator.
        
        Args:
            logits: Raw logits (before sigmoid), shape (N,) or (N, 1)
            targets: Binary targets, shape (N,) or (N, 1)
            loss: Optional loss value for this batch
        """
        logits_np = logits.detach().cpu().view(-1).numpy()
        targets_np = targets.detach().cpu().view(-1).numpy()
        
        self._logits.append(logits_np)
        self._targets.append(targets_np)
        
        if loss is not None:
            self._losses.append(loss)
            self._batch_sizes.append(len(logits_np))
    
    def compute(self, tune_threshold: bool = False) -> ClassificationMetrics:
        """
        Compute metrics from accumulated predictions.
        
        Args:
            tune_threshold: If True, find optimal threshold for F1
        
        Returns:
            ClassificationMetrics with all computed values
        """
        if len(self._logits) == 0:
            return ClassificationMetrics()
        
        logits = np.concatenate(self._logits)
        targets = np.concatenate(self._targets)
        probs = 1 / (1 + np.exp(-logits))  # Sigmoid
        
        metrics = ClassificationMetrics()
        metrics.n_positive = int(targets.sum())
        metrics.n_negative = len(targets) - metrics.n_positive
        
        # Compute loss
        if self._losses:
            total_samples = sum(self._batch_sizes)
            metrics.loss = sum(
                l * n for l, n in zip(self._losses, self._batch_sizes)
            ) / total_samples
        
        # Handle edge cases
        if metrics.n_positive == 0 or metrics.n_negative == 0:
            # Cannot compute meaningful metrics with only one class
            metrics.auroc = 0.5
            metrics.auprc = metrics.n_positive / len(targets)
            metrics.threshold = self.threshold
            preds = (probs >= metrics.threshold).astype(int)
            self._compute_confusion_metrics(metrics, preds, targets)
            return metrics
        
        # AUROC and AUPRC
        try:
            metrics.auroc = roc_auc_score(targets, probs)
        except ValueError:
            metrics.auroc = 0.5
        
        try:
            metrics.auprc = average_precision_score(targets, probs)
        except ValueError:
            metrics.auprc = 0.0
        
        # Find optimal threshold if requested
        if tune_threshold:
            metrics.threshold = self._find_optimal_threshold(targets, probs)
        else:
            metrics.threshold = self.threshold
        
        # Compute threshold-dependent metrics
        preds = (probs >= metrics.threshold).astype(int)
        self._compute_confusion_metrics(metrics, preds, targets)
        
        return metrics
    
    def _find_optimal_threshold(
        self,
        targets: np.ndarray,
        probs: np.ndarray
    ) -> float:
        """Find threshold that maximizes F1 score."""
        precisions, recalls, thresholds = precision_recall_curve(targets, probs)
        
        # Compute F1 for each threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        
        # Find best threshold
        best_idx = np.argmax(f1_scores[:-1])  # Last element is placeholder
        return float(thresholds[best_idx])
    
    def _compute_confusion_metrics(
        self,
        metrics: ClassificationMetrics,
        preds: np.ndarray,
        targets: np.ndarray
    ) -> None:
        """Compute confusion matrix and derived metrics."""
        # Confusion matrix
        if len(np.unique(preds)) == 1 or len(np.unique(targets)) == 1:
            # Handle edge case where all predictions or targets are same class
            metrics.tp = int(((preds == 1) & (targets == 1)).sum())
            metrics.fp = int(((preds == 1) & (targets == 0)).sum())
            metrics.tn = int(((preds == 0) & (targets == 0)).sum())
            metrics.fn = int(((preds == 0) & (targets == 1)).sum())
        else:
            tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
            metrics.tp = int(tp)
            metrics.fp = int(fp)
            metrics.tn = int(tn)
            metrics.fn = int(fn)
        
        # Precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, preds, average='binary', zero_division=0
        )
        metrics.precision = float(precision)
        metrics.recall = float(recall)
        metrics.f1 = float(f1)


@dataclass 
class PerTimestepMetrics:
    """Metrics broken down by timestep for temporal analysis."""
    
    timesteps: List[int] = field(default_factory=list)
    aurocs: List[float] = field(default_factory=list)
    auprcs: List[float] = field(default_factory=list)
    f1s: List[float] = field(default_factory=list)
    n_positives: List[int] = field(default_factory=list)
    n_negatives: List[int] = field(default_factory=list)
    
    def add_timestep(
        self,
        timestep: int,
        metrics: ClassificationMetrics
    ) -> None:
        self.timesteps.append(timestep)
        self.aurocs.append(metrics.auroc)
        self.auprcs.append(metrics.auprc)
        self.f1s.append(metrics.f1)
        self.n_positives.append(metrics.n_positive)
        self.n_negatives.append(metrics.n_negative)
    
    @property
    def mean_auroc(self) -> float:
        return np.mean(self.aurocs) if self.aurocs else 0.0
    
    @property
    def mean_auprc(self) -> float:
        return np.mean(self.auprcs) if self.auprcs else 0.0
    
    @property
    def mean_f1(self) -> float:
        return np.mean(self.f1s) if self.f1s else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timesteps": self.timesteps,
            "aurocs": self.aurocs,
            "auprcs": self.auprcs,
            "f1s": self.f1s,
            "n_positives": self.n_positives,
            "n_negatives": self.n_negatives,
            "mean_auroc": self.mean_auroc,
            "mean_auprc": self.mean_auprc,
            "mean_f1": self.mean_f1,
        }


def compute_per_timestep_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    timesteps: torch.Tensor,
    threshold: float = 0.5,
) -> PerTimestepMetrics:
    """
    Compute metrics for each unique timestep.
    
    Args:
        logits: Raw logits, shape (N,)
        targets: Binary targets, shape (N,)
        timesteps: Timestep for each sample, shape (N,)
        threshold: Classification threshold
    
    Returns:
        PerTimestepMetrics with per-timestep breakdown
    """
    logits = logits.detach().cpu().view(-1)
    targets = targets.detach().cpu().view(-1)
    timesteps = timesteps.detach().cpu().view(-1)
    
    unique_timesteps = torch.unique(timesteps).numpy()
    per_ts_metrics = PerTimestepMetrics()
    
    for ts in sorted(unique_timesteps):
        mask = timesteps == ts
        ts_logits = logits[mask]
        ts_targets = targets[mask]
        
        computer = MetricsComputer(threshold=threshold)
        computer.update(ts_logits, ts_targets)
        metrics = computer.compute()
        
        per_ts_metrics.add_timestep(int(ts), metrics)
    
    return per_ts_metrics
