"""
Trainer for TrajectoryRAPID model.

Handles trajectory-level training with:
- Sequence loss over full trajectories
- Transition weighting (optional)
- Evaluation on val/test portions of trajectory
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.data.trajectory_dataset import TrajectoryDataModule
from src.models.trajectory_rapid import TrajectoryRAPIDModel


class TrajectoryMetrics:
    """Compute and store trajectory prediction metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.n_batches = 0
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.all_probs = []
        self.all_labels = []

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss: float,
        threshold: float = 0.5,
    ):
        """Update metrics with batch results."""
        self.total_loss += loss
        self.n_batches += 1

        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        self.tp += ((preds == 1) & (labels == 1)).sum().item()
        self.fp += ((preds == 1) & (labels == 0)).sum().item()
        self.tn += ((preds == 0) & (labels == 0)).sum().item()
        self.fn += ((preds == 0) & (labels == 1)).sum().item()

        self.all_probs.append(probs.detach().cpu())
        self.all_labels.append(labels.detach().cpu())

    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        loss = self.total_loss / max(self.n_batches, 1)
        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        accuracy = (self.tp + self.tn) / max(self.tp + self.fp + self.tn + self.fn, 1)

        # Compute AUROC and AUPRC
        if self.all_probs:
            all_probs = torch.cat(self.all_probs).numpy().flatten()
            all_labels = torch.cat(self.all_labels).numpy().flatten()

            try:
                from sklearn.metrics import roc_auc_score, average_precision_score

                auroc = roc_auc_score(all_labels, all_probs)
                auprc = average_precision_score(all_labels, all_probs)
            except Exception:
                auroc = 0.0
                auprc = 0.0
        else:
            auroc = 0.0
            auprc = 0.0

        return {
            "loss": loss,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "auroc": auroc,
            "auprc": auprc,
        }


class TransitionMetrics:
    """Metrics specifically for transition events."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.trans_tp = 0
        self.trans_fp = 0
        self.trans_fn = 0

    def update(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        history_last: torch.Tensor,
        threshold: float = 0.5,
    ):
        """
        Update transition metrics.

        Args:
            probs: Predicted probabilities (batch, traj_len)
            labels: Ground truth (batch, traj_len)
            history_last: Last history state (batch,)
            threshold: Classification threshold
        """
        preds = (probs > threshold).float()

        # Build full sequence for transition detection
        prev_state = history_last.unsqueeze(1)  # (batch, 1)
        full_labels = torch.cat([prev_state, labels], dim=1)
        full_preds = torch.cat([prev_state, preds], dim=1)

        # Detect transitions
        true_trans = (full_labels[:, 1:] != full_labels[:, :-1]).float()
        pred_trans = (full_preds[:, 1:] != full_preds[:, :-1]).float()

        self.trans_tp += ((pred_trans == 1) & (true_trans == 1)).sum().item()
        self.trans_fp += ((pred_trans == 1) & (true_trans == 0)).sum().item()
        self.trans_fn += ((pred_trans == 0) & (true_trans == 1)).sum().item()

    def compute(self) -> Dict[str, float]:
        """Compute transition-specific metrics."""
        precision = self.trans_tp / max(self.trans_tp + self.trans_fp, 1)
        recall = self.trans_tp / max(self.trans_tp + self.trans_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        return {
            "trans_precision": precision,
            "trans_recall": recall,
            "trans_f1": f1,
        }


class TrajectoryTrainer:
    """
    Trainer for trajectory-level prediction.

    Args:
        model: TrajectoryRAPIDModel
        data_module: TrajectoryDataModule
        learning_rate: Learning rate
        weight_decay: Weight decay
        patience: Early stopping patience
        checkpoint_dir: Directory for checkpoints
        device: Torch device
        transition_weight: Weight for transition timesteps in loss
        use_focal_loss: Use focal loss instead of BCE
        focal_gamma: Focal loss gamma parameter
    """

    def __init__(
        self,
        model: TrajectoryRAPIDModel,
        data_module: TrajectoryDataModule,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 10,
        checkpoint_dir: Path = Path("checkpoints"),
        log_dir: Path = Path("logs"),
        device: torch.device = torch.device("cpu"),
        transition_weight: float = 1.0,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
    ):
        self.model = model.to(device)
        self.data_module = data_module
        self.device = device
        self.patience = patience
        self.transition_weight = transition_weight
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma

        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=3,
            verbose=True,
        )

        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

        self.best_val_auprc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

        self.history: Dict[str, List] = {
            "train_loss": [],
            "train_f1": [],
            "train_auprc": [],
            "val_loss": [],
            "val_f1": [],
            "val_auprc": [],
            "val_trans_f1": [],
        }

    def _compute_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss with optional transition weighting and focal loss.

        Args:
            logits: Predicted logits (batch, traj_len)
            target: Ground truth (batch, traj_len)
            history: History states (batch, hist_len)

        Returns:
            Scalar loss
        """
        # Base loss
        if self.use_focal_loss:
            # Focal loss: FL = -alpha * (1-p)^gamma * log(p)
            probs = torch.sigmoid(logits)
            pt = torch.where(target == 1, probs, 1 - probs)
            focal_weight = (1 - pt) ** self.focal_gamma
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            loss_per_elem = focal_weight * bce
        else:
            loss_per_elem = self.criterion(logits, target)

        # Transition weighting
        if self.transition_weight > 1.0:
            # Detect transitions
            last_hist = history[:, -1].unsqueeze(1)
            full_seq = torch.cat([last_hist, target], dim=1)
            is_transition = (full_seq[:, 1:] != full_seq[:, :-1]).float()

            # Weight transitions higher
            weights = 1.0 + (self.transition_weight - 1.0) * is_transition
            loss = (loss_per_elem * weights).mean()
        else:
            loss = loss_per_elem.mean()

        return loss

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metrics = TrajectoryMetrics()

        dataloader = self.data_module.get_train_dataloader()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:03d} [Train]")

        for batch in pbar:
            # Move to device
            entity1 = batch["entity1"].to(self.device)
            entity2 = batch["entity2"].to(self.device)
            history = batch["history"].to(self.device)
            target = batch["target"].to(self.device)
            history_timesteps = batch["history_timesteps"].to(self.device)
            neighbor_e1 = batch["neighbor_entity1"].to(self.device)
            neighbor_e2 = batch["neighbor_entity2"].to(self.device)
            neighbor_hist = batch["neighbor_history"].to(self.device)
            neighbor_mask = batch["neighbor_mask"].to(self.device)

            traj_len = target.size(1)

            # Forward
            logits = self.model(
                entity1_ids=entity1,
                entity2_ids=entity2,
                history=history,
                history_timesteps=history_timesteps,
                neighbor_entity1=neighbor_e1,
                neighbor_entity2=neighbor_e2,
                neighbor_history=neighbor_hist,
                neighbor_mask=neighbor_mask,
                traj_len=traj_len,
                graph_dict=self.data_module.graph_dict,
            )

            # Loss
            loss = self._compute_loss(logits, target, history)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            metrics.update(logits, target, loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return metrics.compute()

    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
        timestep_range: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Evaluate model.

        Args:
            dataloader: DataLoader to evaluate on
            timestep_range: Optional (start, end) to evaluate specific timesteps

        Returns:
            Tuple of (general_metrics, transition_metrics)
        """
        self.model.eval()
        metrics = TrajectoryMetrics()
        trans_metrics = TransitionMetrics()

        for batch in dataloader:
            entity1 = batch["entity1"].to(self.device)
            entity2 = batch["entity2"].to(self.device)
            history = batch["history"].to(self.device)
            target = batch["target"].to(self.device)
            history_timesteps = batch["history_timesteps"].to(self.device)
            neighbor_e1 = batch["neighbor_entity1"].to(self.device)
            neighbor_e2 = batch["neighbor_entity2"].to(self.device)
            neighbor_hist = batch["neighbor_history"].to(self.device)
            neighbor_mask = batch["neighbor_mask"].to(self.device)

            full_traj_len = target.size(1)

            # Predict full trajectory
            logits = self.model(
                entity1_ids=entity1,
                entity2_ids=entity2,
                history=history,
                history_timesteps=history_timesteps,
                neighbor_entity1=neighbor_e1,
                neighbor_entity2=neighbor_e2,
                neighbor_history=neighbor_hist,
                neighbor_mask=neighbor_mask,
                traj_len=full_traj_len,
                graph_dict=self.data_module.graph_dict,
            )

            # Slice to evaluation range if specified
            if timestep_range is not None:
                start, end = timestep_range
                logits = logits[:, start:end]
                target = target[:, start:end]

            loss = self.criterion(logits, target).mean()
            metrics.update(logits, target, loss.item())

            probs = torch.sigmoid(logits)
            trans_metrics.update(probs, target, history[:, -1])

        return metrics.compute(), trans_metrics.compute()

    def train(self, n_epochs: int = 50) -> Dict[str, Any]:
        """
        Full training loop.

        Returns:
            Training history and final metrics
        """
        print(f"\n{'='*60}")
        print("Starting TrajectoryRAPID Training")
        print(f"{'='*60}")
        print(f"Model: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"Device: {self.device}")
        print(f"Transition weight: {self.transition_weight}")
        print(f"Use focal loss: {self.use_focal_loss}")
        print(f"{'='*60}\n")

        val_dataloader = self.data_module.get_val_dataloader()
        val_range = self.data_module.val_timestep_indices

        for epoch in range(1, n_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics, val_trans = self.evaluate(val_dataloader, val_range)

            # Update scheduler
            self.scheduler.step(val_metrics["auprc"])

            # Log
            print(f"\nEpoch {epoch:03d}")
            print(
                f"  Train: Loss={train_metrics['loss']:.4f}, "
                f"F1={train_metrics['f1']:.4f}, AUPRC={train_metrics['auprc']:.4f}"
            )
            print(
                f"  Val:   Loss={val_metrics['loss']:.4f}, "
                f"F1={val_metrics['f1']:.4f}, AUPRC={val_metrics['auprc']:.4f}"
            )
            print(
                f"  Trans: P={val_trans['trans_precision']:.4f}, "
                f"R={val_trans['trans_recall']:.4f}, F1={val_trans['trans_f1']:.4f}"
            )

            # Update history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_f1"].append(train_metrics["f1"])
            self.history["train_auprc"].append(train_metrics["auprc"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["val_auprc"].append(val_metrics["auprc"])
            self.history["val_trans_f1"].append(val_trans["trans_f1"])

            # Check for improvement
            if val_metrics["auprc"] > self.best_val_auprc:
                self.best_val_auprc = val_metrics["auprc"]
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"Best epoch: {self.best_epoch} with AUPRC: {self.best_val_auprc:.4f}")
                break

        # Save final checkpoint
        self._save_checkpoint(epoch, val_metrics, is_best=False)
        self._save_history()

        return {
            "history": self.history,
            "best_epoch": self.best_epoch,
            "best_val_auprc": self.best_val_auprc,
        }

    def evaluate_test(self) -> Dict[str, float]:
        """Evaluate on test set."""
        print("\nEvaluating on test set...")

        test_dataloader = self.data_module.get_test_dataloader()
        test_range = self.data_module.test_timestep_indices

        test_metrics, test_trans = self.evaluate(test_dataloader, test_range)

        print(f"\nTest Results:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  AUROC: {test_metrics['auroc']:.4f}")
        print(f"  AUPRC: {test_metrics['auprc']:.4f}")
        print(f"  F1: {test_metrics['f1']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  Trans F1: {test_trans['trans_f1']:.4f}")

        return {**test_metrics, **test_trans}

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "best_val_auprc": self.best_val_auprc,
            "config": {
                "hidden_dim": self.model.hidden_dim,
                "use_rgcn": self.model.use_rgcn,
                "use_global_context": self.model.use_global_context,
                "use_neighbor_attention": self.model.use_neighbor_attention,
                "transition_weight": self.transition_weight,
            },
        }

        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / "latest.pth")

        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pth")
            print(f"  ✓ New best model saved (AUPRC: {metrics['auprc']:.4f})")

    def _save_history(self):
        """Save training history."""
        with open(self.log_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

    def load_checkpoint(self, path: Path):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {path}")
        return checkpoint
