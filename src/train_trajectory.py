"""
Trainer for trajectory-level prediction model.

Trains on full trajectory sequences instead of per-timestep samples.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.data.trajectory_dataset import TrajectoryDataModule
from src.models.trajectory import TrajectoryModel


class TrajectoryTrainer:
    """
    Trainer for trajectory-level prediction.

    Differences from per-timestep trainer:
    - Loss computed over full trajectory (sequence-level)
    - No autoregressive during training (teacher forcing)
    - Batch = edges, not (edge, timestep) pairs
    """

    def __init__(
        self,
        model: TrajectoryModel,
        data_module: TrajectoryDataModule,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 5,
        checkpoint_dir: Path = Path("checkpoints"),
        device: torch.device = torch.device("cpu"),
        transition_weight: float = 1.0,
    ):
        self.model = model.to(device)
        self.data_module = data_module
        self.device = device
        self.patience = patience
        self.transition_weight = transition_weight
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

        self.best_loss = float("inf")
        self.epochs_without_improvement = 0
        self.history: Dict[str, List] = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
        }

    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch."""
        self.model.train()

        dataloader = self.data_module.get_train_dataloader()

        total_loss = 0.0
        total_tp = total_fp = total_tn = total_fn = 0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch:03d} [Train]")

        for batch in pbar:
            # Move to device
            entity1 = batch["entity1"].to(self.device)
            entity2 = batch["entity2"].to(self.device)
            history = batch["history"].to(self.device)
            target = batch["target"].to(self.device)
            neighbor_e1 = batch["neighbor_entity1"].to(self.device)
            neighbor_e2 = batch["neighbor_entity2"].to(self.device)
            neighbor_hist = batch["neighbor_history"].to(self.device)
            neighbor_mask = batch["neighbor_mask"].to(self.device)

            traj_len = target.size(1)

            # Forward pass
            logits = self.model(
                entity1_ids=entity1,
                entity2_ids=entity2,
                history=history,
                neighbor_entity1=neighbor_e1,
                neighbor_entity2=neighbor_e2,
                neighbor_history=neighbor_hist,
                traj_len=traj_len,
                neighbor_mask=neighbor_mask,
            )

            # Sequence loss with transition weighting
            loss_per_elem = self.criterion(logits, target)

            # Weight timesteps where transitions occur in target
            if self.transition_weight > 1.0:
                # Detect transitions: compare each timestep to the previous
                # For first timestep, compare to last history state
                last_history_state = history[:, -1].unsqueeze(1)  # (batch, 1)
                full_seq = torch.cat(
                    [last_history_state, target], dim=1
                )  # (batch, traj_len+1)
                is_transition = (
                    full_seq[:, 1:] != full_seq[:, :-1]
                ).float()  # (batch, traj_len)

                # Create weights: base 1.0, add (weight-1) for transitions
                weights = 1.0 + (self.transition_weight - 1.0) * is_transition
                loss = (loss_per_elem * weights).mean()
            else:
                loss = loss_per_elem.mean()

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Metrics
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_tp += ((preds == 1) & (target == 1)).sum().item()
            total_fp += ((preds == 1) & (target == 0)).sum().item()
            total_tn += ((preds == 0) & (target == 0)).sum().item()
            total_fn += ((preds == 0) & (target == 1)).sum().item()

            total_loss += loss.item()
            n_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Compute metrics
        avg_loss = total_loss / max(n_batches, 1)
        precision = total_tp / max(total_tp + total_fp, 1)
        recall = total_tp / max(total_tp + total_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        return {
            "loss": avg_loss,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": (total_tp + total_tn)
            / max(total_tp + total_fp + total_tn + total_fn, 1),
        }

    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict:
        """Evaluate on a dataloader."""
        self.model.eval()

        total_loss = 0.0
        total_tp = total_fp = total_tn = total_fn = 0
        n_batches = 0

        all_preds = []
        all_targets = []

        for batch in dataloader:
            entity1 = batch["entity1"].to(self.device)
            entity2 = batch["entity2"].to(self.device)
            history = batch["history"].to(self.device)
            target = batch["target"].to(self.device)
            neighbor_e1 = batch["neighbor_entity1"].to(self.device)
            neighbor_e2 = batch["neighbor_entity2"].to(self.device)
            neighbor_hist = batch["neighbor_history"].to(self.device)
            neighbor_mask = batch["neighbor_mask"].to(self.device)

            traj_len = target.size(1)

            logits = self.model(
                entity1_ids=entity1,
                entity2_ids=entity2,
                history=history,
                neighbor_entity1=neighbor_e1,
                neighbor_entity2=neighbor_e2,
                neighbor_history=neighbor_hist,
                traj_len=traj_len,
                neighbor_mask=neighbor_mask,
            )

            loss = self.criterion(logits, target).mean()

            preds = (torch.sigmoid(logits) > 0.5).float()
            total_tp += ((preds == 1) & (target == 1)).sum().item()
            total_fp += ((preds == 1) & (target == 0)).sum().item()
            total_tn += ((preds == 0) & (target == 0)).sum().item()
            total_fn += ((preds == 0) & (target == 1)).sum().item()

            total_loss += loss.item()
            n_batches += 1

            all_preds.append(torch.sigmoid(logits).cpu())
            all_targets.append(target.cpu())

        avg_loss = total_loss / max(n_batches, 1)
        precision = total_tp / max(total_tp + total_fp, 1)
        recall = total_tp / max(total_tp + total_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        # Compute transition metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        trans_metrics = self._compute_transition_metrics(all_preds, all_targets)

        return {
            "loss": avg_loss,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": (total_tp + total_tn)
            / max(total_tp + total_fp + total_tn + total_fn, 1),
            **trans_metrics,
        }

    def _compute_transition_metrics(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict:
        """Compute transition-specific metrics."""
        batch_size, traj_len = preds.shape

        if traj_len < 2:
            return {"transition_f1": 0.0, "persistence_acc": 1.0}

        # Binary predictions
        pred_binary = (preds > 0.5).float()

        # Compute transitions between consecutive timesteps
        pred_trans = (pred_binary[:, 1:] != pred_binary[:, :-1]).float()
        target_trans = (targets[:, 1:] != targets[:, :-1]).float()

        # Transition metrics
        trans_tp = ((pred_trans == 1) & (target_trans == 1)).sum().item()
        trans_fp = ((pred_trans == 1) & (target_trans == 0)).sum().item()
        trans_fn = ((pred_trans == 0) & (target_trans == 1)).sum().item()

        trans_precision = trans_tp / max(trans_tp + trans_fp, 1)
        trans_recall = trans_tp / max(trans_tp + trans_fn, 1)
        trans_f1 = (
            2
            * trans_precision
            * trans_recall
            / max(trans_precision + trans_recall, 1e-8)
        )

        # Persistence accuracy (how often no-transition was correct)
        persist_correct = ((pred_trans == 0) & (target_trans == 0)).sum().item()
        persist_total = (target_trans == 0).sum().item()
        persistence_acc = persist_correct / max(persist_total, 1)

        return {
            "transition_f1": trans_f1,
            "transition_precision": trans_precision,
            "transition_recall": trans_recall,
            "persistence_acc": persistence_acc,
        }

    def train(self, n_epochs: int) -> Dict:
        """Full training loop."""
        print(f"\nStarting training for {n_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Patience: {self.patience}")
        print()

        for epoch in range(1, n_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_metrics"].append(train_metrics)

            # Validate on test set (since we predict val trajectory from train)
            test_loader = self.data_module.get_test_dataloader()
            val_metrics = self.evaluate(test_loader)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_metrics"].append(val_metrics)

            # Log
            print(f"\nEpoch {epoch:03d}")
            print(
                f"  Train: Loss={train_metrics['loss']:.4f} | F1={train_metrics['f1']:.4f}"
            )
            print(
                f"  Test:  Loss={val_metrics['loss']:.4f} | F1={val_metrics['f1']:.4f} | "
                f"Trans F1={val_metrics['transition_f1']:.4f} | Persist={val_metrics['persistence_acc']:.4f}"
            )

            # Checkpointing
            if val_metrics["loss"] < self.best_loss:
                self.best_loss = val_metrics["loss"]
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, is_best=True)
                print(f"  ✓ New best model saved")
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        print(f"\nTraining complete! Best loss: {self.best_loss:.4f}")
        return self.history

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
        }

        path = self.checkpoint_dir / ("best.pth" if is_best else f"epoch_{epoch}.pth")
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_loss = checkpoint.get("best_loss", float("inf"))
