"""Training module for RAPID - Protein Interaction Dynamics prediction."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from tqdm import tqdm

from src.config import TrainingConfig
from src.data.dataset import PPIDataModule
from src.losses import get_loss_function
from src.metrics import ClassificationMetrics, MetricsComputer, find_optimal_threshold
from src.models.global_model import PPIGlobalModel
from src.models.rapid import RAPIDModel


class Trainer:
    """
    Trainer for RAPID model.

    Handles:
    - Training loop with oracle (ground-truth) history
    - Validation with autoregressive (predicted) history
    - Checkpointing and early stopping
    - Detailed metric logging

    Args:
        model: RAPIDModel instance
        data_module: PPIDataModule with train/val/test data
        config: Training configuration
        device: torch device
    """

    def __init__(
        self,
        model: RAPIDModel,
        data_module: PPIDataModule,
        config: TrainingConfig,
        device: torch.device,
        checkpoint_dir: Path,
        log_dir: Path,
        global_model: Optional[PPIGlobalModel] = None,
    ):
        self.model = model.to(device)
        self.data_module = data_module
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        self.global_model = global_model
        if global_model is not None:
            self.global_model = global_model.to(device)
            self.global_model.eval()
            self.global_emb = global_model.global_emb
        else:
            self.global_emb = None

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.criterion = get_loss_function(
            loss_type="focal",
            gamma=config.focal_gamma,
            alpha=config.focal_alpha,
        )

        self.train_metrics = MetricsComputer()
        self.val_metrics = MetricsComputer()

        self.best_val_auprc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

        self.optimal_threshold = 0.5

        self.history: Dict[str, list] = {
            "train_loss": [],
            "train_auprc": [],
            "train_auroc": [],
            "train_f1": [],
            "val_loss": [],
            "val_auprc": [],
            "val_auroc": [],
            "val_f1": [],
        }

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int) -> ClassificationMetrics:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()

        dataloader = self.data_module.get_train_dataloader()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch:03d} [Train]")

        for batch in pbar:
            entity1 = batch["entity1"].to(self.device)
            entity2 = batch["entity2"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits = self.model(
                entity1_ids=entity1,
                entity2_ids=entity2,
                entity1_history=batch["entity1_history"],
                entity2_history=batch["entity2_history"],
                entity1_history_t=batch["entity1_history_t"],
                entity2_history_t=batch["entity2_history_t"],
                graph_dict=self.data_module.graph_dict,
                global_emb=self.global_emb,
            )
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()

            if self.config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip_norm,
                )

            self.optimizer.step()
            self.train_metrics.update(logits, labels, loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return self.train_metrics.compute()

    @torch.no_grad()
    def validate(self, tune_threshold: bool = True) -> ClassificationMetrics:
        """
        Validate model using autoregressive inference.

        Args:
            tune_threshold: Whether to find optimal classification threshold

        Returns:
            ClassificationMetrics for the validation set
        """
        self.model.eval()
        self.val_metrics.reset()

        # Collect all predictions for threshold tuning
        all_logits = []
        all_labels = []

        # Initialize model with training history
        self.model.reset_inference_state()
        self.model.init_from_train_history(
            graph_dict=self.data_module.graph_dict,
            entity_history=self.data_module.entity_history,
            entity_history_t=self.data_module.entity_history_t,
            global_emb=self.global_emb,
            global_model=self.global_model,
        )

        # Get validation timesteps
        timesteps = sorted(self.data_module.val_dataset.unique_timesteps)

        pbar = tqdm(timesteps, desc="Validation")

        for t in pbar:
            # Get ALL known pairs with ground truth labels
            pairs, labels_np = self.data_module.get_history_pairs_for_timestep(
                t, split="valid"
            )

            # Process in batches
            batch_size = 128
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i : i + batch_size]
                batch_labels = labels_np[i : i + batch_size]

                entity1 = torch.LongTensor(batch_pairs[:, 0]).to(self.device)
                entity2 = torch.LongTensor(batch_pairs[:, 1]).to(self.device)
                labels = torch.FloatTensor(batch_labels).to(self.device)

                probs, preds = self.model.predict_batch(
                    entity1_ids=entity1,
                    entity2_ids=entity2,
                    timestep=t,
                    threshold=self.optimal_threshold,
                    update_history=True,
                )

                # Convert probs to logits for metrics
                probs_clamped = probs.clamp(1e-7, 1 - 1e-7)
                logits = torch.log(probs_clamped / (1 - probs_clamped))

                loss = self.criterion(logits, labels)
                self.val_metrics.update(logits, labels, loss.item())

                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        # Compute metrics
        metrics = self.val_metrics.compute(tune_threshold=tune_threshold)

        if tune_threshold and all_logits:
            # Find optimal threshold on validation predictions
            all_logits_cat = torch.cat(all_logits)
            all_labels_cat = torch.cat(all_labels)
            self.optimal_threshold = find_optimal_threshold(
                all_logits_cat, all_labels_cat
            )
            metrics.threshold = self.optimal_threshold

        return metrics

    def train(self) -> Dict[str, Any]:
        """
        Full training loop.

        Returns:
            Training history and final metrics
        """
        print(f"\n{'=' * 60}")
        print("Starting training")
        print(f"{'=' * 60}")
        print(f"Entities: {self.data_module.num_entities}")
        print(f"Relations: {self.data_module.num_rels}")
        print(f"Device: {self.device}")
        print(f"{'=' * 60}\n")

        for epoch in range(1, self.config.max_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            if epoch % self.config.eval_interval == 0:
                val_metrics = self.validate(tune_threshold=True)

                # Log metrics
                self._log_metrics(epoch, train_metrics, val_metrics)

                # Check for improvement
                if val_metrics.auprc > self.best_val_auprc:
                    self.best_val_auprc = val_metrics.auprc
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, val_metrics, is_best=True)
                else:
                    self.patience_counter += 1

                # Early stopping
                if self.patience_counter >= self.config.patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    print(
                        f"Best epoch: {self.best_epoch} with AUPRC: {self.best_val_auprc:.4f}"
                    )
                    break

            # Update history
            self.history["train_loss"].append(train_metrics.loss)
            self.history["train_auprc"].append(train_metrics.auprc)
            self.history["train_auroc"].append(train_metrics.auroc)
            self.history["train_f1"].append(train_metrics.f1)

            if epoch % self.config.eval_interval == 0:
                self.history["val_loss"].append(val_metrics.loss)
                self.history["val_auprc"].append(val_metrics.auprc)
                self.history["val_auroc"].append(val_metrics.auroc)
                self.history["val_f1"].append(val_metrics.f1)

        # Save final checkpoint
        self._save_checkpoint(epoch, val_metrics, is_best=False)

        # Save history
        self._save_history()

        return {
            "history": self.history,
            "best_epoch": self.best_epoch,
            "best_val_auprc": self.best_val_auprc,
            "optimal_threshold": self.optimal_threshold,
        }

    def _log_metrics(
        self,
        epoch: int,
        train_metrics: ClassificationMetrics,
        val_metrics: ClassificationMetrics,
    ) -> None:
        """Log metrics to console."""
        print(f"\nEpoch {epoch:03d}")
        print(f"  Train: {train_metrics.short_str()}")
        print(f"  Val:   {val_metrics.short_str()}")
        print(f"  Threshold: {self.optimal_threshold:.3f}")

    def _save_checkpoint(
        self, epoch: int, metrics: ClassificationMetrics, is_best: bool = False
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": {
                "training": {
                    "learning_rate": self.config.learning_rate,
                    "max_epochs": self.config.max_epochs,
                    "patience": self.config.patience,
                    "focal_gamma": self.config.focal_gamma,
                },
                "model": {
                    "hidden_dim": self.model.hidden_dim,
                    "seq_len": self.model.seq_len,
                    "num_entities": self.model.num_entities,
                    "num_rels": self.model.num_rels,
                },
            },
            "metrics": metrics.to_dict(),
            "optimal_threshold": self.optimal_threshold,
            "best_val_auprc": self.best_val_auprc,
        }

        # Save latest
        latest_path = self.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
            print(f"  ✓ New best model saved (AUPRC: {metrics.auprc:.4f})")

    def _save_history(self) -> None:
        """Save training history."""
        history_path = self.log_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
