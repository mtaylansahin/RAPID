"""Training module for RAPID encoder-decoder architecture."""

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from src.config import TrainingConfig
from src.data.dataset import PPIDataModule
from src.losses import get_loss_function
from src.metrics import ClassificationMetrics, MetricsComputer
from src.models.decoder import TemporalEdgeDecoder
from src.models.encoder import RAPIDEncoder


class Trainer:
    """
    Trainer for RAPID encoder-decoder model.

    Trains decoder using validation timesteps as targets.
    Encoder can be frozen or fine-tuned with lower learning rate.

    Args:
        encoder: RAPIDEncoder wrapper
        decoder: TemporalEdgeDecoder
        data_module: PPIDataModule with train/val/test data
        config: TrainingConfig
        device: Torch device
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory to save logs
    """

    def __init__(
        self,
        encoder: RAPIDEncoder,
        decoder: TemporalEdgeDecoder,
        data_module: PPIDataModule,
        config: TrainingConfig,
        device: torch.device,
        checkpoint_dir: Path,
        log_dir: Path,
    ):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.data_module = data_module
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)

        # Optimizer: different LR for encoder vs decoder
        param_groups = [
            {"params": self.decoder.parameters(), "lr": config.learning_rate},
        ]
        if not config.freeze_encoder:
            param_groups.append(
                {
                    "params": self.encoder.parameters(),
                    "lr": config.encoder_lr,
                }
            )

        self.optimizer = torch.optim.Adam(
            param_groups, weight_decay=config.weight_decay
        )
        self.criterion = get_loss_function(loss_type="focal", gamma=config.focal_gamma)

        # Get known pairs and timesteps
        self.known_pairs = self._get_known_pairs()
        self.train_max_time = data_module.train_max_time
        self.val_timesteps = sorted(data_module.val_dataset.unique_timesteps)
        self.test_timesteps = sorted(data_module.test_dataset.unique_timesteps)

        # Tracking
        self.best_val_auprc = 0.0
        self.best_epoch = 0
        self.optimal_threshold = 0.5
        self.patience_counter = 0
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_auprc": [],
            "val_f1": [],
        }

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _get_known_pairs(self) -> torch.Tensor:
        """Get all known pairs as tensor."""
        if not hasattr(self.data_module, "known_pairs_list"):
            # Trigger lazy computation
            self.data_module.get_history_pairs_for_timestep(0, split="test")
        return torch.tensor(self.data_module.known_pairs_list, dtype=torch.long)

    def _get_target_matrix(self, timesteps: List[int], split: str) -> torch.Tensor:
        """
        Build target matrix for all pairs × timesteps.

        Args:
            timesteps: List of timesteps
            split: 'valid' or 'test'

        Returns:
            targets: (num_pairs, num_timesteps) binary tensor
        """
        dataset = (
            self.data_module.val_dataset
            if split == "valid"
            else self.data_module.test_dataset
        )
        num_pairs = len(self.known_pairs)
        num_timesteps = len(timesteps)

        targets = torch.zeros(num_pairs, num_timesteps)

        for t_idx, t in enumerate(timesteps):
            pos_edges = dataset.positives_by_timestep.get(t, set())
            for p_idx, (e1, e2) in enumerate(self.known_pairs.tolist()):
                if (e1, e2) in pos_edges or (e2, e1) in pos_edges:
                    targets[p_idx, t_idx] = 1.0

        return targets

    def _get_previous_states(self, timesteps: List[int], split: str) -> torch.Tensor:
        """
        Get edge states at timestep before each target timestep.

        Returns:
            prev_states: (num_pairs, num_timesteps) with state at t-1 for each t
        """
        # Build list of all timesteps we have data for
        all_times = sorted(
            set(self.data_module.train_dataset.timesteps)
            | set(self.data_module.val_dataset.timesteps)
        )

        num_pairs = len(self.known_pairs)
        num_timesteps = len(timesteps)
        prev_states = torch.zeros(num_pairs, num_timesteps)

        for t_idx, t in enumerate(timesteps):
            # Find previous timestep
            prev_t = None
            for candidate in reversed(all_times):
                if candidate < t:
                    prev_t = candidate
                    break

            if prev_t is None:
                continue

            # Get edges at prev_t
            prev_edges = set()
            for ds in [self.data_module.train_dataset, self.data_module.val_dataset]:
                prev_edges.update(ds.positives_by_timestep.get(prev_t, set()))

            for p_idx, (e1, e2) in enumerate(self.known_pairs.tolist()):
                if (e1, e2) in prev_edges or (e2, e1) in prev_edges:
                    prev_states[p_idx, t_idx] = 1.0

        return prev_states

    def _get_edge_history(self) -> torch.Tensor:
        """
        Get FULL edge state history for each pair from training data.

        Returns:
            edge_history: (num_pairs, num_train_timesteps) binary tensor
                          ordered chronologically (oldest first, newest last)
        """
        # Get all train timesteps sorted chronologically
        train_times = sorted(set(self.data_module.train_dataset.timesteps))

        num_pairs = len(self.known_pairs)
        num_timesteps = len(train_times)
        edge_history = torch.zeros(num_pairs, num_timesteps)

        # Fill history chronologically
        for t_idx, t in enumerate(train_times):
            pos_edges = self.data_module.train_dataset.positives_by_timestep.get(
                t, set()
            )
            for p_idx, (e1, e2) in enumerate(self.known_pairs.tolist()):
                if (e1, e2) in pos_edges or (e2, e1) in pos_edges:
                    edge_history[p_idx, t_idx] = 1.0

        return edge_history

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch on validation timesteps."""
        if self.config.freeze_encoder:
            self.encoder.eval()
        else:
            self.encoder.train()
        self.decoder.train()

        # Encode context from training data
        with torch.no_grad() if self.config.freeze_encoder else torch.enable_grad():
            entity_context = self.encoder(
                self.data_module.graph_dict,
                self.data_module.entity_history,
                self.data_module.entity_history_t,
            )

        # Prepare targets
        target_matrix = self._get_target_matrix(self.val_timesteps, "valid").to(
            self.device
        )

        # Relative timesteps (from train boundary)
        relative_t = torch.tensor(
            [t - self.train_max_time for t in self.val_timesteps],
            dtype=torch.long,
            device=self.device,
        )

        # Get previous states for transition weighting
        prev_states = None
        if self.config.transition_weight > 1.0:
            prev_states = self._get_previous_states(self.val_timesteps, "valid").to(
                self.device
            )

        # Get edge history if decoder uses it
        edge_history = None
        if self.decoder.use_edge_history:
            edge_history = self._get_edge_history().to(self.device)

        # Forward pass (process pairs in batches for memory)
        pair_batch_size = 256
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            range(0, len(self.known_pairs), pair_batch_size),
            desc=f"Epoch {epoch:03d}",
        )

        for start_idx in pbar:
            end_idx = min(start_idx + pair_batch_size, len(self.known_pairs))
            batch_pairs = self.known_pairs[start_idx:end_idx].to(self.device)
            batch_targets = target_matrix[start_idx:end_idx]

            # Get edge history batch
            batch_edge_history = None
            if edge_history is not None:
                batch_edge_history = edge_history[start_idx:end_idx]

            # Get subgraph context batch if decoder uses edge history encoder
            batch_subgraph_context = None
            if self.decoder.use_edge_history and hasattr(
                self.data_module, "get_subgraph_context"
            ):
                batch_subgraph_context = self.data_module.get_subgraph_context(
                    batch_pairs.tolist()
                )
                if batch_subgraph_context is not None:
                    # Move tensors to device
                    batch_subgraph_context = {
                        k: v.to(self.device) for k, v in batch_subgraph_context.items()
                    }

            # Decode
            logits = self.decoder(
                entity_context,
                batch_pairs,
                relative_t,
                edge_history=batch_edge_history,
                subgraph_context=batch_subgraph_context,
            )

            # Compute transition weights
            if self.config.transition_weight > 1.0:
                batch_prev = prev_states[start_idx:end_idx]
                # Identify transitions: where prev != target
                is_transition = (batch_prev != batch_targets).float()
                # Weight: transition_weight for transitions, 1.0 for persistence
                weights = 1.0 + (self.config.transition_weight - 1.0) * is_transition
                weights = weights.view(-1)

                # Compute per-sample loss and apply weights
                per_sample_loss = self.criterion(
                    logits.view(-1), batch_targets.view(-1)
                )
                if per_sample_loss.dim() == 0:
                    # Criterion returns mean, need to recompute
                    import torch.nn.functional as F

                    per_sample_loss = F.binary_cross_entropy_with_logits(
                        logits.view(-1), batch_targets.view(-1), reduction="none"
                    )
                loss = (per_sample_loss * weights).mean()
            else:
                # Standard loss without weighting
                loss = self.criterion(logits.view(-1), batch_targets.view(-1))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                self.config.grad_clip_norm,
            )
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self) -> ClassificationMetrics:
        """Validate on test timesteps."""
        self.encoder.eval()
        self.decoder.eval()

        # Use train+val context for test evaluation
        context_graph_dict, context_history, context_history_t = (
            self.data_module.get_train_val_context()
        )

        entity_context = self.encoder(
            context_graph_dict, context_history, context_history_t
        )

        # Get test targets
        target_matrix = self._get_target_matrix(self.test_timesteps, "test").to(
            self.device
        )

        # Relative timesteps (from train boundary)
        relative_t = torch.tensor(
            [t - self.train_max_time for t in self.test_timesteps],
            dtype=torch.long,
            device=self.device,
        )

        # Get edge history for validation if decoder uses it
        val_edge_history = None
        if self.decoder.use_edge_history:
            # For validation on test set, we use history from train data
            val_edge_history = self._get_edge_history().to(self.device)

        # Predict all at once (in batches for memory)
        all_logits = []
        pair_batch_size = 512

        for start_idx in range(0, len(self.known_pairs), pair_batch_size):
            end_idx = min(start_idx + pair_batch_size, len(self.known_pairs))
            batch_pairs = self.known_pairs[start_idx:end_idx].to(self.device)
            batch_edge_history = None
            if val_edge_history is not None:
                batch_edge_history = val_edge_history[start_idx:end_idx]

            # Get subgraph context batch if decoder uses edge history encoder
            batch_subgraph_context = None
            if self.decoder.use_edge_history and hasattr(
                self.data_module, "get_subgraph_context"
            ):
                batch_subgraph_context = self.data_module.get_subgraph_context(
                    batch_pairs.tolist()
                )
                if batch_subgraph_context is not None:
                    batch_subgraph_context = {
                        k: v.to(self.device) for k, v in batch_subgraph_context.items()
                    }

            logits = self.decoder(
                entity_context,
                batch_pairs,
                relative_t,
                edge_history=batch_edge_history,
                subgraph_context=batch_subgraph_context,
            )
            all_logits.append(logits.cpu())

        all_logits = torch.cat(all_logits, dim=0)  # (num_pairs, num_timesteps)

        # Flatten for metrics
        logits_flat = all_logits.view(-1)
        targets_flat = target_matrix.view(-1).cpu()

        metrics_computer = MetricsComputer()
        metrics_computer.update(logits_flat, targets_flat)
        return metrics_computer.compute(tune_threshold=True)

    def train(self) -> Dict[str, Any]:
        """Full training loop."""
        print(f"\n{'=' * 60}")
        print("Training Encoder-Decoder Model")
        print(f"{'=' * 60}")
        print(f"  Known pairs: {len(self.known_pairs)}")
        print(f"  Val timesteps: {len(self.val_timesteps)}")
        print(f"  Test timesteps: {len(self.test_timesteps)}")
        print(f"  Encoder frozen: {self.config.freeze_encoder}")
        print(f"{'=' * 60}\n")

        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self.train_epoch(epoch)
            self.history["train_loss"].append(train_loss)

            if epoch % self.config.eval_interval == 0:
                val_metrics = self.validate()
                self.history["val_auprc"].append(val_metrics.auprc)
                self.history["val_f1"].append(val_metrics.f1)

                print(
                    f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | "
                    f"{val_metrics.short_str()}"
                )

                if val_metrics.auprc > self.best_val_auprc:
                    self.best_val_auprc = val_metrics.auprc
                    self.best_epoch = epoch
                    self.optimal_threshold = val_metrics.threshold
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, val_metrics, is_best=True)
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.config.patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        self._save_history()

        return {
            "history": self.history,
            "best_epoch": self.best_epoch,
            "best_val_auprc": self.best_val_auprc,
            "optimal_threshold": self.optimal_threshold,
        }

    def _save_checkpoint(
        self, epoch: int, metrics: ClassificationMetrics, is_best: bool
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "encoder_state_dict": self.encoder.rapid.state_dict(),
            "decoder_state_dict": self.decoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics.to_dict(),
            "optimal_threshold": metrics.threshold,
            "config": {
                "hidden_dim": self.encoder.hidden_dim,
                "num_entities": self.encoder.num_entities,
                "decoder": {
                    "num_layers": self.decoder.num_layers,
                    "max_timesteps": self.decoder.max_timesteps,
                    "use_edge_history": self.decoder.use_edge_history,
                },
            },
        }

        torch.save(checkpoint, self.checkpoint_dir / "latest.pth")
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pth")
            print(f"  ✓ New best model (AUPRC: {metrics.auprc:.4f})")

    def _save_history(self):
        """Save training history."""
        with open(self.log_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
