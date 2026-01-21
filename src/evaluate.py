"""Evaluation module for RAPID encoder-decoder architecture."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.data.dataset import PPIDataModule
from src.metrics import (
    ClassificationMetrics,
    MetricsComputer,
    PerTimestepMetrics,
    TransitionMetrics,
    compute_per_timestep_metrics,
    compute_transition_metrics,
)
from src.models.decoder import TemporalEdgeDecoder
from src.models.encoder import RAPIDEncoder


class Evaluator:
    """
    Evaluator for RAPID encoder-decoder model.

    Performs seq2seq evaluation on test timesteps with transition metrics.

    Args:
        encoder: RAPIDEncoder wrapper
        decoder: TemporalEdgeDecoder
        data_module: Data module with test data
        device: Torch device
        threshold: Classification threshold
    """

    def __init__(
        self,
        encoder: RAPIDEncoder,
        decoder: TemporalEdgeDecoder,
        data_module: PPIDataModule,
        device: torch.device,
        threshold: float = 0.5,
    ):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.data_module = data_module
        self.device = device
        self.threshold = threshold

        # Prepare known pairs and timesteps
        self.known_pairs = self._get_known_pairs()
        self.train_max_time = data_module.train_max_time
        self.test_timesteps = sorted(data_module.test_dataset.unique_timesteps)

        # Storage for predictions
        self.predictions: List[Tuple[int, int, int, int, float, int]] = []
        self._cached_results: Optional[Dict[str, Any]] = None

    def _get_known_pairs(self) -> torch.Tensor:
        """Get all known pairs as tensor."""
        if not hasattr(self.data_module, "known_pairs_list"):
            self.data_module.get_history_pairs_for_timestep(0, split="test")
        return torch.tensor(self.data_module.known_pairs_list, dtype=torch.long)

    def _get_target_matrix(self, timesteps: List[int]) -> torch.Tensor:
        """Build target matrix for all pairs × timesteps."""
        dataset = self.data_module.test_dataset
        num_pairs = len(self.known_pairs)
        num_timesteps = len(timesteps)

        targets = torch.zeros(num_pairs, num_timesteps)

        for t_idx, t in enumerate(timesteps):
            pos_edges = dataset.positives_by_timestep.get(t, set())
            for p_idx, (e1, e2) in enumerate(self.known_pairs.tolist()):
                if (e1, e2) in pos_edges or (e2, e1) in pos_edges:
                    targets[p_idx, t_idx] = 1.0

        return targets

    def _get_previous_states(self, timesteps: List[int]) -> torch.Tensor:
        """Get edge states at the timestep before the first test timestep."""
        first_test_t = min(timesteps)

        all_times = sorted(
            set(self.data_module.train_dataset.timesteps)
            | set(self.data_module.val_dataset.timesteps)
        )
        prev_t = None
        for t in reversed(all_times):
            if t < first_test_t:
                prev_t = t
                break

        prev_states = torch.zeros(len(self.known_pairs))

        if prev_t is not None:
            prev_edges = set()
            for ds in [
                self.data_module.train_dataset,
                self.data_module.val_dataset,
            ]:
                prev_edges.update(ds.positives_by_timestep.get(prev_t, set()))

            for p_idx, (e1, e2) in enumerate(self.known_pairs.tolist()):
                if (e1, e2) in prev_edges or (e2, e1) in prev_edges:
                    prev_states[p_idx] = 1.0

        return prev_states

    def _get_subgraph_context(self, batch_pairs: torch.Tensor) -> dict:
        """Get subgraph context for a batch of pairs (required)."""
        batch_subgraph_context = self.data_module.get_subgraph_context(
            batch_pairs.tolist()
        )
        if batch_subgraph_context is None:
            raise RuntimeError(
                "Subgraph context is required but data_module.get_subgraph_context() "
                "returned None. Ensure SubgraphExtractor is properly initialized."
            )
        return {k: v.to(self.device) for k, v in batch_subgraph_context.items()}

    @torch.no_grad()
    def run_inference(self, force_rerun: bool = False) -> Dict[str, Any]:
        """Run seq2seq inference on test set."""
        if not force_rerun and self._cached_results is not None:
            return self._cached_results

        self.encoder.eval()
        self.decoder.eval()

        print("\nRunning seq2seq inference...")

        # Encode entities - simplified: only needs graph_dict
        entity_context = self.encoder(self.data_module.graph_dict)

        # Get targets
        target_matrix = self._get_target_matrix(self.test_timesteps)
        prev_states = self._get_previous_states(self.test_timesteps)

        # Relative timesteps
        relative_t = torch.tensor(
            [t - self.train_max_time for t in self.test_timesteps],
            dtype=torch.long,
            device=self.device,
        )

        # Predict in batches
        all_logits = []
        all_probs = []
        pair_batch_size = 512

        for start_idx in tqdm(
            range(0, len(self.known_pairs), pair_batch_size),
            desc="Predicting",
        ):
            end_idx = min(start_idx + pair_batch_size, len(self.known_pairs))
            batch_pairs = self.known_pairs[start_idx:end_idx].to(self.device)

            # Get subgraph context (required)
            batch_subgraph_context = self._get_subgraph_context(batch_pairs)

            probs, preds, logits = self.decoder.predict(
                entity_context,
                batch_pairs,
                relative_t,
                subgraph_context=batch_subgraph_context,
                threshold=self.threshold,
            )
            all_logits.append(logits.cpu())
            all_probs.append(probs.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_probs = torch.cat(all_probs, dim=0)
        all_preds = (all_probs >= self.threshold).long()

        # Store detailed predictions
        self.predictions = []
        for p_idx, (e1, e2) in enumerate(self.known_pairs.tolist()):
            for t_idx, t in enumerate(self.test_timesteps):
                prob = float(all_probs[p_idx, t_idx])
                pred = int(all_preds[p_idx, t_idx])
                self.predictions.append((e1, 1, e2, t, prob, pred))

        results = {
            "logits": all_logits,
            "probs": all_probs,
            "predictions": all_preds,
            "targets": target_matrix,
            "prev_states": prev_states,
            "timesteps": self.test_timesteps,
        }

        self._cached_results = results
        return results

    def save_predictions(self, output_path: Path) -> None:
        """Save predicted interactions to file."""
        if not self.predictions:
            print("Warning: No predictions to save. Run evaluation first.")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for e1, rel, e2, t, score, pred in self.predictions:
                if pred == 1:
                    f.write(f"{e1}\t{e2}\t{t}\n")

        num_positive = sum(1 for p in self.predictions if p[5] == 1)
        print(f"\nPredictions saved to: {output_path}")
        print(f"  Positive predictions: {num_positive}")

    def evaluate(self) -> ClassificationMetrics:
        """Compute overall classification metrics."""
        results = self.run_inference()

        logits_flat = results["logits"].view(-1)
        targets_flat = results["targets"].view(-1)

        n_pos = targets_flat.sum().item()
        n_neg = len(targets_flat) - n_pos
        print(
            f"\n  Total samples: {len(targets_flat)} ({int(n_pos)} positive, {int(n_neg)} negative)"
        )
        print(f"  Class ratio: 1:{n_neg / max(n_pos, 1):.1f}")

        metrics_computer = MetricsComputer(threshold=self.threshold)
        metrics_computer.update(logits_flat, targets_flat)

        return metrics_computer.compute(tune_threshold=True)

    def evaluate_per_timestep(self) -> PerTimestepMetrics:
        """Compute metrics per timestep."""
        results = self.run_inference()

        logits_flat = results["logits"].view(-1)
        targets_flat = results["targets"].view(-1)

        num_pairs = len(self.known_pairs)
        timesteps_per_sample = (
            torch.tensor(self.test_timesteps)
            .unsqueeze(0)
            .expand(num_pairs, -1)
            .reshape(-1)
        )

        return compute_per_timestep_metrics(
            logits_flat,
            targets_flat,
            timesteps_per_sample,
            threshold=self.threshold,
        )

    def evaluate_transitions(self) -> TransitionMetrics:
        """Compute transition-focused metrics."""
        results = self.run_inference()

        return compute_transition_metrics(
            results["predictions"],
            results["targets"],
            results["prev_states"],
        )

    def full_evaluation(self) -> Dict[str, Any]:
        """Run full evaluation with all analyses."""
        self.run_inference(force_rerun=True)

        output = {}

        print("\n" + "=" * 50)
        print("Test Results")
        print("=" * 50)

        metrics = self.evaluate()
        output["metrics"] = metrics.to_dict()
        print(f"\n{metrics}")

        per_ts_metrics = self.evaluate_per_timestep()
        output["per_timestep"] = per_ts_metrics.to_dict()
        print(f"\nPer-Timestep Analysis:")
        print(f"  Mean AUPRC: {per_ts_metrics.mean_auprc:.4f}")
        print(f"  Mean F1: {per_ts_metrics.mean_f1:.4f}")

        trans_metrics = self.evaluate_transitions()
        output["transitions"] = trans_metrics.to_dict()
        print(f"\nTransition Analysis:")
        print(f"  {trans_metrics}")
        print(f"  ON→OFF Recall: {trans_metrics.on_to_off_recall:.3f}")
        print(f"  OFF→ON Recall: {trans_metrics.off_to_on_recall:.3f}")

        if len(per_ts_metrics.auprcs) > 5:
            early = np.mean(per_ts_metrics.auprcs[:5])
            late = np.mean(per_ts_metrics.auprcs[-5:])
            if late < early * 0.9:
                print(
                    f"\n  ⚠️ Temporal degradation detected: "
                    f"early AUPRC {early:.4f} vs late {late:.4f}"
                )

        print("=" * 50)

        return output
