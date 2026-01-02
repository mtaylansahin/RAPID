"""Evaluation module for RAPID - Protein Interaction Dynamics prediction."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.data.dataset import PPIDataModule
from src.losses import get_loss_function
from src.metrics import (
    ClassificationMetrics,
    MetricsComputer,
    PerTimestepMetrics,
    compute_per_timestep_metrics,
)
from src.models.global_model import PPIGlobalModel
from src.models.rapid import RAPIDModel


class Evaluator:
    """
    Evaluator for RAPID model.

    Uses all-pairs evaluation for unbiased metrics.
    Supports autoregressive inference with predicted history.

    Args:
        model: Trained RAPIDModel
        data_module: Data module with test data
        device: torch device
        threshold: Classification threshold
    """

    def __init__(
        self,
        model: RAPIDModel,
        data_module: PPIDataModule,
        device: torch.device,
        threshold: float = 0.5,
        global_model: Optional[PPIGlobalModel] = None,
    ):
        self.model = model.to(device)
        self.data_module = data_module
        self.device = device
        self.threshold = threshold

        # Global model (optional)
        self.global_model = global_model
        if global_model is not None:
            self.global_model = global_model.to(device)
            self.global_model.eval()
            self.global_emb = global_model.global_emb
        else:
            self.global_emb = None

        self.criterion = get_loss_function(loss_type="focal", gamma=2.0)

        # Storage for predictions (to save to file)
        self.predictions: List[Tuple[int, int, int, int, float, int]] = []

    def save_predictions(
        self,
        output_path: Path,
    ) -> None:
        """
        Save predicted interactions to a text file.

        Args:
            output_path: Path to save the predictions file
        """
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

    def run_inference(
        self,
        split: str = "test",
        force_rerun: bool = False,
    ) -> Dict[str, Any]:
        """
        Run autoregressive inference loop and cache results.

        Args:
            split: 'valid' or 'test'
            force_rerun: If True, ignore cache and rerun

        Returns:
            Dictionary containing:
            - logits: torch.Tensor
            - labels: torch.Tensor
            - timesteps: torch.Tensor
            - predictions: List of tuples (e1, rel, e2, t, score, pred)
        """
        # Return cached results if available
        if (
            not force_rerun
            and hasattr(self, "_cached_results")
            and self._cached_results.get("split") == split
        ):
            return self._cached_results

        self.model.eval()

        # Get dataset and timesteps
        if split == "valid":
            dataset = self.data_module.val_dataset
            # For validation, use train context only
            context_graph_dict = self.data_module.graph_dict
            context_history = self.data_module.entity_history
            context_history_t = self.data_module.entity_history_t
        else:
            dataset = self.data_module.test_dataset
            # For test, use train + val context
            context_graph_dict, context_history, context_history_t = (
                self.data_module.get_train_val_context()
            )

        # Extend global embeddings if needed
        if self.global_model is not None:
            print("Extending global embeddings...")
            self.global_model.extend_embeddings(context_graph_dict)
            self.global_emb = self.global_model.global_emb

        # Initialize model with historical context
        self.model.reset_inference_state()
        self.model.init_from_train_history(
            graph_dict=context_graph_dict,
            entity_history=context_history,
            entity_history_t=context_history_t,
            global_emb=self.global_emb,
            global_model=self.global_model,
        )

        # Collect all predictions
        all_logits = []
        all_labels = []
        all_timesteps = []
        detailed_predictions = []

        print(f"\nRunning inference on {split} set...")

        # Get unique timesteps from dataset
        timesteps = sorted(dataset.unique_timesteps)

        for t in tqdm(timesteps, desc="Timesteps"):
            # Get known history pairs
            pairs, labels_np = self.data_module.get_history_pairs_for_timestep(
                t, split=split
            )

            # Process in batches
            batch_size = 128
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i : i + batch_size]
                batch_labels = labels_np[i : i + batch_size]

                entity1 = torch.LongTensor(batch_pairs[:, 0]).to(self.device)
                entity2 = torch.LongTensor(batch_pairs[:, 1]).to(self.device)
                labels = torch.FloatTensor(batch_labels).to(self.device)

                # Get predictions (updates internal state)
                probs, preds = self.model.predict_batch(
                    entity1_ids=entity1,
                    entity2_ids=entity2,
                    timestep=t,
                    threshold=self.threshold,
                    update_history=True,
                )

                # Convert probs to logits for metrics
                probs_clamped = probs.clamp(1e-7, 1 - 1e-7)
                logits = torch.log(probs_clamped / (1 - probs_clamped))

                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
                all_timesteps.append(torch.full((len(labels),), t))

                # Store detailed predictions
                probs_np = probs.cpu().numpy()
                preds_np = preds.cpu().numpy()
                e1_np = entity1.cpu().numpy()
                e2_np = entity2.cpu().numpy()

                for j in range(len(e1_np)):
                    detailed_predictions.append(
                        (
                            int(e1_np[j]),
                            1,  # relation (assumed 1 for now or from graph)
                            int(e2_np[j]),
                            int(t),
                            float(probs_np[j]),
                            int(preds_np[j]),
                        )
                    )

        # Concatenate results
        results = {
            "split": split,
            "logits": torch.cat(all_logits),
            "labels": torch.cat(all_labels),
            "timesteps": torch.cat(all_timesteps),
            "predictions": detailed_predictions,
        }

        # Cache results
        self._cached_results = results
        return results

    @torch.no_grad()
    def evaluate(
        self,
        split: str = "test",
        collect_predictions: bool = False,
    ) -> ClassificationMetrics:
        """
        Compute classification metrics.

        Uses cached inference results if available.
        """
        results = self.run_inference(split=split)

        logits = results["logits"]
        labels = results["labels"]
        self.predictions = results["predictions"] if collect_predictions else []

        # Report class balance
        n_pos = labels.sum().item()
        n_neg = len(labels) - n_pos
        print(f"  Total pairs: {len(labels)} ({n_pos} positive, {n_neg} negative)")
        print(f"  Class ratio: 1:{n_neg / max(n_pos, 1):.1f}")

        metrics_computer = MetricsComputer(threshold=self.threshold)
        metrics_computer.update(logits, labels)

        return metrics_computer.compute()

    @torch.no_grad()
    def evaluate_per_timestep(self, split: str = "test") -> PerTimestepMetrics:
        """
        Compute metrics per timestep.

        Uses cached inference results if available.
        """
        results = self.run_inference(split=split)

        return compute_per_timestep_metrics(
            results["logits"],
            results["labels"],
            results["timesteps"],
            threshold=self.threshold,
        )

    def full_evaluation(self, split: str = "test") -> Dict[str, Any]:
        """
        Run full evaluation with all analyses.

        Runs inference ONCE and computes both overall and per-timestep metrics.
        """
        # Force a fresh run for full evaluation
        self.run_inference(split=split, force_rerun=True)

        results = {}

        # Main evaluation (uses cache)
        metrics = self.evaluate(split=split, collect_predictions=True)
        results["metrics"] = metrics.to_dict()
        print(f"\n{split.capitalize()} Results:")
        print(f"  {metrics}")

        # Per-timestep analysis (uses cache)
        per_ts_metrics = self.evaluate_per_timestep(split=split)
        results["per_timestep"] = per_ts_metrics.to_dict()
        print("\nPer-Timestep Analysis:")
        print(f"  Mean AUPRC: {per_ts_metrics.mean_auprc:.4f}")
        print(f"  Mean F1: {per_ts_metrics.mean_f1:.4f}")

        # Check for temporal degradation
        if len(per_ts_metrics.auprcs) > 5:
            early = np.mean(per_ts_metrics.auprcs[:5])
            late = np.mean(per_ts_metrics.auprcs[-5:])
            if late < early * 0.9:
                print(
                    f"  ⚠️ Temporal degradation detected: "
                    f"early AUPRC {early:.4f} vs late {late:.4f}"
                )

        return results
