#!/usr/bin/env python
"""
Evaluate trajectory model and generate analysis outputs.

Converts trajectory predictions to per-timestep format for the existing analysis pipeline.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.data.trajectory_dataset import TrajectoryDataModule
from src.models.trajectory import TrajectoryModel
from src.metrics import compute_transition_metrics, TransitionMetrics


def evaluate_trajectory_model(
    model: TrajectoryModel,
    data_module: TrajectoryDataModule,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict:
    """
    Evaluate trajectory model and return results in standard format.
    """
    model.eval()
    model.to(device)

    # Get test dataset
    test_dataset = data_module.get_test_dataset()
    test_loader = data_module.get_test_dataloader()

    # Collect predictions
    all_probs = []
    all_targets = []
    all_entity1 = []
    all_entity2 = []

    print("Running trajectory inference...")
    for batch in tqdm(test_loader):
        entity1 = batch["entity1"].to(device)
        entity2 = batch["entity2"].to(device)
        history = batch["history"].to(device)
        target = batch["target"].to(device)
        neighbor_e1 = batch["neighbor_entity1"].to(device)
        neighbor_e2 = batch["neighbor_entity2"].to(device)
        neighbor_hist = batch["neighbor_history"].to(device)
        neighbor_mask = batch["neighbor_mask"].to(device)

        traj_len = target.size(1)

        with torch.no_grad():
            logits = model(
                entity1_ids=entity1,
                entity2_ids=entity2,
                history=history,
                neighbor_entity1=neighbor_e1,
                neighbor_entity2=neighbor_e2,
                neighbor_history=neighbor_hist,
                traj_len=traj_len,
                neighbor_mask=neighbor_mask,
            )
            probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu())
        all_targets.append(target.cpu())
        all_entity1.append(entity1.cpu())
        all_entity2.append(entity2.cpu())

    # Stack results
    all_probs = torch.cat(all_probs, dim=0)  # (n_edges, traj_len)
    all_targets = torch.cat(all_targets, dim=0)
    all_entity1 = torch.cat(all_entity1, dim=0)
    all_entity2 = torch.cat(all_entity2, dim=0)

    n_edges, traj_len = all_probs.shape
    test_timesteps = data_module.test_timesteps

    print(f"  Edges: {n_edges}, Trajectory length: {traj_len}")

    # Compute overall metrics
    preds = (all_probs > threshold).float()
    tp = ((preds == 1) & (all_targets == 1)).sum().item()
    fp = ((preds == 1) & (all_targets == 0)).sum().item()
    tn = ((preds == 0) & (all_targets == 0)).sum().item()
    fn = ((preds == 0) & (all_targets == 1)).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)

    # Compute AUPRC
    from sklearn.metrics import average_precision_score, roc_auc_score

    auprc = average_precision_score(
        all_targets.numpy().flatten(), all_probs.numpy().flatten()
    )
    auroc = roc_auc_score(all_targets.numpy().flatten(), all_probs.numpy().flatten())

    print(f"\nOverall Metrics:")
    print(f"  AUPRC: {auprc:.4f} | AUROC: {auroc:.4f}")
    print(f"  F1: {f1:.4f} (P: {precision:.4f}, R: {recall:.4f})")
    print(f"  Accuracy: {accuracy:.4f}")

    # Compute transition metrics
    pred_binary = (all_probs > threshold).numpy()
    target_binary = all_targets.numpy()

    # Transitions between consecutive timesteps
    all_predictions = []
    all_labels = []
    all_prev_labels = []

    for edge_idx in range(n_edges):
        for t_idx in range(1, traj_len):
            all_predictions.append(pred_binary[edge_idx, t_idx])
            all_labels.append(target_binary[edge_idx, t_idx])
            # Use predicted t-1 as prev_label (test true dynamics learning)
            all_prev_labels.append(pred_binary[edge_idx, t_idx - 1])

    trans_metrics = compute_transition_metrics(
        np.array(all_predictions),
        np.array(all_labels),
        np.array(all_prev_labels),
    )

    print(f"\nTransition Metrics:")
    print(f"  {trans_metrics}")
    print(f"  Persistence Accuracy: {trans_metrics.persistence_accuracy:.4f}")

    # Per-timestep metrics
    per_timestep_auprc = []
    per_timestep_f1 = []

    for t_idx in range(traj_len):
        t_probs = all_probs[:, t_idx].numpy()
        t_targets = all_targets[:, t_idx].numpy()
        t_preds = (t_probs > threshold).astype(int)

        if t_targets.sum() > 0 and (1 - t_targets).sum() > 0:
            per_timestep_auprc.append(average_precision_score(t_targets, t_probs))
        else:
            per_timestep_auprc.append(np.nan)

        t_tp = ((t_preds == 1) & (t_targets == 1)).sum()
        t_fp = ((t_preds == 1) & (t_targets == 0)).sum()
        t_fn = ((t_preds == 0) & (t_targets == 1)).sum()
        t_prec = t_tp / max(t_tp + t_fp, 1)
        t_rec = t_tp / max(t_tp + t_fn, 1)
        t_f1 = 2 * t_prec * t_rec / max(t_prec + t_rec, 1e-8)
        per_timestep_f1.append(t_f1)

    mean_auprc = np.nanmean(per_timestep_auprc)
    mean_f1 = np.nanmean(per_timestep_f1)

    print(f"\nPer-Timestep Analysis:")
    print(f"  Mean AUPRC: {mean_auprc:.4f}")
    print(f"  Mean F1: {mean_f1:.4f}")

    # Build predictions list for analysis pipeline
    predictions_list = []
    for edge_idx in range(n_edges):
        e1 = int(all_entity1[edge_idx])
        e2 = int(all_entity2[edge_idx])
        for t_idx, t in enumerate(test_timesteps):
            prob = float(all_probs[edge_idx, t_idx])
            pred = int(pred_binary[edge_idx, t_idx])
            label = int(target_binary[edge_idx, t_idx])
            predictions_list.append((e1, 1, e2, t, prob, pred, label))

    return {
        "metrics": {
            "auprc": auprc,
            "auroc": auroc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
        "transition_metrics": trans_metrics.to_dict(),
        "per_timestep": {
            "auprcs": per_timestep_auprc,
            "f1s": per_timestep_f1,
            "mean_auprc": mean_auprc,
            "mean_f1": mean_f1,
        },
        "predictions": predictions_list,
        "all_probs": all_probs.numpy(),
        "all_targets": all_targets.numpy(),
        "entity1": all_entity1.numpy(),
        "entity2": all_entity2.numpy(),
        "test_timesteps": test_timesteps,
    }


def save_results(results: Dict, output_dir: Path):
    """Save evaluation results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_file = output_dir / "PerformanceMetrics.txt"
    with open(metrics_file, "w") as f:
        f.write("=== Overall Metrics ===\n")
        for k, v in results["metrics"].items():
            f.write(f"{k}: {v}\n")
        f.write("\n=== Transition Metrics ===\n")
        for k, v in results["transition_metrics"].items():
            f.write(f"{k}: {v}\n")
        f.write("\n=== Per-Timestep ===\n")
        f.write(f"Mean AUPRC: {results['per_timestep']['mean_auprc']:.4f}\n")
        f.write(f"Mean F1: {results['per_timestep']['mean_f1']:.4f}\n")

    # Save structured metrics
    metrics_json = output_dir / "metrics_structured.json"
    json_results = {
        "metrics": results["metrics"],
        "transition_metrics": results["transition_metrics"],
        "per_timestep": {
            "mean_auprc": results["per_timestep"]["mean_auprc"],
            "mean_f1": results["per_timestep"]["mean_f1"],
        },
    }
    with open(metrics_json, "w") as f:
        json.dump(json_results, f, indent=2)

    # Save predictions
    pred_file = output_dir / "predictions.txt"
    with open(pred_file, "w") as f:
        for e1, rel, e2, t, prob, pred, label in results["predictions"]:
            if pred == 1:
                f.write(f"{e1}\t{e2}\t{t}\n")

    print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trajectory model")
    parser.add_argument("--dataset", type=str, default="1JPS")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/trajectory_full/best.pth"
    )
    parser.add_argument(
        "--output_dir", type=str, default="analysis_outputs/trajectory_full"
    )
    parser.add_argument("--test_ratio", type=float, default=0.25)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n_hops", type=int, default=1, help="Number of neighbor hops")
    parser.add_argument(
        "--use_full_neighbors",
        action="store_true",
        help="Include intrachain edges as neighbors",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data_path = Path(args.data_dir) / args.dataset
    print(f"\nLoading dataset: {args.dataset}")

    data_module = TrajectoryDataModule(
        data_path=data_path,
        test_ratio=args.test_ratio,
        batch_size=64,
        use_full_neighbors=args.use_full_neighbors,
        n_hops=args.n_hops,
    )

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model = TrajectoryModel(
        num_entities=data_module.num_entities,
        hidden_dim=args.hidden_dim,
        max_traj_len=data_module.traj_len + 10,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate
    results = evaluate_trajectory_model(
        model=model,
        data_module=data_module,
        device=device,
        threshold=args.threshold,
    )

    # Save results
    output_dir = Path(args.output_dir)
    save_results(results, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
