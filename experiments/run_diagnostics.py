#!/usr/bin/env python
"""
Diagnostic experiments to validate hypotheses about RAPID model behavior.

This script runs a series of experiments to test:
1. Whether temporal embeddings provide useful signal
2. Whether the model is just predicting persistence
3. Whether batch variance causes training instability
4. Effects of different focal loss configurations
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ModelConfig, NodeFeatureConfig, TrainingConfig
from src.data.dataset import PPIDataModule
from src.data.node_features import compute_node_features
from src.losses import get_loss_function
from src.metrics import ClassificationMetrics, MetricsComputer, find_optimal_threshold
from src.models.rapid import create_model, RAPIDModel
from src.train import Trainer


DATA_DIR = Path("./data")
RESULTS_DIR = Path("./experiments/diagnostic_results")


def setup_env(seed: int = 42, gpu: int = -1) -> torch.device:
    """Set random seeds and setup compute device."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu >= 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = torch.device(f"cuda:{gpu}")
        torch.cuda.set_device(gpu)
    else:
        device = torch.device("cpu")
    return device


def load_data_and_model(
    dataset: str,
    device: torch.device,
    hidden_dim: int = 200,
    seq_len: int = 10,
    hard_ratio: float = 0.5,
    batch_size: int = 128,
    seed: int = 42,
) -> Tuple[PPIDataModule, RAPIDModel, torch.Tensor]:
    """Load data module and create model."""
    data_path = DATA_DIR / "processed" / dataset

    data_module = PPIDataModule(
        data_path=data_path,
        batch_size=batch_size,
        neg_ratio=1.0,
        hard_ratio=hard_ratio,
        seed=seed,
    )

    model_config = ModelConfig(
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        dropout=0.2,
    )

    # Compute node features
    node_feature_config = NodeFeatureConfig(enabled=True)
    train_cutoff = data_module.train_max_time
    node_features = compute_node_features(
        config=node_feature_config,
        data_dir=data_path,
        train_cutoff=train_cutoff,
    )
    model_config.node_features = node_feature_config

    model = create_model(
        num_entities=data_module.num_entities,
        num_rels=data_module.num_rels,
        config=model_config,
        node_features=node_features,
    )

    return data_module, model, node_features


def run_baseline_training(
    data_module: PPIDataModule,
    model: RAPIDModel,
    device: torch.device,
    epochs: int = 10,
    patience: int = 5,
    lr: float = 1e-3,
    focal_gamma: float = 2.0,
    focal_alpha: float = None,
    pair_masking: float = 0.0,
    experiment_name: str = "baseline",
) -> Dict:
    """Run a training experiment and return metrics."""
    checkpoint_dir = RESULTS_DIR / experiment_name / "checkpoints"
    log_dir = RESULTS_DIR / experiment_name / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    training_config = TrainingConfig(
        learning_rate=lr,
        max_epochs=epochs,
        patience=patience,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        pair_masking_prob=pair_masking,
        pair_masking_warmup=0,
    )

    # Reset model weights
    model = model.to(device)
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()

    trainer = Trainer(
        model=model,
        data_module=data_module,
        config=training_config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        global_model=None,
    )

    result = trainer.train()

    return {
        "best_epoch": result["best_epoch"],
        "best_val_auprc": result["best_val_auprc"],
        "history": result["history"],
        "optimal_threshold": result["optimal_threshold"],
    }


def experiment_1_temporal_ablation(
    device: torch.device,
    dataset: str = "1JPS-full",
    epochs: int = 10,
) -> Dict:
    """
    Experiment 1: Test if temporal embeddings provide useful signal.

    Compare:
    - Normal model (with temporal)
    - Model with zeroed temporal embeddings
    """
    print("\n" + "=" * 60)
    print("Experiment 1: Temporal Embedding Ablation")
    print("=" * 60)

    results = {}

    # Run normal training first
    print("\n[1a] Training with normal temporal embeddings...")
    data_module, model, node_features = load_data_and_model(dataset, device)

    results["with_temporal"] = run_baseline_training(
        data_module=data_module,
        model=model,
        device=device,
        epochs=epochs,
        patience=5,
        experiment_name="exp1_with_temporal",
    )

    # Now create a model variant that zeros temporal embeddings
    # We'll do this by testing on the trained model with zeroed temporal
    print("\n[1b] Testing model behavior without temporal (zeroing embeddings)...")

    # Load best checkpoint
    checkpoint_path = RESULTS_DIR / "exp1_with_temporal" / "checkpoints" / "best.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reload model and weights
    data_module2, model2, _ = load_data_and_model(dataset, device)
    model2.load_state_dict(checkpoint["model_state_dict"])
    model2 = model2.to(device)
    model2.eval()

    # Evaluate with normal temporal
    print("  Evaluating with normal temporal...")
    normal_metrics = evaluate_model(model2, data_module2, device, zero_temporal=False)

    # Evaluate with zeroed temporal
    print("  Evaluating with zeroed temporal...")
    ablated_metrics = evaluate_model(model2, data_module2, device, zero_temporal=True)

    results["eval_with_temporal"] = normal_metrics
    results["eval_without_temporal"] = ablated_metrics
    results["temporal_impact"] = {
        "auprc_drop": normal_metrics["auprc"] - ablated_metrics["auprc"],
        "f1_drop": normal_metrics["f1"] - ablated_metrics["f1"],
    }

    print(f"\n  Results:")
    print(
        f"    With temporal:    AUPRC={normal_metrics['auprc']:.4f}, F1={normal_metrics['f1']:.4f}"
    )
    print(
        f"    Without temporal: AUPRC={ablated_metrics['auprc']:.4f}, F1={ablated_metrics['f1']:.4f}"
    )
    print(
        f"    Temporal impact:  AUPRC drop={results['temporal_impact']['auprc_drop']:.4f}"
    )

    return results


def evaluate_model(
    model: RAPIDModel,
    data_module: PPIDataModule,
    device: torch.device,
    zero_temporal: bool = False,
    threshold: float = 0.5,
) -> Dict:
    """Evaluate model on validation set, optionally zeroing temporal embeddings."""
    model.eval()
    metrics = MetricsComputer()

    all_logits = []
    all_labels = []

    # Initialize model with training history
    model.reset_inference_state()
    model.init_from_train_history(
        graph_dict=data_module.graph_dict,
        entity_history=data_module.entity_history,
        entity_history_t=data_module.entity_history_t,
        global_emb=None,
        global_model=None,
    )

    timesteps = sorted(data_module.val_dataset.unique_timesteps)

    for t in tqdm(timesteps, desc="Evaluating", leave=False):
        pairs, labels_np = data_module.get_history_pairs_for_timestep(t, split="valid")

        batch_size = 128
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]
            batch_labels = labels_np[i : i + batch_size]

            entity1 = torch.LongTensor(batch_pairs[:, 0]).to(device)
            entity2 = torch.LongTensor(batch_pairs[:, 1]).to(device)
            labels = torch.FloatTensor(batch_labels).to(device)

            with torch.no_grad():
                # Get embeddings
                entity1_embed = model.get_entity_embed(entity1)
                entity2_embed = model.get_entity_embed(entity2)

                # Get temporal embeddings
                if zero_temporal:
                    entity1_temporal = torch.zeros(
                        len(entity1), model.hidden_dim, device=device
                    )
                    entity2_temporal = torch.zeros(
                        len(entity2), model.hidden_dim, device=device
                    )
                else:
                    entity1_temporal = torch.zeros(
                        len(entity1), model.hidden_dim, device=device
                    )
                    entity2_temporal = torch.zeros(
                        len(entity2), model.hidden_dim, device=device
                    )
                    for j in range(len(entity1)):
                        entity1_temporal[j] = model._get_entity_temporal_embed(
                            entity1[j].item()
                        )
                        entity2_temporal[j] = model._get_entity_temporal_embed(
                            entity2[j].item()
                        )

                logits = model.classifier(
                    entity1_embed,
                    entity2_embed,
                    entity1_temporal,
                    entity2_temporal,
                )

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Compute metrics
    probs = torch.sigmoid(all_logits)
    preds = (probs >= threshold).long()

    from sklearn.metrics import precision_recall_curve, auc, f1_score, roc_auc_score

    precision, recall, _ = precision_recall_curve(all_labels.numpy(), probs.numpy())
    auprc = auc(recall, precision)
    auroc = roc_auc_score(all_labels.numpy(), probs.numpy())
    f1 = f1_score(all_labels.numpy(), preds.numpy())

    return {
        "auprc": auprc,
        "auroc": auroc,
        "f1": f1,
    }


def experiment_2_persistence_correlation(
    device: torch.device,
    dataset: str = "1JPS-full",
    epochs: int = 10,
) -> Dict:
    """
    Experiment 2: Check if model predictions correlate with persistence baseline.

    For each pair, compute:
    - Model prediction
    - Persistence baseline (previous state)
    And measure correlation.
    """
    print("\n" + "=" * 60)
    print("Experiment 2: Persistence Correlation Analysis")
    print("=" * 60)

    # First train a model
    print("\n[2a] Training model...")
    data_module, model, _ = load_data_and_model(dataset, device)

    train_result = run_baseline_training(
        data_module=data_module,
        model=model,
        device=device,
        epochs=epochs,
        patience=5,
        experiment_name="exp2_persistence",
    )

    # Load best model
    checkpoint_path = RESULTS_DIR / "exp2_persistence" / "checkpoints" / "best.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    data_module2, model2, _ = load_data_and_model(dataset, device)
    model2.load_state_dict(checkpoint["model_state_dict"])
    model2 = model2.to(device)
    model2.eval()

    # Analyze predictions vs persistence
    print("\n[2b] Analyzing prediction patterns...")

    model2.reset_inference_state()
    model2.init_from_train_history(
        graph_dict=data_module2.graph_dict,
        entity_history=data_module2.entity_history,
        entity_history_t=data_module2.entity_history_t,
        global_emb=None,
        global_model=None,
    )

    timesteps = sorted(data_module2.val_dataset.unique_timesteps)

    # Track pair states over time
    pair_predictions = {}  # (e1, e2) -> list of (t, pred, label, prev_state)

    # Get all known pairs
    known_pairs = set()
    for ds in [data_module2.train_dataset, data_module2.val_dataset]:
        for e1, rel, e2, t in ds.data:
            known_pairs.add(tuple(sorted((int(e1), int(e2)))))

    # Build previous state from training data
    prev_states = {}  # (e1, e2) -> last known state before val
    train_max_t = max(data_module2.train_dataset.unique_timesteps)
    for t in sorted(data_module2.train_dataset.unique_timesteps):
        pos_at_t = data_module2.train_dataset.positives_by_timestep.get(t, set())
        for pair in known_pairs:
            if pair in pos_at_t:
                prev_states[pair] = 1
            elif pair not in prev_states:
                prev_states[pair] = 0

    # Now evaluate on validation
    current_states = prev_states.copy()

    for t in tqdm(timesteps, desc="Analyzing persistence"):
        pairs, labels_np = data_module2.get_history_pairs_for_timestep(t, split="valid")

        for i in range(0, len(pairs), 128):
            batch_pairs = pairs[i : i + 128]
            batch_labels = labels_np[i : i + 128]

            entity1 = torch.LongTensor(batch_pairs[:, 0]).to(device)
            entity2 = torch.LongTensor(batch_pairs[:, 1]).to(device)

            with torch.no_grad():
                probs, preds = model2.predict_batch(
                    entity1, entity2, t, threshold=0.5, update_history=True
                )

            for j in range(len(batch_pairs)):
                e1, e2 = int(batch_pairs[j][0]), int(batch_pairs[j][1])
                pair = tuple(sorted((e1, e2)))
                pred = preds[j].item()
                label = batch_labels[j]
                prev_state = current_states.get(pair, 0)

                if pair not in pair_predictions:
                    pair_predictions[pair] = []
                pair_predictions[pair].append((t, pred, label, prev_state))

        # Update current states based on labels (ground truth for next step)
        pos_at_t = data_module2.val_dataset.positives_by_timestep.get(t, set())
        for pair in known_pairs:
            current_states[pair] = 1 if pair in pos_at_t else 0

    # Analyze correlation
    all_preds = []
    all_prev_states = []
    all_labels = []

    for pair, records in pair_predictions.items():
        for t, pred, label, prev_state in records:
            all_preds.append(pred)
            all_prev_states.append(prev_state)
            all_labels.append(label)

    all_preds = np.array(all_preds)
    all_prev_states = np.array(all_prev_states)
    all_labels = np.array(all_labels)

    # Compute correlations
    pred_prev_corr = np.corrcoef(all_preds, all_prev_states)[0, 1]
    label_prev_corr = np.corrcoef(all_labels, all_prev_states)[0, 1]

    # Compute persistence baseline accuracy
    persistence_acc = np.mean(all_prev_states == all_labels)
    model_acc = np.mean(all_preds == all_labels)

    # Count transitions
    transitions = np.sum(all_labels != all_prev_states)
    total = len(all_labels)
    transition_rate = transitions / total

    # Model's accuracy on transitions vs non-transitions
    transition_mask = all_labels != all_prev_states
    if transition_mask.sum() > 0:
        transition_acc = np.mean(
            all_preds[transition_mask] == all_labels[transition_mask]
        )
    else:
        transition_acc = 0.0

    non_transition_mask = ~transition_mask
    if non_transition_mask.sum() > 0:
        non_transition_acc = np.mean(
            all_preds[non_transition_mask] == all_labels[non_transition_mask]
        )
    else:
        non_transition_acc = 0.0

    results = {
        "training": train_result,
        "analysis": {
            "pred_prev_correlation": float(pred_prev_corr),
            "label_prev_correlation": float(label_prev_corr),
            "persistence_baseline_accuracy": float(persistence_acc),
            "model_accuracy": float(model_acc),
            "transition_rate": float(transition_rate),
            "transition_accuracy": float(transition_acc),
            "non_transition_accuracy": float(non_transition_acc),
        },
    }

    print(f"\n  Results:")
    print(f"    Correlation (pred ↔ prev_state): {pred_prev_corr:.4f}")
    print(f"    Correlation (label ↔ prev_state): {label_prev_corr:.4f}")
    print(f"    Persistence baseline accuracy:    {persistence_acc:.4f}")
    print(f"    Model accuracy:                   {model_acc:.4f}")
    print(f"    Transition rate in data:          {transition_rate:.4f}")
    print(f"    Model accuracy on transitions:    {transition_acc:.4f}")
    print(f"    Model accuracy on non-transitions: {non_transition_acc:.4f}")

    return results


def experiment_3_loss_stability(
    device: torch.device,
    dataset: str = "1JPS-full",
    epochs: int = 10,
) -> Dict:
    """
    Experiment 3: Test different loss configurations for stability.

    Compare:
    - Focal loss γ=2.0, α=None (baseline)
    - Focal loss γ=1.0, α=None
    - Focal loss γ=2.0, α=0.3
    - BCE loss
    """
    print("\n" + "=" * 60)
    print("Experiment 3: Loss Function Stability")
    print("=" * 60)

    results = {}

    configs = [
        {"name": "focal_g2_noalpha", "focal_gamma": 2.0, "focal_alpha": None},
        {"name": "focal_g1_noalpha", "focal_gamma": 1.0, "focal_alpha": None},
        {"name": "focal_g2_alpha03", "focal_gamma": 2.0, "focal_alpha": 0.3},
        {"name": "focal_g1_alpha03", "focal_gamma": 1.0, "focal_alpha": 0.3},
    ]

    for config in configs:
        print(f"\n[3] Testing {config['name']}...")
        data_module, model, _ = load_data_and_model(dataset, device, seed=42)

        result = run_baseline_training(
            data_module=data_module,
            model=model,
            device=device,
            epochs=epochs,
            patience=5,
            focal_gamma=config["focal_gamma"],
            focal_alpha=config["focal_alpha"],
            experiment_name=f"exp3_{config['name']}",
        )

        # Compute loss variance
        train_losses = result["history"]["train_loss"]
        loss_variance = np.var(train_losses) if len(train_losses) > 1 else 0
        loss_range = max(train_losses) - min(train_losses) if train_losses else 0

        results[config["name"]] = {
            "best_val_auprc": result["best_val_auprc"],
            "best_epoch": result["best_epoch"],
            "loss_variance": float(loss_variance),
            "loss_range": float(loss_range),
            "train_losses": train_losses,
        }

        print(f"    Best AUPRC: {result['best_val_auprc']:.4f}")
        print(f"    Loss variance: {loss_variance:.6f}")
        print(f"    Loss range: {loss_range:.4f}")

    return results


def experiment_4_hard_ratio_effect(
    device: torch.device,
    dataset: str = "1JPS-full",
    epochs: int = 10,
) -> Dict:
    """
    Experiment 4: Test different hard_ratio values.

    Compare:
    - hard_ratio=0.0 (all easy negatives)
    - hard_ratio=0.5 (baseline)
    - hard_ratio=1.0 (all hard negatives)
    """
    print("\n" + "=" * 60)
    print("Experiment 4: Hard Ratio Effect")
    print("=" * 60)

    results = {}

    for hard_ratio in [0.0, 0.5, 1.0]:
        print(f"\n[4] Testing hard_ratio={hard_ratio}...")
        data_module, model, node_features = load_data_and_model(
            dataset, device, hard_ratio=hard_ratio, seed=42
        )

        result = run_baseline_training(
            data_module=data_module,
            model=model,
            device=device,
            epochs=epochs,
            patience=5,
            experiment_name=f"exp4_hard{hard_ratio}",
        )

        # Compute train/val gap from first complete epoch
        train_auprc = result["history"]["train_auprc"]
        val_auprc = result["history"]["val_auprc"]
        if train_auprc and val_auprc:
            avg_gap = np.mean([t - v for t, v in zip(train_auprc, val_auprc)])
        else:
            avg_gap = 0

        results[f"hard_{hard_ratio}"] = {
            "best_val_auprc": result["best_val_auprc"],
            "best_epoch": result["best_epoch"],
            "avg_train_val_gap": float(avg_gap),
        }

        print(f"    Best AUPRC: {result['best_val_auprc']:.4f}")
        print(f"    Avg train/val gap: {avg_gap:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run diagnostic experiments")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU device (-1 for CPU)")
    parser.add_argument("--dataset", type=str, default="1JPS-full", help="Dataset name")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per experiment")
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=["1", "2", "3", "4"],
        help="Which experiments to run (1, 2, 3, 4)",
    )
    args = parser.parse_args()

    device = setup_env(seed=42, gpu=args.gpu)
    print(f"Device: {device}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    if "1" in args.experiments:
        all_results["exp1_temporal_ablation"] = experiment_1_temporal_ablation(
            device, args.dataset, args.epochs
        )

    if "2" in args.experiments:
        all_results["exp2_persistence"] = experiment_2_persistence_correlation(
            device, args.dataset, args.epochs
        )

    if "3" in args.experiments:
        all_results["exp3_loss_stability"] = experiment_3_loss_stability(
            device, args.dataset, args.epochs
        )

    if "4" in args.experiments:
        all_results["exp4_hard_ratio"] = experiment_4_hard_ratio_effect(
            device, args.dataset, args.epochs
        )

    # Save all results
    results_path = RESULTS_DIR / "diagnostic_results.json"

    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    serializable_results = convert_to_serializable(all_results)

    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print("\n" + "=" * 60)
    print("All experiments complete!")
    print(f"Results saved to: {results_path}")
    print("=" * 60)

    # Print summary
    print("\n=== SUMMARY ===\n")

    if "exp1_temporal_ablation" in all_results:
        exp1 = all_results["exp1_temporal_ablation"]
        print("Experiment 1 (Temporal Ablation):")
        print(
            f"  AUPRC drop when zeroing temporal: {exp1['temporal_impact']['auprc_drop']:.4f}"
        )
        if abs(exp1["temporal_impact"]["auprc_drop"]) < 0.02:
            print("  → CONFIRMS: Temporal embeddings provide minimal signal!")
        else:
            print("  → Temporal embeddings do provide some signal.")

    if "exp2_persistence" in all_results:
        exp2 = all_results["exp2_persistence"]["analysis"]
        print("\nExperiment 2 (Persistence Correlation):")
        print(f"  Pred↔Prev correlation: {exp2['pred_prev_correlation']:.4f}")
        print(f"  Accuracy on transitions: {exp2['transition_accuracy']:.4f}")
        if exp2["pred_prev_correlation"] > 0.8:
            print("  → CONFIRMS: Model mostly predicts persistence!")
        elif exp2["transition_accuracy"] < 0.5:
            print("  → Model struggles with transitions (dynamics).")

    if "exp3_loss_stability" in all_results:
        exp3 = all_results["exp3_loss_stability"]
        print("\nExperiment 3 (Loss Stability):")
        for name, data in exp3.items():
            print(
                f"  {name}: variance={data['loss_variance']:.6f}, AUPRC={data['best_val_auprc']:.4f}"
            )

    if "exp4_hard_ratio" in all_results:
        exp4 = all_results["exp4_hard_ratio"]
        print("\nExperiment 4 (Hard Ratio Effect):")
        for name, data in exp4.items():
            print(
                f"  {name}: gap={data['avg_train_val_gap']:.4f}, AUPRC={data['best_val_auprc']:.4f}"
            )


if __name__ == "__main__":
    main()
