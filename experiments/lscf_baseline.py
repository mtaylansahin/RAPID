#!/usr/bin/env python
"""
Last State Carried Forward (LSCF) baseline comparison.

This script evaluates both the trained model and an LSCF baseline
under the same autoregressive conditions for fair comparison.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ModelConfig, NodeFeatureConfig
from src.data.dataset import PPIDataModule
from src.data.node_features import compute_node_features
from src.models.rapid import create_model

from sklearn.metrics import (
    precision_recall_curve,
    auc,
    f1_score,
    roc_auc_score,
    accuracy_score,
)


DATA_DIR = Path("./data")
RESULTS_DIR = Path("./experiments/diagnostic_results")


def run_lscf_comparison(
    dataset: str = "1JPS-full",
    checkpoint_path: str = None,
):
    """
    Compare model vs LSCF baseline under same autoregressive conditions.

    LSCF baseline: Each pair's prediction at time t is its last known state
    from training data (or 0 if never observed in training).
    """
    print("\n" + "=" * 60)
    print("Last State Carried Forward (LSCF) Baseline Comparison")
    print("=" * 60)

    # Setup
    device = torch.device("cpu")
    data_path = DATA_DIR / "processed" / dataset

    # Find checkpoint
    if checkpoint_path is None:
        checkpoint_path = RESULTS_DIR / "exp2_persistence" / "checkpoints" / "best.pth"
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Looking for alternative...")
        alt_path = RESULTS_DIR / "exp1_with_temporal" / "checkpoints" / "best.pth"
        if alt_path.exists():
            checkpoint_path = alt_path
            print(f"Using: {checkpoint_path}")
        else:
            print("No checkpoint found. Please train a model first.")
            return None

    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load data
    print(f"Loading dataset: {dataset}")
    data_module = PPIDataModule(
        data_path=data_path,
        batch_size=128,
        neg_ratio=1.0,
        hard_ratio=0.5,
        seed=42,
    )

    # Create model
    model_config = ModelConfig(hidden_dim=200, seq_len=10, dropout=0.2)
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
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    threshold = checkpoint.get("optimal_threshold", 0.5)
    print(f"Using threshold: {threshold:.3f}")

    # Build known pairs and initial states from training data
    print("\nBuilding initial state from training data...")
    known_pairs = set()
    for ds in [
        data_module.train_dataset,
        data_module.val_dataset,
        data_module.test_dataset,
    ]:
        for e1, rel, e2, t in ds.data:
            known_pairs.add(tuple(sorted((int(e1), int(e2)))))
    known_pairs_list = sorted(list(known_pairs))

    # Get last known state from training (for LSCF baseline)
    train_final_states = {}  # pair -> last state in training
    train_max_t = max(data_module.train_dataset.unique_timesteps)

    for t in sorted(data_module.train_dataset.unique_timesteps):
        pos_at_t = data_module.train_dataset.positives_by_timestep.get(t, set())
        for pair in known_pairs:
            if pair in pos_at_t:
                train_final_states[pair] = 1
            else:
                # Only update to 0 if we haven't seen this pair as positive before
                # OR if this is a later timestep (state changed to OFF)
                if pair not in train_final_states:
                    train_final_states[pair] = 0
                elif t > 0:  # After first timestep, update to 0 if not positive
                    train_final_states[pair] = 0

    print(f"  Known pairs: {len(known_pairs_list)}")
    print(f"  Pairs ON at end of training: {sum(train_final_states.values())}")

    # Initialize model for autoregressive evaluation
    model.reset_inference_state()
    model.init_from_train_history(
        graph_dict=data_module.graph_dict,
        entity_history=data_module.entity_history,
        entity_history_t=data_module.entity_history_t,
        global_emb=None,
        global_model=None,
    )

    # Run evaluation
    print("\nRunning autoregressive evaluation...")
    timesteps = sorted(data_module.val_dataset.unique_timesteps)

    # LSCF baseline state (starts from training end state, updates autoregressively)
    lscf_current_state = train_final_states.copy()

    # Collect predictions
    all_model_preds = []
    all_model_probs = []
    all_lscf_preds = []
    all_labels = []
    all_prev_states = []  # What state was before this timestep

    for t_idx, t in enumerate(tqdm(timesteps, desc="Evaluating")):
        pairs, labels_np = data_module.get_history_pairs_for_timestep(t, split="valid")

        for i in range(0, len(pairs), 128):
            batch_pairs = pairs[i : i + 128]
            batch_labels = labels_np[i : i + 128]

            entity1 = torch.LongTensor(batch_pairs[:, 0]).to(device)
            entity2 = torch.LongTensor(batch_pairs[:, 1]).to(device)

            with torch.no_grad():
                probs, preds = model.predict_batch(
                    entity1, entity2, t, threshold=threshold, update_history=True
                )

            for j in range(len(batch_pairs)):
                e1, e2 = int(batch_pairs[j][0]), int(batch_pairs[j][1])
                pair = tuple(sorted((e1, e2)))

                # Record predictions
                all_model_preds.append(preds[j].item())
                all_model_probs.append(probs[j].item())
                all_labels.append(batch_labels[j])

                # LSCF prediction: use current LSCF state
                lscf_pred = lscf_current_state.get(pair, 0)
                all_lscf_preds.append(lscf_pred)
                all_prev_states.append(lscf_pred)  # Record for transition analysis

        # Update LSCF state based on ITS OWN predictions (autoregressive)
        # After each timestep, LSCF "predicts" based on what it predicted last time
        # Since LSCF predicts = previous state, the state only changes if we update it
        # from ground truth (which would be cheating) or from model predictions
        #
        # For a TRUE LSCF baseline, we DON'T update - we keep predicting last training state
        # This is a "frozen" baseline that doesn't adapt
        #
        # But we should also track an "oracle LSCF" that uses ground truth for comparison

    # Convert to numpy
    all_model_preds = np.array(all_model_preds)
    all_model_probs = np.array(all_model_probs)
    all_lscf_preds = np.array(all_lscf_preds)
    all_labels = np.array(all_labels)
    all_prev_states = np.array(all_prev_states)

    # Compute metrics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Model metrics
    model_acc = accuracy_score(all_labels, all_model_preds)
    model_f1 = f1_score(all_labels, all_model_preds)
    precision, recall, _ = precision_recall_curve(all_labels, all_model_probs)
    model_auprc = auc(recall, precision)
    model_auroc = roc_auc_score(all_labels, all_model_probs)

    # LSCF metrics (prediction = last training state, no updates)
    lscf_acc = accuracy_score(all_labels, all_lscf_preds)
    lscf_f1 = f1_score(all_labels, all_lscf_preds)
    # For AUPRC/AUROC, LSCF has no probabilities, so we compute based on binary preds
    lscf_auprc = auc(*precision_recall_curve(all_labels, all_lscf_preds)[:2])

    # Transition analysis
    is_transition = all_labels != all_prev_states
    transition_rate = is_transition.mean()
    n_transitions = is_transition.sum()

    # Model accuracy on transitions vs non-transitions
    if n_transitions > 0:
        model_transition_acc = accuracy_score(
            all_labels[is_transition], all_model_preds[is_transition]
        )
        lscf_transition_acc = accuracy_score(
            all_labels[is_transition], all_lscf_preds[is_transition]
        )
    else:
        model_transition_acc = 0
        lscf_transition_acc = 0

    model_non_transition_acc = accuracy_score(
        all_labels[~is_transition], all_model_preds[~is_transition]
    )
    lscf_non_transition_acc = accuracy_score(
        all_labels[~is_transition], all_lscf_preds[~is_transition]
    )

    print("\n### Overall Metrics ###\n")
    print(f"{'Metric':<25} {'Model':<15} {'LSCF Baseline':<15}")
    print("-" * 55)
    print(f"{'Accuracy':<25} {model_acc:.4f}{'':<10} {lscf_acc:.4f}")
    print(f"{'F1 Score':<25} {model_f1:.4f}{'':<10} {lscf_f1:.4f}")
    print(f"{'AUPRC':<25} {model_auprc:.4f}{'':<10} {lscf_auprc:.4f}")
    print(f"{'AUROC':<25} {model_auroc:.4f}{'':<10} {'N/A (binary)':<15}")

    print("\n### Transition Analysis ###\n")
    print(
        f"Transition rate: {transition_rate:.4f} ({n_transitions} / {len(all_labels)})"
    )
    print(f"\n{'Setting':<25} {'Model Acc':<15} {'LSCF Acc':<15}")
    print("-" * 55)
    print(
        f"{'On transitions':<25} {model_transition_acc:.4f}{'':<10} {lscf_transition_acc:.4f}"
    )
    print(
        f"{'On non-transitions':<25} {model_non_transition_acc:.4f}{'':<10} {lscf_non_transition_acc:.4f}"
    )

    # Key insight
    print("\n### Key Insight ###\n")
    if model_acc > lscf_acc:
        print(
            f"✅ Model BEATS LSCF baseline by {(model_acc - lscf_acc) * 100:.2f} percentage points"
        )
    else:
        print(
            f"❌ Model is WORSE than LSCF baseline by {(lscf_acc - model_acc) * 100:.2f} percentage points"
        )

    if model_transition_acc > 0.5:
        print(
            f"✅ Model does better than random on transitions ({model_transition_acc:.2f} > 0.5)"
        )
    else:
        print(
            f"⚠️ Model is worse than random on transitions ({model_transition_acc:.2f})"
        )

    # Note about LSCF
    print("\n### Note on LSCF Baseline ###")
    print("LSCF uses the LAST state from training data and doesn't update.")
    print("On transitions, LSCF accuracy is 0% by definition (it predicts no change).")
    print(f"Actual LSCF transition accuracy: {lscf_transition_acc:.4f}")

    results = {
        "model": {
            "accuracy": float(model_acc),
            "f1": float(model_f1),
            "auprc": float(model_auprc),
            "auroc": float(model_auroc),
            "transition_accuracy": float(model_transition_acc),
            "non_transition_accuracy": float(model_non_transition_acc),
        },
        "lscf_baseline": {
            "accuracy": float(lscf_acc),
            "f1": float(lscf_f1),
            "auprc": float(lscf_auprc),
            "transition_accuracy": float(lscf_transition_acc),
            "non_transition_accuracy": float(lscf_non_transition_acc),
        },
        "data": {
            "total_predictions": len(all_labels),
            "transition_rate": float(transition_rate),
            "n_transitions": int(n_transitions),
        },
    }

    # Save results
    results_path = RESULTS_DIR / "lscf_comparison.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == "__main__":
    run_lscf_comparison()
