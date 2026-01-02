"""Pretraining script for the global RGCN model (RAPID)."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.data.dataset import PPIDataModule


def get_true_distribution(
    train_data: np.ndarray,
    num_entities: int,
    timesteps: List[int],
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Compute true entity distribution per timestep for soft cross-entropy.

    For each timestep, computes the probability distribution over entities
    that appear in edges at that timestep.

    Args:
        train_data: Training quadruples (e1, rel, e2, t)
        num_entities: Number of entities
        timesteps: List of timesteps

    Returns:
        Tuple of (entity1_probs, entity2_probs) dicts mapping timestep -> distribution
    """
    # Group edges by timestep
    edges_by_t: Dict[int, List] = {t: [] for t in timesteps}

    for row in train_data:
        e1, _rel, e2, t = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        if t in edges_by_t:
            edges_by_t[t].append((e1, e2))

    # Compute distributions
    e1_probs = {}
    e2_probs = {}

    for t in timesteps:
        edges = edges_by_t[t]

        e1_counts = torch.zeros(num_entities)
        e2_counts = torch.zeros(num_entities)

        for e1, e2 in edges:
            e1_counts[e1] += 1
            e2_counts[e2] += 1

        # Normalize to probabilities (add small epsilon to avoid division by zero)
        e1_probs[t] = e1_counts / (e1_counts.sum() + 1e-8)
        e2_probs[t] = e2_counts / (e2_counts.sum() + 1e-8)

    return e1_probs, e2_probs


def make_batch(
    timesteps: List[int],
    e1_probs: Dict[int, torch.Tensor],
    e2_probs: Dict[int, torch.Tensor],
    batch_size: int,
):
    """
    Generate batches for training.

    Yields:
        Tuple of (batch_timesteps, batch_e1_probs, batch_e2_probs)
    """
    indices = np.random.permutation(len(timesteps))

    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i : i + batch_size]
        batch_t = torch.LongTensor([timesteps[j] for j in batch_idx])
        batch_e1 = torch.stack([e1_probs[timesteps[j]] for j in batch_idx])
        batch_e2 = torch.stack([e2_probs[timesteps[j]] for j in batch_idx])
        yield batch_t, batch_e1, batch_e2


def train_global_model(
    model: torch.nn.Module,
    data_module: PPIDataModule,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
    output_path: Optional[Path] = None,
    batch_size: int = 64,
    weight_decay: float = 1e-5,
    grad_norm: float = 1.0,
):
    """
    Train the global model (callable from main.py).

    Args:
        model: Global model to train
        data_module: Data module with training data
        device: Torch device
        epochs: Number of training epochs
        lr: Learning rate
        output_path: Path to save model checkpoint
        batch_size: Training batch size
        weight_decay: L2 regularization
        grad_norm: Gradient clipping norm
    """
    print(f"\nEntities: {data_module.num_entities}")
    print(f"Relations: {data_module.num_rels}")
    print(f"Train timesteps: {len(data_module.train_times)}")

    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Compute true distributions for training
    print("\nComputing entity distributions...")
    train_data = data_module.train_data
    timesteps = sorted(data_module.train_times)
    e1_probs, e2_probs = get_true_distribution(
        train_data, data_module.num_entities, timesteps
    )

    # Setup checkpointing
    if output_path is None:
        output_path = Path(f"./models/{data_module.dataset}/global.pth")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_loss = float("inf")

    print("\nStarting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        n_batches = 0

        t0 = time.time()

        pbar = tqdm(
            make_batch(timesteps, e1_probs, e2_probs, batch_size),
            total=len(timesteps) // batch_size + 1,
            desc=f"Epoch {epoch:03d}",
        )

        for batch_t, batch_e1, batch_e2 in pbar:
            batch_t = batch_t.to(device)
            batch_e1 = batch_e1.to(device)
            batch_e2 = batch_e2.to(device)

            # Forward pass for both entity predictions
            loss_e1 = model(batch_t, batch_e1, data_module.graph_dict)
            loss_e2 = model(batch_t, batch_e2, data_module.graph_dict)
            loss = (loss_e1 + loss_e2) / 2

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Compute global embeddings
        model.eval()
        with torch.no_grad():
            global_emb = model.compute_global_embeddings(
                timesteps, data_module.graph_dict
            )

        elapsed = time.time() - t0
        avg_loss = epoch_loss / n_batches

        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

        # Save if best
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"  -> New best! Saving to {output_path}")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "global_emb": {k: v.cpu() for k, v in global_emb.items()},
                    "config": {
                        "num_entities": data_module.num_entities,
                        "num_rels": data_module.num_rels,
                        "hidden_dim": model.hidden_dim,
                    },
                },
                output_path,
            )

    print(f"\nPretraining complete! Best loss: {best_loss:.4f}")
    print(f"Model saved to: {output_path}")
