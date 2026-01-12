#!/usr/bin/env python
"""
Main script for trajectory-level prediction.

Trains a model to predict full edge trajectories informed by neighbor context.
"""

import argparse
from pathlib import Path

import torch

from src.data.trajectory_dataset import TrajectoryDataModule
from src.models.trajectory import TrajectoryModel
from src.train_trajectory import TrajectoryTrainer


def main():
    parser = argparse.ArgumentParser(description="Trajectory-level PPI prediction")

    # Data
    parser.add_argument("--dataset", type=str, default="1JPS", help="Dataset name")
    parser.add_argument(
        "--data_dir", type=str, default="data/processed", help="Data directory"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.25, help="Fraction for test"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1, help="Fraction of remaining for val"
    )

    # Model
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Attention heads")
    parser.add_argument(
        "--n_encoder_layers", type=int, default=2, help="Encoder layers"
    )
    parser.add_argument(
        "--n_decoder_layers", type=int, default=2, help="Decoder layers"
    )
    parser.add_argument(
        "--max_neighbors", type=int, default=50, help="Max neighbor edges"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--transition_weight",
        type=float,
        default=1.0,
        help="Weight for transition timesteps (1.0 = no weighting)",
    )
    parser.add_argument(
        "--use_full_neighbors",
        action="store_true",
        help="Include intrachain edges as neighbors (not just interchain)",
    )
    parser.add_argument(
        "--n_hops",
        type=int,
        default=1,
        help="Number of hops for neighbor expansion (1=direct, 2=neighbors of neighbors)",
    )

    # Output
    parser.add_argument(
        "--experiment_name", type=str, default="trajectory", help="Experiment name"
    )

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    data_path = Path(args.data_dir) / args.dataset
    print(f"\nLoading dataset: {args.dataset}")

    data_module = TrajectoryDataModule(
        data_path=data_path,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        max_neighbors=args.max_neighbors,
        batch_size=args.batch_size,
        seed=args.seed,
        use_full_neighbors=args.use_full_neighbors,
        n_hops=args.n_hops,
    )

    # Model
    print("\nCreating model...")
    model = TrajectoryModel(
        num_entities=data_module.num_entities,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        max_neighbors=args.max_neighbors,
        max_traj_len=data_module.traj_len + 10,  # Buffer
        dropout=args.dropout,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    if args.transition_weight > 1.0:
        print(f"Transition weight: {args.transition_weight}x")

    # Trainer
    checkpoint_dir = Path("checkpoints") / args.experiment_name

    trainer = TrajectoryTrainer(
        model=model,
        data_module=data_module,
        learning_rate=args.lr,
        patience=args.patience,
        checkpoint_dir=checkpoint_dir,
        device=device,
        transition_weight=args.transition_weight,
    )

    # Train
    print(f"\nStarting training...")
    history = trainer.train(n_epochs=args.epochs)

    print("\nDone!")


if __name__ == "__main__":
    main()
