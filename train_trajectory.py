#!/usr/bin/env python
"""
Main script for TrajectoryRAPID training.

Hybrid model combining RGCN structure encoding with trajectory-level prediction.
"""

import argparse
from pathlib import Path

import torch

from src.config import TrajectoryModelConfig, TrajectoryTrainingConfig
from src.data.trajectory_dataset import TrajectoryDataModule
from src.models.trajectory_rapid import TrajectoryRAPIDModel
from src.train_trajectory import TrajectoryTrainer


def main():
    parser = argparse.ArgumentParser(
        description="TrajectoryRAPID: Hybrid trajectory prediction for PPI dynamics"
    )

    # Data
    parser.add_argument("--dataset", type=str, default="1JPS", help="Dataset name")
    parser.add_argument(
        "--data_dir", type=str, default="data/processed", help="Data directory"
    )
    parser.add_argument(
        "--history_ratio",
        type=float,
        default=0.5,
        help="Fraction of timesteps for history",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Fraction of prediction timesteps for validation",
    )

    # Model architecture
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="Hidden dimension"
    )
    parser.add_argument("--n_heads", type=int, default=4, help="Attention heads")
    parser.add_argument(
        "--n_encoder_layers", type=int, default=2, help="Encoder layers"
    )
    parser.add_argument(
        "--n_decoder_layers", type=int, default=2, help="Decoder layers"
    )
    parser.add_argument(
        "--n_neighbor_layers", type=int, default=1, help="Neighbor attention layers"
    )
    parser.add_argument(
        "--max_neighbors", type=int, default=50, help="Max neighbor edges"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Feature flags (for ablation)
    parser.add_argument(
        "--no_rgcn",
        action="store_true",
        help="Disable RGCN structural features",
    )
    parser.add_argument(
        "--no_global",
        action="store_true",
        help="Disable global context",
    )
    parser.add_argument(
        "--no_neighbors",
        action="store_true",
        help="Disable neighbor attention",
    )
    parser.add_argument(
        "--use_node_features",
        action="store_true",
        help="Enable physicochemical node features",
    )
    parser.add_argument(
        "--n_hops",
        type=int,
        default=1,
        help="Number of hops for neighbor expansion",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--transition_weight",
        type=float,
        default=1.0,
        help="Weight for transition timesteps in loss",
    )
    parser.add_argument(
        "--focal_loss",
        action="store_true",
        help="Use focal loss instead of BCE",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma parameter",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="trajectory_rapid",
        help="Experiment name",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run evaluation (requires checkpoint)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for eval or resume",
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
        history_ratio=args.history_ratio,
        val_ratio=args.val_ratio,
        max_neighbors=args.max_neighbors,
        batch_size=args.batch_size,
        seed=args.seed,
        n_hops=args.n_hops,
    )

    # Model
    print("\nCreating model...")
    print(f"  RGCN: {not args.no_rgcn}")
    print(f"  Global context: {not args.no_global}")
    print(f"  Neighbor attention: {not args.no_neighbors}")
    print(f"  Node features: {args.use_node_features}")
    print(f"  N-hops: {args.n_hops}")

    model = TrajectoryRAPIDModel(
        num_entities=data_module.num_entities,
        num_rels=data_module.num_rels,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        n_neighbor_layers=args.n_neighbor_layers,
        max_neighbors=args.max_neighbors,
        max_seq_len=data_module.hist_len + data_module.traj_len + 10,
        dropout=args.dropout,
        use_rgcn=not args.no_rgcn,
        use_global_context=not args.no_global,
        use_neighbor_attention=not args.no_neighbors,
        use_node_features=args.use_node_features,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Trainer
    checkpoint_dir = Path("checkpoints") / args.experiment_name
    log_dir = Path("logs") / args.experiment_name

    trainer = TrajectoryTrainer(
        model=model,
        data_module=data_module,
        learning_rate=args.lr,
        weight_decay=1e-5,
        patience=args.patience,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        device=device,
        transition_weight=args.transition_weight,
        use_focal_loss=args.focal_loss,
        focal_gamma=args.focal_gamma,
    )

    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(Path(args.checkpoint))

    if args.eval_only:
        # Evaluation only
        if not args.checkpoint:
            # Try to load best checkpoint
            best_ckpt = checkpoint_dir / "best.pth"
            if best_ckpt.exists():
                trainer.load_checkpoint(best_ckpt)
            else:
                print("Error: No checkpoint found for evaluation")
                return

        trainer.evaluate_test()
    else:
        # Train
        print(f"\nStarting training...")
        if args.transition_weight > 1.0:
            print(f"Transition weight: {args.transition_weight}x")
        if args.focal_loss:
            print(f"Using focal loss with gamma={args.focal_gamma}")

        history = trainer.train(n_epochs=args.epochs)

        # Evaluate on test
        trainer.evaluate_test()

    print("\nDone!")


if __name__ == "__main__":
    main()
