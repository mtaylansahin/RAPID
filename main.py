#!/usr/bin/env python
"""
RAPID: Encoder-Decoder Architecture for Predicting Protein Interaction Dynamics

This script orchestrates all functionality:
- pretrain: Train the encoder on link prediction
- train: Train the decoder on validation timesteps
- evaluate: Evaluate the encoder-decoder model
- all: Run full pipeline (pretrain -> train -> evaluate)

Examples:
    # Full pipeline
    uv run python main.py all --dataset RAPID --epochs 50

    # Pretrain encoder only
    uv run python main.py pretrain --dataset RAPID --pretrain_epochs 30

    # Train decoder (requires pretrained encoder)
    uv run python main.py train --dataset RAPID --epochs 50 --encoder_path ./models/RAPID/encoder.pth

    # Evaluate
    uv run python main.py evaluate --checkpoint ./checkpoints/RAPID_*/best.pth
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Internal imports
from src.config import DecoderConfig, ModelConfig, NodeFeatureConfig, TrainingConfig
from src.data.dataset import PPIDataModule
from src.data.node_features import compute_node_features
from src.data.preprocessing import PreprocessingConfig, run_preprocessing
from src.evaluate import Evaluator
from src.analysis import AnalysisConfig, ResultsManager
from src.models.decoder import TemporalEdgeDecoder, create_decoder
from src.models.encoder import RAPIDEncoder
from src.models.rapid import create_model
from src.pretrain import pretrain_encoder
from src.train import Trainer

# Constants
DATA_DIR = Path("./data")
MODELS_DIR = Path("./models")
CHECKPOINTS_DIR = Path("./checkpoints")
LOGS_DIR = Path("./logs")
PREDICTIONS_DIR = Path("./predictions")


def get_base_args():
    """Get argument parser with common arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--dataset", type=str, default="RAPID", help="Dataset name (folder in data/)"
    )
    parser.add_argument("--hidden_dim", type=int, default=200, help="Hidden dimension")
    parser.add_argument(
        "--seq_len", type=int, default=10, help="History sequence length"
    )
    parser.add_argument("--num_bases", type=int, default=5, help="Number of RGCN bases")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU device (-1 for CPU)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser


def setup_env(args) -> torch.device:
    """Set random seeds and setup compute device."""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cpu")
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.gpu)

    print(f"Using device: {device}")
    return device


def load_encoder(
    path: Path,
    num_entities: int,
    num_rels: int,
    hidden_dim: int,
    device: torch.device,
    node_features: Optional[torch.Tensor] = None,
    freeze: bool = True,
) -> RAPIDEncoder:
    """Load a pretrained encoder from checkpoint."""
    print(f"\nLoading encoder from: {path}")
    checkpoint = torch.load(path, map_location=device)

    # Create model config
    model_config = ModelConfig(hidden_dim=hidden_dim)

    # Create underlying RAPID model
    rapid_model = create_model(
        num_entities=num_entities,
        num_rels=num_rels,
        config=model_config,
        node_features=node_features,
    )
    rapid_model.load_state_dict(checkpoint["model_state_dict"])

    # Wrap in encoder
    encoder = RAPIDEncoder(rapid_model, freeze=freeze)
    encoder = encoder.to(device)

    print(f"  Encoder loaded (frozen: {freeze})")
    return encoder


def run_pretrain(args) -> Path:
    """Run encoder pretraining."""
    print("\n" + "=" * 60)
    print("Stage 1: Pretraining Encoder")
    print("=" * 60)

    device = setup_env(args)

    # Load data
    data_path = DATA_DIR / args.dataset
    print(f"\nLoading dataset: {args.dataset}")
    data_module = PPIDataModule(
        data_path=data_path,
        batch_size=args.batch_size,
        neg_ratio=1.0,
        seed=args.seed,
    )

    # Create model config
    model_config = ModelConfig(
        hidden_dim=args.hidden_dim,
        seq_len=args.seq_len,
        dropout=args.dropout,
    )

    # Compute node features if enabled
    node_features = None
    if not getattr(args, "no_node_features", False):
        print("\nComputing node features...")
        node_feature_config = NodeFeatureConfig(
            enabled=True,
            use_physicochemical=not getattr(args, "no_physicochemical", False),
            use_intrachain=not getattr(args, "no_intrachain_features", False),
        )
        train_cutoff = data_module.train_max_time
        node_features = compute_node_features(
            config=node_feature_config,
            data_dir=DATA_DIR / args.dataset,
            train_cutoff=train_cutoff,
        )
        if node_features is not None:
            print(f"  Node features shape: {node_features.shape}")
        model_config.node_features = node_feature_config
    else:
        print("\nNode features disabled.")
        model_config.node_features = NodeFeatureConfig(enabled=False)

    # Create RAPID model for pretraining
    print("\nCreating encoder model...")
    model = create_model(
        num_entities=data_module.num_entities,
        num_rels=data_module.num_rels,
        config=model_config,
        node_features=node_features,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Output path
    output_dir = MODELS_DIR / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "encoder.pth"

    # Train
    pretrain_encoder(
        model=model,
        data_module=data_module,
        device=device,
        epochs=args.pretrain_epochs,
        lr=args.pretrain_lr,
        output_path=output_path,
        patience=5,
        focal_gamma=args.focal_gamma if hasattr(args, "focal_gamma") else 2.0,
    )

    return output_path


def run_train(args) -> Path:
    """Train decoder model."""
    print("\n" + "=" * 60)
    print("Stage 2: Training Decoder")
    print("=" * 60)

    device = setup_env(args)

    # Load data (PPIDataModule now has integrated subgraph support)
    print(f"\nLoading dataset: {args.dataset}")
    data_module = PPIDataModule(
        data_path=DATA_DIR / args.dataset,
        batch_size=args.batch_size,
        neg_ratio=args.neg_ratio,
        hard_ratio=args.hard_ratio,
        seed=args.seed,
    )

    # Check if subgraph data is available
    if getattr(args, "use_edge_history", False) and data_module.full_graphs is not None:
        print("  Full graph data available for edge-centric encoding")

    # Compute node features if needed
    node_features = None
    if not getattr(args, "no_node_features", False):
        print("\nComputing node features...")
        node_feature_config = NodeFeatureConfig(enabled=True)
        train_cutoff = data_module.train_max_time
        node_features = compute_node_features(
            config=node_feature_config,
            data_dir=DATA_DIR / args.dataset,
            train_cutoff=train_cutoff,
        )

    # Load pretrained encoder
    encoder_path = (
        Path(args.encoder_path)
        if args.encoder_path
        else MODELS_DIR / args.dataset / "encoder.pth"
    )
    if not encoder_path.exists():
        print(f"Error: Encoder not found at {encoder_path}")
        print("Run 'pretrain' first or specify --encoder_path")
        sys.exit(1)

    encoder = load_encoder(
        path=encoder_path,
        num_entities=data_module.num_entities,
        num_rels=data_module.num_rels,
        hidden_dim=args.hidden_dim,
        device=device,
        node_features=node_features,
        freeze=args.freeze_encoder,
    )

    # Create decoder
    print("\nCreating decoder...")
    use_edge_history = getattr(args, "use_edge_history", False)
    if use_edge_history:
        print("  Using EdgeCentricSubgraphEncoder")

    decoder = create_decoder(
        hidden_dim=args.hidden_dim,
        num_layers=args.decoder_layers,
        num_heads=args.decoder_heads,
        max_timesteps=200,
        dropout=args.dropout,
        use_edge_history=use_edge_history,
    )
    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")

    # Create training config
    training_config = TrainingConfig(
        learning_rate=args.lr,
        max_epochs=args.epochs,
        patience=args.patience,
        focal_gamma=args.focal_gamma,
        freeze_encoder=args.freeze_encoder,
        encoder_lr=args.encoder_lr if not args.freeze_encoder else 0.0,
        transition_weight=getattr(args, "transition_weight", 1.0),
    )

    # Setup paths
    checkpoint_dir = CHECKPOINTS_DIR / args.experiment_name
    log_dir = LOGS_DIR / args.experiment_name

    # Create trainer
    trainer = Trainer(
        encoder=encoder,
        decoder=decoder,
        data_module=data_module,
        config=training_config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
    )

    # Train
    result = trainer.train()

    print("\nTraining complete!")
    print(f"  Best epoch: {result['best_epoch']}")
    print(f"  Best AUPRC: {result['best_val_auprc']:.4f}")
    print(f"  Optimal threshold: {result['optimal_threshold']:.3f}")

    return checkpoint_dir / "best.pth"


def run_evaluate(args) -> bool:
    """Run evaluation."""
    print("\n" + "=" * 60)
    print("Stage 3: Evaluation")
    print("=" * 60)

    # Find checkpoint
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        if CHECKPOINTS_DIR.exists():
            experiment_dirs = sorted(
                [d for d in CHECKPOINTS_DIR.iterdir() if d.is_dir()],
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            if experiment_dirs:
                best_ckpt = experiment_dirs[0] / "best.pth"
                if best_ckpt.exists():
                    checkpoint_path = str(best_ckpt)

    if not checkpoint_path:
        print("Error: No checkpoint found. Please specify --checkpoint")
        sys.exit(1)

    device = setup_env(args)

    # Load data
    data_path = DATA_DIR / args.dataset
    print(f"\nLoading dataset: {args.dataset}")
    data_module = PPIDataModule(
        data_path=data_path,
        batch_size=128,
        neg_ratio=1.0,
    )

    # Load checkpoint
    print(f"\nLoading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint
    hidden_dim = checkpoint.get("config", {}).get("hidden_dim", args.hidden_dim)
    decoder_config = checkpoint.get("config", {}).get("decoder", {})

    # Compute node features
    node_features = None
    if not getattr(args, "no_node_features", False):
        print("\nComputing node features...")
        node_feature_config = NodeFeatureConfig(enabled=True)
        train_cutoff = data_module.train_max_time
        node_features = compute_node_features(
            config=node_feature_config,
            data_dir=DATA_DIR / args.dataset,
            train_cutoff=train_cutoff,
        )

    # Create encoder
    model_config = ModelConfig(hidden_dim=hidden_dim)
    rapid_model = create_model(
        num_entities=data_module.num_entities,
        num_rels=data_module.num_rels,
        config=model_config,
        node_features=node_features,
    )
    rapid_model.load_state_dict(checkpoint["encoder_state_dict"])
    encoder = RAPIDEncoder(rapid_model, freeze=True)

    # Create decoder
    decoder = create_decoder(
        hidden_dim=hidden_dim,
        num_layers=decoder_config.get("num_layers", 4),
        num_heads=8,
        max_timesteps=decoder_config.get("max_timesteps", 200),
        dropout=0.1,
        use_edge_history=decoder_config.get("use_edge_history", False),
    )
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    # Get threshold
    threshold = checkpoint.get("optimal_threshold", 0.5)
    print(f"Using threshold: {threshold:.3f}")

    # Create evaluator
    evaluator = Evaluator(
        encoder=encoder,
        decoder=decoder,
        data_module=data_module,
        device=device,
        threshold=threshold,
    )

    # Predictions directory
    predictions_dir = Path(args.predictions_dir) / args.dataset
    predictions_path = predictions_dir / "predictions.txt"

    # Run evaluation
    evaluator.full_evaluation()
    evaluator.save_predictions(predictions_path)

    # Run analysis + visualization
    analysis_output_dir = Path("analysis_outputs") / Path(checkpoint_path).parent.name
    analysis_config = AnalysisConfig(
        input_directory=str(data_path),
        output_directory=str(analysis_output_dir),
        output_file_path=str(predictions_path),
    )
    results_manager = ResultsManager(analysis_config)
    analysis_success = results_manager.run_complete_analysis()
    if analysis_success:
        print(f"\nAnalysis outputs saved to: {analysis_output_dir}")
    else:
        print("\nWarning: Analysis pipeline failed. Check logs for details.")

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAPID: Encoder-Decoder for Protein Interaction Dynamics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # === Pretrain command ===
    pretrain_parser = subparsers.add_parser(
        "pretrain",
        help="Pretrain encoder on link prediction",
        parents=[get_base_args()],
    )
    pretrain_parser.add_argument(
        "--pretrain_epochs", type=int, default=30, help="Number of pretraining epochs"
    )
    pretrain_parser.add_argument(
        "--pretrain_lr", type=float, default=1e-3, help="Pretraining learning rate"
    )
    pretrain_parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size"
    )
    pretrain_parser.add_argument(
        "--focal_gamma", type=float, default=2.0, help="Focal loss gamma"
    )
    pretrain_parser.add_argument(
        "--no_node_features", action="store_true", help="Disable node features"
    )

    # === Train command ===
    train_parser = subparsers.add_parser(
        "train",
        help="Train decoder model",
        parents=[get_base_args()],
    )
    train_parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    train_parser.add_argument(
        "--lr", type=float, default=1e-3, help="Decoder learning rate"
    )
    train_parser.add_argument(
        "--encoder_lr", type=float, default=1e-5, help="Encoder fine-tuning LR"
    )
    train_parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    train_parser.add_argument(
        "--neg_ratio", type=float, default=1.0, help="Negative ratio"
    )
    train_parser.add_argument(
        "--hard_ratio", type=float, default=0.5, help="Hard negative ratio"
    )
    train_parser.add_argument(
        "--focal_gamma", type=float, default=2.0, help="Focal loss gamma"
    )
    train_parser.add_argument(
        "--transition_weight",
        type=float,
        default=1.0,
        help="Weight multiplier for transitions vs persistence in loss",
    )
    train_parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    train_parser.add_argument(
        "--encoder_path", type=str, default=None, help="Path to pretrained encoder"
    )
    train_parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        default=True,
        help="Freeze encoder weights",
    )
    train_parser.add_argument(
        "--fine_tune_encoder",
        dest="freeze_encoder",
        action="store_false",
        help="Fine-tune encoder with lower LR",
    )
    train_parser.add_argument(
        "--decoder_layers", type=int, default=4, help="Number of decoder layers"
    )
    train_parser.add_argument(
        "--decoder_heads", type=int, default=8, help="Number of decoder attention heads"
    )
    train_parser.add_argument(
        "--experiment_name", type=str, default=None, help="Experiment name"
    )
    train_parser.add_argument(
        "--no_node_features", action="store_true", help="Disable node features"
    )
    train_parser.add_argument(
        "--use_edge_history",
        action="store_true",
        help="Enable edge-centric subgraph encoder for N-hop temporal context",
    )

    # === Evaluate command ===
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate trained model",
        parents=[get_base_args()],
    )
    eval_parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint path"
    )
    eval_parser.add_argument(
        "--predictions_dir",
        type=str,
        default=str(PREDICTIONS_DIR),
        help="Predictions dir",
    )
    eval_parser.add_argument(
        "--no_node_features", action="store_true", help="Disable node features"
    )

    # === All command ===
    all_parser = subparsers.add_parser(
        "all",
        help="Run full pipeline: pretrain -> train -> evaluate",
        parents=[get_base_args()],
    )
    # Pretrain args
    all_parser.add_argument(
        "--pretrain_epochs", type=int, default=30, help="Pretrain epochs"
    )
    all_parser.add_argument(
        "--pretrain_lr", type=float, default=1e-3, help="Pretrain LR"
    )
    # Train args
    all_parser.add_argument("--epochs", type=int, default=50, help="Train epochs")
    all_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    all_parser.add_argument("--encoder_lr", type=float, default=1e-5, help="Encoder LR")
    all_parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    all_parser.add_argument(
        "--neg_ratio", type=float, default=1.0, help="Negative ratio"
    )
    all_parser.add_argument("--hard_ratio", type=float, default=0.5, help="Hard ratio")
    all_parser.add_argument(
        "--focal_gamma", type=float, default=2.0, help="Focal gamma"
    )
    all_parser.add_argument("--patience", type=int, default=10, help="Patience")
    all_parser.add_argument("--freeze_encoder", action="store_true", default=True)
    all_parser.add_argument(
        "--fine_tune_encoder", dest="freeze_encoder", action="store_false"
    )
    all_parser.add_argument("--decoder_layers", type=int, default=4)
    all_parser.add_argument("--decoder_heads", type=int, default=8)
    all_parser.add_argument("--encoder_path", type=str, default=None)
    all_parser.add_argument("--experiment_name", type=str, default=None)
    all_parser.add_argument("--predictions_dir", type=str, default=str(PREDICTIONS_DIR))
    all_parser.add_argument("--checkpoint", type=str, default=None)
    all_parser.add_argument("--no_node_features", action="store_true")
    all_parser.add_argument(
        "--use_edge_history",
        action="store_true",
        help="Enable edge-centric subgraph encoder",
    )
    # Preprocess args
    all_parser.add_argument("--data_dir", type=str, default=None)
    all_parser.add_argument("--replica", type=str, default=None)
    all_parser.add_argument("--test_ratio", type=float, default=0.2)
    all_parser.add_argument(
        "--transition_weight",
        type=float,
        default=1.0,
        help="Weight multiplier for transition samples",
    )

    # === Preprocess command ===
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Preprocess raw MD simulation data",
    )
    preprocess_parser.add_argument("--data_dir", type=str, required=True)
    preprocess_parser.add_argument("--output_dir", type=str, required=True)
    preprocess_parser.add_argument("--replica", type=str, required=True)
    preprocess_parser.add_argument("--test_ratio", type=float, default=0.2)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    print("\n" + "=" * 60)
    print("RAPID: Encoder-Decoder for Protein Interaction Dynamics")
    print("=" * 60)
    print(f"Command: {args.command}")
    if args.command != "preprocess":
        print(f"Dataset: {args.dataset}")
        print(f"GPU: {args.gpu}")
        print(f"Seed: {args.seed}")

    # Execute command
    if args.command == "pretrain":
        run_pretrain(args)

    elif args.command == "train":
        if args.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.experiment_name = f"{args.dataset}_{timestamp}"
        run_train(args)

    elif args.command == "evaluate":
        run_evaluate(args)

    elif args.command == "preprocess":
        config = PreprocessingConfig(
            data_directory=Path(args.data_dir),
            output_directory=Path(args.output_dir),
            replica=args.replica,
            test_ratio=args.test_ratio,
        )
        result = run_preprocessing(config)
        if result.success:
            print("\n✓ Preprocessing complete!")
            print(f"  Entities:    {result.num_entities}")
            print(f"  Relations:   {result.num_relations}")
            print(f"  Timesteps:   {result.num_timesteps}")
            print(f"  Train:       {result.train_samples} samples")
            print(f"  Valid:       {result.valid_samples} samples")
            print(f"  Test:        {result.test_samples} samples")
        else:
            print(f"\n✗ Preprocessing failed: {result.error_message}")
            sys.exit(1)

    elif args.command == "all":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not args.experiment_name:
            args.experiment_name = f"{args.dataset}_{timestamp}"

        # Step 0: Preprocess (if raw data provided)
        if hasattr(args, "data_dir") and args.data_dir:
            preprocess_config = PreprocessingConfig(
                data_directory=Path(args.data_dir),
                output_directory=DATA_DIR / args.dataset,
                replica=args.replica,
                test_ratio=args.test_ratio,
            )
            result = run_preprocessing(preprocess_config)
            if not result.success:
                print(f"\n✗ Preprocessing failed: {result.error_message}")
                sys.exit(1)
            print("\n✓ Preprocessing complete!")

        # Step 1: Pretrain encoder
        encoder_path = run_pretrain(args)
        args.encoder_path = str(encoder_path)

        # Step 2: Train decoder
        best_model_path = run_train(args)

        # Step 3: Evaluate
        args.checkpoint = str(best_model_path)
        run_evaluate(args)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
