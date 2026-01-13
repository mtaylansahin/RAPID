#!/usr/bin/env python
"""
RAPID: A Recurrent Architecture for Predicting Protein Interaction Dynamics

Unified pipeline supporting:
- preprocess: Convert raw MD simulation data to RAPID format
- pretrain: Train the global RGCN model
- train: Train the TrajectoryRAPID model (hybrid RGCN + Transformer)
- evaluate: Evaluate a trained model
- all: Run full pipeline (preprocess -> pretrain -> train -> evaluate)

The architecture uses TrajectoryRAPID - a hybrid model combining:
- RGCN for structure-aware entity embeddings
- Transformer encoder for edge history
- Cross-attention to neighbor edges
- Transformer decoder for full trajectory prediction

Examples:
    # Full pipeline
    uv run python main.py all --dataset 1JPS --epochs 50

    # Pretrain global model only
    uv run python main.py pretrain --dataset 1JPS --epochs 30

    # Train with all features
    uv run python main.py train --dataset 1JPS --epochs 50

    # Ablation: disable RGCN
    uv run python main.py train --dataset 1JPS --no_rgcn

    # Evaluate
    uv run python main.py evaluate --checkpoint ./checkpoints/1JPS_*/best.pth
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

# Internal imports
from src.config import TrajectoryModelConfig, TrajectoryTrainingConfig
from src.data.trajectory_dataset import TrajectoryDataModule
from src.data.dataset import PPIDataModule  # Still needed for pretrain
from src.data.preprocessing import PreprocessingConfig, run_preprocessing
from src.models.trajectory_rapid import TrajectoryRAPIDModel
from src.models.global_model import create_global_model
from src.pretrain import train_global_model
from src.train_trajectory import TrajectoryTrainer

# Constants
DATA_DIR = Path("./data/processed")
RAW_DATA_DIR = Path("./data/raw")
MODELS_DIR = Path("./models")
CHECKPOINTS_DIR = Path("./checkpoints")
LOGS_DIR = Path("./logs")
PREDICTIONS_DIR = Path("./predictions")
ANALYSIS_DIR = Path("./analysis_outputs")


def get_base_args():
    """Get argument parser with common arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Common arguments
    parser.add_argument(
        "--dataset", type=str, default="1JPS", help="Dataset name (folder in data/processed/)"
    )
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU device (-1 for CPU)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser


def setup_env(args) -> torch.device:
    """Set random seeds and setup compute device."""
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup device
    device = torch.device("cpu")
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.gpu)

    print(f"Using device: {device}")
    return device


def load_global_model(
    path: Union[str, Path],
    num_entities: int,
    num_rels: int,
    device: torch.device,
    hidden_dim: int = 128,
) -> Optional[nn.Module]:
    """Load a pretrained global RGCN model from checkpoint."""
    path = Path(path)
    if not path.exists():
        print(f"\nWarning: Global model path not found: {path}")
        return None

    print(f"\nLoading global model from: {path}")
    checkpoint = torch.load(path, map_location=device)

    gm_config = checkpoint.get("config", {})

    model = create_global_model(
        num_entities=num_entities,
        num_rels=num_rels,
        hidden_dim=gm_config.get("hidden_dim", hidden_dim),
        num_bases=gm_config.get("num_bases", 5),
        seq_len=gm_config.get("seq_len", 10),
        pooling=gm_config.get("pooling", "max"),
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.global_emb = checkpoint.get("global_emb", {})
    model = model.to(device)

    print(f"  Global embeddings loaded for {len(model.global_emb)} timesteps")
    return model


def run_pretrain(args) -> Path:
    """Run global model pretraining."""
    print("\n" + "=" * 60)
    print("Stage: Pretraining Global Model")
    print("=" * 60)

    device = setup_env(args)

    # Load data using PPIDataModule (for pretraining compatibility)
    data_path = DATA_DIR / args.dataset
    print(f"\nLoading dataset: {args.dataset}")
    data_module = PPIDataModule(
        data_path=data_path,
        batch_size=args.batch_size,
        neg_ratio=1.0,
        seed=args.seed,
    )

    # Create global model
    print("\nCreating global RGCN model...")
    model = create_global_model(
        num_entities=data_module.num_entities,
        num_rels=data_module.num_rels,
        hidden_dim=args.hidden_dim,
        num_bases=args.num_bases,
        seq_len=10,
        pooling=args.pooling,
    )
    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    output_dir = MODELS_DIR / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.pooling}_global.pth"

    train_global_model(
        model=model,
        data_module=data_module,
        device=device,
        epochs=args.pretrain_epochs,
        lr=args.pretrain_lr,
        output_path=output_path,
    )

    return output_path


def run_train(args) -> Path:
    """Run TrajectoryRAPID training."""
    print("\n" + "=" * 60)
    print("Stage: Training TrajectoryRAPID Model")
    print("=" * 60)

    device = setup_env(args)

    # Setup paths
    checkpoint_dir = CHECKPOINTS_DIR / args.experiment_name
    log_dir = LOGS_DIR / args.experiment_name

    # Load data
    data_path = DATA_DIR / args.dataset
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

    # Create model
    print("\nCreating TrajectoryRAPID model...")
    print(f"  RGCN: {not args.no_rgcn}")
    print(f"  Global context: {not args.no_global}")
    print(f"  Neighbor attention: {not args.no_neighbors}")
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
        num_rgcn_layers=args.num_rgcn_layers,
        num_bases=args.num_bases,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Create trainer
    trainer = TrajectoryTrainer(
        model=model,
        data_module=data_module,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        device=device,
        transition_weight=args.transition_weight,
        use_focal_loss=args.focal_loss,
        focal_gamma=args.focal_gamma,
    )

    # Train
    result = trainer.train(n_epochs=args.epochs)

    print("\nTraining complete!")
    print(f"  Best epoch: {result['best_epoch']}")
    print(f"  Best AUPRC: {result['best_val_auprc']:.4f}")

    return checkpoint_dir / "best.pth"


def run_evaluate(args) -> Dict:
    """Run evaluation."""
    print("\n" + "=" * 60)
    print("Stage: Evaluation")
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

    data_module = TrajectoryDataModule(
        data_path=data_path,
        history_ratio=args.history_ratio,
        val_ratio=args.val_ratio,
        max_neighbors=args.max_neighbors,
        batch_size=args.batch_size,
        seed=args.seed,
        n_hops=args.n_hops,
    )

    # Load checkpoint
    print(f"\nLoading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint
    ckpt_config = checkpoint.get("config", {})

    # Create model with checkpoint config
    model = TrajectoryRAPIDModel(
        num_entities=data_module.num_entities,
        num_rels=data_module.num_rels,
        hidden_dim=ckpt_config.get("hidden_dim", args.hidden_dim),
        n_heads=args.n_heads,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        n_neighbor_layers=args.n_neighbor_layers,
        max_neighbors=args.max_neighbors,
        max_seq_len=data_module.hist_len + data_module.traj_len + 10,
        dropout=args.dropout,
        use_rgcn=ckpt_config.get("use_rgcn", not args.no_rgcn),
        use_global_context=ckpt_config.get("use_global_context", not args.no_global),
        use_neighbor_attention=ckpt_config.get("use_neighbor_attention", not args.no_neighbors),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Create trainer for evaluation
    trainer = TrajectoryTrainer(
        model=model,
        data_module=data_module,
        learning_rate=args.lr,
        device=device,
        checkpoint_dir=CHECKPOINTS_DIR / "eval_temp",
        log_dir=LOGS_DIR / "eval_temp",
    )

    # Run test evaluation
    test_metrics = trainer.evaluate_test()

    # Save predictions
    predictions_dir = PREDICTIONS_DIR / args.dataset
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = predictions_dir / "predictions.txt"
    _save_trajectory_predictions(
        model=model,
        data_module=data_module,
        device=device,
        output_path=predictions_path,
    )

    # Run analysis if available
    try:
        from src.analysis import AnalysisConfig, ResultsManager
        
        analysis_output_dir = ANALYSIS_DIR / Path(checkpoint_path).parent.name
        analysis_config = AnalysisConfig(
            input_directory=str(data_path),
            output_directory=str(analysis_output_dir),
            output_file_path=str(predictions_path),
        )
        results_manager = ResultsManager(analysis_config)
        if results_manager.run_complete_analysis():
            print(f"\nAnalysis outputs saved to: {analysis_output_dir}")
    except Exception as e:
        print(f"\nWarning: Analysis failed: {e}")

    return test_metrics


def _save_trajectory_predictions(
    model: TrajectoryRAPIDModel,
    data_module: TrajectoryDataModule,
    device: torch.device,
    output_path: Path,
    threshold: float = 0.45,  # Selective threshold based on probability distribution
):
    """Save trajectory predictions to file.
    
    Saves ALL predictions with probabilities for proper analysis,
    using threshold only for the binary prediction column.
    """
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_predictions = []  # Store all predictions with probabilities
    positive_count = 0
    dataloader = data_module.get_test_dataloader()

    with torch.no_grad():
        for batch in dataloader:
            entity1 = batch["entity1"].to(device)
            entity2 = batch["entity2"].to(device)
            history = batch["history"].to(device)
            history_timesteps = batch["history_timesteps"].to(device)
            neighbor_e1 = batch["neighbor_entity1"].to(device)
            neighbor_e2 = batch["neighbor_entity2"].to(device)
            neighbor_hist = batch["neighbor_history"].to(device)
            neighbor_mask = batch["neighbor_mask"].to(device)
            target = batch["target"].to(device)

            traj_len = target.size(1)

            logits = model(
                entity1_ids=entity1,
                entity2_ids=entity2,
                history=history,
                history_timesteps=history_timesteps,
                neighbor_entity1=neighbor_e1,
                neighbor_entity2=neighbor_e2,
                neighbor_history=neighbor_hist,
                neighbor_mask=neighbor_mask,
                traj_len=traj_len,
                graph_dict=data_module.graph_dict,
            )

            probs = torch.sigmoid(logits)
            preds = (probs > threshold).long()
            ground_truth = target

            # Store ALL predictions with entity IDs, timesteps, probs, and ground truth
            for b in range(entity1.size(0)):
                e1, e2 = entity1[b].item(), entity2[b].item()
                for t_idx in range(traj_len):
                    actual_t = data_module.hist_len + t_idx
                    prob = probs[b, t_idx].item()
                    pred = preds[b, t_idx].item()
                    gt = ground_truth[b, t_idx].item()
                    all_predictions.append((e1, e2, actual_t, prob, pred, gt))
                    if pred == 1:
                        positive_count += 1

    # Write ALL predictions with probabilities
    with open(output_path, "w") as f:
        # Header for clarity
        f.write("# e1\te2\ttimestep\tprobability\tprediction\tground_truth\n")
        for e1, e2, t, prob, pred, gt in all_predictions:
            f.write(f"{e1}\t{e2}\t{t}\t{prob:.4f}\t{pred}\t{gt}\n")

    print(f"\nPredictions saved to: {output_path}")
    print(f"  Total predictions: {len(all_predictions)}")
    print(f"  Positive predictions (threshold={threshold}): {positive_count}")
    
    # Also save a summary of probability distribution
    probs_list = [p[3] for p in all_predictions]
    if probs_list:
        print(f"  Probability stats: min={min(probs_list):.4f}, max={max(probs_list):.4f}, mean={sum(probs_list)/len(probs_list):.4f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAPID: Recurrent Architecture for Predicting Protein Interaction Dynamics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # === Pretrain command ===
    pretrain_parser = subparsers.add_parser(
        "pretrain",
        help="Pretrain global RGCN model",
        parents=[get_base_args()],
    )
    pretrain_parser.add_argument(
        "--pretrain_epochs", type=int, default=30, help="Pretraining epochs"
    )
    pretrain_parser.add_argument(
        "--pretrain_lr", type=float, default=1e-3, help="Pretraining learning rate"
    )
    pretrain_parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    pretrain_parser.add_argument("--num_bases", type=int, default=5, help="RGCN bases")
    pretrain_parser.add_argument(
        "--pooling", type=str, default="max", choices=["max", "mean"],
        help="Graph pooling method"
    )

    # === Train command ===
    train_parser = subparsers.add_parser(
        "train",
        help="Train TrajectoryRAPID model",
        parents=[get_base_args()],
    )
    # Architecture
    train_parser.add_argument("--n_heads", type=int, default=4, help="Attention heads")
    train_parser.add_argument("--n_encoder_layers", type=int, default=2, help="Encoder layers")
    train_parser.add_argument("--n_decoder_layers", type=int, default=2, help="Decoder layers")
    train_parser.add_argument("--n_neighbor_layers", type=int, default=1, help="Neighbor attn layers")
    train_parser.add_argument("--max_neighbors", type=int, default=50, help="Max neighbors")
    train_parser.add_argument("--num_rgcn_layers", type=int, default=2, help="RGCN layers")
    train_parser.add_argument("--num_bases", type=int, default=100, help="RGCN bases")
    # Feature flags
    train_parser.add_argument("--no_rgcn", action="store_true", help="Disable RGCN")
    train_parser.add_argument("--no_global", action="store_true", help="Disable global context")
    train_parser.add_argument("--no_neighbors", action="store_true", help="Disable neighbor attention")
    train_parser.add_argument("--use_node_features", action="store_true", help="Enable node features")
    train_parser.add_argument("--n_hops", type=int, default=1, help="Neighbor hops")
    # Training
    train_parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    train_parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    train_parser.add_argument("--transition_weight", type=float, default=1.0, help="Transition weight")
    train_parser.add_argument("--focal_loss", action="store_true", help="Use focal loss")
    train_parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal gamma")
    # Data
    train_parser.add_argument("--history_ratio", type=float, default=0.5, help="History ratio")
    train_parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio")
    train_parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")

    # === Evaluate command ===
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate trained model",
        parents=[get_base_args()],
    )
    eval_parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    eval_parser.add_argument("--n_heads", type=int, default=4, help="Attention heads")
    eval_parser.add_argument("--n_encoder_layers", type=int, default=2, help="Encoder layers")
    eval_parser.add_argument("--n_decoder_layers", type=int, default=2, help="Decoder layers")
    eval_parser.add_argument("--n_neighbor_layers", type=int, default=1, help="Neighbor attn layers")
    eval_parser.add_argument("--max_neighbors", type=int, default=50, help="Max neighbors")
    eval_parser.add_argument("--no_rgcn", action="store_true", help="Disable RGCN")
    eval_parser.add_argument("--no_global", action="store_true", help="Disable global context")
    eval_parser.add_argument("--no_neighbors", action="store_true", help="Disable neighbor attention")
    eval_parser.add_argument("--n_hops", type=int, default=1, help="Neighbor hops")
    eval_parser.add_argument("--history_ratio", type=float, default=0.5, help="History ratio")
    eval_parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio")
    eval_parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    eval_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (unused)")

    # === All command (full pipeline) ===
    all_parser = subparsers.add_parser(
        "all",
        help="Run full pipeline: preprocess -> pretrain -> train -> evaluate",
        parents=[get_base_args()],
    )
    # Preprocess
    all_parser.add_argument("--raw_data_dir", type=str, default=None, help="Raw data directory")
    all_parser.add_argument("--replica", type=str, default=None, help="Replica name")
    all_parser.add_argument("--test_ratio", type=float, default=0.2, help="Test ratio")
    # Pretrain
    all_parser.add_argument("--pretrain_epochs", type=int, default=30, help="Pretrain epochs")
    all_parser.add_argument("--pretrain_lr", type=float, default=1e-3, help="Pretrain LR")
    all_parser.add_argument("--pooling", type=str, default="max", help="Pooling method")
    all_parser.add_argument("--use_global_model", action="store_true", help="Use global model")
    all_parser.add_argument("--num_bases", type=int, default=100, help="RGCN bases")
    # Architecture
    all_parser.add_argument("--n_heads", type=int, default=4, help="Attention heads")
    all_parser.add_argument("--n_encoder_layers", type=int, default=2, help="Encoder layers")
    all_parser.add_argument("--n_decoder_layers", type=int, default=2, help="Decoder layers")
    all_parser.add_argument("--n_neighbor_layers", type=int, default=1, help="Neighbor attn layers")
    all_parser.add_argument("--max_neighbors", type=int, default=50, help="Max neighbors")
    all_parser.add_argument("--num_rgcn_layers", type=int, default=2, help="RGCN layers")
    # Feature flags
    all_parser.add_argument("--no_rgcn", action="store_true", help="Disable RGCN")
    all_parser.add_argument("--no_global", action="store_true", help="Disable global context")
    all_parser.add_argument("--no_neighbors", action="store_true", help="Disable neighbor attention")
    all_parser.add_argument("--use_node_features", action="store_true", help="Enable node features")
    all_parser.add_argument("--n_hops", type=int, default=1, help="Neighbor hops")
    # Training
    all_parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    all_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    all_parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    all_parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    all_parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    all_parser.add_argument("--transition_weight", type=float, default=1.0, help="Transition weight")
    all_parser.add_argument("--focal_loss", action="store_true", help="Use focal loss")
    all_parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal gamma")
    # Data
    all_parser.add_argument("--history_ratio", type=float, default=0.5, help="History ratio")
    all_parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio")
    all_parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")
    all_parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint (eval only)")

    # === Preprocess command ===
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Preprocess raw MD simulation data",
    )
    preprocess_parser.add_argument("--data_dir", type=str, required=True, help="Raw data directory")
    preprocess_parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    preprocess_parser.add_argument("--replica", type=str, required=True, help="Replica name")
    preprocess_parser.add_argument("--test_ratio", type=float, default=0.2, help="Test ratio")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    print("\n" + "=" * 60)
    print("RAPID: Recurrent Architecture for Predicting Protein Interaction Dynamics")
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
            print(f"  Output:      {result.output_directory}")
        else:
            print(f"\n✗ Preprocessing failed: {result.error_message}")
            sys.exit(1)

    elif args.command == "all":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not args.experiment_name:
            args.experiment_name = f"{args.dataset}_{timestamp}"

        # Step 0: Preprocess (if raw data provided)
        if hasattr(args, "raw_data_dir") and args.raw_data_dir:
            preprocess_config = PreprocessingConfig(
                data_directory=Path(args.raw_data_dir),
                output_directory=DATA_DIR / args.dataset,
                replica=args.replica,
                test_ratio=args.test_ratio if hasattr(args, "test_ratio") else 0.2,
            )
            result = run_preprocessing(preprocess_config)
            if not result.success:
                print(f"\n✗ Preprocessing failed: {result.error_message}")
                sys.exit(1)
            print("\n✓ Preprocessing complete!")

        # Step 1: Pretrain (if using global model)
        if args.use_global_model:
            pretrain_path = run_pretrain(args)
            args.global_model_path = str(pretrain_path)

        # Step 2: Train
        best_model_path = run_train(args)

        # Step 3: Evaluate
        args.checkpoint = str(best_model_path)
        run_evaluate(args)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
