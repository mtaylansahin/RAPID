#!/usr/bin/env python
"""
RAPID: A Recurrent Architecture for Predicting Protein Interaction Dynamics

This script orchestrates all functionality:
- pretrain: Train the global RGCN model
- train: Train the main RAPID model
- evaluate: Evaluate a trained model
- all: Run full pipeline (pretrain -> train -> evaluate)

Examples:
    # Full pipeline
    uv run python main.py all --dataset RAPID --epochs 100

    # Pretrain global model only
    uv run python main.py pretrain --dataset RAPID --epochs 30

    # Train with global model
    uv run python main.py train --dataset RAPID --use_global_model --epochs 100

    # Train without global model
    uv run python main.py train --dataset RAPID --epochs 100

    # Evaluate
    uv run python main.py evaluate --checkpoint ./checkpoints/RAPID_*/best.pth
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn

# Internal imports
from src.config import ModelConfig, TrainingConfig
from src.data.dataset import PPIDataModule
from src.data.preprocessing import PreprocessingConfig, run_preprocessing
from src.evaluate import Evaluator
from src.models.global_model import create_global_model
from src.models.rapid import create_model
from src.pretrain import train_global_model
from src.train import Trainer

# Constants
DATA_DIR = Path("./data")
MODELS_DIR = Path("./models")
CHECKPOINTS_DIR = Path("./checkpoints")
LOGS_DIR = Path("./logs")
PREDICTIONS_DIR = Path("./predictions")
ANALYSIS_DIR = Path("./analysis")


def get_base_args():
    """Get argument parser with common arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Common arguments
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
    default_hidden_dim: int = 200,
    default_seq_len: int = 10,
) -> nn.Module:
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
        hidden_dim=gm_config.get("hidden_dim", default_hidden_dim),
        num_bases=gm_config.get("num_bases", 5),
        seq_len=gm_config.get("seq_len", default_seq_len),
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
    print("Stage 1: Pretraining Global Model")
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

    # Create global model
    print("\nCreating global RGCN model...")
    model = create_global_model(
        num_entities=data_module.num_entities,
        num_rels=data_module.num_rels,
        hidden_dim=args.hidden_dim,
        num_bases=args.num_bases,
        seq_len=args.seq_len,
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
    """Run main model training."""
    print("\n" + "=" * 60)
    print("Stage 2: Training Main Model")
    print("=" * 60)

    device = setup_env(args)

    # Create configs
    model_config = ModelConfig(
        hidden_dim=args.hidden_dim,
        seq_len=args.seq_len,
        dropout=args.dropout,
    )

    training_config = TrainingConfig(
        learning_rate=args.lr,
        max_epochs=args.epochs,
        patience=args.patience,
        focal_gamma=args.focal_gamma,
    )

    # Setup paths
    checkpoint_dir = CHECKPOINTS_DIR / args.experiment_name
    log_dir = LOGS_DIR / args.experiment_name

    # Load data
    print(f"\nLoading dataset: {args.dataset}")
    data_module = PPIDataModule(
        data_path=DATA_DIR / args.dataset,
        batch_size=args.batch_size,
        neg_ratio=args.neg_ratio,
        hard_ratio=args.hard_ratio,
        seed=args.seed,
    )

    # Create model
    print("\nCreating model...")
    model = create_model(
        num_entities=data_module.num_entities,
        num_rels=data_module.num_rels,
        config=model_config,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load global model if specified
    global_model = None
    if args.use_global_model:
        if args.global_model_path:
            global_model_path = Path(args.global_model_path)
        else:
            global_model_path = MODELS_DIR / args.dataset / "max_global.pth"

        global_model = load_global_model(
            path=global_model_path,
            num_entities=data_module.num_entities,
            num_rels=data_module.num_rels,
            device=device,
            default_hidden_dim=args.hidden_dim,
            default_seq_len=args.seq_len,
        )
        if global_model is None:
            print("  Training without global model.")

    # Create trainer
    trainer = Trainer(
        model=model,
        data_module=data_module,
        config=training_config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        global_model=global_model,
    )

    # Train
    result = trainer.train()

    print("\nTraining complete!")
    print(f"  Best epoch: {result['best_epoch']}")
    print(f"  Best AUPRC: {result['best_val_auprc']:.4f}")
    print(f"  Optimal threshold: {result['optimal_threshold']:.3f}")

    return checkpoint_dir / "best.pth"


def run_evaluate(args) -> bool:
    """Run evaluation with analysis and visualization."""
    print("\n" + "=" * 60)
    print("Stage 3: Evaluation")
    print("=" * 60)

    # Find checkpoint if not specified
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

    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "config" in checkpoint and "model" in checkpoint["config"]:
        model_config = ModelConfig(**checkpoint["config"]["model"])
    else:
        model_config = ModelConfig()

    model = create_model(
        num_entities=data_module.num_entities,
        num_rels=data_module.num_rels,
        config=model_config,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # Get threshold (from checkpoint)
    threshold = checkpoint.get("optimal_threshold", 0.5)
    print(f"Using threshold: {threshold:.3f}")

    # Load global model if specified
    global_model = None
    if args.use_global_model:
        if args.global_model_path:
            global_model_path = Path(args.global_model_path)
        else:
            global_model_path = MODELS_DIR / args.dataset / "max_global.pth"

        global_model = load_global_model(
            path=global_model_path,
            num_entities=data_module.num_entities,
            num_rels=data_module.num_rels,
            device=device,
            default_hidden_dim=200,
            default_seq_len=10,
        )

    # Create evaluator
    evaluator = Evaluator(
        model=model,
        data_module=data_module,
        device=device,
        threshold=threshold,
        global_model=global_model,
    )

    # Prepare predictions directory
    predictions_dir = PREDICTIONS_DIR / args.dataset
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    results = evaluator.full_evaluation()
    evaluator.save_predictions(predictions_dir / "predictions.txt")

    # Run analysis and visualization
    from src.analysis import AnalysisManager

    analysis_dir = ANALYSIS_DIR / args.dataset
    analysis_manager = AnalysisManager(analysis_dir)

    # Prepare data for analysis
    train_data = data_module.train_data
    valid_data = data_module.val_dataset.data
    test_timesteps = list(data_module.test_dataset.timesteps)

    # Get ground truth from test set
    test_data = data_module.test_dataset.data
    ground_truth = [
        (int(row[0]), int(row[1]), int(row[2]), int(row[3])) for row in test_data
    ]

    # Prepare overall metrics
    overall_metrics = results.get("metrics", {})

    # Prepare per-timestep metrics
    per_ts = results.get("per_timestep", {})
    per_timestep_metrics = {
        "f1": {
            ts: f1 for ts, f1 in zip(per_ts.get("timesteps", []), per_ts.get("f1s", []))
        },
        "auprc": {
            ts: auprc
            for ts, auprc in zip(per_ts.get("timesteps", []), per_ts.get("auprcs", []))
        },
    }

    # Dataset info
    dataset_info = {
        "name": args.dataset,
        "num_entities": data_module.num_entities,
        "num_relations": data_module.num_rels,
        "train_timesteps": len(data_module.train_times),
        "test_timesteps": len(test_timesteps),
    }

    # Run analysis
    analysis_manager.run_analysis(
        predictions=evaluator.predictions,
        ground_truth=ground_truth,
        train_data=train_data,
        valid_data=valid_data,
        test_timesteps=test_timesteps,
        overall_metrics=overall_metrics,
        per_timestep_metrics=per_timestep_metrics,
        dataset_info=dataset_info,
    )

    return True


def main():
    """Main entry point."""
    # Create main parser
    parser = argparse.ArgumentParser(
        description="RAPID: A Recurrent Architecture for Predicting Protein Interaction Dynamics",
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
        "--pretrain_epochs", type=int, default=30, help="Number of pretraining epochs"
    )
    pretrain_parser.add_argument(
        "--pretrain_lr", type=float, default=1e-3, help="Pretraining learning rate"
    )
    pretrain_parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size"
    )
    pretrain_parser.add_argument(
        "--pooling",
        type=str,
        default="max",
        choices=["max", "mean"],
        help="Graph pooling method",
    )

    # === Train command ===
    train_parser = subparsers.add_parser(
        "train",
        help="Train main PPI dynamics model",
        parents=[get_base_args()],
    )
    train_parser.add_argument(
        "--epochs", type=int, default=100, help="Maximum training epochs"
    )
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    train_parser.add_argument(
        "--neg_ratio", type=float, default=1.0, help="Negative sampling ratio"
    )
    train_parser.add_argument(
        "--hard_ratio",
        type=float,
        default=0.5,
        help="Hard negative ratio (history constrained)",
    )
    train_parser.add_argument(
        "--focal_gamma", type=float, default=2.0, help="Focal loss gamma"
    )
    train_parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    train_parser.add_argument(
        "--use_global_model", action="store_true", help="Use pretrained global model"
    )
    train_parser.add_argument(
        "--global_model_path",
        type=str,
        default=None,
        help="Path to global model checkpoint",
    )
    train_parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name for checkpoints",
    )

    # === Evaluate command ===
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate trained model",
        parents=[get_base_args()],
    )
    eval_parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to model checkpoint"
    )
    eval_parser.add_argument(
        "--use_global_model", action="store_true", help="Use pretrained global model"
    )
    eval_parser.add_argument(
        "--global_model_path",
        type=str,
        default=None,
        help="Path to global model checkpoint",
    )
    # Prediction output options
    eval_parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save predicted interactions to text file",
    )
    eval_parser.add_argument(
        "--predictions_dir",
        type=str,
        default="./predictions",
        help="Directory to save prediction files",
    )

    # === All command (full pipeline) ===
    all_parser = subparsers.add_parser(
        "all",
        help="Run full pipeline: preprocess -> pretrain (optional) -> train -> evaluate",
        parents=[get_base_args()],
    )
    # Preprocess args
    all_parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Raw data directory with .interfacea files (optional, skip if data preprocessed)",
    )
    all_parser.add_argument(
        "--replica",
        type=str,
        default=None,
        help="Replica name for preprocessing (e.g., replica1)",
    )
    all_parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Test/validation ratio for preprocessing (default: 0.2)",
    )
    # Pretrain args
    all_parser.add_argument(
        "--pretrain_epochs", type=int, default=30, help="Number of pretraining epochs"
    )
    all_parser.add_argument(
        "--pretrain_lr", type=float, default=1e-3, help="Pretraining learning rate"
    )
    all_parser.add_argument(
        "--pooling",
        type=str,
        default="max",
        choices=["max", "mean"],
        help="Graph pooling method",
    )
    # Train args
    all_parser.add_argument(
        "--epochs", type=int, default=100, help="Maximum training epochs"
    )
    all_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    all_parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    all_parser.add_argument(
        "--neg_ratio", type=float, default=1.0, help="Negative sampling ratio"
    )
    all_parser.add_argument(
        "--hard_ratio",
        type=float,
        default=0.5,
        help="Hard negative ratio (history constrained)",
    )
    all_parser.add_argument(
        "--focal_gamma", type=float, default=2.0, help="Focal loss gamma"
    )
    all_parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    all_parser.add_argument(
        "--use_global_model", action="store_true", help="Use pretrained global model"
    )
    all_parser.add_argument(
        "--global_model_path",
        type=str,
        default=None,
        help="Path to global model checkpoint",
    )
    all_parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name for checkpoints",
    )
    all_parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (for evaluate only)",
    )
    # Prediction output options
    all_parser.add_argument(
        "--save_predictions",
        action="store_true",
        default=True,
        help="Save predicted interactions to text file (default: True)",
    )
    all_parser.add_argument(
        "--no_save_predictions",
        action="store_false",
        dest="save_predictions",
        help="Disable saving predictions to file",
    )
    all_parser.add_argument(
        "--predictions_dir",
        type=str,
        default="./predictions",
        help="Directory to save prediction files",
    )

    # === Preprocess command ===
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Preprocess raw MD simulation data into RAPID format",
    )
    preprocess_parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing replica folders with .interfacea files",
    )
    preprocess_parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for processed data files",
    )
    preprocess_parser.add_argument(
        "--replica",
        type=str,
        required=True,
        help="Replica name (e.g., replica1)",
    )
    preprocess_parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Fraction of timeline for test set; validation uses same ratio (default: 0.2)",
    )

    # === Multi-Analyze command ===
    multi_analyze_parser = subparsers.add_parser(
        "multi-analyze",
        help="Aggregate and visualize results from multiple replica runs",
    )
    multi_analyze_parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Base directory containing replica result subdirectories (e.g., results_dir/replica1/analysis/)",
    )
    multi_analyze_parser.add_argument(
        "--pattern",
        type=str,
        default="replica*/analysis",
        help="Glob pattern to find analysis directories (default: replica*/analysis)",
    )
    multi_analyze_parser.add_argument(
        "--output_dir",
        type=str,
        default="./multi_replica_analysis",
        help="Output directory for aggregated plots (default: ./multi_replica_analysis)",
    )
    multi_analyze_parser.add_argument(
        "--export_json",
        type=str,
        default=None,
        help="Export aggregated metrics to JSON file",
    )

    # === Analyze-Raw command ===
    analyze_raw_parser = subparsers.add_parser(
        "analyze-raw",
        help="Analyze raw .interfacea simulation data (temporal dynamics and stability)",
    )
    analyze_raw_parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing complex/replica/.interfacea structure",
    )
    analyze_raw_parser.add_argument(
        "--output_dir",
        type=str,
        default="./raw_analysis",
        help="Output directory for plots (default: ./raw_analysis)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    print("\n" + "=" * 60)
    print("RAPID: Recurrent Architecture for Predicting Protein Interaction Dynamics")
    print("=" * 60)
    print(f"Command: {args.command}")
    if args.command not in ("preprocess", "multi-analyze", "analyze-raw"):
        print(f"Dataset: {args.dataset}")
        print(f"GPU: {args.gpu}")
        print(f"Seed: {args.seed}")

    # Execute command
    if args.command == "pretrain":
        run_pretrain(args)

    elif args.command == "train":
        # Setup experiment name
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
        if hasattr(args, "data_dir") and args.data_dir:
            preprocess_config = PreprocessingConfig(
                data_directory=Path(args.data_dir),
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

    elif args.command == "multi-analyze":
        from src.analysis.multi_replica import MultiReplicaAnalyzer, MultiReplicaPlotter
        import json

        print("\n" + "=" * 60)
        print("Multi-Replica Analysis")
        print("=" * 60)

        analyzer = MultiReplicaAnalyzer()
        replica_paths = analyzer.discover_replica_directories(
            Path(args.results_dir), args.pattern
        )

        if not replica_paths:
            print("Error: No replica directories found")
            sys.exit(1)

        if not analyzer.load_replica_metrics(replica_paths):
            print("Error: Failed to load replica metrics")
            sys.exit(1)

        aggregated = analyzer.aggregate_metrics()
        if aggregated is None:
            print("Error: Failed to aggregate metrics")
            sys.exit(1)

        # Print summary
        print(f"\nAggregated {aggregated.n_replicas} replicas:")
        print(
            f"  F1:     {aggregated.overall_stats.f1.mean:.4f} ± {aggregated.overall_stats.f1.std:.4f}"
        )
        print(
            f"  Recall: {aggregated.overall_stats.recall.mean:.4f} ± {aggregated.overall_stats.recall.std:.4f}"
        )
        print(
            f"  MCC:    {aggregated.overall_stats.mcc.mean:.4f} ± {aggregated.overall_stats.mcc.std:.4f}"
        )

        # Generate plots
        output_dir = Path(args.output_dir)
        plotter = MultiReplicaPlotter(output_dir)
        plots = plotter.generate_plots(aggregated)
        print(f"\nGenerated {len(plots)} plots in {output_dir}")

        # Export JSON if requested
        if args.export_json:
            with open(args.export_json, "w") as f:
                json.dump(aggregated.to_dict(), f, indent=2)
            print(f"Exported metrics to {args.export_json}")

    elif args.command == "analyze-raw":
        from src.analysis.raw_analysis import TemporalAnalyzer, StabilityAnalyzer

        print("\n" + "=" * 60)
        print("Raw Simulation Data Analysis")
        print("=" * 60)

        output_dir = Path(args.output_dir)

        # Temporal analysis
        temporal = TemporalAnalyzer(output_dir)
        if not temporal.load_data(Path(args.data_dir)):
            print("Error: Failed to load data")
            sys.exit(1)

        temporal_plots = temporal.generate_plots()
        print(f"Generated {len(temporal_plots)} temporal dynamics plots")

        # Stability analysis (reuses loaded data)
        stability = StabilityAnalyzer(output_dir)
        stability.set_data(temporal.all_data)
        stability_plots = stability.generate_plots()
        print(f"Generated {len(stability_plots)} stability distribution plots")

        print(f"\n✓ All plots saved to {output_dir}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
