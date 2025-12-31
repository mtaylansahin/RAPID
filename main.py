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
import os
from pathlib import Path
from datetime import datetime

# Add project root and src/ to path for imports (Colab safety)
PROJECT_ROOT = Path(__file__).parent.resolve()
SRC_ROOT = PROJECT_ROOT / 'src'
for p in (PROJECT_ROOT, SRC_ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)


def get_base_args():
    """Get argument parser with common arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    
    # Common arguments
    parser.add_argument('--dataset', type=str, default='RAPID',
                        help='Dataset name (folder in data/)')
    parser.add_argument('--hidden_dim', type=int, default=200,
                        help='Hidden dimension')
    parser.add_argument('--seq_len', type=int, default=10,
                        help='History sequence length')
    parser.add_argument('--num_bases', type=int, default=5,
                        help='Number of RGCN bases')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU device (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser


def run_pretrain(args):
    """Run global model pretraining."""
    print("\n" + "="*60)
    print("Stage 1: Pretraining Global Model")
    print("="*60)
    
    # Import here to avoid import errors at module load time
    import numpy as np
    import torch
    from src.models.global_model import create_global_model
    from src.data.dataset import PPIDataModule
    from src.pretrain import train_global_model
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    device = torch.device('cpu')
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
    
    print(f"Using device: {device}")
    
    # Load data
    data_path = Path('./data') / args.dataset
    print(f"\nLoading dataset: {args.dataset}")
    data_module = PPIDataModule(
        data_path=data_path,
        batch_size=args.batch_size,
        neg_ratio=1.0,
        seed=args.seed,
    )
    
    # Create global model
    print(f"\nCreating global RGCN model...")
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
    output_dir = Path('./models') / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{args.pooling}_global.pth'
    
    train_global_model(
        model=model,
        data_module=data_module,
        device=device,
        epochs=args.pretrain_epochs,
        lr=args.pretrain_lr,
        output_path=output_path,
    )
    
    return True


def run_train(args):
    """Run main model training."""
    print("\n" + "="*60)
    print("Stage 2: Training Main Model")
    print("="*60)
    
    # Import here to avoid import errors at module load time
    import numpy as np
    import torch
    from src.config import DataConfig, ModelConfig, TrainingConfig
    from src.models.rapid import create_model
    from src.models.global_model import create_global_model
    from src.data.dataset import PPIDataModule
    from src.train import Trainer
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create configs
    data_config = DataConfig(
        dataset=args.dataset,
        data_dir=Path('./data'),
        neg_ratio=args.neg_ratio,
        hard_ratio=args.hard_ratio,
        batch_size=args.batch_size,
    )
    
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
        gpu=args.gpu,
        seed=args.seed,
    )
    
    # Setup paths
    checkpoint_dir = Path('./checkpoints') / args.experiment_name
    log_dir = Path('./logs') / args.experiment_name
    
    # Setup device
    device = torch.device('cpu')
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
    
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading dataset: {args.dataset}")
    data_module = PPIDataModule(
        data_path=data_config.dataset_path,
        batch_size=data_config.batch_size,
        neg_ratio=data_config.neg_ratio,
        hard_ratio=data_config.hard_ratio,
        seed=args.seed,
    )
    
    # Create model
    print(f"\nCreating model...")
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
            global_model_path = Path(f'./models/{args.dataset}/max_global.pth')
        
        if global_model_path.exists():
            print(f"\nLoading global model from: {global_model_path}")
            checkpoint = torch.load(global_model_path, map_location=device)
            
            gm_config = checkpoint.get('config', {})
            global_model = create_global_model(
                num_entities=data_module.num_entities,
                num_rels=data_module.num_rels,
                hidden_dim=gm_config.get('hidden_dim', args.hidden_dim),
                num_bases=gm_config.get('num_bases', 5),
                seq_len=gm_config.get('seq_len', args.seq_len),
                pooling=gm_config.get('pooling', 'max'),
            )
            global_model.load_state_dict(checkpoint['state_dict'])
            global_model.global_emb = checkpoint.get('global_emb', {})
            print(f"  Global embeddings loaded for {len(global_model.global_emb)} timesteps")
        else:
            print(f"\nWarning: Global model path not found: {global_model_path}")
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
    
    print(f"\nTraining complete!")
    print(f"  Best epoch: {result['best_epoch']}")
    print(f"  Best AUPRC: {result['best_val_auprc']:.4f}")
    print(f"  Optimal threshold: {result['optimal_threshold']:.3f}")
    
    return True


def run_evaluate(args):
    """Run evaluation."""
    print("\n" + "="*60)
    print("Stage 3: Evaluation")
    print("="*60)
    
    # Import here to avoid import errors at module load time
    import torch
    from src.config import ModelConfig
    from src.models.rapid import create_model
    from src.models.global_model import create_global_model
    from src.data.dataset import PPIDataModule
    from src.evaluate import Evaluator
    
    # Find checkpoint if not specified
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoint_dir = Path('./checkpoints')
        if checkpoint_dir.exists():
            experiment_dirs = sorted(
                [d for d in checkpoint_dir.iterdir() if d.is_dir()],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            if experiment_dirs:
                best_ckpt = experiment_dirs[0] / 'best.pth'
                if best_ckpt.exists():
                    checkpoint_path = str(best_ckpt)
    
    if not checkpoint_path:
        print("Error: No checkpoint found. Please specify --checkpoint")
        sys.exit(1)
    
    # Setup device
    device = torch.device('cpu')
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    
    print(f"Using device: {device}")
    
    # Load data
    data_path = Path('./data') / args.dataset
    print(f"\nLoading dataset: {args.dataset}")
    data_module = PPIDataModule(
        data_path=data_path,
        batch_size=128,
        neg_ratio=1.0,
    )
    
    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'config' in checkpoint and 'model' in checkpoint['config']:
        model_config = ModelConfig(**checkpoint['config']['model'])
    else:
        model_config = ModelConfig()
    
    model = create_model(
        num_entities=data_module.num_entities,
        num_rels=data_module.num_rels,
        config=model_config,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get threshold (from args or checkpoint)
    if hasattr(args, 'threshold') and args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = checkpoint.get('optimal_threshold', 0.5)
    print(f"Using threshold: {threshold:.3f}")
    
    # Load global model if specified
    global_model = None
    if args.use_global_model:
        if args.global_model_path:
            global_model_path = Path(args.global_model_path)
        else:
            global_model_path = Path(f'./models/{args.dataset}/max_global.pth')
        
        if global_model_path.exists():
            print(f"\nLoading global model from: {global_model_path}")
            gm_checkpoint = torch.load(global_model_path, map_location=device)
            gm_config = gm_checkpoint.get('config', {})
            
            global_model = create_global_model(
                num_entities=data_module.num_entities,
                num_rels=data_module.num_rels,
                hidden_dim=gm_config.get('hidden_dim', 200),
                num_bases=gm_config.get('num_bases', 5),
                seq_len=gm_config.get('seq_len', 10),
                pooling=gm_config.get('pooling', 'max'),
            )
            global_model.load_state_dict(gm_checkpoint['state_dict'])
            global_model.global_emb = gm_checkpoint.get('global_emb', {})
            print(f"  Global embeddings for {len(global_model.global_emb)} timesteps")
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        data_module=data_module,
        device=device,
        threshold=threshold,
        global_model=global_model,
    )
    
    # Prepare predictions directory if saving predictions
    predictions_dir = Path(args.predictions_dir) / args.dataset if args.save_predictions else None
    
    # Run evaluation
    results = evaluator.full_evaluation()
    if args.save_predictions:
        evaluator.evaluate(collect_predictions=True)
        evaluator.save_predictions(
            predictions_dir / 'predictions.txt',
            include_negative=getattr(args, 'include_negative', False),
            include_scores=getattr(args, 'include_scores', False),
        )
    
    return True


def main():
    """Main entry point."""
    # Create main parser
    parser = argparse.ArgumentParser(
        description='RAPID: A Recurrent Architecture for Predicting Protein Interaction Dynamics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # === Pretrain command ===
    pretrain_parser = subparsers.add_parser(
        'pretrain',
        help='Pretrain global RGCN model',
        parents=[get_base_args()],
    )
    pretrain_parser.add_argument('--pretrain_epochs', type=int, default=30,
                                  help='Number of pretraining epochs')
    pretrain_parser.add_argument('--pretrain_lr', type=float, default=1e-3,
                                  help='Pretraining learning rate')
    pretrain_parser.add_argument('--batch_size', type=int, default=64,
                                  help='Batch size')
    pretrain_parser.add_argument('--pooling', type=str, default='max',
                                  choices=['max', 'mean'],
                                  help='Graph pooling method')
    
    # === Train command ===
    train_parser = subparsers.add_parser(
        'train',
        help='Train main PPI dynamics model',
        parents=[get_base_args()],
    )
    train_parser.add_argument('--epochs', type=int, default=100,
                               help='Maximum training epochs')
    train_parser.add_argument('--lr', type=float, default=1e-3,
                               help='Learning rate')
    train_parser.add_argument('--batch_size', type=int, default=128,
                               help='Batch size')
    train_parser.add_argument('--neg_ratio', type=float, default=1.0,
                               help='Negative sampling ratio')
    train_parser.add_argument('--hard_ratio', type=float, default=0.5,
                               help='Hard negative ratio (history constrained)')
    train_parser.add_argument('--focal_gamma', type=float, default=2.0,
                               help='Focal loss gamma')
    train_parser.add_argument('--patience', type=int, default=10,
                               help='Early stopping patience')
    train_parser.add_argument('--use_global_model', action='store_true',
                               help='Use pretrained global model')
    train_parser.add_argument('--global_model_path', type=str, default=None,
                               help='Path to global model checkpoint')
    train_parser.add_argument('--experiment_name', type=str, default=None,
                               help='Experiment name for checkpoints')
    
    # === Evaluate command ===
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate trained model',
        parents=[get_base_args()],
    )
    eval_parser.add_argument('--checkpoint', type=str, default=None,
                              help='Path to model checkpoint')
    eval_parser.add_argument('--threshold', type=float, default=None,
                              help='Classification threshold (default: use checkpoint value)')
    eval_parser.add_argument('--use_global_model', action='store_true',
                              help='Use pretrained global model')
    eval_parser.add_argument('--global_model_path', type=str, default=None,
                              help='Path to global model checkpoint')
    # Prediction output options
    eval_parser.add_argument('--save_predictions', action='store_true',
                              help='Save predicted interactions to text file')
    eval_parser.add_argument('--predictions_dir', type=str, default='./predictions',
                              help='Directory to save prediction files')
    eval_parser.add_argument('--include_scores', action='store_true',
                              help='Include prediction scores in output file')
    eval_parser.add_argument('--include_negative', action='store_true',
                              help='Include negative predictions in output file')
    
    # === All command (full pipeline) ===
    all_parser = subparsers.add_parser(
        'all',
        help='Run full pipeline: pretrain -> train -> evaluate',
        parents=[get_base_args()],
    )
    # Pretrain args
    all_parser.add_argument('--pretrain_epochs', type=int, default=30,
                             help='Number of pretraining epochs')
    all_parser.add_argument('--pretrain_lr', type=float, default=1e-3,
                             help='Pretraining learning rate')
    all_parser.add_argument('--pooling', type=str, default='max',
                             choices=['max', 'mean'],
                             help='Graph pooling method')
    # Train args
    all_parser.add_argument('--epochs', type=int, default=100,
                             help='Maximum training epochs')
    all_parser.add_argument('--lr', type=float, default=1e-3,
                             help='Learning rate')
    all_parser.add_argument('--batch_size', type=int, default=128,
                             help='Batch size')
    all_parser.add_argument('--neg_ratio', type=float, default=1.0,
                             help='Negative sampling ratio')
    all_parser.add_argument('--hard_ratio', type=float, default=0.5,
                             help='Hard negative ratio (history constrained)')
    all_parser.add_argument('--focal_gamma', type=float, default=2.0,
                             help='Focal loss gamma')
    all_parser.add_argument('--patience', type=int, default=10,
                             help='Early stopping patience')
    all_parser.add_argument('--use_global_model', action='store_true',
                             help='Use pretrained global model')
    all_parser.add_argument('--global_model_path', type=str, default=None,
                             help='Path to global model checkpoint')
    all_parser.add_argument('--experiment_name', type=str, default=None,
                             help='Experiment name for checkpoints')
    all_parser.add_argument('--checkpoint', type=str, default=None,
                             help='Path to model checkpoint (for evaluate only)')
    # Prediction output options
    all_parser.add_argument('--save_predictions', action='store_true', default=True,
                             help='Save predicted interactions to text file (default: True)')
    all_parser.add_argument('--no_save_predictions', action='store_false', dest='save_predictions',
                             help='Disable saving predictions to file')
    all_parser.add_argument('--predictions_dir', type=str, default='./predictions',
                             help='Directory to save prediction files')
    all_parser.add_argument('--include_scores', action='store_true',
                             help='Include prediction scores in output file')
    all_parser.add_argument('--include_negative', action='store_true',
                             help='Include negative predictions in output file')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    print("\n" + "="*60)
    print("RAPID: Recurrent Architecture for Predicting Protein Interaction Dynamics")
    print("="*60)
    print(f"Command: {args.command}")
    print(f"Dataset: {args.dataset}")
    print(f"GPU: {args.gpu}")
    print(f"Seed: {args.seed}")
    
    # Execute command
    if args.command == 'pretrain':
        run_pretrain(args)
    
    elif args.command == 'train':
        # Setup experiment name
        if args.experiment_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            args.experiment_name = f"{args.dataset}_{timestamp}"
        run_train(args)
    
    elif args.command == 'evaluate':
        run_evaluate(args)
    
    elif args.command == 'all':
        # Run full pipeline
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if not args.experiment_name:
            args.experiment_name = f"{args.dataset}_{timestamp}"
        
        # Step 1: Pretrain (if using global model)
        if args.use_global_model:
            run_pretrain(args)
            args.global_model_path = f'./models/{args.dataset}/{args.pooling}_global.pth'
        
        # Step 2: Train
        run_train(args)
        
        # Step 3: Evaluate
        if not args.checkpoint:
            args.checkpoint = f'./checkpoints/{args.experiment_name}/best.pth'
        run_evaluate(args)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == '__main__':
    main()
