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
import subprocess
import sys
from pathlib import Path
from datetime import datetime


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
    
    cmd = [
        sys.executable, 'src/pretrain.py',
        '--dataset', args.dataset,
        '--hidden_dim', str(args.hidden_dim),
        '--seq_len', str(args.seq_len),
        '--num_bases', str(args.num_bases),
        '--dropout', str(args.dropout),
        '--epochs', str(args.pretrain_epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.pretrain_lr),
        '--pooling', args.pooling,
        '--gpu', str(args.gpu),
        '--seed', str(args.seed),
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("Error: Pretraining failed!")
        sys.exit(1)
    
    return result.returncode == 0


def run_train(args):
    """Run main model training."""
    print("\n" + "="*60)
    print("Stage 2: Training Main Model")
    print("="*60)
    
    cmd = [
        sys.executable, 'src/train.py',
        '--dataset', args.dataset,
        '--hidden_dim', str(args.hidden_dim),
        '--seq_len', str(args.seq_len),
        '--dropout', str(args.dropout),
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--neg_ratio', str(args.neg_ratio),
        '--focal_gamma', str(args.focal_gamma),
        '--patience', str(args.patience),
        '--gpu', str(args.gpu),
        '--seed', str(args.seed),
    ]
    
    if args.use_global_model:
        cmd.append('--use_global_model')
        if args.global_model_path:
            cmd.extend(['--global_model_path', args.global_model_path])
    
    if args.experiment_name:
        cmd.extend(['--experiment_name', args.experiment_name])
    
    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("Error: Training failed!")
        sys.exit(1)
    
    return result.returncode == 0


def run_evaluate(args):
    """Run evaluation."""
    print("\n" + "="*60)
    print("Stage 3: Evaluation")
    print("="*60)
    
    # Find checkpoint if not specified
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        # Look for most recent checkpoint
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
    
    cmd = [
        sys.executable, 'src/evaluate.py',
        '--checkpoint', checkpoint_path,
        '--dataset', args.dataset,
        '--mode', args.eval_mode,
        '--gpu', str(args.gpu),
    ]
    
    if args.use_global_model:
        cmd.append('--use_global_model')
        if args.global_model_path:
            cmd.extend(['--global_model_path', args.global_model_path])
    
    # Add prediction saving options
    if getattr(args, 'save_predictions', False):
        cmd.append('--save_predictions')
        if args.predictions_dir:
            cmd.extend(['--predictions_dir', args.predictions_dir])
        if getattr(args, 'include_scores', False):
            cmd.append('--include_scores')
        if getattr(args, 'include_negative', False):
            cmd.append('--include_negative')
    
    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    
    return result.returncode == 0


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
    eval_parser.add_argument('--eval_mode', type=str, default='all',
                              choices=['teacher_forcing', 'autoregressive', 'all'],
                              help='Evaluation mode')
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
    # Eval args
    all_parser.add_argument('--eval_mode', type=str, default='all',
                             choices=['teacher_forcing', 'autoregressive', 'all'],
                             help='Evaluation mode')
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
