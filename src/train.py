"""Training script for RAPID - Protein Interaction Dynamics prediction."""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from src.config import Config, DataConfig, ModelConfig, TrainingConfig
from src.models.rapid import RAPIDModel, create_model
from src.models.global_model import PPIGlobalModel, create_global_model
from src.data.dataset import PPIDataModule
from src.losses import FocalLoss, get_loss_function
from src.metrics import MetricsComputer, ClassificationMetrics


class Trainer:
    """
    Trainer for RAPID model.
    
    Handles:
    - Training loop with teacher forcing
    - Validation with threshold tuning
    - Checkpointing and early stopping
    - Detailed metric logging
    
    Args:
        model: RAPIDModel instance
        data_module: PPIDataModule with train/val/test data
        config: Training configuration
        device: torch device
    """
    
    def __init__(
        self,
        model: RAPIDModel,
        data_module: PPIDataModule,
        config: TrainingConfig,
        device: torch.device,
        checkpoint_dir: Path,
        log_dir: Path,
        global_model: Optional[PPIGlobalModel] = None,
    ):
        self.model = model.to(device)
        self.data_module = data_module
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # Global model (optional)
        self.global_model = global_model
        if global_model is not None:
            self.global_model = global_model.to(device)
            self.global_model.eval()  # Always in eval mode
            self.global_emb = global_model.global_emb
        else:
            self.global_emb = None
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Setup loss function
        self.criterion = get_loss_function(
            loss_type='focal',
            gamma=config.focal_gamma,
            alpha=config.focal_alpha,
        )
        
        # Metrics computer
        self.train_metrics = MetricsComputer()
        self.val_metrics = MetricsComputer()
        
        # Best validation metrics for checkpointing
        self.best_val_auprc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Optimal threshold (tuned on validation)
        self.optimal_threshold = 0.5
        
        # Training history
        self.history: Dict[str, list] = {
            'train_loss': [],
            'train_auprc': [],
            'train_auroc': [],
            'train_f1': [],
            'val_loss': [],
            'val_auprc': [],
            'val_auroc': [],
            'val_f1': [],
        }
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, epoch: int) -> ClassificationMetrics:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        dataloader = self.data_module.get_train_dataloader()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:03d} [Train]")
        
        for batch in pbar:
            # Move to device
            entity1 = batch['entity1'].to(self.device)
            entity2 = batch['entity2'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            logits = self.model(
                entity1_ids=entity1,
                entity2_ids=entity2,
                entity1_history=batch['entity1_history'],
                entity2_history=batch['entity2_history'],
                entity1_history_t=batch['entity1_history_t'],
                entity2_history_t=batch['entity2_history_t'],
                graph_dict=self.data_module.graph_dict,
                global_emb=self.global_emb,
            )
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip_norm,
                )
            
            self.optimizer.step()
            
            # Update metrics
            self.train_metrics.update(logits, labels, loss.item())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute epoch metrics
        metrics = self.train_metrics.compute()
        return metrics
    
    @torch.no_grad()
    def validate(self, tune_threshold: bool = True) -> ClassificationMetrics:
        """Validate model and optionally tune threshold."""
        self.model.eval()
        self.val_metrics.reset()
        
        # Collect all predictions
        all_logits = []
        all_labels = []
        
        # Get validation timesteps
        timesteps = sorted(self.data_module.val_dataset.unique_timesteps)
        
        pbar = tqdm(timesteps, desc="Validation")
        
        for t in pbar:
            # Get ALL pairs with ground truth labels
            pairs, labels_np = self.data_module.get_all_pairs_for_timestep(t, split='valid')
            
            # Process in batches
            batch_size = 128
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                batch_labels = labels_np[i:i + batch_size]
                
                entity1 = torch.LongTensor(batch_pairs[:, 0]).to(self.device)
                entity2 = torch.LongTensor(batch_pairs[:, 1]).to(self.device)
                labels = torch.FloatTensor(batch_labels).to(self.device)
                
                # Get entity histories for this timestep
                entity1_history = []
                entity1_history_t = []
                entity2_history = []
                entity2_history_t = []
                
                for j in range(len(batch_pairs)):
                    e1, e2 = batch_pairs[j]
                    # Get history up to (not including) current timestep
                    e1_hist = [h for h, ht in zip(
                        self.data_module.entity_history[e1],
                        self.data_module.entity_history_t[e1]
                    ) if ht < t]
                    e1_hist_t = [ht for ht in self.data_module.entity_history_t[e1] if ht < t]
                    
                    e2_hist = [h for h, ht in zip(
                        self.data_module.entity_history[e2],
                        self.data_module.entity_history_t[e2]
                    ) if ht < t]
                    e2_hist_t = [ht for ht in self.data_module.entity_history_t[e2] if ht < t]
                    
                    entity1_history.append(e1_hist)
                    entity1_history_t.append(e1_hist_t)
                    entity2_history.append(e2_hist)
                    entity2_history_t.append(e2_hist_t)
                
                logits = self.model(
                    entity1_ids=entity1,
                    entity2_ids=entity2,
                    entity1_history=entity1_history,
                    entity2_history=entity2_history,
                    entity1_history_t=entity1_history_t,
                    entity2_history_t=entity2_history_t,
                    graph_dict=self.data_module.graph_dict,
                    global_emb=self.global_emb,
                )
                
                loss = self.criterion(logits, labels)
                self.val_metrics.update(logits, labels, loss.item())
        
        # Compute metrics with optional threshold tuning
        metrics = self.val_metrics.compute(tune_threshold=tune_threshold)
        
        if tune_threshold:
            self.optimal_threshold = metrics.threshold
        
        return metrics
    
    def train(self) -> Dict[str, Any]:
        """
        Full training loop.
        
        Returns:
            Training history and final metrics
        """
        print(f"\n{'='*60}")
        print("Starting training")
        print(f"{'='*60}")
        print(f"Entities: {self.data_module.num_entities}")
        print(f"Relations: {self.data_module.num_rels}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, self.config.max_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            if epoch % self.config.eval_interval == 0:
                val_metrics = self.validate(tune_threshold=True)
                
                # Log metrics
                self._log_metrics(epoch, train_metrics, val_metrics)
                
                # Check for improvement
                if val_metrics.auprc > self.best_val_auprc:
                    self.best_val_auprc = val_metrics.auprc
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, val_metrics, is_best=True)
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= self.config.patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    print(f"Best epoch: {self.best_epoch} with AUPRC: {self.best_val_auprc:.4f}")
                    break
            
            # Update history
            self.history['train_loss'].append(train_metrics.loss)
            self.history['train_auprc'].append(train_metrics.auprc)
            self.history['train_auroc'].append(train_metrics.auroc)
            self.history['train_f1'].append(train_metrics.f1)
            
            if epoch % self.config.eval_interval == 0:
                self.history['val_loss'].append(val_metrics.loss)
                self.history['val_auprc'].append(val_metrics.auprc)
                self.history['val_auroc'].append(val_metrics.auroc)
                self.history['val_f1'].append(val_metrics.f1)
        
        # Save final checkpoint
        self._save_checkpoint(epoch, val_metrics, is_best=False)
        
        # Save history
        self._save_history()
        
        return {
            'history': self.history,
            'best_epoch': self.best_epoch,
            'best_val_auprc': self.best_val_auprc,
            'optimal_threshold': self.optimal_threshold,
        }
    
    def _log_metrics(
        self,
        epoch: int,
        train_metrics: ClassificationMetrics,
        val_metrics: ClassificationMetrics,
    ) -> None:
        """Log metrics to console."""
        print(f"\nEpoch {epoch:03d}:")
        print(f"  Train: {train_metrics.short_str()}")
        print(f"  Valid: {val_metrics.short_str()}")
        print(f"  Threshold: {val_metrics.threshold:.3f}")
        print(f"  Confusion: TP={val_metrics.tp}, FP={val_metrics.fp}, "
              f"TN={val_metrics.tn}, FN={val_metrics.fn}")
        
        # Check for warning signs
        if val_metrics.tp == 0 and val_metrics.fn > 0:
            print("  ⚠️ WARNING: No positive predictions (all negatives)")
        if val_metrics.fp == 0 and val_metrics.tn > 0:
            print("  ⚠️ WARNING: No negative predictions (all positives)")
    
    def _save_checkpoint(
        self,
        epoch: int,
        metrics: ClassificationMetrics,
        is_best: bool,
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics.to_dict(),
            'optimal_threshold': self.optimal_threshold,
            'config': {
                'model': self.model.config.__dict__,
                'training': self.config.__dict__,
            },
        }
        
        # Save latest
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"  ✓ New best model saved (AUPRC: {metrics.auprc:.4f})")
    
    def _save_history(self) -> None:
        """Save training history."""
        history_path = self.log_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Train PPI dynamics model')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='RAPID',
                        help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=200,
                        help='Hidden dimension')
    parser.add_argument('--seq_len', type=int, default=10,
                        help='History sequence length')
    parser.add_argument('--num_rgcn_layers', type=int, default=2,
                        help='Number of RGCN layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--neg_ratio', type=float, default=1.0,
                        help='Negative sampling ratio')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU ID (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Global model arguments
    parser.add_argument('--use_global_model', action='store_true',
                        help='Use global RGCN model')
    parser.add_argument('--global_model_path', type=str, default=None,
                        help='Path to pretrained global model checkpoint')
    
    # Output arguments
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Log directory')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create configs
    data_config = DataConfig(
        dataset=args.dataset,
        data_dir=Path(args.data_dir),
        neg_ratio=args.neg_ratio,
        batch_size=args.batch_size,
    )
    
    model_config = ModelConfig(
        hidden_dim=args.hidden_dim,
        seq_len=args.seq_len,
        num_rgcn_layers=args.num_rgcn_layers,
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
    
    # Setup experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f"{args.dataset}_{timestamp}"
    
    checkpoint_dir = Path(args.checkpoint_dir) / args.experiment_name
    log_dir = Path(args.log_dir) / args.experiment_name
    
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
            # Default path
            global_model_path = Path(f'./models/{args.dataset}/max_global.pth')
        
        if global_model_path.exists():
            print(f"\nLoading global model from: {global_model_path}")
            checkpoint = torch.load(global_model_path, map_location=device)
            
            # Get config from checkpoint or use defaults
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
            print("  Training without global model. Run pretrain first:")
            print(f"  python src/pretrain.py --dataset {args.dataset}")
    
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
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best epoch: {result['best_epoch']}")
    print(f"Best validation AUPRC: {result['best_val_auprc']:.4f}")
    print(f"Optimal threshold: {result['optimal_threshold']:.3f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"{'='*60}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
