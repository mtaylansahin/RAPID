"""Evaluation script for RAPID - Protein Interaction Dynamics prediction."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ModelConfig
from src.models.rapid import RAPIDModel, create_model
from src.models.global_model import PPIGlobalModel, create_global_model
from src.data.dataset import PPIDataModule
from src.losses import get_loss_function
from src.metrics import (
    MetricsComputer,
    ClassificationMetrics,
    PerTimestepMetrics,
    compute_per_timestep_metrics,
)


class Evaluator:
    """
    Evaluator for RAPID model.
    
    Supports two modes:
    1. Teacher forcing: Use ground-truth history (like training)
    2. Autoregressive: Use predicted history (true inference)
    
    Args:
        model: Trained RAPIDModel
        data_module: Data module with test data
        device: torch device
        threshold: Classification threshold
    """
    
    def __init__(
        self,
        model: RAPIDModel,
        data_module: PPIDataModule,
        device: torch.device,
        threshold: float = 0.5,
        global_model: Optional[PPIGlobalModel] = None,
    ):
        self.model = model.to(device)
        self.data_module = data_module
        self.device = device
        self.threshold = threshold
        
        # Global model (optional)
        self.global_model = global_model
        if global_model is not None:
            self.global_model = global_model.to(device)
            self.global_model.eval()
            self.global_emb = global_model.global_emb
        else:
            self.global_emb = None
        
        self.criterion = get_loss_function(loss_type='focal', gamma=2.0)
    
    @torch.no_grad()
    def evaluate_teacher_forcing(self) -> ClassificationMetrics:
        """
        Evaluate with teacher forcing (ground-truth history).
        
        This is the standard evaluation matching training.
        """
        self.model.eval()
        metrics_computer = MetricsComputer(threshold=self.threshold)
        
        dataloader = self.data_module.get_test_dataloader()
        
        print("\nEvaluating with teacher forcing...")
        pbar = tqdm(dataloader, desc="Test")
        
        for batch in pbar:
            entity1 = batch['entity1'].to(self.device)
            entity2 = batch['entity2'].to(self.device)
            labels = batch['labels'].to(self.device)
            
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
            
            loss = self.criterion(logits, labels)
            metrics_computer.update(logits, labels, loss.item())
        
        return metrics_computer.compute()
    
    @torch.no_grad()
    def evaluate_autoregressive(self) -> ClassificationMetrics:
        """
        Evaluate autoregressively (predicted history).
        
        This simulates true inference where we don't know future interactions.
        """
        self.model.eval()
        
        # Initialize model with training history
        self.model.reset_inference_state()
        self.model.init_from_train_history(
            graph_dict=self.data_module.graph_dict,
            entity_history=self.data_module.entity_history,
            entity_history_t=self.data_module.entity_history_t,
        )
        
        # Collect all predictions
        all_logits = []
        all_labels = []
        
        print("\nEvaluating autoregressively...")
        
        # Get test data sorted by timestep
        self.data_module.test_dataset.neg_ratio = 1.0
        self.data_module.test_dataset.prepare_epoch()
        
        # Group samples by timestep for proper autoregressive processing
        samples_by_t: Dict[int, list] = {}
        for sample in self.data_module.test_dataset.samples:
            e1, e2, t, label = sample
            if t not in samples_by_t:
                samples_by_t[t] = []
            samples_by_t[t].append((e1, e2, label))
        
        # Process timesteps in order
        for t in tqdm(sorted(samples_by_t.keys()), desc="Timesteps"):
            samples = samples_by_t[t]
            
            # Process in batches
            batch_size = 128
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i + batch_size]
                
                entity1 = torch.LongTensor([s[0] for s in batch]).to(self.device)
                entity2 = torch.LongTensor([s[1] for s in batch]).to(self.device)
                labels = torch.FloatTensor([s[2] for s in batch]).to(self.device)
                
                # Get predictions (updates internal state)
                probs, preds = self.model.predict_batch(
                    entity1_ids=entity1,
                    entity2_ids=entity2,
                    timestep=t,
                    threshold=self.threshold,
                    update_history=True,
                )
                
                # Convert probs to logits for metrics
                probs_clamped = probs.clamp(1e-7, 1 - 1e-7)
                logits = torch.log(probs_clamped / (1 - probs_clamped))
                
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
        
        # Compute metrics
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        
        metrics_computer = MetricsComputer(threshold=self.threshold)
        metrics_computer.update(all_logits, all_labels)
        
        return metrics_computer.compute()
    
    @torch.no_grad()
    def evaluate_per_timestep(self) -> PerTimestepMetrics:
        """
        Evaluate and return metrics per timestep.
        
        Useful for analyzing temporal degradation.
        """
        self.model.eval()
        
        # Collect all predictions with timesteps
        all_logits = []
        all_labels = []
        all_timesteps = []
        
        dataloader = self.data_module.get_test_dataloader()
        
        print("\nComputing per-timestep metrics...")
        for batch in tqdm(dataloader):
            entity1 = batch['entity1'].to(self.device)
            entity2 = batch['entity2'].to(self.device)
            labels = batch['labels']
            timesteps = batch['timesteps']
            
            logits = self.model(
                entity1_ids=entity1,
                entity2_ids=entity2,
                entity1_history=batch['entity1_history'],
                entity2_history=batch['entity2_history'],
                entity1_history_t=batch['entity1_history_t'],
                entity2_history_t=batch['entity2_history_t'],
                graph_dict=self.data_module.graph_dict,
                global_emb=None,
            )
            
            all_logits.append(logits.cpu())
            all_labels.append(labels)
            all_timesteps.append(timesteps)
        
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        all_timesteps = torch.cat(all_timesteps)
        
        return compute_per_timestep_metrics(
            all_logits, all_labels, all_timesteps,
            threshold=self.threshold,
        )
    
    def full_evaluation(self) -> Dict[str, Any]:
        """
        Run full evaluation with all modes and analyses.
        
        Returns:
            Dictionary with all evaluation results
        """
        results = {}
        
        # Teacher forcing evaluation
        tf_metrics = self.evaluate_teacher_forcing()
        results['teacher_forcing'] = tf_metrics.to_dict()
        print(f"\nTeacher Forcing Results:")
        print(f"  {tf_metrics}")
        
        # Autoregressive evaluation
        ar_metrics = self.evaluate_autoregressive()
        results['autoregressive'] = ar_metrics.to_dict()
        print(f"\nAutoregressive Results:")
        print(f"  {ar_metrics}")
        
        # Per-timestep analysis (teacher forcing)
        per_ts_metrics = self.evaluate_per_timestep()
        results['per_timestep'] = per_ts_metrics.to_dict()
        print(f"\nPer-Timestep Analysis:")
        print(f"  Mean AUPRC: {per_ts_metrics.mean_auprc:.4f}")
        print(f"  Mean F1: {per_ts_metrics.mean_f1:.4f}")
        
        # Check for temporal degradation
        if len(per_ts_metrics.auprcs) > 5:
            early = np.mean(per_ts_metrics.auprcs[:5])
            late = np.mean(per_ts_metrics.auprcs[-5:])
            if late < early * 0.9:
                print(f"  ⚠️ Temporal degradation detected: "
                      f"early AUPRC {early:.4f} vs late {late:.4f}")
        
        return results


def load_model(
    checkpoint_path: Path,
    data_module: PPIDataModule,
    device: torch.device,
) -> RAPIDModel:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint or use defaults
    if 'config' in checkpoint and 'model' in checkpoint['config']:
        model_config = ModelConfig(**checkpoint['config']['model'])
    else:
        model_config = ModelConfig()
    
    # Create model
    model = create_model(
        num_entities=data_module.num_entities,
        num_rels=data_module.num_rels,
        config=model_config,
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description='Evaluate RAPID model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='RAPID',
                        help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Classification threshold (use checkpoint value if None)')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU ID (-1 for CPU)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results (JSON)')
    
    # Global model
    parser.add_argument('--use_global_model', action='store_true',
                        help='Use pretrained global model')
    parser.add_argument('--global_model_path', type=str, default=None,
                        help='Path to global model checkpoint')
    
    # Evaluation mode
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'teacher_forcing', 'autoregressive', 'per_timestep'],
                        help='Evaluation mode')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cpu')
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading dataset: {args.dataset}")
    data_module = PPIDataModule(
        data_path=Path(args.data_dir) / args.dataset,
        batch_size=args.batch_size,
        neg_ratio=1.0,  # Always use 1:1 for evaluation
    )
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model = load_model(Path(args.checkpoint), data_module, device)
    
    # Get threshold from checkpoint if not specified
    if args.threshold is None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        args.threshold = checkpoint.get('optimal_threshold', 0.5)
    
    print(f"Using threshold: {args.threshold:.3f}")
    
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
        else:
            print(f"\nWarning: Global model not found: {global_model_path}")
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        data_module=data_module,
        device=device,
        threshold=args.threshold,
        global_model=global_model,
    )
    
    # Run evaluation
    if args.mode == 'all':
        results = evaluator.full_evaluation()
    elif args.mode == 'teacher_forcing':
        metrics = evaluator.evaluate_teacher_forcing()
        results = {'teacher_forcing': metrics.to_dict()}
        print(f"\n{metrics}")
    elif args.mode == 'autoregressive':
        metrics = evaluator.evaluate_autoregressive()
        results = {'autoregressive': metrics.to_dict()}
        print(f"\n{metrics}")
    elif args.mode == 'per_timestep':
        per_ts = evaluator.evaluate_per_timestep()
        results = {'per_timestep': per_ts.to_dict()}
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
