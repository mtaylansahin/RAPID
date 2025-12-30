"""Evaluation script for RAPID - Protein Interaction Dynamics prediction."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json

import torch
import numpy as np
from tqdm import tqdm

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
    
    Uses all-pairs evaluation for unbiased metrics.
    Supports autoregressive inference with predicted history.
    
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
        
        # Storage for predictions (to save to file)
        self.predictions: List[Tuple[int, int, int, int, float, int]] = []
    
    def save_predictions(
        self,
        output_path: Path,
        include_negative: bool = False,
        include_scores: bool = False,
    ) -> None:
        """
        Save predicted interactions to a text file.
        
        Args:
            output_path: Path to save the predictions file
            include_negative: If True, include negative predictions
            include_scores: If True, add prediction probability as 4th column
        """
        if not self.predictions:
            print("Warning: No predictions to save. Run evaluation first.")
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for e1, rel, e2, t, score, pred in self.predictions:
                if pred == 1 or include_negative:
                    if include_scores:
                        f.write(f"{e1}\t{e2}\t{t}\t{score:.4f}\n")
                    else:
                        f.write(f"{e1}\t{e2}\t{t}\n")
        
        num_positive = sum(1 for p in self.predictions if p[5] == 1)
        num_total = len(self.predictions)
        print(f"\nPredictions saved to: {output_path}")
        print(f"  Positive predictions: {num_positive}")
        if include_negative:
            print(f"  Total predictions: {num_total}")
    
    @torch.no_grad()
    def evaluate(
        self,
        split: str = 'test',
        collect_predictions: bool = False,
    ) -> ClassificationMetrics:
        """
        Evaluate autoregressively on ALL pairs.
        
        Uses train + validation data as historical context for test evaluation.
        Evaluates on all N×N pairs for unbiased metrics.
        
        Args:
            split: 'valid' or 'test'
            collect_predictions: If True, store predictions for later saving
        """
        self.model.eval()
        
        if collect_predictions:
            self.predictions = []
        
        # Get dataset and timesteps
        if split == 'valid':
            dataset = self.data_module.val_dataset
            # For validation, use train context only
            context_graph_dict = self.data_module.graph_dict
            context_history = self.data_module.entity_history
            context_history_t = self.data_module.entity_history_t
        else:
            dataset = self.data_module.test_dataset
            # For test, use train + val context
            context_graph_dict, context_history, context_history_t = \
                self.data_module.get_train_val_context()
        
        # Extend global embeddings if needed
        if self.global_model is not None:
            print("Extending global embeddings...")
            self.global_model.extend_embeddings(context_graph_dict)
            self.global_emb = self.global_model.global_emb
        
        # Initialize model with historical context
        self.model.reset_inference_state()
        self.model.init_from_train_history(
            graph_dict=context_graph_dict,
            entity_history=context_history,
            entity_history_t=context_history_t,
            global_emb=self.global_emb,
            global_model=self.global_model,
        )
        
        # Collect all predictions
        all_logits = []
        all_labels = []
        
        print(f"\nEvaluating on {split} set (all pairs)...")
        
        # Get unique timesteps from dataset
        timesteps = sorted(dataset.unique_timesteps)
        
        for t in tqdm(timesteps, desc="Timesteps"):
            # Get ALL pairs with ground truth labels
            pairs, labels_np = self.data_module.get_all_pairs_for_timestep(t, split=split)
            
            # Process in batches
            batch_size = 128
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                batch_labels = labels_np[i:i + batch_size]
                
                entity1 = torch.LongTensor(batch_pairs[:, 0]).to(self.device)
                entity2 = torch.LongTensor(batch_pairs[:, 1]).to(self.device)
                labels = torch.FloatTensor(batch_labels).to(self.device)
                
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
                
                # Collect predictions if requested
                if collect_predictions:
                    probs_np = probs.cpu().numpy()
                    preds_np = preds.cpu().numpy()
                    e1_np = entity1.cpu().numpy()
                    e2_np = entity2.cpu().numpy()
                    
                    for j in range(len(e1_np)):
                        self.predictions.append((
                            int(e1_np[j]),
                            1,  # relation
                            int(e2_np[j]),
                            int(t),
                            float(probs_np[j]),
                            int(preds_np[j])
                        ))
        
        # Compute metrics
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        
        # Report class balance
        n_pos = all_labels.sum().item()
        n_neg = len(all_labels) - n_pos
        print(f"  Total pairs: {len(all_labels)} ({n_pos} positive, {n_neg} negative)")
        print(f"  Class ratio: 1:{n_neg/max(n_pos, 1):.1f}")
        
        metrics_computer = MetricsComputer(threshold=self.threshold)
        metrics_computer.update(all_logits, all_labels)
        
        return metrics_computer.compute()
    
    @torch.no_grad()
    def evaluate_per_timestep(self, split: str = 'test') -> PerTimestepMetrics:
        """
        Evaluate and return metrics per timestep.
        
        Useful for analyzing temporal degradation.
        """
        self.model.eval()
        
        # Get dataset and context
        if split == 'valid':
            dataset = self.data_module.val_dataset
            context_graph_dict = self.data_module.graph_dict
            context_history = self.data_module.entity_history
            context_history_t = self.data_module.entity_history_t
        else:
            dataset = self.data_module.test_dataset
            context_graph_dict, context_history, context_history_t = \
                self.data_module.get_train_val_context()
        
        # Initialize model
        self.model.reset_inference_state()
        self.model.init_from_train_history(
            graph_dict=context_graph_dict,
            entity_history=context_history,
            entity_history_t=context_history_t,
            global_emb=self.global_emb,
            global_model=self.global_model,
        )
        
        # Collect predictions with timesteps
        all_logits = []
        all_labels = []
        all_timesteps = []
        
        print(f"\nComputing per-timestep metrics ({split} set)...")
        
        timesteps = sorted(dataset.unique_timesteps)
        
        for t in tqdm(timesteps, desc="Timesteps"):
            pairs, labels_np = self.data_module.get_all_pairs_for_timestep(t, split=split)
            
            batch_size = 128
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                batch_labels = labels_np[i:i + batch_size]
                
                entity1 = torch.LongTensor(batch_pairs[:, 0]).to(self.device)
                entity2 = torch.LongTensor(batch_pairs[:, 1]).to(self.device)
                labels = torch.FloatTensor(batch_labels)
                
                probs, _ = self.model.predict_batch(
                    entity1_ids=entity1,
                    entity2_ids=entity2,
                    timestep=t,
                    threshold=self.threshold,
                    update_history=True,
                )
                
                probs_clamped = probs.clamp(1e-7, 1 - 1e-7)
                logits = torch.log(probs_clamped / (1 - probs_clamped))
                
                all_logits.append(logits.cpu())
                all_labels.append(labels)
                all_timesteps.append(torch.full((len(labels),), t))
        
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        all_timesteps = torch.cat(all_timesteps)
        
        return compute_per_timestep_metrics(
            all_logits, all_labels, all_timesteps,
            threshold=self.threshold,
        )
    
    def full_evaluation(self, split: str = 'test') -> Dict[str, Any]:
        """
        Run full evaluation with all analyses.
        
        Returns:
            Dictionary with all evaluation results
        """
        results = {}
        
        # Main evaluation
        metrics = self.evaluate(split=split)
        results['metrics'] = metrics.to_dict()
        print(f"\n{split.capitalize()} Results:")
        print(f"  {metrics}")
        
        # Per-timestep analysis
        per_ts_metrics = self.evaluate_per_timestep(split=split)
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
    parser.add_argument('--split', type=str, default='test',
                        choices=['valid', 'test'],
                        help='Data split to evaluate')
    
    # Prediction output options
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save predicted interactions to text file')
    parser.add_argument('--predictions_dir', type=str, default='./predictions',
                        help='Directory to save prediction files')
    parser.add_argument('--include_scores', action='store_true',
                        help='Include prediction scores in output file')
    parser.add_argument('--include_negative', action='store_true',
                        help='Include negative predictions in output file')
    
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
        neg_ratio=1.0,
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
    
    # Prepare predictions directory if saving predictions
    predictions_dir = Path(args.predictions_dir) / args.dataset if args.save_predictions else None
    
    # Run evaluation
    results = evaluator.full_evaluation(split=args.split)
    if args.save_predictions:
        evaluator.evaluate(split=args.split, collect_predictions=True)
        evaluator.save_predictions(
            predictions_dir / f'predictions_{args.split}.txt',
            include_negative=args.include_negative,
            include_scores=args.include_scores,
        )
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
