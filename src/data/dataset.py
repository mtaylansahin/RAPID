"""Dataset classes for PPI dynamics."""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import pickle
import dgl

from src.data.sampler import NegativeSampler
from src.models.rgcn import build_undirected_graph


class PPIDataset(Dataset):
    """
    Dataset for PPI dynamics with on-the-fly negative sampling.
    
    Handles:
    - Loading quadruples (entity1, relation, entity2, timestep)
    - Converting to undirected format with canonical ordering
    - Generating negative samples per epoch
    - Providing history for each entity pair
    
    Args:
        data_path: Path to dataset directory
        split: 'train', 'valid', or 'test'
        neg_ratio: Ratio of negative to positive samples
        temporal_neg_ratio: Fraction of negatives that are temporal
        seed: Random seed
    """
    
    def __init__(
        self,
        data_path: Path,
        split: str = 'train',
        neg_ratio: float = 1.0,
        temporal_neg_ratio: float = 0.5,
        seed: Optional[int] = None,
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.neg_ratio = neg_ratio
        self.temporal_neg_ratio = temporal_neg_ratio
        
        # Load dataset statistics
        self.num_entities, self.num_rels = self._load_stats()
        
        # Load quadruples
        self.data, self.timesteps = self._load_data()
        
        # Convert to undirected (canonical ordering)
        self._canonicalize_edges()
        
        # Get unique timesteps
        self.unique_timesteps = sorted(set(self.timesteps))
        
        # Initialize negative sampler
        self.neg_sampler = NegativeSampler(
            num_entities=self.num_entities,
            temporal_ratio=temporal_neg_ratio,
            seed=seed,
        )
        
        # Pre-compute positives per timestep
        self.positives_by_timestep = self._group_by_timestep()
        
        # Negative samples (regenerated each epoch)
        self.negatives: List[Tuple[int, int, int]] = []  # (e1, e2, timestep)
        
        # Combined samples for iteration
        self.samples: List[Tuple[int, int, int, int]] = []  # (e1, e2, timestep, label)
    
    def _load_stats(self) -> Tuple[int, int]:
        """Load dataset statistics from stat.txt."""
        stat_file = self.data_path / 'stat.txt'
        with open(stat_file, 'r') as f:
            line = f.readline().strip()
            parts = line.split()
            return int(parts[0]), int(parts[1])
    
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load quadruples from split file."""
        if self.split == 'train':
            filename = 'train.txt'
        elif self.split == 'valid':
            filename = 'valid.txt'
        elif self.split == 'test':
            filename = 'test.txt'
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        filepath = self.data_path / filename
        
        quadruples = []
        timesteps = []
        
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    e1 = int(parts[0])
                    rel = int(parts[1])
                    e2 = int(parts[2])
                    t = int(parts[3])
                    quadruples.append([e1, rel, e2, t])
                    timesteps.append(t)
        
        return np.array(quadruples), np.array(timesteps)
    
    def _canonicalize_edges(self) -> None:
        """Convert edges to canonical (min, max) ordering."""
        for i in range(len(self.data)):
            e1, rel, e2, t = self.data[i]
            if e1 > e2:
                self.data[i] = [e2, rel, e1, t]
    
    def _group_by_timestep(self) -> Dict[int, Set[Tuple[int, int]]]:
        """Group positive edges by timestep."""
        positives = {}
        for e1, rel, e2, t in self.data:
            if t not in positives:
                positives[t] = set()
            positives[t].add((e1, e2))
        return positives
    
    def prepare_epoch(self) -> None:
        """
        Prepare samples for a new epoch.
        
        Regenerates negative samples and combines with positives.
        """
        self.neg_sampler.reset()
        self.negatives = []
        self.samples = []
        
        # Process each timestep in order
        for t in self.unique_timesteps:
            pos_edges = self.positives_by_timestep.get(t, set())
            if len(pos_edges) == 0:
                continue
            
            # Add positive samples
            for e1, e2 in pos_edges:
                self.samples.append((e1, e2, t, 1))
            
            # Generate negative samples
            pos_array = np.array(list(pos_edges))
            n_neg = int(len(pos_edges) * self.neg_ratio)
            
            neg_e1, neg_e2, _ = self.neg_sampler.sample(
                n_samples=n_neg,
                timestep=t,
                positive_edges=pos_array,
            )
            
            for i in range(len(neg_e1)):
                self.samples.append((int(neg_e1[i]), int(neg_e2[i]), t, 0))
        
        # Shuffle
        np.random.shuffle(self.samples)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample."""
        e1, e2, t, label = self.samples[idx]
        return {
            'entity1': e1,
            'entity2': e2,
            'timestep': t,
            'label': label,
        }


class PPIDataModule:
    """
    Data module managing train/val/test splits and dataloaders.
    
    Also handles:
    - Graph construction per timestep
    - History computation
    - Global embedding storage
    
    Args:
        data_path: Path to dataset
        batch_size: Batch size for dataloaders
        neg_ratio: Negative sampling ratio
        temporal_neg_ratio: Temporal negative fraction
        seed: Random seed
    """
    
    def __init__(
        self,
        data_path: Path,
        batch_size: int = 128,
        neg_ratio: float = 1.0,
        temporal_neg_ratio: float = 0.5,
        seed: Optional[int] = None,
    ):
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.neg_ratio = neg_ratio
        self.temporal_neg_ratio = temporal_neg_ratio
        self.seed = seed
        
        # Load datasets
        self.train_dataset = PPIDataset(
            data_path, 'train', neg_ratio, temporal_neg_ratio, seed
        )
        self.val_dataset = PPIDataset(
            data_path, 'valid', neg_ratio=0, seed=seed  # No negative sampling for val
        )
        self.test_dataset = PPIDataset(
            data_path, 'test', neg_ratio=0, seed=seed
        )
        
        self.num_entities = self.train_dataset.num_entities
        self.num_rels = self.train_dataset.num_rels
        
        # Build graphs for all timesteps
        self.graph_dict = self._build_graph_dict()
        
        # Compute entity histories
        self.entity_history, self.entity_history_t = self._build_histories()
    
    def _build_graph_dict(self) -> Dict[int, dgl.DGLGraph]:
        """Build DGL graphs for each timestep from training data."""
        graph_dict = {}
        
        # Group edges by timestep
        edges_by_t: Dict[int, List[Tuple[int, int, int]]] = {}
        for e1, rel, e2, t in self.train_dataset.data:
            if t not in edges_by_t:
                edges_by_t[t] = []
            edges_by_t[t].append((e1, e2, rel))
        
        # Build graph for each timestep
        for t, edges in edges_by_t.items():
            edge_array = torch.LongTensor([[e[0], e[1]] for e in edges])
            rel_array = torch.LongTensor([e[2] for e in edges])
            
            g = build_undirected_graph(
                edges=edge_array,
                rel_types=rel_array,
                num_nodes=self.num_entities,
                node_ids=torch.arange(self.num_entities),
            )
            
            # Add ids mapping for lookup
            g.ids = {i: i for i in range(self.num_entities)}
            
            graph_dict[t] = g
        
        return graph_dict
    
    def _build_histories(self) -> Tuple[List[List[Dict]], List[List[int]]]:
        """Build entity histories from training data."""
        entity_history: List[List[Dict]] = [[] for _ in range(self.num_entities)]
        entity_history_t: List[List[int]] = [[] for _ in range(self.num_entities)]
        
        # Group edges by (entity, timestep)
        edges_by_entity_t: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        
        for e1, rel, e2, t in self.train_dataset.data:
            # Add to both entities (undirected)
            key1 = (e1, t)
            key2 = (e2, t)
            
            if key1 not in edges_by_entity_t:
                edges_by_entity_t[key1] = []
            edges_by_entity_t[key1].append((e2, rel))
            
            if key2 not in edges_by_entity_t:
                edges_by_entity_t[key2] = []
            edges_by_entity_t[key2].append((e1, rel))
        
        # Build history per entity
        for (entity, t), neighbors in sorted(edges_by_entity_t.items(), key=lambda x: x[0][1]):
            entity_history[entity].append({
                'neighbors': [n[0] for n in neighbors],
                'rel_types': [n[1] for n in neighbors],
            })
            entity_history_t[entity].append(t)
        
        return entity_history, entity_history_t
    
    @property
    def train_data(self) -> np.ndarray:
        """Get training data as numpy array."""
        return self.train_dataset.data
    
    @property
    def train_times(self) -> List[int]:
        """Get sorted list of unique training timesteps."""
        return sorted(self.graph_dict.keys())
    
    def get_train_dataloader(self, shuffle: bool = True):
        """Get training dataloader."""
        # Prepare epoch (regenerate negatives)
        self.train_dataset.prepare_epoch()
        
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
        )
    
    def get_val_dataloader(self):
        """Get validation dataloader."""
        # For validation, we need negatives too
        self.val_dataset.neg_ratio = 1.0
        self.val_dataset.prepare_epoch()
        
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )
    
    def get_test_dataloader(self):
        """Get test dataloader."""
        self.test_dataset.neg_ratio = 1.0
        self.test_dataset.prepare_epoch()
        
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """Collate batch of samples."""
        entity1 = torch.LongTensor([b['entity1'] for b in batch])
        entity2 = torch.LongTensor([b['entity2'] for b in batch])
        timesteps = torch.LongTensor([b['timestep'] for b in batch])
        labels = torch.FloatTensor([b['label'] for b in batch])
        
        # Get histories for each entity
        entity1_history = []
        entity1_history_t = []
        entity2_history = []
        entity2_history_t = []
        
        for b in batch:
            e1, e2, t = b['entity1'], b['entity2'], b['timestep']
            
            # Get history up to (not including) current timestep
            e1_hist = [h for h, ht in zip(self.entity_history[e1], self.entity_history_t[e1]) if ht < t]
            e1_hist_t = [ht for ht in self.entity_history_t[e1] if ht < t]
            
            e2_hist = [h for h, ht in zip(self.entity_history[e2], self.entity_history_t[e2]) if ht < t]
            e2_hist_t = [ht for ht in self.entity_history_t[e2] if ht < t]
            
            entity1_history.append(e1_hist)
            entity1_history_t.append(e1_hist_t)
            entity2_history.append(e2_hist)
            entity2_history_t.append(e2_hist_t)
        
        return {
            'entity1': entity1,
            'entity2': entity2,
            'timesteps': timesteps,
            'labels': labels,
            'entity1_history': entity1_history,
            'entity1_history_t': entity1_history_t,
            'entity2_history': entity2_history,
            'entity2_history_t': entity2_history_t,
        }
