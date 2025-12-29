"""Negative sampling strategies for PPI dynamics."""

import numpy as np
import torch
from typing import Set, Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class NegativeSample:
    """A negative sample (non-interacting pair)."""
    entity1: int
    entity2: int
    timestep: int
    sample_type: str  # 'random' or 'temporal'


class NegativeSampler:
    """
    Mixed negative sampling strategy for PPI dynamics.
    
    Generates two types of negatives:
    1. Random negatives: Pairs that don't interact at a given timestep
    2. Temporal negatives: Pairs that interacted at t-1 but not at t
    
    Args:
        num_entities: Total number of entities (residues)
        temporal_ratio: Fraction of negatives that should be temporal (default 0.5)
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        num_entities: int,
        temporal_ratio: float = 0.5,
        seed: Optional[int] = None,
    ):
        self.num_entities = num_entities
        self.temporal_ratio = temporal_ratio
        self.rng = np.random.RandomState(seed)
        
        # Cache for all possible pairs (for random sampling)
        self._all_pairs: Optional[np.ndarray] = None
        
        # History tracking for temporal negatives
        self._prev_edges: Set[Tuple[int, int]] = set()
        self._current_edges: Set[Tuple[int, int]] = set()
        self._current_timestep: Optional[int] = None
    
    def _get_all_pairs(self) -> np.ndarray:
        """Generate all possible undirected pairs."""
        if self._all_pairs is None:
            pairs = []
            for i in range(self.num_entities):
                for j in range(i + 1, self.num_entities):
                    pairs.append((i, j))
            self._all_pairs = np.array(pairs)
        return self._all_pairs
    
    def _canonical_edge(self, e1: int, e2: int) -> Tuple[int, int]:
        """Return edge in canonical order (smaller id first)."""
        return (min(e1, e2), max(e1, e2))
    
    def update_timestep(self, timestep: int, positive_edges: np.ndarray) -> None:
        """
        Update the sampler with positive edges for a new timestep.
        
        Call this at the start of each timestep before sampling negatives.
        
        Args:
            timestep: Current timestep
            positive_edges: Array of shape (N, 2) with positive (e1, e2) pairs
        """
        if self._current_timestep != timestep:
            # Move current to previous
            self._prev_edges = self._current_edges.copy()
            self._current_edges = set()
            self._current_timestep = timestep
        
        # Add positive edges
        for e1, e2 in positive_edges:
            self._current_edges.add(self._canonical_edge(int(e1), int(e2)))
    
    def sample(
        self,
        n_samples: int,
        timestep: int,
        positive_edges: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample negative edges for a timestep.
        
        Args:
            n_samples: Number of negative samples to generate
            timestep: Current timestep
            positive_edges: If provided, updates internal state first
        
        Returns:
            Tuple of:
            - entities1: Array of first entities in pairs, shape (n_samples,)
            - entities2: Array of second entities in pairs, shape (n_samples,)
            - is_temporal: Boolean array indicating if sample is temporal
        """
        if positive_edges is not None:
            self.update_timestep(timestep, positive_edges)
        
        # Determine split
        n_temporal = int(n_samples * self.temporal_ratio)
        n_random = n_samples - n_temporal
        
        # Get temporal negatives (edges from t-1 that don't exist at t)
        temporal_candidates = self._prev_edges - self._current_edges
        temporal_negatives = self._sample_from_set(temporal_candidates, n_temporal)
        temporal_negatives_set = set(temporal_negatives)
        
        # If not enough temporal negatives, fill with random
        if len(temporal_negatives) < n_temporal:
            n_random += n_temporal - len(temporal_negatives)
        
        # Get random negatives (excluding temporal negatives to avoid duplicates)
        random_negatives = self._sample_random_negatives(n_random, exclude=temporal_negatives_set)
        
        # Combine
        all_negatives = temporal_negatives + random_negatives
        
        # Shuffle
        self.rng.shuffle(all_negatives)
        
        # Convert to arrays
        if len(all_negatives) == 0:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=bool),
            )
        
        entities1 = np.array([e[0] for e in all_negatives], dtype=np.int64)
        entities2 = np.array([e[1] for e in all_negatives], dtype=np.int64)
        is_temporal = np.array(
            [i < len(temporal_negatives) for i in range(len(all_negatives))],
            dtype=bool
        )
        
        return entities1, entities2, is_temporal
    
    def _sample_from_set(
        self,
        edge_set: Set[Tuple[int, int]],
        n_samples: int
    ) -> List[Tuple[int, int]]:
        """Sample edges from a set."""
        if len(edge_set) == 0:
            return []
        
        edges = list(edge_set)
        n_samples = min(n_samples, len(edges))
        indices = self.rng.choice(len(edges), size=n_samples, replace=False)
        return [edges[i] for i in indices]
    
    def _sample_random_negatives(
        self,
        n_samples: int,
        exclude: Optional[Set[Tuple[int, int]]] = None,
    ) -> List[Tuple[int, int]]:
        """Sample random non-edges.
        
        Args:
            n_samples: Number of samples to generate
            exclude: Optional set of edges to exclude (e.g., temporal negatives)
        """
        if n_samples <= 0:
            return []
        
        all_pairs = self._get_all_pairs()
        exclude = exclude or set()
        
        # Filter out current positive edges and excluded edges
        candidates = [
            (int(p[0]), int(p[1])) for p in all_pairs
            if (int(p[0]), int(p[1])) not in self._current_edges
            and (int(p[0]), int(p[1])) not in exclude
        ]
        
        if len(candidates) == 0:
            return []
        
        n_samples = min(n_samples, len(candidates))
        indices = self.rng.choice(len(candidates), size=n_samples, replace=False)
        return [candidates[i] for i in indices]
    
    def reset(self) -> None:
        """Reset temporal state (call between epochs or data splits)."""
        self._prev_edges = set()
        self._current_edges = set()
        self._current_timestep = None


class BatchNegativeSampler:
    """
    Negative sampler optimized for batch processing.
    
    Pre-computes negative samples for all timesteps in a dataset split.
    """
    
    def __init__(
        self,
        num_entities: int,
        neg_ratio: float = 1.0,
        temporal_ratio: float = 0.5,
        seed: Optional[int] = None,
    ):
        self.num_entities = num_entities
        self.neg_ratio = neg_ratio
        self.temporal_ratio = temporal_ratio
        self.seed = seed
        self.sampler = NegativeSampler(
            num_entities=num_entities,
            temporal_ratio=temporal_ratio,
            seed=seed,
        )
    
    def prepare_epoch(
        self,
        data: np.ndarray,
        timesteps: np.ndarray,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Prepare negative samples for all timesteps in an epoch.
        
        Args:
            data: Array of shape (N, 2) with (entity1, entity2) pairs
            timesteps: Array of shape (N,) with timestep for each sample
        
        Returns:
            Dict mapping timestep to (neg_entities1, neg_entities2) arrays
        """
        self.sampler.reset()
        negatives_by_timestep: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        
        unique_timesteps = np.unique(timesteps)
        
        for t in sorted(unique_timesteps):
            # Get positives for this timestep
            mask = timesteps == t
            pos_edges = data[mask]
            
            # Number of negatives
            n_neg = int(len(pos_edges) * self.neg_ratio)
            
            # Sample negatives
            neg_e1, neg_e2, _ = self.sampler.sample(
                n_samples=n_neg,
                timestep=int(t),
                positive_edges=pos_edges,
            )
            
            negatives_by_timestep[int(t)] = (neg_e1, neg_e2)
        
        return negatives_by_timestep
