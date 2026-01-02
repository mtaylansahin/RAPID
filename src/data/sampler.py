"""Negative sampling strategies for PPI dynamics."""

from typing import List, Optional, Set, Tuple

import numpy as np


class NegativeSampler:
    """
    Hard/easy negative sampling strategy for PPI dynamics.

    Generates two types of negatives:
    1. Hard negatives: Pairs that have interacted before but are OFF now
    2. Easy negatives: Pairs that have never interacted

    Args:
        num_entities: Total number of entities (residues)
        hard_ratio: Fraction of negatives that should be hard (default 0.5)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        num_entities: int,
        hard_ratio: float = 0.5,
        seed: Optional[int] = None,
    ):
        self.num_entities = num_entities
        self.hard_ratio = hard_ratio
        self.rng = np.random.RandomState(seed)

        # Cache for all possible pairs
        self._all_pairs: Optional[Set[Tuple[int, int]]] = None

        # Track all edges ever seen (for hard negatives)
        self._ever_positive: Set[Tuple[int, int]] = set()

        # Current timestep's positive edges
        self._current_edges: Set[Tuple[int, int]] = set()
        self._current_timestep: Optional[int] = None

    def _get_all_pairs(self) -> Set[Tuple[int, int]]:
        """Generate all possible undirected pairs."""
        if self._all_pairs is None:
            pairs = set()
            for i in range(self.num_entities):
                for j in range(i + 1, self.num_entities):
                    pairs.add((i, j))
            self._all_pairs = pairs
        return self._all_pairs

    def _canonical_edge(self, e1: int, e2: int) -> Tuple[int, int]:
        """Return edge in canonical order (smaller id first)."""
        return (min(e1, e2), max(e1, e2))

    def update_timestep(self, timestep: int, positive_edges: np.ndarray) -> None:
        """
        Update the sampler with positive edges for a new timestep.

        Args:
            timestep: Current timestep
            positive_edges: Array of shape (N, 2) with positive (e1, e2) pairs
        """
        if self._current_timestep != timestep:
            self._current_edges = set()
            self._current_timestep = timestep

        # Add positive edges to current and ever-seen sets
        for e1, e2 in positive_edges:
            edge = self._canonical_edge(int(e1), int(e2))
            self._current_edges.add(edge)
            self._ever_positive.add(edge)

    def sample(
        self,
        n_samples: int,
        timestep: int,
        positive_edges: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        """
        if positive_edges is not None:
            self.update_timestep(timestep, positive_edges)

        # Hard negatives: edges that have interacted before but are OFF now
        hard_pool = self._ever_positive - self._current_edges

        # Easy negatives: edges that have never interacted
        all_pairs = self._get_all_pairs()
        easy_pool = all_pairs - self._ever_positive

        # Determine split (target 50% hard, but cap at available)
        n_hard = min(int(n_samples * self.hard_ratio), len(hard_pool))
        n_easy = n_samples - n_hard

        # Sample from each pool
        hard_samples = self._sample_from_set(hard_pool, n_hard)
        easy_samples = self._sample_from_set(easy_pool, n_easy)

        # Combine and shuffle
        all_negatives = hard_samples + easy_samples
        self.rng.shuffle(all_negatives)

        # Convert to arrays
        if len(all_negatives) == 0:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
            )

        entities1 = np.array([e[0] for e in all_negatives], dtype=np.int64)
        entities2 = np.array([e[1] for e in all_negatives], dtype=np.int64)

        return entities1, entities2

    def _sample_from_set(
        self, edge_set: Set[Tuple[int, int]], n_samples: int
    ) -> List[Tuple[int, int]]:
        """Sample edges from a set."""
        if len(edge_set) == 0 or n_samples <= 0:
            return []

        edges = list(edge_set)
        n_samples = min(n_samples, len(edges))
        indices = self.rng.choice(len(edges), size=n_samples, replace=False)
        return [edges[i] for i in indices]

    def reset(self) -> None:
        """Reset state. Called at the start of each epoch for reproducible negative sampling."""
        self._ever_positive = set()
        self._current_edges = set()
        self._current_timestep = None
