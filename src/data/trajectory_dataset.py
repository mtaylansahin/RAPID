"""
Trajectory-level dataset for PPI dynamics.

Provides full edge trajectories with neighbor context for trajectory-level prediction.
Each sample is an edge with its history and target trajectory.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import dgl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.models.rgcn import build_undirected_graph


class TrajectoryDataset(Dataset):
    """
    Dataset providing edge trajectories for trajectory-level prediction.

    Each sample contains:
    - Target edge (e1, e2)
    - Binary history: edge states for history timesteps
    - Binary target: edge states for future timesteps (to predict)
    - Neighbor edges with their histories
    - Timestep indices for RGCN feature lookup

    Args:
        edges: List of (e1, e2) tuples - all edges to include
        edge_states: Dict mapping (e1, e2) -> array of states per timestep
        history_timesteps: List of timesteps for history
        target_timesteps: List of timesteps to predict
        max_neighbors: Maximum number of neighbor edges per target
        neighbor_edges: Dict mapping entity -> list of connected edges
        n_hops: Number of hops for neighbor expansion
    """

    def __init__(
        self,
        edges: List[Tuple[int, int]],
        edge_states: Dict[Tuple[int, int], np.ndarray],
        history_timesteps: List[int],
        target_timesteps: List[int],
        max_neighbors: int = 50,
        neighbor_edges: Optional[Dict[int, Set[Tuple[int, int]]]] = None,
        n_hops: int = 1,
    ):
        self.edges = edges
        self.edge_states = edge_states
        self.history_timesteps = history_timesteps
        self.target_timesteps = target_timesteps
        self.max_neighbors = max_neighbors
        self.neighbor_edges = neighbor_edges or {}
        self.n_hops = n_hops

        self.hist_len = len(history_timesteps)
        self.traj_len = len(target_timesteps)

    def __len__(self) -> int:
        return len(self.edges)

    def _get_neighbors(self, e1: int, e2: int) -> List[Tuple[int, int]]:
        """Get neighbor edges for a target edge (e1, e2)."""
        neighbors = set()

        # Direct neighbors: edges sharing a node with target
        for entity in [e1, e2]:
            if entity in self.neighbor_edges:
                neighbors.update(self.neighbor_edges[entity])

        # Remove self
        neighbors.discard((e1, e2))
        neighbors.discard((e2, e1))

        # Multi-hop expansion
        if self.n_hops > 1:
            for _ in range(self.n_hops - 1):
                new_neighbors = set()
                for ne1, ne2 in neighbors:
                    for entity in [ne1, ne2]:
                        if entity in self.neighbor_edges:
                            new_neighbors.update(self.neighbor_edges[entity])
                neighbors.update(new_neighbors)
                neighbors.discard((e1, e2))
                neighbors.discard((e2, e1))

        return list(neighbors)[: self.max_neighbors]

    def __getitem__(self, idx: int) -> Dict:
        e1, e2 = self.edges[idx]

        # Get edge states
        states = self.edge_states.get((e1, e2), np.zeros(self.hist_len + self.traj_len))

        history = states[: self.hist_len].astype(np.float32)
        target = states[self.hist_len : self.hist_len + self.traj_len].astype(
            np.float32
        )

        # Get neighbors
        neighbor_list = self._get_neighbors(e1, e2)
        n_neighbors = len(neighbor_list)

        # Pad neighbors
        neighbor_e1 = np.zeros(self.max_neighbors, dtype=np.int64)
        neighbor_e2 = np.zeros(self.max_neighbors, dtype=np.int64)
        neighbor_history = np.zeros(
            (self.max_neighbors, self.hist_len), dtype=np.float32
        )
        neighbor_mask = np.ones(self.max_neighbors, dtype=bool)  # True = padding

        for i, (ne1, ne2) in enumerate(neighbor_list):
            neighbor_e1[i] = ne1
            neighbor_e2[i] = ne2
            n_states = self.edge_states.get(
                (ne1, ne2), self.edge_states.get((ne2, ne1), np.zeros(self.hist_len))
            )
            neighbor_history[i] = n_states[: self.hist_len]
            neighbor_mask[i] = False

        return {
            "entity1": e1,
            "entity2": e2,
            "history": history,
            "target": target,
            "history_timesteps": np.array(self.history_timesteps, dtype=np.int64),
            "neighbor_entity1": neighbor_e1,
            "neighbor_entity2": neighbor_e2,
            "neighbor_history": neighbor_history,
            "neighbor_mask": neighbor_mask,
        }


class TrajectoryDataModule:
    """
    Data module for trajectory-level prediction.

    Handles:
    - Loading data and building edge trajectories
    - Train/val/test splits based on timesteps
    - Graph construction for RGCN
    - Neighbor edge indexing

    Args:
        data_path: Path to dataset directory
        history_ratio: Fraction of timesteps to use as history
        val_ratio: Fraction of prediction timesteps for validation
        max_neighbors: Max neighbor edges per target
        batch_size: Batch size
        seed: Random seed
        n_hops: Number of hops for neighbor expansion
    """

    def __init__(
        self,
        data_path: Path,
        history_ratio: float = 0.5,
        val_ratio: float = 0.2,
        max_neighbors: int = 50,
        batch_size: int = 64,
        seed: int = 42,
        n_hops: int = 1,
    ):
        self.data_path = Path(data_path)
        self.history_ratio = history_ratio
        self.val_ratio = val_ratio
        self.max_neighbors = max_neighbors
        self.batch_size = batch_size
        self.seed = seed
        self.n_hops = n_hops

        np.random.seed(seed)

        # Load data
        self.num_entities, self.num_rels = self._load_stats()
        self.all_data = self._load_all_data()

        # Build timestep splits
        self._build_splits()

        # Build edge trajectories
        self._build_edge_states()

        # Build neighbor index
        self._build_neighbor_index()

        # Build graphs for RGCN
        self.graph_dict = self._build_graph_dict()

        # Create datasets
        self._create_datasets()

    def _load_stats(self) -> Tuple[int, int]:
        """Load dataset statistics."""
        stat_file = self.data_path / "stat.txt"
        with open(stat_file, "r") as f:
            parts = f.readline().strip().split()
            return int(parts[0]), int(parts[1])

    def _load_all_data(self) -> np.ndarray:
        """Load all quadruples from train/valid/test."""
        all_quads = []
        for split in ["train.txt", "valid.txt", "test.txt"]:
            filepath = self.data_path / split
            if filepath.exists():
                with open(filepath, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            e1, rel, e2, t = (
                                int(parts[0]),
                                int(parts[1]),
                                int(parts[2]),
                                int(parts[3]),
                            )
                            # Canonicalize edge
                            if e1 > e2:
                                e1, e2 = e2, e1
                            all_quads.append([e1, rel, e2, t])
        return np.array(all_quads)

    def _build_splits(self) -> None:
        """Build train/val/test splits based on timesteps."""
        all_timesteps = sorted(set(self.all_data[:, 3]))
        self.total_timesteps = len(all_timesteps)

        # History timesteps (first history_ratio)
        n_history = int(self.total_timesteps * self.history_ratio)
        self.history_timesteps = all_timesteps[:n_history]

        # Remaining timesteps split into val and test
        remaining = all_timesteps[n_history:]
        n_val = int(len(remaining) * self.val_ratio)

        self.val_timesteps = remaining[:n_val]
        self.test_timesteps = remaining[n_val:]

        self.traj_len = len(remaining)
        self.hist_len = len(self.history_timesteps)

        print(f"Timestep splits:")
        print(f"  History: {len(self.history_timesteps)} timesteps")
        print(f"  Val: {len(self.val_timesteps)} timesteps")
        print(f"  Test: {len(self.test_timesteps)} timesteps")

    def _build_edge_states(self) -> None:
        """Build binary state array for each edge across all timesteps."""
        all_timesteps = self.history_timesteps + list(self.val_timesteps) + list(
            self.test_timesteps
        )
        self.timestep_to_idx = {t: i for i, t in enumerate(all_timesteps)}

        # Find all unique edges
        self.all_edges = set()
        for e1, rel, e2, t in self.all_data:
            self.all_edges.add((int(e1), int(e2)))

        self.all_edges = sorted(self.all_edges)
        print(f"Total unique edges: {len(self.all_edges)}")

        # Build state matrix
        n_timesteps = len(all_timesteps)
        self.edge_states = {}

        for e1, e2 in self.all_edges:
            self.edge_states[(e1, e2)] = np.zeros(n_timesteps, dtype=np.float32)

        # Fill in positive states
        for e1, rel, e2, t in self.all_data:
            e1, e2 = int(e1), int(e2)
            if e1 > e2:
                e1, e2 = e2, e1
            if t in self.timestep_to_idx:
                idx = self.timestep_to_idx[t]
                self.edge_states[(e1, e2)][idx] = 1.0

    def _build_neighbor_index(self) -> None:
        """Build index of edges per entity."""
        self.neighbor_edges: Dict[int, Set[Tuple[int, int]]] = {}

        for e1, e2 in self.all_edges:
            if e1 not in self.neighbor_edges:
                self.neighbor_edges[e1] = set()
            if e2 not in self.neighbor_edges:
                self.neighbor_edges[e2] = set()
            self.neighbor_edges[e1].add((e1, e2))
            self.neighbor_edges[e2].add((e1, e2))

    def _build_graph_dict(self) -> Dict[int, dgl.DGLGraph]:
        """Build DGL graphs for each history timestep."""
        graph_dict = {}

        # Group edges by timestep
        edges_by_t: Dict[int, List[Tuple[int, int, int]]] = {}
        for e1, rel, e2, t in self.all_data:
            t = int(t)
            if t in self.history_timesteps:
                if t not in edges_by_t:
                    edges_by_t[t] = []
                edges_by_t[t].append((int(e1), int(e2), int(rel)))

        for t, edges in edges_by_t.items():
            edge_array = torch.LongTensor([[e[0], e[1]] for e in edges])
            rel_array = torch.LongTensor([e[2] for e in edges])

            g = build_undirected_graph(
                edges=edge_array,
                rel_types=rel_array,
                num_nodes=self.num_entities,
                node_ids=torch.arange(self.num_entities),
            )
            g.ids = {i: i for i in range(self.num_entities)}
            graph_dict[t] = g

        return graph_dict

    def _create_datasets(self) -> None:
        """Create train/val/test datasets."""
        # All edges are used for training (predict full trajectory)
        # For evaluation, we split the target into val and test portions

        # Training: predict val + test timesteps from history
        target_timesteps = list(self.val_timesteps) + list(self.test_timesteps)

        self.train_dataset = TrajectoryDataset(
            edges=self.all_edges,
            edge_states=self.edge_states,
            history_timesteps=list(self.history_timesteps),
            target_timesteps=target_timesteps,
            max_neighbors=self.max_neighbors,
            neighbor_edges=self.neighbor_edges,
            n_hops=self.n_hops,
        )

        # Val: same edges, but we'll only evaluate on val timesteps
        self.val_dataset = self.train_dataset

        # Test: same edges, evaluate on test timesteps
        self.test_dataset = self.train_dataset

        print(f"Dataset: {len(self.train_dataset)} edges")
        print(f"  History length: {self.train_dataset.hist_len}")
        print(f"  Target length: {self.train_dataset.traj_len}")

    def get_train_dataloader(self, shuffle: bool = True) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=self._collate_fn,
        )

    def get_val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self._collate_fn,
        )

    def get_test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """Collate batch of samples."""
        return {
            "entity1": torch.LongTensor([b["entity1"] for b in batch]),
            "entity2": torch.LongTensor([b["entity2"] for b in batch]),
            "history": torch.FloatTensor(np.stack([b["history"] for b in batch])),
            "target": torch.FloatTensor(np.stack([b["target"] for b in batch])),
            "history_timesteps": torch.LongTensor(batch[0]["history_timesteps"]),
            "neighbor_entity1": torch.LongTensor(
                np.stack([b["neighbor_entity1"] for b in batch])
            ),
            "neighbor_entity2": torch.LongTensor(
                np.stack([b["neighbor_entity2"] for b in batch])
            ),
            "neighbor_history": torch.FloatTensor(
                np.stack([b["neighbor_history"] for b in batch])
            ),
            "neighbor_mask": torch.BoolTensor(
                np.stack([b["neighbor_mask"] for b in batch])
            ),
        }

    @property
    def val_timestep_indices(self) -> Tuple[int, int]:
        """Return (start, end) indices of val timesteps in target."""
        return (0, len(self.val_timesteps))

    @property
    def test_timestep_indices(self) -> Tuple[int, int]:
        """Return (start, end) indices of test timesteps in target."""
        start = len(self.val_timesteps)
        return (start, start + len(self.test_timesteps))
