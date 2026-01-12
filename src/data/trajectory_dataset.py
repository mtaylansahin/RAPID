"""
Trajectory-level dataset for PPI dynamics.

Provides per-edge trajectories with neighbor context for trajectory-level prediction.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TrajectoryDataModule:
    """
    Data module for trajectory-level prediction.

    Instead of (edge, timestep) samples, provides full trajectories for each edge
    with neighbor context for cross-attention.

    Args:
        data_path: Path to dataset directory
        test_ratio: Fraction of timesteps for test (also used as trajectory length)
        val_ratio: Fraction of remaining timesteps for validation
        max_neighbors: Maximum neighbor edges per target edge
        batch_size: Training batch size
        seed: Random seed
        use_full_neighbors: If True, include intrachain edges as neighbors
    """

    def __init__(
        self,
        data_path: Path,
        test_ratio: float = 0.25,
        val_ratio: float = 0.1,
        max_neighbors: int = 50,
        batch_size: int = 64,
        seed: Optional[int] = None,
        use_full_neighbors: bool = False,
        n_hops: int = 1,
    ):
        self.data_path = Path(data_path)
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.max_neighbors = max_neighbors
        self.batch_size = batch_size
        self.seed = seed
        self.use_full_neighbors = use_full_neighbors
        self.n_hops = n_hops

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Load data
        self.num_entities, self.num_rels = self._load_stats()

        # Load interchain data for targets
        self.all_data = self._load_all_data()

        # Load full data (inter+intra) for neighbors if enabled
        if use_full_neighbors:
            self.full_data = self._load_full_data()
        else:
            self.full_data = self.all_data

        # Get all timesteps and compute splits
        self.all_timesteps = sorted(set(self.all_data[:, 3]))
        self.n_timesteps = len(self.all_timesteps)

        # Compute split points
        n_test = max(1, int(self.n_timesteps * test_ratio))
        n_remaining = self.n_timesteps - n_test
        n_val = max(1, int(n_remaining * val_ratio))
        n_train = n_remaining - n_val

        self.train_timesteps = self.all_timesteps[:n_train]
        self.val_timesteps = self.all_timesteps[n_train : n_train + n_val]
        self.test_timesteps = self.all_timesteps[n_train + n_val :]

        self.train_end_t = self.train_timesteps[-1] if self.train_timesteps else 0
        self.val_end_t = (
            self.val_timesteps[-1] if self.val_timesteps else self.train_end_t
        )

        self.traj_len = len(self.test_timesteps)

        # Build per-timestep edge sets (for targets - interchain only)
        self.edges_at_t = self._build_edges_at_t()

        # Build known pairs (for targets - interchain only)
        self.known_pairs = self._build_known_pairs()

        # Build ALL edge pairs (inter+intra for neighbors)
        self.all_edge_pairs = self._build_all_edge_pairs()

        # Build neighbor graph (using full data)
        self.entity_neighbors = self._build_entity_neighbors()

        # Build ALL edge trajectories (inter+intra for neighbor context)
        self.all_edge_trajectories = self._build_all_edge_trajectories()

        # Build edge trajectories (interchain for targets)
        self.edge_trajectories = self._build_edge_trajectories()

        print(f"TrajectoryDataModule initialized:")
        print(f"  Total timesteps: {self.n_timesteps}")
        print(
            f"  Train: {len(self.train_timesteps)}, Val: {len(self.val_timesteps)}, Test: {len(self.test_timesteps)}"
        )
        print(f"  Interchain pairs (targets): {len(self.known_pairs)}")
        if use_full_neighbors:
            print(f"  Total pairs (neighbors): {len(self.all_edge_pairs)}")
        if n_hops > 1:
            print(f"  Neighbor hops: {n_hops}")
        print(f"  Trajectory length: {self.traj_len}")

    def _load_stats(self) -> Tuple[int, int]:
        """Load dataset statistics."""
        stat_file = self.data_path / "stat.txt"
        with open(stat_file, "r") as f:
            line = f.readline().strip()
            parts = line.split()
            return int(parts[0]), int(parts[1])

    def _load_all_data(self) -> np.ndarray:
        """Load all data from train/valid/test files."""
        all_quads = []

        for filename in ["train.txt", "valid.txt", "test.txt"]:
            filepath = self.data_path / filename
            if filepath.exists():
                with open(filepath, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            e1 = int(parts[0])
                            rel = int(parts[1])
                            e2 = int(parts[2])
                            t = int(parts[3])
                            # Canonicalize edge
                            if e1 > e2:
                                e1, e2 = e2, e1
                            all_quads.append([e1, rel, e2, t])

        return np.array(all_quads)

    def _load_full_data(self) -> np.ndarray:
        """Load full data (inter+intra) from labels.txt for neighbor context."""
        all_quads = []

        # Try to load metadata for entity mapping
        metadata_path = self.data_path / "metadata.json"
        entity_to_id = None
        if metadata_path.exists():
            import json

            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                entity_to_id = metadata.get("entity_to_id", {})

        labels_path = self.data_path / "labels.txt"
        if labels_path.exists() and entity_to_id:
            import pandas as pd

            try:
                df = pd.read_csv(labels_path, sep=" ")
                # Build entity strings and map to IDs
                df["entity_a"] = df["chain_a"].astype(str) + df["resid_a"].astype(str)
                df["entity_b"] = df["chain_b"].astype(str) + df["resid_b"].astype(str)

                for _, row in df.iterrows():
                    e1_str = row["entity_a"]
                    e2_str = row["entity_b"]
                    t = int(row["timestep"])

                    if e1_str in entity_to_id and e2_str in entity_to_id:
                        e1 = entity_to_id[e1_str]
                        e2 = entity_to_id[e2_str]
                        # Canonicalize
                        if e1 > e2:
                            e1, e2 = e2, e1
                        # Use relation=1 for intrachain (0 is typically interchain)
                        rel = 0 if row["chain_a"] != row["chain_b"] else 1
                        all_quads.append([e1, rel, e2, t])

                print(f"  Loaded {len(all_quads)} full interactions from labels.txt")
            except Exception as e:
                print(f"  Warning: Could not load labels.txt: {e}")
                return self.all_data
        else:
            print(
                "  Warning: labels.txt or metadata.json not found, using interchain only"
            )
            return self.all_data

        return np.array(all_quads) if all_quads else self.all_data

    def _build_edges_at_t(self) -> Dict[int, Set[Tuple[int, int]]]:
        """Build set of active edges at each timestep (interchain only for targets)."""
        edges_at_t: Dict[int, Set[Tuple[int, int]]] = {}
        for e1, rel, e2, t in self.all_data:
            t = int(t)
            if t not in edges_at_t:
                edges_at_t[t] = set()
            edges_at_t[t].add((int(e1), int(e2)))
        return edges_at_t

    def _build_full_edges_at_t(self) -> Dict[int, Set[Tuple[int, int]]]:
        """Build set of ALL active edges at each timestep (inter+intra)."""
        edges_at_t: Dict[int, Set[Tuple[int, int]]] = {}
        for e1, rel, e2, t in self.full_data:
            t = int(t)
            if t not in edges_at_t:
                edges_at_t[t] = set()
            edges_at_t[t].add((int(e1), int(e2)))
        return edges_at_t

    def _build_known_pairs(self) -> List[Tuple[int, int]]:
        """Build sorted list of all known interchain edge pairs (for targets)."""
        pairs = set()
        for e1, rel, e2, t in self.all_data:
            pairs.add((int(e1), int(e2)))
        return sorted(pairs)

    def _build_all_edge_pairs(self) -> List[Tuple[int, int]]:
        """Build sorted list of ALL edge pairs including intrachain (for neighbors)."""
        pairs = set()
        for e1, rel, e2, t in self.full_data:
            pairs.add((int(e1), int(e2)))
        return sorted(pairs)

    def _build_entity_neighbors(self) -> Dict[int, Set[int]]:
        """Build neighbor entity set for each entity (using full data)."""
        neighbors: Dict[int, Set[int]] = {i: set() for i in range(self.num_entities)}
        # Use full_data to include intrachain neighbors
        for e1, rel, e2, t in self.full_data:
            neighbors[int(e1)].add(int(e2))
            neighbors[int(e2)].add(int(e1))
        return neighbors

    def _build_edge_trajectories(self) -> Dict[Tuple[int, int], np.ndarray]:
        """Build full trajectory for each interchain edge pair (targets)."""
        trajectories = {}
        for e1, e2 in self.known_pairs:
            traj = np.zeros(self.n_timesteps, dtype=np.float32)
            for i, t in enumerate(self.all_timesteps):
                if t in self.edges_at_t and (e1, e2) in self.edges_at_t[t]:
                    traj[i] = 1.0
            trajectories[(e1, e2)] = traj
        return trajectories

    def _build_all_edge_trajectories(self) -> Dict[Tuple[int, int], np.ndarray]:
        """Build full trajectory for ALL edge pairs (inter+intra for neighbors)."""
        # Build full edges_at_t for neighbor trajectories
        full_edges_at_t = self._build_full_edges_at_t()

        trajectories = {}
        for e1, e2 in self.all_edge_pairs:
            traj = np.zeros(self.n_timesteps, dtype=np.float32)
            for i, t in enumerate(self.all_timesteps):
                if t in full_edges_at_t and (e1, e2) in full_edges_at_t[t]:
                    traj[i] = 1.0
            trajectories[(e1, e2)] = traj
        return trajectories

    def get_edge_neighbors(
        self, e1: int, e2: int, n_hops: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """Get neighbor edges for a target edge (A,B).

        Args:
            e1, e2: Target edge endpoints
            n_hops: Number of hops (default: self.n_hops)

        If use_full_neighbors is True, includes intrachain neighbors.
        """
        if n_hops is None:
            n_hops = self.n_hops
        # Use all_edge_trajectories which includes intrachain
        edge_traj_lookup = (
            self.all_edge_trajectories
            if self.use_full_neighbors
            else self.edge_trajectories
        )

        target_edge = (min(e1, e2), max(e1, e2))

        # Start with entities from target edge
        frontier_entities = {e1, e2}
        all_neighbor_edges = set()

        for hop in range(n_hops):
            next_frontier = set()

            for entity in frontier_entities:
                for neighbor_entity in self.entity_neighbors.get(entity, []):
                    pair = (min(entity, neighbor_entity), max(entity, neighbor_entity))

                    # Skip target edge itself
                    if pair == target_edge:
                        continue

                    # Only include if we have trajectory data for it
                    if pair in edge_traj_lookup:
                        all_neighbor_edges.add(pair)
                        # Add both endpoints to next frontier for next hop
                        next_frontier.add(entity)
                        next_frontier.add(neighbor_entity)

            frontier_entities = next_frontier

        neighbors = list(all_neighbor_edges)

        # Shuffle and limit if needed
        if len(neighbors) > self.max_neighbors:
            np.random.shuffle(neighbors)
            neighbors = neighbors[: self.max_neighbors]

        return neighbors

    def get_train_dataset(self) -> "TrajectoryDataset":
        """Get training dataset."""
        # For training, predict val trajectory from train history
        hist_timesteps = self.train_timesteps
        pred_timesteps = self.val_timesteps

        # Use all_edge_trajectories for neighbor lookup if full neighbors enabled
        neighbor_edge_traj = (
            self.all_edge_trajectories
            if self.use_full_neighbors
            else self.edge_trajectories
        )

        return TrajectoryDataset(
            known_pairs=self.known_pairs,
            edge_trajectories=self.edge_trajectories,
            neighbor_edge_trajectories=neighbor_edge_traj,
            all_timesteps=self.all_timesteps,
            hist_timesteps=hist_timesteps,
            pred_timesteps=pred_timesteps,
            get_neighbors_fn=self.get_edge_neighbors,
            max_neighbors=self.max_neighbors,
        )

    def get_test_dataset(self) -> "TrajectoryDataset":
        """Get test dataset (predict test trajectory from train+val history)."""
        hist_timesteps = self.train_timesteps + self.val_timesteps
        pred_timesteps = self.test_timesteps

        # Use all_edge_trajectories for neighbor lookup if full neighbors enabled
        neighbor_edge_traj = (
            self.all_edge_trajectories
            if self.use_full_neighbors
            else self.edge_trajectories
        )

        return TrajectoryDataset(
            known_pairs=self.known_pairs,
            edge_trajectories=self.edge_trajectories,
            neighbor_edge_trajectories=neighbor_edge_traj,
            all_timesteps=self.all_timesteps,
            hist_timesteps=hist_timesteps,
            pred_timesteps=pred_timesteps,
            get_neighbors_fn=self.get_edge_neighbors,
            max_neighbors=self.max_neighbors,
        )

    def get_train_dataloader(self, shuffle: bool = True) -> DataLoader:
        """Get training dataloader."""
        dataset = self.get_train_dataset()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=trajectory_collate_fn,
            num_workers=0,
            pin_memory=True,
        )

    def get_test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        dataset = self.get_test_dataset()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=trajectory_collate_fn,
            num_workers=0,
            pin_memory=True,
        )


class TrajectoryDataset(Dataset):
    """
    Dataset yielding per-edge trajectories with neighbor context.
    """

    def __init__(
        self,
        known_pairs: List[Tuple[int, int]],
        edge_trajectories: Dict[Tuple[int, int], np.ndarray],
        neighbor_edge_trajectories: Dict[Tuple[int, int], np.ndarray],
        all_timesteps: List[int],
        hist_timesteps: List[int],
        pred_timesteps: List[int],
        get_neighbors_fn,
        max_neighbors: int = 50,
    ):
        self.known_pairs = known_pairs
        self.edge_trajectories = edge_trajectories
        self.neighbor_edge_trajectories = neighbor_edge_trajectories
        self.all_timesteps = all_timesteps
        self.hist_timesteps = hist_timesteps
        self.pred_timesteps = pred_timesteps
        self.get_neighbors_fn = get_neighbors_fn
        self.max_neighbors = max_neighbors

        # Compute indices for history and prediction timesteps
        self.t_to_idx = {t: i for i, t in enumerate(all_timesteps)}
        self.hist_indices = [self.t_to_idx[t] for t in hist_timesteps]
        self.pred_indices = [self.t_to_idx[t] for t in pred_timesteps]

    def __len__(self) -> int:
        return len(self.known_pairs)

    def __getitem__(self, idx: int) -> Dict:
        e1, e2 = self.known_pairs[idx]

        # Get trajectories (interchain for target)
        full_traj = self.edge_trajectories[(e1, e2)]
        history = full_traj[self.hist_indices]
        target = full_traj[self.pred_indices]

        # Get neighbor edges
        neighbor_pairs = self.get_neighbors_fn(e1, e2)

        neighbor_e1 = []
        neighbor_e2 = []
        neighbor_hist = []

        for n1, n2 in neighbor_pairs:
            neighbor_e1.append(n1)
            neighbor_e2.append(n2)
            # Use neighbor_edge_trajectories which may include intrachain
            full_n_traj = self.neighbor_edge_trajectories[(n1, n2)]
            neighbor_hist.append(full_n_traj[self.hist_indices])

        return {
            "entity1": e1,
            "entity2": e2,
            "history": history,
            "target": target,
            "neighbor_entity1": neighbor_e1,
            "neighbor_entity2": neighbor_e2,
            "neighbor_history": neighbor_hist,
        }


def trajectory_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for trajectory batches."""
    batch_size = len(batch)

    # Stack simple tensors
    entity1 = torch.LongTensor([b["entity1"] for b in batch])
    entity2 = torch.LongTensor([b["entity2"] for b in batch])
    history = torch.FloatTensor(np.stack([b["history"] for b in batch]))
    target = torch.FloatTensor(np.stack([b["target"] for b in batch]))

    # Handle variable-length neighbors with padding
    max_neighbors = max(len(b["neighbor_entity1"]) for b in batch) if batch else 0
    max_neighbors = max(max_neighbors, 1)  # At least 1 to avoid empty tensors

    hist_len = history.size(1)

    neighbor_e1 = torch.zeros(batch_size, max_neighbors, dtype=torch.long)
    neighbor_e2 = torch.zeros(batch_size, max_neighbors, dtype=torch.long)
    neighbor_hist = torch.zeros(batch_size, max_neighbors, hist_len)
    neighbor_mask = torch.ones(
        batch_size, max_neighbors, dtype=torch.bool
    )  # True = pad

    for i, b in enumerate(batch):
        n_neighbors = len(b["neighbor_entity1"])
        if n_neighbors > 0:
            neighbor_e1[i, :n_neighbors] = torch.LongTensor(b["neighbor_entity1"])
            neighbor_e2[i, :n_neighbors] = torch.LongTensor(b["neighbor_entity2"])
            neighbor_hist[i, :n_neighbors] = torch.FloatTensor(
                np.stack(b["neighbor_history"])
            )
            neighbor_mask[i, :n_neighbors] = False

    return {
        "entity1": entity1,
        "entity2": entity2,
        "history": history,
        "target": target,
        "neighbor_entity1": neighbor_e1,
        "neighbor_entity2": neighbor_e2,
        "neighbor_history": neighbor_hist,
        "neighbor_mask": neighbor_mask,
    }
