"""
Full graph data loader for loading ALL interactions (inter + intra chain).

This module loads raw .interfacea files and creates complete interaction graphs
per timestep, preserving both interchain and intrachain interactions.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd
import torch

from src.data.preprocessing import discover_interfacea_folder, read_interfacea_files


@dataclass
class FullGraphData:
    """Complete interaction data including intra and interchain.

    Attributes:
        graphs: Per-timestep edge tensors, each (num_edges, 3) with [src, dst, rel_type]
        num_entities: Total number of unique entities
        entity_to_id: Mapping from entity string (e.g., 'A42') to integer ID
        id_to_entity: Reverse mapping from ID to entity string
        is_interchain: Per-timestep boolean masks indicating interchain edges
        all_timesteps: Sorted list of all timesteps
        interchain_pairs: Set of all interchain entity pairs (for prediction candidates)
    """

    graphs: Dict[int, torch.Tensor]
    num_entities: int
    entity_to_id: Dict[str, int]
    id_to_entity: Dict[int, str]
    is_interchain: Dict[int, torch.Tensor]
    all_timesteps: list
    interchain_pairs: Set[Tuple[int, int]]


def load_full_graphs(
    data_dir: Path,
    replica: str,
) -> FullGraphData:
    """Load ALL interactions from raw .interfacea files.

    Args:
        data_dir: Root data directory containing replica folders
        replica: Replica name (e.g., 'replica1')

    Returns:
        FullGraphData with complete interaction information
    """
    interface_folder = discover_interfacea_folder(data_dir, replica)
    df = read_interfacea_files(interface_folder)

    if df.empty:
        raise ValueError(f"No interactions found in {interface_folder}")

    # Create entity identifiers
    df["entity_a"] = df["chain_a"] + df["resid_a"].astype(str)
    df["entity_b"] = df["chain_b"] + df["resid_b"].astype(str)

    # Build unified entity namespace
    all_entities = pd.concat([df["entity_a"], df["entity_b"]]).unique()
    entity_to_id = {e: i for i, e in enumerate(sorted(all_entities))}
    id_to_entity = {i: e for e, i in entity_to_id.items()}

    # Map to integer IDs
    df["src"] = df["entity_a"].map(entity_to_id)
    df["dst"] = df["entity_b"].map(entity_to_id)

    # Encode relation types
    df["rel"] = pd.Categorical(df["itype"]).codes

    # Mark interchain vs intrachain
    df["is_inter"] = df["chain_a"] != df["chain_b"]

    # Build per-timestep graphs
    graphs: Dict[int, torch.Tensor] = {}
    is_interchain: Dict[int, torch.Tensor] = {}
    interchain_pairs: Set[Tuple[int, int]] = set()

    for t, grp in df.groupby("timestep"):
        src = grp["src"].values
        dst = grp["dst"].values
        rel = grp["rel"].values
        is_inter = grp["is_inter"].values

        graphs[t] = torch.tensor(np.stack([src, dst, rel], axis=1), dtype=torch.long)
        is_interchain[t] = torch.tensor(is_inter, dtype=torch.bool)

        # Collect interchain pairs
        for s, d, inter in zip(src, dst, is_inter):
            if inter:
                interchain_pairs.add(tuple(sorted([int(s), int(d)])))

    all_timesteps = sorted(graphs.keys())

    return FullGraphData(
        graphs=graphs,
        num_entities=len(entity_to_id),
        entity_to_id=entity_to_id,
        id_to_entity=id_to_entity,
        is_interchain=is_interchain,
        all_timesteps=all_timesteps,
        interchain_pairs=interchain_pairs,
    )


def get_train_timesteps(
    full_data: FullGraphData,
    train_ratio: float = 0.6,
) -> list:
    """Get timesteps belonging to training split.

    Args:
        full_data: Full graph data
        train_ratio: Fraction of timeline for training

    Returns:
        List of training timesteps
    """
    max_t = max(full_data.all_timesteps)
    train_cutoff = int(max_t * train_ratio)
    return [t for t in full_data.all_timesteps if t <= train_cutoff]
