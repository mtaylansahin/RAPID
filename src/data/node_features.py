"""Node-level feature computation for RAPID.

Computes two types of features for each residue:
1. Physicochemical properties - amino acid characteristics (static)
2. Intrachain-derived features - structural position from contact topology
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

logger = logging.getLogger(__name__)


# Amino acid physicochemical properties
# Sources:
# - Hydrophobicity: Kyte-Doolittle scale, normalized to [-1, 1]
# - Charge: Based on side chain at pH 7 (-1: D/E, +1: K/R, 0.5: H, 0: others)
# - Size: Molecular weight normalized to [0, 1] (Gly=0.19, Trp=0.70)
# - Polarity: Binary (0: nonpolar, 1: polar)
# - Aromaticity: Binary (0: no, 1: F/W/Y/H)
AMINO_ACID_PROPERTIES: Dict[str, list] = {
    # [hydrophobicity, charge, size, polarity, aromaticity]
    "ALA": [0.40, 0.0, 0.28, 0, 0],
    "ARG": [-1.00, 1.0, 0.60, 1, 0],
    "ASN": [-0.78, 0.0, 0.41, 1, 0],
    "ASP": [-0.78, -1.0, 0.41, 1, 0],
    "CYS": [0.56, 0.0, 0.37, 0, 0],
    "GLN": [-0.78, 0.0, 0.49, 1, 0],
    "GLU": [-0.78, -1.0, 0.49, 1, 0],
    "GLY": [-0.09, 0.0, 0.19, 0, 0],
    "HIS": [-0.71, 0.5, 0.53, 1, 1],
    "ILE": [1.00, 0.0, 0.44, 0, 0],
    "LEU": [0.84, 0.0, 0.44, 0, 0],
    "LYS": [-0.87, 1.0, 0.49, 1, 0],
    "MET": [0.42, 0.0, 0.51, 0, 0],
    "PHE": [0.62, 0.0, 0.56, 0, 1],
    "PRO": [-0.36, 0.0, 0.36, 0, 0],
    "SER": [-0.18, 0.0, 0.32, 1, 0],
    "THR": [-0.16, 0.0, 0.38, 1, 0],
    "TRP": [-0.20, 0.0, 0.70, 0, 1],
    "TYR": [-0.29, 0.0, 0.63, 1, 1],
    "VAL": [0.93, 0.0, 0.38, 0, 0],
}

# Default properties for unknown residues (use ALA as default)
DEFAULT_PROPERTIES = AMINO_ACID_PROPERTIES["ALA"]


@dataclass
class NodeFeatureConfig:
    """Configuration for node-level features."""

    enabled: bool = True
    use_physicochemical: bool = True
    use_intrachain: bool = True

    @property
    def num_physicochemical(self) -> int:
        """Number of physicochemical features."""
        return 5 if self.use_physicochemical else 0

    @property
    def num_intrachain(self) -> int:
        """Number of intrachain-derived features."""
        return 3 if self.use_intrachain else 0

    @property
    def total_features(self) -> int:
        """Total number of node features."""
        if not self.enabled:
            return 0
        return self.num_physicochemical + self.num_intrachain


def compute_physicochemical_features(
    num_entities: int,
    entity_to_id: Dict[str, int],
    resname_map: Dict[str, str],
) -> torch.Tensor:
    """Compute physicochemical features for all entities.

    Args:
        num_entities: Total number of entities
        entity_to_id: Mapping from entity string to integer ID
        resname_map: Mapping from entity string to 3-letter residue name

    Returns:
        Tensor of shape (num_entities, 5) with physicochemical features
    """
    features = torch.zeros(num_entities, 5)

    for entity, idx in entity_to_id.items():
        resname = resname_map.get(entity, "ALA")
        props = AMINO_ACID_PROPERTIES.get(resname, DEFAULT_PROPERTIES)
        features[idx] = torch.tensor(props, dtype=torch.float32)

    return features


def compute_intrachain_features(
    num_entities: int,
    entity_to_id: Dict[str, int],
    intrachain_df: pd.DataFrame,
    interface_entities: Set[str],
) -> torch.Tensor:
    """Compute intrachain-derived structural features.

    Features computed:
    1. mean_dist_to_interface: Average graph distance to interface residues
    2. intrachain_degree: Number of unique intrachain neighbors
    3. interface_neighbor_frac: Fraction of neighbors that are interface residues

    Args:
        num_entities: Total number of entities
        entity_to_id: Mapping from entity string to integer ID
        intrachain_df: DataFrame with intrachain interactions
        interface_entities: Set of entity strings that participate in interchain

    Returns:
        Tensor of shape (num_entities, 3) with intrachain features
    """
    features = torch.zeros(num_entities, 3)

    if intrachain_df.empty:
        logger.warning("No intrachain interactions provided, using zero features")
        return features

    # Build entity strings for intrachain interactions
    intrachain_df = intrachain_df.copy()
    intrachain_df["entity_a"] = intrachain_df["chain_a"] + intrachain_df[
        "resid_a"
    ].astype(str)
    intrachain_df["entity_b"] = intrachain_df["chain_b"] + intrachain_df[
        "resid_b"
    ].astype(str)

    # Build neighbor sets
    neighbors: Dict[str, Set[str]] = {entity: set() for entity in entity_to_id}
    for _, row in intrachain_df.iterrows():
        e_a, e_b = row["entity_a"], row["entity_b"]
        if e_a in entity_to_id and e_b in entity_to_id:
            neighbors[e_a].add(e_b)
            neighbors[e_b].add(e_a)

    # Build graph distance matrix
    entities = sorted(entity_to_id.keys())
    entity_idx_local = {e: i for i, e in enumerate(entities)}
    n = len(entities)

    adj = np.zeros((n, n))
    for entity, nbrs in neighbors.items():
        if entity not in entity_idx_local:
            continue
        i = entity_idx_local[entity]
        for nbr in nbrs:
            if nbr in entity_idx_local:
                j = entity_idx_local[nbr]
                adj[i, j] = 1

    logger.info("Computing shortest paths in intrachain graph...")
    dist_matrix = shortest_path(csr_matrix(adj), directed=False, unweighted=True)

    # Get interface indices in local graph
    interface_local_indices = [
        entity_idx_local[e] for e in interface_entities if e in entity_idx_local
    ]

    # Compute features for each entity
    for entity, global_idx in entity_to_id.items():
        if entity not in entity_idx_local:
            continue

        local_idx = entity_idx_local[entity]

        # Feature 1: Mean distance to interface residues
        if interface_local_indices:
            dists = [
                dist_matrix[local_idx, j]
                for j in interface_local_indices
                if not np.isinf(dist_matrix[local_idx, j])
            ]
            features[global_idx, 0] = np.mean(dists) if dists else 10.0
        else:
            features[global_idx, 0] = 10.0

        # Feature 2: Intrachain degree
        features[global_idx, 1] = float(len(neighbors[entity]))

        # Feature 3: Interface neighbor fraction
        if neighbors[entity]:
            n_interface = sum(
                1 for nbr in neighbors[entity] if nbr in interface_entities
            )
            features[global_idx, 2] = n_interface / len(neighbors[entity])

    # Normalize features to [0, 1]
    for i in range(3):
        col = features[:, i]
        col_min, col_max = col.min(), col.max()
        if col_max > col_min:
            features[:, i] = (col - col_min) / (col_max - col_min)

    return features


def compute_node_features(
    config: NodeFeatureConfig,
    data_dir: Path,
    train_cutoff: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Compute all node features from preprocessed data.

    Args:
        config: Node feature configuration
        data_dir: Directory containing preprocessed data (metadata.json, labels.txt)
        train_cutoff: Maximum timestep to use for intrachain features (leak prevention)

    Returns:
        Tensor of shape (num_entities, num_features) or None if features disabled
    """
    if not config.enabled:
        return None

    # Load metadata
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found in {data_dir}. "
            "Please re-run preprocessing to generate metadata."
        )

    with open(metadata_path) as f:
        metadata = json.load(f)

    entity_to_id = metadata["entity_to_id"]
    resname_map = metadata["resname_map"]
    num_entities = len(entity_to_id)

    logger.info(f"Computing node features for {num_entities} entities")

    features_list = []

    # Physicochemical features
    if config.use_physicochemical:
        logger.info("Computing physicochemical features...")
        phys_features = compute_physicochemical_features(
            num_entities, entity_to_id, resname_map
        )
        features_list.append(phys_features)
        logger.info(f"  Physicochemical: {phys_features.shape[1]} features")

    # Intrachain features
    if config.use_intrachain:
        logger.info("Computing intrachain-derived features...")

        # Load full interaction data
        labels_path = data_dir / "labels.txt"
        if not labels_path.exists():
            raise FileNotFoundError(f"labels.txt not found in {data_dir}")

        df = pd.read_csv(labels_path, sep=" ")

        # Filter to intrachain only
        intrachain_df = df[df["chain_a"] == df["chain_b"]].copy()

        # Apply train cutoff to prevent data leakage
        if train_cutoff is not None:
            intrachain_df = intrachain_df[intrachain_df["timestep"] <= train_cutoff]
            logger.info(f"  Using intrachain from timesteps <= {train_cutoff}")

        # Identify interface entities (from interchain interactions in training)
        interchain_df = df[df["chain_a"] != df["chain_b"]]
        if train_cutoff is not None:
            interchain_df = interchain_df[interchain_df["timestep"] <= train_cutoff]

        interchain_df = interchain_df.copy()
        interchain_df["entity_a"] = interchain_df["chain_a"] + interchain_df[
            "resid_a"
        ].astype(str)
        interchain_df["entity_b"] = interchain_df["chain_b"] + interchain_df[
            "resid_b"
        ].astype(str)
        interface_entities = set(interchain_df["entity_a"]) | set(
            interchain_df["entity_b"]
        )

        intra_features = compute_intrachain_features(
            num_entities, entity_to_id, intrachain_df, interface_entities
        )
        features_list.append(intra_features)
        logger.info(f"  Intrachain: {intra_features.shape[1]} features")

    if not features_list:
        return None

    all_features = torch.cat(features_list, dim=1)
    logger.info(f"Total node features: {all_features.shape}")

    return all_features


def save_node_features(features: torch.Tensor, output_path: Path) -> None:
    """Save computed node features to disk."""
    torch.save(features, output_path)
    logger.info(f"Saved node features to {output_path}")


def load_node_features(path: Path) -> torch.Tensor:
    """Load pre-computed node features from disk."""
    return torch.load(path)
