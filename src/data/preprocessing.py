"""Data preprocessing to convert raw MD simulation output to RAPID format.

This module reads .interfacea files produced by molecular dynamics simulation
analysis tools and converts them into the train/valid/test splits expected by
RAPID's dataset loader.

Output files:
    train.txt, valid.txt, test.txt: Quadruples (entity1, relation, entity2, timestep)
    stat.txt: Dataset statistics (num_entities, num_relations, num_timesteps)
    labels.txt: Raw interaction labels for debugging/analysis
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""

    data_directory: Path
    output_directory: Path
    replica: str
    test_ratio: float = 0.2

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.data_directory = Path(self.data_directory)
        self.output_directory = Path(self.output_directory)

        if not 0.05 <= self.test_ratio <= 0.45:
            raise ValueError(
                f"test_ratio must be in [0.05, 0.45], got {self.test_ratio}"
            )

    @property
    def valid_ratio(self) -> float:
        """Validation ratio equals test ratio."""
        return self.test_ratio

    @property
    def train_ratio(self) -> float:
        """Training ratio is the remainder after valid and test."""
        return 1.0 - 2 * self.test_ratio


@dataclass
class PreprocessingResult:
    """Result of preprocessing operation."""

    success: bool
    num_entities: int = 0
    num_relations: int = 0
    num_timesteps: int = 0
    train_samples: int = 0
    valid_samples: int = 0
    test_samples: int = 0
    output_directory: Path = Path(".")
    error_message: str = ""


def discover_interfacea_folder(data_directory: Path, replica: str) -> Path:
    """Locate the folder containing .interfacea files for a replica.

    Expected structure:
        <data_directory>/<replica>/rep<NUM>-interfacea/

    Args:
        data_directory: Root data directory
        replica: Replica name (e.g., "replica1")

    Returns:
        Path to the interfacea folder

    Raises:
        FileNotFoundError: If the expected directory structure is not found
    """
    replica_num = replica.replace("replica", "")
    replica_dir = data_directory / replica
    if not replica_dir.exists():
        raise FileNotFoundError(f"Replica directory not found: {replica_dir}")

    interface_folder = replica_dir / f"rep{replica_num}-interfacea"
    if not interface_folder.exists():
        raise FileNotFoundError(f"Interfacea folder not found: {interface_folder}")

    return interface_folder


def read_interfacea_files(folder: Path) -> pd.DataFrame:
    """Parse all .interfacea files into a unified DataFrame.

    Args:
        folder: Directory containing .interfacea files

    Returns:
        DataFrame with all interactions and computed timesteps
    """
    all_frames: list[pd.DataFrame] = []
    timesteps: list[int] = []

    files = sorted(folder.iterdir())
    for filepath in files:
        if not filepath.suffix == ".interfacea":
            continue

        matches = re.findall(r"[0-9]+", filepath.name)
        if not matches:
            continue
        timestep = int(matches[0]) - 1

        try:
            df = pd.read_table(
                filepath,
                header=0,
                names=[
                    "itype",
                    "chain_a",
                    "chain_b",
                    "resname_a",
                    "resname_b",
                    "resid_a",
                    "resid_b",
                    "atom_a",
                    "atom_b",
                ],
                sep=r"\s+",
            )
        except Exception as e:
            logger.warning(f"Failed to read {filepath}: {e}")
            continue

        all_frames.append(df)
        timesteps.extend([timestep] * len(df))

    if not all_frames:
        return pd.DataFrame(
            columns=[
                "itype",
                "chain_a",
                "chain_b",
                "resname_a",
                "resname_b",
                "resid_a",
                "resid_b",
                "atom_a",
                "atom_b",
                "timestep",
            ]
        )

    df = pd.concat(all_frames, ignore_index=True)
    df["timestep"] = np.array(timesteps, dtype=int)

    return df


def filter_interchain(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only interchain interactions (different chains).

    Args:
        df: DataFrame with chain_a and chain_b columns

    Returns:
        Filtered DataFrame containing only interchain interactions
    """
    return df[df["chain_a"] != df["chain_b"]].copy()


def encode_entities(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Convert residue identifiers to integer entity codes using unified namespace.

    Uses a single ID space for all entities, so the same residue gets the same ID
    regardless of whether it appears in entity_a or entity_b.

    Args:
        df: DataFrame with interaction data

    Returns:
        Tuple of:
            - DataFrame with encoded columns (subject, relation, object, time)
            - Metadata dict with entity_to_id mapping and resname info
    """
    df = df.copy()
    df["relation"] = pd.Categorical(df["itype"]).codes
    df["entity_a"] = df["chain_a"] + df["resid_a"].astype(str)
    df["entity_b"] = df["chain_b"] + df["resid_b"].astype(str)

    # Build unified entity namespace from ALL entities in both columns
    all_entities = pd.concat([df["entity_a"], df["entity_b"]]).unique()
    entity_to_id = {entity: idx for idx, entity in enumerate(sorted(all_entities))}

    df["subject"] = df["entity_a"].map(entity_to_id)
    df["object"] = df["entity_b"].map(entity_to_id)

    # Build resname mapping (entity_str -> 3-letter residue name)
    resname_map = {}
    for _, row in df.drop_duplicates(subset=["entity_a"])[
        ["entity_a", "resname_a"]
    ].iterrows():
        resname_map[row["entity_a"]] = row["resname_a"]
    for _, row in df.drop_duplicates(subset=["entity_b"])[
        ["entity_b", "resname_b"]
    ].iterrows():
        if row["entity_b"] not in resname_map:
            resname_map[row["entity_b"]] = row["resname_b"]

    dataset = pd.DataFrame(
        {
            "subject": df["subject"],
            "relation": df["relation"],
            "object": df["object"],
            "time": df["timestep"],
        }
    )

    dataset = dataset.sort_values("time").drop_duplicates().reset_index(drop=True)

    metadata = {
        "entity_to_id": entity_to_id,
        "id_to_entity": {v: k for k, v in entity_to_id.items()},
        "resname_map": resname_map,
    }

    return dataset, metadata


def split_by_time(
    dataset: pd.DataFrame, train_ratio: float, valid_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically into train/valid/test.

    Args:
        dataset: DataFrame with columns (subject, relation, object, time)
        train_ratio: Fraction of timeline for training
        valid_ratio: Fraction of timeline for validation (test gets the same)

    Returns:
        Tuple of (train_df, valid_df, test_df)
    """
    max_time = dataset["time"].max()

    train_cutoff = int(max_time * train_ratio)
    valid_cutoff = int(max_time * (train_ratio + valid_ratio))

    train = dataset[dataset["time"] <= train_cutoff]
    valid = dataset[
        (dataset["time"] > train_cutoff) & (dataset["time"] <= valid_cutoff)
    ]
    test = dataset[dataset["time"] > valid_cutoff]

    return train, valid, test


def compute_statistics(dataset: pd.DataFrame, metadata: dict) -> Tuple[int, int, int]:
    """Compute dataset statistics.

    Args:
        dataset: Full dataset
        metadata: Entity metadata from encode_entities

    Returns:
        Tuple of (num_entities, num_relations, num_timesteps)
    """
    num_entities = len(metadata["entity_to_id"])
    num_relations = len(set(dataset["relation"]))
    num_timesteps = len(set(dataset["time"]))

    return num_entities, num_relations, num_timesteps


def run_preprocessing(config: PreprocessingConfig) -> PreprocessingResult:
    """Execute the full preprocessing pipeline.

    Args:
        config: Preprocessing configuration

    Returns:
        PreprocessingResult with operation details and statistics
    """
    try:
        config.output_directory.mkdir(parents=True, exist_ok=True)
        interface_folder = discover_interfacea_folder(
            config.data_directory, config.replica
        )

        logger.info(f"Reading interfacea files from {interface_folder}")
        df = read_interfacea_files(interface_folder)
        if df.empty:
            return PreprocessingResult(
                success=False,
                error_message="No interactions found",
                output_directory=config.output_directory,
            )

        # Encode all entities with unified namespace
        dataset, metadata = encode_entities(df)

        # Filter to interchain only for training/evaluation
        interchain_df = filter_interchain(df)
        interchain_dataset, _ = encode_entities(interchain_df)

        # Use interchain dataset for train/valid/test splits
        train, valid, test = split_by_time(
            interchain_dataset, config.train_ratio, config.valid_ratio
        )

        num_entities, num_relations, num_timesteps = compute_statistics(
            dataset, metadata
        )

        logger.info(f"Writing output files to {config.output_directory}")

        stat_path = config.output_directory / "stat.txt"
        np.savetxt(stat_path, [[num_entities, num_relations, num_timesteps]], fmt="%d")

        np.savetxt(config.output_directory / "train.txt", train.values, fmt="%d")
        np.savetxt(config.output_directory / "valid.txt", valid.values, fmt="%d")
        np.savetxt(config.output_directory / "test.txt", test.values, fmt="%d")

        # Save raw labels
        labels_path = config.output_directory / "labels.txt"
        df.to_csv(labels_path, sep=" ", index=False, header=True)

        # Save metadata for node features
        import json

        metadata_path = config.output_directory / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Preprocessing complete")

        return PreprocessingResult(
            success=True,
            num_entities=num_entities,
            num_relations=num_relations,
            num_timesteps=num_timesteps,
            train_samples=len(train),
            valid_samples=len(valid),
            test_samples=len(test),
            output_directory=config.output_directory,
        )

    except FileNotFoundError as e:
        return PreprocessingResult(
            success=False,
            error_message=str(e),
            output_directory=config.output_directory,
        )
    except Exception as e:
        logger.exception("Preprocessing failed")
        return PreprocessingResult(
            success=False,
            error_message=str(e),
            output_directory=config.output_directory,
        )
