#!/usr/bin/env python3
"""
Dynamics Signal Analysis for RAPID

This script analyzes the training data to quantify:
1. Transition rates (OFF→ON and ON→OFF)
2. Persistence patterns (how long pairs stay ON/OFF)
3. Predictability of transitions from available features
4. Information content comparison: t-1 state vs. other signals

Run with: python scripts/analyze_dynamics_signal.py --data_path data/processed/1JPS
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy import stats


def load_quadruples(filepath: Path) -> np.ndarray:
    """Load quadruples from file."""
    data = []
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
                # Canonicalize
                if e1 > e2:
                    e1, e2 = e2, e1
                data.append([e1, rel, e2, t])
    return np.array(data)


def analyze_transitions(data: np.ndarray) -> Dict:
    """Analyze transition patterns in the data."""
    # Group edges by timestep
    edges_by_t: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
    for e1, rel, e2, t in data:
        edges_by_t[t].add((e1, e2))

    timesteps = sorted(edges_by_t.keys())

    # Track transitions
    transitions = {
        "off_to_on": 0,  # Was OFF at t-1, ON at t
        "on_to_off": 0,  # Was ON at t-1, OFF at t
        "on_to_on": 0,  # Was ON at t-1, ON at t
        "off_to_off": 0,  # Was OFF at t-1, OFF at t (for known pairs)
    }

    # Get all pairs that ever interact
    all_pairs = set()
    for edges in edges_by_t.values():
        all_pairs.update(edges)

    print(f"Total unique pairs: {len(all_pairs)}")
    print(f"Total timesteps: {len(timesteps)}")
    print(f"Timestep range: {min(timesteps)} - {max(timesteps)}")

    # Count transitions
    for i in range(1, len(timesteps)):
        t_prev = timesteps[i - 1]
        t_curr = timesteps[i]

        edges_prev = edges_by_t[t_prev]
        edges_curr = edges_by_t[t_curr]

        for pair in all_pairs:
            was_on = pair in edges_prev
            is_on = pair in edges_curr

            if was_on and is_on:
                transitions["on_to_on"] += 1
            elif was_on and not is_on:
                transitions["on_to_off"] += 1
            elif not was_on and is_on:
                transitions["off_to_on"] += 1
            else:
                transitions["off_to_off"] += 1

    return {
        "transitions": transitions,
        "timesteps": timesteps,
        "all_pairs": all_pairs,
        "edges_by_t": edges_by_t,
    }


def analyze_persistence_patterns(
    edges_by_t: Dict[int, Set], all_pairs: Set, timesteps: List[int]
) -> Dict:
    """Analyze how long pairs stay in ON/OFF states."""

    # Track consecutive runs
    on_run_lengths = []
    off_run_lengths = []

    for pair in all_pairs:
        # Build state sequence for this pair
        states = [pair in edges_by_t[t] for t in timesteps]

        # Find consecutive runs
        run_start = 0
        current_state = states[0]

        for i in range(1, len(states)):
            if states[i] != current_state:
                run_length = i - run_start
                if current_state:
                    on_run_lengths.append(run_length)
                else:
                    off_run_lengths.append(run_length)
                run_start = i
                current_state = states[i]

        # Don't count final run (it's censored)

    return {
        "on_run_lengths": np.array(on_run_lengths),
        "off_run_lengths": np.array(off_run_lengths),
    }


def analyze_neighbor_signal(
    edges_by_t: Dict[int, Set], all_pairs: Set, timesteps: List[int]
) -> Dict:
    """
    Analyze whether neighbor features predict transitions.

    Hypothesis: Pairs sharing more active neighbors are more likely to turn ON.
    """
    # For each pair at each timestep, compute:
    # - Number of shared active neighbors (both entities have active edge to same neighbor)
    # - Whether pair transitions OFF→ON next timestep

    shared_neighbor_forming = []  # Shared neighbors for pairs that form
    shared_neighbor_not_forming = []  # Shared neighbors for pairs that don't form

    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        t_next = timesteps[i + 1]

        edges_t = edges_by_t[t]
        edges_next = edges_by_t[t_next]

        # Build neighbor sets for each entity
        neighbors: Dict[int, Set[int]] = defaultdict(set)
        for e1, e2 in edges_t:
            neighbors[e1].add(e2)
            neighbors[e2].add(e1)

        # For pairs that were OFF at t
        for pair in all_pairs:
            if pair in edges_t:
                continue  # Skip pairs that are already ON

            e1, e2 = pair
            shared = len(neighbors[e1] & neighbors[e2])

            if pair in edges_next:
                shared_neighbor_forming.append(shared)
            else:
                shared_neighbor_not_forming.append(shared)

    return {
        "forming": np.array(shared_neighbor_forming),
        "not_forming": np.array(shared_neighbor_not_forming),
    }


def analyze_activity_signal(
    edges_by_t: Dict[int, Set],
    all_pairs: Set,
    timesteps: List[int],
) -> Dict:
    """
    Analyze whether entity activity level predicts pair activation.

    Hypothesis: Pairs with highly active entities are more likely to interact.
    """
    activity_forming = []
    activity_not_forming = []

    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        t_next = timesteps[i + 1]

        edges_t = edges_by_t[t]
        edges_next = edges_by_t[t_next]

        # Compute activity (degree) for each entity
        degree: Dict[int, int] = defaultdict(int)
        for e1, e2 in edges_t:
            degree[e1] += 1
            degree[e2] += 1

        # For pairs that were OFF at t
        for pair in all_pairs:
            if pair in edges_t:
                continue

            e1, e2 = pair
            combined_activity = degree[e1] + degree[e2]

            if pair in edges_next:
                activity_forming.append(combined_activity)
            else:
                activity_not_forming.append(combined_activity)

    return {
        "forming": np.array(activity_forming),
        "not_forming": np.array(activity_not_forming),
    }


def compute_baseline_accuracy(
    edges_by_t: Dict[int, Set], all_pairs: Set, timesteps: List[int]
) -> Dict:
    """
    Compute accuracy of simple baselines:
    - Persistence: predict same as t-1
    - Always ON: predict 1 for all pairs
    - Always OFF: predict 0 for all pairs
    - Random: random 0/1
    """
    correct_persistence = 0
    correct_always_on = 0
    correct_always_off = 0
    total = 0

    positive_rate = []

    for i in range(1, len(timesteps)):
        t_prev = timesteps[i - 1]
        t_curr = timesteps[i]

        edges_prev = edges_by_t[t_prev]
        edges_curr = edges_by_t[t_curr]

        positive_at_t = len(edges_curr)
        positive_rate.append(positive_at_t / len(all_pairs))

        for pair in all_pairs:
            total += 1
            was_on = pair in edges_prev
            is_on = pair in edges_curr

            # Persistence baseline
            if was_on == is_on:
                correct_persistence += 1

            # Always ON
            if is_on:
                correct_always_on += 1

            # Always OFF
            if not is_on:
                correct_always_off += 1

    return {
        "persistence_accuracy": correct_persistence / total,
        "always_on_accuracy": correct_always_on / total,
        "always_off_accuracy": correct_always_off / total,
        "random_expected": 0.5,
        "mean_positive_rate": np.mean(positive_rate),
        "std_positive_rate": np.std(positive_rate),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dynamics signal in RAPID data"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to processed data directory"
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)

    print("=" * 60)
    print("RAPID Dynamics Signal Analysis")
    print("=" * 60)
    print(f"\nData path: {data_path}\n")

    # Load all data
    train_data = load_quadruples(data_path / "train.txt")
    valid_data = load_quadruples(data_path / "valid.txt")
    test_data = load_quadruples(data_path / "test.txt")

    all_data = np.concatenate([train_data, valid_data, test_data], axis=0)

    print(f"Train samples: {len(train_data)}")
    print(f"Valid samples: {len(valid_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Total samples: {len(all_data)}")

    # Analyze transitions
    print("\n" + "=" * 60)
    print("TRANSITION ANALYSIS")
    print("=" * 60)

    result = analyze_transitions(all_data)
    trans = result["transitions"]

    total_trans = sum(trans.values())
    print(f"\nTransition counts (for known pairs across all timesteps):")
    print(
        f"  ON→ON (persistence):  {trans['on_to_on']:>8} ({100 * trans['on_to_on'] / total_trans:.2f}%)"
    )
    print(
        f"  OFF→OFF (persistence): {trans['off_to_off']:>8} ({100 * trans['off_to_off'] / total_trans:.2f}%)"
    )
    print(
        f"  ON→OFF (breaking):    {trans['on_to_off']:>8} ({100 * trans['on_to_off'] / total_trans:.2f}%)"
    )
    print(
        f"  OFF→ON (forming):     {trans['off_to_on']:>8} ({100 * trans['off_to_on'] / total_trans:.2f}%)"
    )

    persistence_rate = (trans["on_to_on"] + trans["off_to_off"]) / total_trans
    transition_rate = (trans["on_to_off"] + trans["off_to_on"]) / total_trans
    print(f"\n  Total persistence rate: {100 * persistence_rate:.2f}%")
    print(f"  Total transition rate:  {100 * transition_rate:.2f}%")

    # Conditional transition probabilities
    on_total = trans["on_to_on"] + trans["on_to_off"]
    off_total = trans["off_to_off"] + trans["off_to_on"]

    if on_total > 0:
        print(f"\n  P(stay ON | was ON):    {100 * trans['on_to_on'] / on_total:.2f}%")
        print(f"  P(break | was ON):      {100 * trans['on_to_off'] / on_total:.2f}%")
    if off_total > 0:
        print(f"  P(stay OFF | was OFF):  {100 * trans['off_to_off'] / off_total:.2f}%")
        print(f"  P(form | was OFF):      {100 * trans['off_to_on'] / off_total:.2f}%")

    # Persistence patterns
    print("\n" + "=" * 60)
    print("PERSISTENCE PATTERNS")
    print("=" * 60)

    persist = analyze_persistence_patterns(
        result["edges_by_t"], result["all_pairs"], result["timesteps"]
    )

    on_runs = persist["on_run_lengths"]
    off_runs = persist["off_run_lengths"]

    if len(on_runs) > 0:
        print(f"\nON run length (consecutive timesteps pair stays ON):")
        print(f"  Mean: {np.mean(on_runs):.2f}")
        print(f"  Median: {np.median(on_runs):.2f}")
        print(f"  Std: {np.std(on_runs):.2f}")
        print(f"  Min/Max: {np.min(on_runs)}/{np.max(on_runs)}")

    if len(off_runs) > 0:
        print(f"\nOFF run length (consecutive timesteps pair stays OFF):")
        print(f"  Mean: {np.mean(off_runs):.2f}")
        print(f"  Median: {np.median(off_runs):.2f}")
        print(f"  Std: {np.std(off_runs):.2f}")
        print(f"  Min/Max: {np.min(off_runs)}/{np.max(off_runs)}")

    # Baseline performance
    print("\n" + "=" * 60)
    print("BASELINE ACCURACY")
    print("=" * 60)

    baselines = compute_baseline_accuracy(
        result["edges_by_t"], result["all_pairs"], result["timesteps"]
    )

    print(f"\n  Persistence baseline: {100 * baselines['persistence_accuracy']:.2f}%")
    print(f"  Always ON baseline:   {100 * baselines['always_on_accuracy']:.2f}%")
    print(f"  Always OFF baseline:  {100 * baselines['always_off_accuracy']:.2f}%")
    print(f"  Random baseline:      50.00%")
    print(f"\n  Mean positive rate:   {100 * baselines['mean_positive_rate']:.2f}%")
    print(f"  Std positive rate:    {100 * baselines['std_positive_rate']:.3f}%")

    # Signal analysis
    print("\n" + "=" * 60)
    print("SHARED NEIGHBOR SIGNAL (for OFF→ON prediction)")
    print("=" * 60)

    neighbor_signal = analyze_neighbor_signal(
        result["edges_by_t"], result["all_pairs"], result["timesteps"]
    )

    forming = neighbor_signal["forming"]
    not_forming = neighbor_signal["not_forming"]

    if len(forming) > 0 and len(not_forming) > 0:
        print(f"\nShared neighbors when pair FORMS (OFF→ON next timestep):")
        print(f"  Mean: {np.mean(forming):.3f}")
        print(f"  Median: {np.median(forming):.1f}")

        print(f"\nShared neighbors when pair DOESN'T form (stays OFF):")
        print(f"  Mean: {np.mean(not_forming):.3f}")
        print(f"  Median: {np.median(not_forming):.1f}")

        # Statistical test
        stat, p_value = stats.mannwhitneyu(forming, not_forming, alternative="greater")
        print(f"\n  Mann-Whitney U test (forming > not forming): p = {p_value:.2e}")

        # Effect size
        if np.std(forming) + np.std(not_forming) > 0:
            cohens_d = (np.mean(forming) - np.mean(not_forming)) / np.sqrt(
                (np.var(forming) + np.var(not_forming)) / 2
            )
            print(f"  Cohen's d effect size: {cohens_d:.3f}")

    # Activity signal
    print("\n" + "=" * 60)
    print("ENTITY ACTIVITY SIGNAL (for OFF→ON prediction)")
    print("=" * 60)

    activity_signal = analyze_activity_signal(
        result["edges_by_t"], result["all_pairs"], result["timesteps"]
    )

    forming = activity_signal["forming"]
    not_forming = activity_signal["not_forming"]

    if len(forming) > 0 and len(not_forming) > 0:
        print(f"\nCombined entity activity when pair FORMS (OFF→ON):")
        print(f"  Mean: {np.mean(forming):.3f}")
        print(f"  Median: {np.median(forming):.1f}")

        print(f"\nCombined entity activity when pair DOESN'T form (stays OFF):")
        print(f"  Mean: {np.mean(not_forming):.3f}")
        print(f"  Median: {np.median(not_forming):.1f}")

        # Statistical test
        stat, p_value = stats.mannwhitneyu(forming, not_forming, alternative="greater")
        print(f"\n  Mann-Whitney U test (forming > not forming): p = {p_value:.2e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
Key Findings:
1. Persistence rate: {100 * persistence_rate:.1f}% of pair states stay the same between timesteps
2. Transition rate: {100 * transition_rate:.1f}% of pair states change between timesteps
3. A simple "predict same as yesterday" achieves {100 * baselines["persistence_accuracy"]:.1f}% accuracy
4. The model must beat this to demonstrate any dynamics understanding

Implications for Model Improvement:
- If persistence rate is very high (>90%), consider transition-weighted loss
- If shared neighbor signal is significant, RGCN is capturing relevant information
- History length of {len(result["timesteps"])} total timesteps available, using seq_len=10

To improve beyond persistence, the model needs to:
- Identify which currently-ON pairs are about to break
- Identify which currently-OFF pairs are about to form
- Use neighbor/activity signals to predict these transitions
""")


if __name__ == "__main__":
    main()
