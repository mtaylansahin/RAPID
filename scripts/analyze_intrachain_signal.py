#!/usr/bin/env python
"""
Analysis: Do intrachain interactions provide predictive signal for interchain interactions?

This script tests several hypotheses:
1. Temporal correlation: Do changes in intrachain interactions precede/predict changes in interchain?
2. Spatial correlation: Do residues with more intrachain contacts have more dynamic interchain behavior?
3. Co-occurrence patterns: Do specific intrachain contacts correlate with specific interchain contacts?

Key questions:
- Is there information in intrachain dynamics that we're losing by not including them?
- Would incorporating intrachain edges improve prediction of interchain dynamics?
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Paths
FULL_DATA_DIR = Path("data/raw/replica1-full/rep1-interfacea")
INTERCHAIN_DATA_DIR = Path("data/raw/replica1-interchain/rep1-interfacea")


def parse_interfacea_file(filepath: Path) -> pd.DataFrame:
    """Parse a single .interfacea file."""
    try:
        df = pd.read_table(
            filepath,
            header=0,
            sep=r"\s+",
        )
        # Extract timestep from filename
        matches = re.findall(r"[0-9]+", filepath.name)
        if matches:
            df["timestep"] = int(matches[0]) - 1
        return df
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return pd.DataFrame()


def load_all_data(data_dir: Path) -> pd.DataFrame:
    """Load all timesteps from a directory."""
    all_frames = []
    for filepath in sorted(data_dir.glob("*.interfacea")):
        df = parse_interfacea_file(filepath)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()

    return pd.concat(all_frames, ignore_index=True)


def classify_interaction(row) -> str:
    """Classify interaction as intrachain or interchain."""
    if row["chain_a"] == row["chain_b"]:
        return f"intra_{row['chain_a']}"
    else:
        return "interchain"


def create_residue_id(chain: str, resid: int) -> str:
    """Create unique residue identifier."""
    return f"{chain}{resid}"


def analyze_data():
    """Main analysis."""
    print("=" * 70)
    print("INTRACHAIN SIGNAL ANALYSIS")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    full_df = load_all_data(FULL_DATA_DIR)
    interchain_df = load_all_data(INTERCHAIN_DATA_DIR)

    print(
        f"   Full data: {len(full_df)} interactions across {full_df['timestep'].nunique()} timesteps"
    )
    print(f"   Interchain data: {len(interchain_df)} interactions")

    # Classify interactions
    full_df["interaction_type"] = full_df.apply(classify_interaction, axis=1)

    # Create residue IDs
    full_df["res_a"] = full_df.apply(
        lambda r: create_residue_id(r["chain_a"], r["resid_a"]), axis=1
    )
    full_df["res_b"] = full_df.apply(
        lambda r: create_residue_id(r["chain_b"], r["resid_b"]), axis=1
    )

    interchain_df["res_a"] = interchain_df.apply(
        lambda r: create_residue_id(r["chain_a"], r["resid_a"]), axis=1
    )
    interchain_df["res_b"] = interchain_df.apply(
        lambda r: create_residue_id(r["chain_b"], r["resid_b"]), axis=1
    )

    # Summary stats
    print("\n2. Interaction breakdown per timestep (averaged):")
    type_counts = (
        full_df.groupby(["timestep", "interaction_type"]).size().unstack(fill_value=0)
    )
    print(
        f"   Intra-A:     {type_counts['intra_A'].mean():.1f} ± {type_counts['intra_A'].std():.1f}"
    )
    print(
        f"   Intra-C:     {type_counts['intra_C'].mean():.1f} ± {type_counts['intra_C'].std():.1f}"
    )
    print(
        f"   Interchain:  {type_counts['interchain'].mean():.1f} ± {type_counts['interchain'].std():.1f}"
    )

    # ==========================================================================
    # ANALYSIS 1: Temporal dynamics - how much do interactions change over time?
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: TEMPORAL DYNAMICS")
    print("=" * 70)

    # Get unique edges per timestep for each type
    def get_edges_at_timestep(
        df: pd.DataFrame, t: int, interaction_type: str = None
    ) -> Set[Tuple[str, str]]:
        sub_df = df[df["timestep"] == t]
        if interaction_type:
            sub_df = sub_df[sub_df["interaction_type"] == interaction_type]
        edges = set()
        for _, row in sub_df.iterrows():
            edge = tuple(sorted([row["res_a"], row["res_b"]]))
            edges.add(edge)
        return edges

    timesteps = sorted(full_df["timestep"].unique())

    # Calculate Jaccard similarity between consecutive timesteps
    def jaccard(set1: Set, set2: Set) -> float:
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    jaccard_intra_A = []
    jaccard_intra_C = []
    jaccard_inter = []

    for i in range(len(timesteps) - 1):
        t1, t2 = timesteps[i], timesteps[i + 1]

        intra_A_1 = get_edges_at_timestep(full_df, t1, "intra_A")
        intra_A_2 = get_edges_at_timestep(full_df, t2, "intra_A")
        jaccard_intra_A.append(jaccard(intra_A_1, intra_A_2))

        intra_C_1 = get_edges_at_timestep(full_df, t1, "intra_C")
        intra_C_2 = get_edges_at_timestep(full_df, t2, "intra_C")
        jaccard_intra_C.append(jaccard(intra_C_1, intra_C_2))

        inter_1 = get_edges_at_timestep(full_df, t1, "interchain")
        inter_2 = get_edges_at_timestep(full_df, t2, "interchain")
        jaccard_inter.append(jaccard(inter_1, inter_2))

    print("\n   Jaccard similarity between consecutive timesteps (stability):")
    print(
        f"   Intra-A:     {np.mean(jaccard_intra_A):.3f} ± {np.std(jaccard_intra_A):.3f}"
    )
    print(
        f"   Intra-C:     {np.mean(jaccard_intra_C):.3f} ± {np.std(jaccard_intra_C):.3f}"
    )
    print(f"   Interchain:  {np.mean(jaccard_inter):.3f} ± {np.std(jaccard_inter):.3f}")

    print("\n   Interpretation: Higher = more stable, lower = more dynamic")

    # ==========================================================================
    # ANALYSIS 2: Cross-correlation between intrachain and interchain dynamics
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: CROSS-CORRELATION (Intrachain ↔ Interchain)")
    print("=" * 70)

    # Count number of interactions per timestep
    intra_A_counts = type_counts["intra_A"].values
    intra_C_counts = type_counts["intra_C"].values
    inter_counts = type_counts["interchain"].values

    # Cross-correlation with lags
    print("\n   Pearson correlation of interaction counts:")

    # Same-time correlation
    r_A_inter, p_A = stats.pearsonr(intra_A_counts, inter_counts)
    r_C_inter, p_C = stats.pearsonr(intra_C_counts, inter_counts)
    print(f"   Intra-A ↔ Interchain (lag 0): r={r_A_inter:.3f}, p={p_A:.3e}")
    print(f"   Intra-C ↔ Interchain (lag 0): r={r_C_inter:.3f}, p={p_C:.3e}")

    # Lagged correlations (does intrachain at t predict interchain at t+1?)
    print("\n   Lagged correlations (intrachain at t → interchain at t+k):")
    for lag in [1, 2, 3, 5]:
        if lag < len(intra_A_counts):
            r_A_lag, p_A_lag = stats.pearsonr(intra_A_counts[:-lag], inter_counts[lag:])
            r_C_lag, p_C_lag = stats.pearsonr(intra_C_counts[:-lag], inter_counts[lag:])
            print(
                f"   Lag {lag}: Intra-A r={r_A_lag:.3f} (p={p_A_lag:.3e}), Intra-C r={r_C_lag:.3f} (p={p_C_lag:.3e})"
            )

    # ==========================================================================
    # ANALYSIS 3: Residue-level analysis
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 3: RESIDUE-LEVEL CORRELATIONS")
    print("=" * 70)

    # Get residues involved in interchain interactions
    interchain_residues_A = set()
    interchain_residues_C = set()

    interchain_only = full_df[full_df["interaction_type"] == "interchain"]
    for _, row in interchain_only.iterrows():
        if row["chain_a"] == "A":
            interchain_residues_A.add(row["res_a"])
            interchain_residues_C.add(row["res_b"])
        else:
            interchain_residues_A.add(row["res_b"])
            interchain_residues_C.add(row["res_a"])

    print(f"\n   Residues involved in interchain contacts:")
    print(f"   Chain A: {len(interchain_residues_A)} residues")
    print(f"   Chain C: {len(interchain_residues_C)} residues")

    # For each interchain residue, count its intrachain contacts over time
    def get_intrachain_contact_count(
        df: pd.DataFrame, residue: str, timestep: int, chain: str
    ) -> int:
        """Count intrachain contacts for a residue at a timestep."""
        intra_type = f"intra_{chain}"
        sub_df = df[
            (df["timestep"] == timestep) & (df["interaction_type"] == intra_type)
        ]
        count = len(sub_df[(sub_df["res_a"] == residue) | (sub_df["res_b"] == residue)])
        return count

    def is_interchain_active(df: pd.DataFrame, residue: str, timestep: int) -> bool:
        """Check if residue has any interchain contact at timestep."""
        sub_df = df[
            (df["timestep"] == timestep) & (df["interaction_type"] == "interchain")
        ]
        return (
            len(sub_df[(sub_df["res_a"] == residue) | (sub_df["res_b"] == residue)]) > 0
        )

    # Build per-residue time series
    print("\n   Building residue-level time series...")

    residue_data = []
    for residue in list(interchain_residues_A)[:20]:  # Sample for speed
        chain = "A"
        for t in timesteps:
            intra_count = get_intrachain_contact_count(full_df, residue, t, chain)
            inter_active = is_interchain_active(full_df, residue, t)
            residue_data.append(
                {
                    "residue": residue,
                    "timestep": t,
                    "intra_contacts": intra_count,
                    "inter_active": int(inter_active),
                }
            )

    residue_df = pd.DataFrame(residue_data)

    # Point-biserial correlation: does intrachain count correlate with interchain activity?
    r_pb, p_pb = stats.pointbiserialr(
        residue_df["inter_active"], residue_df["intra_contacts"]
    )
    print(f"\n   Point-biserial correlation (intrachain count ↔ interchain active):")
    print(f"   r = {r_pb:.3f}, p = {p_pb:.3e}")

    # Same analysis with lag
    print("\n   Lagged correlation (intrachain at t → interchain active at t+1):")
    lagged_data = []
    for residue in residue_df["residue"].unique():
        res_ts = residue_df[residue_df["residue"] == residue].sort_values("timestep")
        for i in range(len(res_ts) - 1):
            lagged_data.append(
                {
                    "intra_t": res_ts.iloc[i]["intra_contacts"],
                    "inter_t1": res_ts.iloc[i + 1]["inter_active"],
                }
            )

    lagged_df = pd.DataFrame(lagged_data)
    if len(lagged_df) > 0:
        r_lag, p_lag = stats.pointbiserialr(lagged_df["inter_t1"], lagged_df["intra_t"])
        print(f"   r = {r_lag:.3f}, p = {p_lag:.3e}")

    # ==========================================================================
    # ANALYSIS 4: Specific pair co-occurrence
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 4: INTERCHAIN PAIR DYNAMICS vs INTRACHAIN STATE")
    print("=" * 70)

    # Get all interchain pairs
    interchain_pairs = set()
    for _, row in interchain_only.iterrows():
        pair = tuple(sorted([row["res_a"], row["res_b"]]))
        interchain_pairs.add(pair)

    print(f"\n   Total unique interchain pairs: {len(interchain_pairs)}")

    # For each interchain pair, track:
    # - When it's ON vs OFF
    # - Total intrachain contacts of both residues when ON vs OFF

    on_intra_counts = []
    off_intra_counts = []

    for pair in list(interchain_pairs)[:50]:  # Sample for speed
        res_a, res_b = pair
        chain_a = res_a[0]
        chain_b = res_b[0]

        for t in timesteps:
            # Is pair active at this timestep?
            pair_active = get_edges_at_timestep(full_df, t, "interchain")
            is_on = pair in pair_active

            # Count intrachain contacts for both residues
            intra_a = get_intrachain_contact_count(full_df, res_a, t, chain_a)
            intra_b = get_intrachain_contact_count(full_df, res_b, t, chain_b)
            total_intra = intra_a + intra_b

            if is_on:
                on_intra_counts.append(total_intra)
            else:
                off_intra_counts.append(total_intra)

    if on_intra_counts and off_intra_counts:
        print(f"\n   Intrachain contacts of interchain pair residues:")
        print(
            f"   When pair is ON:  {np.mean(on_intra_counts):.2f} ± {np.std(on_intra_counts):.2f}"
        )
        print(
            f"   When pair is OFF: {np.mean(off_intra_counts):.2f} ± {np.std(off_intra_counts):.2f}"
        )

        # Statistical test
        t_stat, p_val = stats.ttest_ind(on_intra_counts, off_intra_counts)
        print(f"   t-test: t={t_stat:.3f}, p={p_val:.3e}")

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.std(on_intra_counts) ** 2 + np.std(off_intra_counts) ** 2) / 2
        )
        cohens_d = (
            (np.mean(on_intra_counts) - np.mean(off_intra_counts)) / pooled_std
            if pooled_std > 0
            else 0
        )
        print(f"   Cohen's d: {cohens_d:.3f} (effect size)")

    # ==========================================================================
    # ANALYSIS 5: Mutual information
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 5: MUTUAL INFORMATION")
    print("=" * 70)

    # Discretize intrachain count changes and interchain state changes
    def get_change_series(df: pd.DataFrame, interaction_type: str) -> List[int]:
        """Get series of edge count changes between timesteps."""
        changes = []
        for i in range(len(timesteps) - 1):
            t1, t2 = timesteps[i], timesteps[i + 1]
            edges_1 = get_edges_at_timestep(df, t1, interaction_type)
            edges_2 = get_edges_at_timestep(df, t2, interaction_type)

            new_edges = len(edges_2 - edges_1)
            lost_edges = len(edges_1 - edges_2)
            net_change = new_edges - lost_edges

            # Discretize: -1 (decrease), 0 (stable), +1 (increase)
            if net_change < -2:
                changes.append(-1)
            elif net_change > 2:
                changes.append(1)
            else:
                changes.append(0)

        return changes

    intra_A_changes = get_change_series(full_df, "intra_A")
    intra_C_changes = get_change_series(full_df, "intra_C")
    inter_changes = get_change_series(full_df, "interchain")

    def mutual_info(x: List[int], y: List[int]) -> float:
        """Calculate mutual information between two discrete series."""
        from collections import Counter

        n = len(x)
        if n == 0:
            return 0.0

        # Joint distribution
        joint = Counter(zip(x, y))

        # Marginal distributions
        px = Counter(x)
        py = Counter(y)

        mi = 0.0
        for (xi, yi), count in joint.items():
            pxy = count / n
            pxi = px[xi] / n
            pyi = py[yi] / n
            if pxy > 0 and pxi > 0 and pyi > 0:
                mi += pxy * np.log2(pxy / (pxi * pyi))

        return mi

    mi_A_inter = mutual_info(intra_A_changes, inter_changes)
    mi_C_inter = mutual_info(intra_C_changes, inter_changes)

    print(f"\n   Mutual information (discretized change direction):")
    print(f"   I(Intra-A changes; Interchain changes) = {mi_A_inter:.4f} bits")
    print(f"   I(Intra-C changes; Interchain changes) = {mi_C_inter:.4f} bits")

    # Lagged MI
    mi_A_lag1 = mutual_info(intra_A_changes[:-1], inter_changes[1:])
    mi_C_lag1 = mutual_info(intra_C_changes[:-1], inter_changes[1:])
    print(f"\n   Lagged MI (intrachain at t → interchain at t+1):")
    print(f"   I(Intra-A[t]; Interchain[t+1]) = {mi_A_lag1:.4f} bits")
    print(f"   I(Intra-C[t]; Interchain[t+1]) = {mi_C_lag1:.4f} bits")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY & CONCLUSIONS")
    print("=" * 70)

    print("""
    Key findings:
    
    1. STABILITY: Compare Jaccard similarities to see which class is more dynamic.
       - If interchain << intrachain: interchain contacts are more dynamic,
         and intrachain provides stable structural context.
       - If similar: both change at similar rates.
    
    2. CORRELATION: If lagged correlations are significant (p < 0.05),
       intrachain dynamics may have predictive value for interchain.
    
    3. RESIDUE-LEVEL: Point-biserial correlation shows if residues with
       more intrachain contacts are more likely to be in interchain contacts.
    
    4. PAIR-LEVEL: t-test shows if interchain pairs have systematically
       different intrachain neighborhood when ON vs OFF.
    
    5. MUTUAL INFORMATION: Quantifies shared information between
       intrachain and interchain dynamics.
    
    RECOMMENDATION:
    - If correlations are low (<0.1) and MI is near zero: intrachain signal
      is likely not useful for predicting interchain dynamics.
    - If correlations are moderate (>0.2) or MI is substantial: consider
      incorporating intrachain structure into the model.
    """)


if __name__ == "__main__":
    analyze_data()
