#!/usr/bin/env python
"""
Extended analysis addressing specific concerns:

1. NEIGHBORHOOD EFFECTS: Do intrachain neighbors of interface residues show
   correlated behavior that predicts interchain dynamics?

2. KEY RESIDUE ANALYSIS: Are there specific residues where intrachain state
   IS highly predictive, even if the average is noisy?

3. STRUCTURAL PROXY: Does intrachain contact graph encode spatial structure
   that's relevant for interchain prediction?

4. LEARNABILITY: Given the data size, would signal be learnable if it exists?
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

FULL_DATA_DIR = Path("data/raw/replica1-full/rep1-interfacea")


def parse_interfacea_file(filepath: Path) -> pd.DataFrame:
    try:
        df = pd.read_table(filepath, header=0, sep=r"\s+")
        matches = re.findall(r"[0-9]+", filepath.name)
        if matches:
            df["timestep"] = int(matches[0]) - 1
        return df
    except Exception:
        return pd.DataFrame()


def load_all_data(data_dir: Path) -> pd.DataFrame:
    all_frames = []
    for filepath in sorted(data_dir.glob("*.interfacea")):
        df = parse_interfacea_file(filepath)
        if not df.empty:
            all_frames.append(df)
    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()


def create_residue_id(chain: str, resid: int) -> str:
    return f"{chain}{resid}"


def run_extended_analysis():
    print("=" * 70)
    print("EXTENDED INTRACHAIN SIGNAL ANALYSIS")
    print("=" * 70)

    df = load_all_data(FULL_DATA_DIR)
    df["res_a"] = df.apply(
        lambda r: create_residue_id(r["chain_a"], r["resid_a"]), axis=1
    )
    df["res_b"] = df.apply(
        lambda r: create_residue_id(r["chain_b"], r["resid_b"]), axis=1
    )
    df["is_interchain"] = df["chain_a"] != df["chain_b"]

    timesteps = sorted(df["timestep"].unique())
    interchain_df = df[df["is_interchain"]]
    intrachain_df = df[~df["is_interchain"]]

    # Get interface residues
    interface_residues = set(interchain_df["res_a"]) | set(interchain_df["res_b"])
    interface_A = {r for r in interface_residues if r.startswith("A")}
    interface_C = {r for r in interface_residues if r.startswith("C")}

    print(f"\nData summary:")
    print(f"  Timesteps: {len(timesteps)}")
    print(f"  Interface residues: A={len(interface_A)}, C={len(interface_C)}")

    # ==========================================================================
    # ANALYSIS 1: NEIGHBORHOOD EFFECTS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: NEIGHBORHOOD EFFECTS")
    print("=" * 70)

    print("\n   Question: When an interface residue X engages in interchain contact,")
    print("   does the intrachain state of X's NEIGHBORS change?")

    # Build aggregated intrachain neighbor graph (across all timesteps)
    intra_neighbors = defaultdict(set)
    for _, row in intrachain_df.iterrows():
        intra_neighbors[row["res_a"]].add(row["res_b"])
        intra_neighbors[row["res_b"]].add(row["res_a"])

    print(
        f"\n   Average intrachain neighbors per residue: {np.mean([len(v) for v in intra_neighbors.values()]):.1f}"
    )

    # For each interface residue, get its intrachain neighborhood
    def get_neighborhood_state(residue: str, timestep: int) -> int:
        """Count total intrachain contacts in the neighborhood at a timestep."""
        neighbors = intra_neighbors.get(residue, set())
        t_intra = intrachain_df[intrachain_df["timestep"] == timestep]

        count = 0
        for neighbor in neighbors:
            count += len(
                t_intra[(t_intra["res_a"] == neighbor) | (t_intra["res_b"] == neighbor)]
            )
        return count

    def is_interchain_active(residue: str, timestep: int) -> bool:
        t_inter = interchain_df[interchain_df["timestep"] == timestep]
        return (
            len(t_inter[(t_inter["res_a"] == residue) | (t_inter["res_b"] == residue)])
            > 0
        )

    # Collect neighborhood state when interface residue is ON vs OFF
    neighborhood_on = []
    neighborhood_off = []

    print("\n   Analyzing neighborhood states for interface residues...")

    for residue in list(interface_A):  # All interface residues on chain A
        for t in timesteps:
            ns = get_neighborhood_state(residue, t)
            if is_interchain_active(residue, t):
                neighborhood_on.append(ns)
            else:
                neighborhood_off.append(ns)

    print(f"\n   Neighborhood intrachain contact count:")
    print(
        f"   When interface residue ON:  {np.mean(neighborhood_on):.2f} ± {np.std(neighborhood_on):.2f} (n={len(neighborhood_on)})"
    )
    print(
        f"   When interface residue OFF: {np.mean(neighborhood_off):.2f} ± {np.std(neighborhood_off):.2f} (n={len(neighborhood_off)})"
    )

    t_stat, p_val = stats.ttest_ind(neighborhood_on, neighborhood_off)
    print(f"   t-test: t={t_stat:.3f}, p={p_val:.3e}")

    # Correlation at neighborhood level
    all_ns = []
    all_inter = []
    for residue in list(interface_A):
        for t in timesteps:
            all_ns.append(get_neighborhood_state(residue, t))
            all_inter.append(int(is_interchain_active(residue, t)))

    r_pb, p_pb = stats.pointbiserialr(all_inter, all_ns)
    print(f"\n   Point-biserial (neighborhood state ↔ interchain active):")
    print(f"   r = {r_pb:.3f}, p = {p_pb:.3e}")

    # ==========================================================================
    # ANALYSIS 2: INTERCHAIN PAIR NEIGHBORHOOD CORRELATION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: INTERCHAIN PAIR NEIGHBORHOOD")
    print("=" * 70)

    print("\n   Question: For an interchain pair A-C, does the combined neighborhood")
    print("   state of residues A and C predict pair dynamics?")

    # Get all interchain pairs
    interchain_pairs = set()
    for _, row in interchain_df.iterrows():
        pair = tuple(sorted([row["res_a"], row["res_b"]]))
        interchain_pairs.add(pair)

    print(f"\n   Total unique interchain pairs: {len(interchain_pairs)}")

    # For each pair, track combined neighborhood state vs pair activity
    pair_data = []

    for pair in list(interchain_pairs):
        res_a, res_c = pair if pair[0].startswith("A") else (pair[1], pair[0])

        for t in timesteps:
            # Combined neighborhood state
            ns_a = get_neighborhood_state(res_a, t)
            ns_c = get_neighborhood_state(res_c, t)
            combined_ns = ns_a + ns_c

            # Is pair active?
            t_inter = interchain_df[interchain_df["timestep"] == t]
            pair_edges = set()
            for _, row in t_inter.iterrows():
                pair_edges.add(tuple(sorted([row["res_a"], row["res_b"]])))
            is_active = pair in pair_edges

            pair_data.append(
                {
                    "pair": f"{res_a}-{res_c}",
                    "timestep": t,
                    "ns_a": ns_a,
                    "ns_c": ns_c,
                    "combined_ns": combined_ns,
                    "active": int(is_active),
                }
            )

    pair_df = pd.DataFrame(pair_data)

    r_comb, p_comb = stats.pointbiserialr(pair_df["active"], pair_df["combined_ns"])
    print(f"\n   Combined neighborhood ↔ pair activity:")
    print(f"   r = {r_comb:.3f}, p = {p_comb:.3e}")

    # ==========================================================================
    # ANALYSIS 3: KEY RESIDUE ANALYSIS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 3: KEY RESIDUE ANALYSIS")
    print("=" * 70)

    print("\n   Question: Are there specific residues where intrachain state")
    print("   is highly predictive of interchain activity?")

    residue_correlations = []

    for residue in list(interface_A):
        # Build per-residue time series
        intra_series = []
        inter_series = []

        for t in timesteps:
            t_intra = intrachain_df[intrachain_df["timestep"] == t]
            intra_count = len(
                t_intra[(t_intra["res_a"] == residue) | (t_intra["res_b"] == residue)]
            )
            intra_series.append(intra_count)

            inter_active = is_interchain_active(residue, t)
            inter_series.append(int(inter_active))

        # Calculate correlation for this residue
        if np.std(inter_series) > 0 and np.std(intra_series) > 0:
            r, p = stats.pointbiserialr(inter_series, intra_series)
            residue_correlations.append(
                {
                    "residue": residue,
                    "r": r,
                    "p": p,
                    "inter_rate": np.mean(inter_series),
                    "intra_mean": np.mean(intra_series),
                }
            )

    corr_df = pd.DataFrame(residue_correlations)
    corr_df = corr_df.sort_values("r")

    print(f"\n   Correlation distribution across {len(corr_df)} interface residues:")
    print(f"   Min r:    {corr_df['r'].min():.3f}")
    print(f"   Median r: {corr_df['r'].median():.3f}")
    print(f"   Max r:    {corr_df['r'].max():.3f}")

    print("\n   Top 5 residues with STRONGEST correlations (positive or negative):")
    top_residues = corr_df.loc[corr_df["r"].abs().nlargest(5).index]
    for _, row in top_residues.iterrows():
        sig = (
            "***"
            if row["p"] < 0.001
            else "**"
            if row["p"] < 0.01
            else "*"
            if row["p"] < 0.05
            else ""
        )
        print(f"   {row['residue']}: r={row['r']:.3f} (p={row['p']:.3e}) {sig}")

    # Count significant residues
    n_sig = len(corr_df[corr_df["p"] < 0.05])
    print(
        f"\n   Residues with significant correlation (p<0.05): {n_sig}/{len(corr_df)}"
    )

    # ==========================================================================
    # ANALYSIS 4: INTRACHAIN AS STRUCTURAL PROXY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 4: INTRACHAIN AS STRUCTURAL PROXY")
    print("=" * 70)

    print("\n   Question: Does intrachain graph topology encode spatial structure")
    print("   relevant for interchain prediction?")

    # Build intrachain contact graph (aggregated) - for chain A
    residues_A = sorted(
        [
            r
            for r in set(intrachain_df["res_a"]) | set(intrachain_df["res_b"])
            if r.startswith("A")
        ]
    )
    res_to_idx = {r: i for i, r in enumerate(residues_A)}
    n = len(residues_A)

    # Count intrachain contacts between residue pairs (as edge weight)
    adj = np.zeros((n, n))
    for _, row in intrachain_df[intrachain_df["chain_a"] == "A"].iterrows():
        if row["res_a"] in res_to_idx and row["res_b"] in res_to_idx:
            i, j = res_to_idx[row["res_a"]], res_to_idx[row["res_b"]]
            adj[i, j] += 1
            adj[j, i] += 1

    # Binary adjacency (any contact ever)
    adj_binary = (adj > 0).astype(int)

    # Compute shortest path distances in intrachain graph
    adj_sparse = csr_matrix(adj_binary)
    dist_matrix = shortest_path(adj_sparse, directed=False, unweighted=True)

    print(f"\n   Intrachain graph (Chain A): {n} residues")
    print(f"   Connected pairs: {np.sum(adj_binary) // 2}")

    # Analyze: do interface residues cluster in the intrachain graph?
    interface_indices = [res_to_idx[r] for r in interface_A if r in res_to_idx]
    non_interface_indices = [i for i in range(n) if i not in interface_indices]

    # Average distance within interface vs between interface and non-interface
    interface_interface_dists = []
    interface_noninterface_dists = []

    for i in interface_indices:
        for j in interface_indices:
            if i < j and not np.isinf(dist_matrix[i, j]):
                interface_interface_dists.append(dist_matrix[i, j])
        for j in non_interface_indices:
            if not np.isinf(dist_matrix[i, j]):
                interface_noninterface_dists.append(dist_matrix[i, j])

    print(f"\n   Graph distances (in intrachain contact graph):")
    print(
        f"   Interface ↔ Interface:     {np.mean(interface_interface_dists):.2f} ± {np.std(interface_interface_dists):.2f}"
    )
    print(
        f"   Interface ↔ Non-interface: {np.mean(interface_noninterface_dists):.2f} ± {np.std(interface_noninterface_dists):.2f}"
    )

    t_stat, p_val = stats.ttest_ind(
        interface_interface_dists, interface_noninterface_dists
    )
    print(f"   t-test: t={t_stat:.3f}, p={p_val:.3e}")

    print("\n   → If interface residues are closer in intrachain graph,")
    print("     intrachain topology encodes spatial clustering of interface.")

    # ==========================================================================
    # ANALYSIS 5: CORRELATED DYNAMICS BETWEEN INTRACHAIN NEIGHBORS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 5: COORDINATED INTERCHAIN BEHAVIOR OF INTRACHAIN NEIGHBORS")
    print("=" * 70)

    print("\n   Question: If X and Y are intrachain neighbors and both are interface")
    print("   residues, do their interchain activities correlate?")

    # Find interface residue pairs that are also intrachain neighbors
    interface_neighbor_pairs = []
    for res_a in interface_A:
        neighbors_of_a = intra_neighbors.get(res_a, set())
        for res_b in neighbors_of_a:
            if res_b in interface_A and res_a < res_b:
                interface_neighbor_pairs.append((res_a, res_b))

    print(
        f"\n   Interface residue pairs that are intrachain neighbors: {len(interface_neighbor_pairs)}"
    )

    if len(interface_neighbor_pairs) > 0:
        # For each pair, compute correlation of their interchain activity time series
        pair_correlations = []

        for res_a, res_b in interface_neighbor_pairs:
            inter_a = [int(is_interchain_active(res_a, t)) for t in timesteps]
            inter_b = [int(is_interchain_active(res_b, t)) for t in timesteps]

            if np.std(inter_a) > 0 and np.std(inter_b) > 0:
                r, p = stats.pearsonr(inter_a, inter_b)
                pair_correlations.append({"pair": f"{res_a}-{res_b}", "r": r, "p": p})

        if pair_correlations:
            pc_df = pd.DataFrame(pair_correlations)
            print(
                f"\n   Correlation of interchain activity between intrachain neighbors:"
            )
            print(f"   Mean r: {pc_df['r'].mean():.3f}")
            print(f"   Median r: {pc_df['r'].median():.3f}")
            print(f"   Max r: {pc_df['r'].max():.3f}")
            print(
                f"   Significant pairs (p<0.05): {len(pc_df[pc_df['p'] < 0.05])}/{len(pc_df)}"
            )

            # Compare to random (non-neighbor) pairs
            non_neighbor_pairs = []
            for res_a in list(interface_A):
                neighbors = intra_neighbors.get(res_a, set())
                for res_b in interface_A:
                    if res_a < res_b and res_b not in neighbors:
                        non_neighbor_pairs.append((res_a, res_b))

            if len(non_neighbor_pairs) > 0:
                random_correlations = []
                for res_a, res_b in non_neighbor_pairs[
                    : len(interface_neighbor_pairs)
                ]:  # Match sample size
                    inter_a = [int(is_interchain_active(res_a, t)) for t in timesteps]
                    inter_b = [int(is_interchain_active(res_b, t)) for t in timesteps]

                    if np.std(inter_a) > 0 and np.std(inter_b) > 0:
                        r, p = stats.pearsonr(inter_a, inter_b)
                        random_correlations.append(r)

                print(f"\n   Comparison to NON-neighbor interface pairs:")
                print(f"   Intrachain neighbors mean r: {pc_df['r'].mean():.3f}")
                print(f"   Non-neighbors mean r: {np.mean(random_correlations):.3f}")

                t_stat, p_val = stats.ttest_ind(pc_df["r"].values, random_correlations)
                print(f"   t-test: t={t_stat:.3f}, p={p_val:.3e}")

    # ==========================================================================
    # ANALYSIS 6: LEARNABILITY ASSESSMENT
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 6: LEARNABILITY ASSESSMENT")
    print("=" * 70)

    print("\n   Question: Given limited data, would a model be able to learn")
    print("   any signal that exists?")

    # Sample sizes
    total_samples = len(pair_df)
    positive_samples = pair_df["active"].sum()
    negative_samples = total_samples - positive_samples

    print(f"\n   Training signal statistics:")
    print(f"   Total samples: {total_samples}")
    print(
        f"   Positive (interchain ON): {positive_samples} ({100 * positive_samples / total_samples:.1f}%)"
    )
    print(
        f"   Negative (interchain OFF): {negative_samples} ({100 * negative_samples / total_samples:.1f}%)"
    )

    # Unique interchain pairs per timestep
    pairs_per_timestep = interchain_df.groupby("timestep").apply(
        lambda x: len(set(zip(x["res_a"], x["res_b"])))
    )

    print(
        f"\n   Interchain pairs per timestep: {pairs_per_timestep.mean():.1f} ± {pairs_per_timestep.std():.1f}"
    )
    print(f"   Total unique pairs: {len(interchain_pairs)}")
    print(f"   Observation per pair: {total_samples / len(interchain_pairs):.1f}")

    # Effect size consideration
    print("\n   Effect size context:")
    effect_sizes = {
        "Neighborhood correlation": abs(r_pb) if "r_pb" in dir() else 0,
        "Combined neighborhood": abs(r_comb) if "r_comb" in dir() else 0,
        "Best single residue": corr_df["r"].abs().max() if len(corr_df) > 0 else 0,
    }

    for name, es in effect_sizes.items():
        learnability = (
            "Likely learnable"
            if es > 0.3
            else "Possibly learnable"
            if es > 0.15
            else "Unlikely learnable"
        )
        print(f"   {name}: |r|={es:.3f} → {learnability}")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("EXTENDED ANALYSIS SUMMARY")
    print("=" * 70)

    print("""
    1. NEIGHBORHOOD EFFECTS:
       - Tested whether intrachain neighborhood state correlates with
         interface residue activity
       - Result shows whether local structural context matters
    
    2. KEY RESIDUES:
       - Some individual residues may show stronger correlations
       - If a few residues drive the average, targeted modeling could help
    
    3. STRUCTURAL PROXY:
       - Intrachain graph distances reveal spatial clustering
       - If interface residues cluster, topology encodes interface geometry
    
    4. COORDINATED BEHAVIOR:
       - If intrachain neighbors have correlated interchain dynamics,
         neighborhood info could propagate through message passing
    
    5. LEARNABILITY:
       - Effect size < 0.15 typically too weak to learn reliably
       - Data size limits ability to capture subtle signals
    
    RECOMMENDATION CRITERIA:
    - If neighborhood correlation > 0.2 OR
    - If interface clustering is significant (p < 0.05) OR  
    - If coordinated behavior of neighbors > non-neighbors:
      → Intrachain structure adds learnable structural context
      
    - Otherwise:
      → Current interchain-only approach is sufficient
    """)


if __name__ == "__main__":
    run_extended_analysis()
