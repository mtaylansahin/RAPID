#!/usr/bin/env python
"""
Targeted hypothesis testing for specific intrachain signal scenarios.

Tests:
1. TEMPORAL LAG: Does 1A-2X interaction predict future 1B-2X interaction
   (where 1A-1B are intrachain neighbors)?

2. GRAPH DISTANCE: Do intrachain-distant interface residues interact with
   different regions of the partner chain?

3. COMPETITION: Are intrachain neighbors anti-correlated in interchain binding
   (steric competition)?

4. CORE VS SURFACE: More rigorous test with all residues, not just sampled.

5. ALLOSTERIC PROPAGATION: Do key residues' intrachain changes precede
   other interface residues' interchain changes?
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import warnings

warnings.filterwarnings("ignore")

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


def run_hypothesis_tests():
    print("=" * 70)
    print("TARGETED HYPOTHESIS TESTING")
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

    # Build intrachain neighbor graph
    intra_neighbors = defaultdict(set)
    for _, row in intrachain_df.iterrows():
        intra_neighbors[row["res_a"]].add(row["res_b"])
        intra_neighbors[row["res_b"]].add(row["res_a"])

    # Get interface residues
    interface_A = set()
    interface_C = set()
    for _, row in interchain_df.iterrows():
        if row["chain_a"] == "A":
            interface_A.add(row["res_a"])
            interface_C.add(row["res_b"])
        else:
            interface_A.add(row["res_b"])
            interface_C.add(row["res_a"])

    print(
        f"\nData: {len(timesteps)} timesteps, {len(interface_A)} interface residues per chain"
    )

    # Helper functions
    def get_interchain_partners(residue: str, timestep: int) -> Set[str]:
        """Get all interchain partners of a residue at a timestep."""
        t_inter = interchain_df[interchain_df["timestep"] == timestep]
        partners = set()
        for _, row in t_inter.iterrows():
            if row["res_a"] == residue:
                partners.add(row["res_b"])
            elif row["res_b"] == residue:
                partners.add(row["res_a"])
        return partners

    def is_pair_interacting(res1: str, res2: str, timestep: int) -> bool:
        """Check if a specific pair is interacting at a timestep."""
        t_inter = interchain_df[interchain_df["timestep"] == timestep]
        for _, row in t_inter.iterrows():
            if (row["res_a"] == res1 and row["res_b"] == res2) or (
                row["res_a"] == res2 and row["res_b"] == res1
            ):
                return True
        return False

    # ==========================================================================
    # HYPOTHESIS 1: TEMPORAL LAG THROUGH NEIGHBORS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("HYPOTHESIS 1: TEMPORAL PREDICTION THROUGH INTRACHAIN NEIGHBORS")
    print("=" * 70)
    print("\n   If 1A-2X interacts at time t, does this predict 1B-2X at time t+1")
    print("   (where 1A and 1B are intrachain neighbors on chain A)?")

    # Find interface neighbor pairs on chain A
    interface_neighbor_pairs_A = []
    for res_a in interface_A:
        for res_b in intra_neighbors.get(res_a, set()):
            if res_b in interface_A and res_a < res_b:
                interface_neighbor_pairs_A.append((res_a, res_b))

    print(
        f"\n   Interface neighbor pairs on chain A: {len(interface_neighbor_pairs_A)}"
    )

    # For each pair, test: does one's interchain activity predict the other's future activity?
    lag_correlations = []

    for res_a, res_b in interface_neighbor_pairs_A:
        # Get interchain time series (as int for correlation)
        inter_a = [int(len(get_interchain_partners(res_a, t)) > 0) for t in timesteps]
        inter_b = [int(len(get_interchain_partners(res_b, t)) > 0) for t in timesteps]

        # Test: A[t] → B[t+1]
        if np.std(inter_a[:-1]) > 0 and np.std(inter_b[1:]) > 0:
            r_ab, p_ab = stats.pearsonr(inter_a[:-1], inter_b[1:])
            lag_correlations.append({"type": "A→B", "r": r_ab, "p": p_ab})

        # Test: B[t] → A[t+1]
        if np.std(inter_b[:-1]) > 0 and np.std(inter_a[1:]) > 0:
            r_ba, p_ba = stats.pearsonr(inter_b[:-1], inter_a[1:])
            lag_correlations.append({"type": "B→A", "r": r_ba, "p": p_ba})

    if lag_correlations:
        lag_df = pd.DataFrame(lag_correlations)
        print(f"\n   Lagged correlation (neighbor[t] → neighbor[t+1]):")
        print(f"   Mean r: {lag_df['r'].mean():.3f}")
        print(f"   Range: [{lag_df['r'].min():.3f}, {lag_df['r'].max():.3f}]")
        print(
            f"   Significant (p<0.05): {len(lag_df[lag_df['p'] < 0.05])}/{len(lag_df)}"
        )

        # Compare to same-time correlation
        same_time_r = []
        for res_a, res_b in interface_neighbor_pairs_A:
            inter_a = [
                int(len(get_interchain_partners(res_a, t)) > 0) for t in timesteps
            ]
            inter_b = [
                int(len(get_interchain_partners(res_b, t)) > 0) for t in timesteps
            ]
            if np.std(inter_a) > 0 and np.std(inter_b) > 0:
                r, _ = stats.pearsonr(inter_a, inter_b)
                same_time_r.append(r)

        print(
            f"\n   Same-time correlation (neighbor[t] ↔ neighbor[t]): {np.mean(same_time_r):.3f}"
        )
        print(
            f"   Lagged correlation (neighbor[t] → neighbor[t+1]): {lag_df['r'].mean():.3f}"
        )

    # ==========================================================================
    # HYPOTHESIS 2: GRAPH DISTANCE → DIFFERENT PARTNER REGIONS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("HYPOTHESIS 2: GRAPH DISTANCE AND PARTNER OVERLAP")
    print("=" * 70)
    print("\n   Do intrachain-distant interface residues interact with DIFFERENT")
    print("   residues on the partner chain?")

    # Build graph distance matrix for chain A
    residues_A = sorted(
        [
            r
            for r in set(intrachain_df["res_a"]) | set(intrachain_df["res_b"])
            if r.startswith("A")
        ]
    )
    res_to_idx = {r: i for i, r in enumerate(residues_A)}
    n = len(residues_A)

    adj = np.zeros((n, n))
    for _, row in intrachain_df[intrachain_df["chain_a"] == "A"].iterrows():
        if row["res_a"] in res_to_idx and row["res_b"] in res_to_idx:
            i, j = res_to_idx[row["res_a"]], res_to_idx[row["res_b"]]
            adj[i, j] = 1
            adj[j, i] = 1

    dist_matrix = shortest_path(csr_matrix(adj), directed=False, unweighted=True)

    # For each interface residue pair, compute: graph distance vs partner overlap
    partner_overlap_data = []

    interface_A_list = list(interface_A)
    for i, res_a in enumerate(interface_A_list):
        for res_b in interface_A_list[i + 1 :]:
            # Get graph distance
            if res_a in res_to_idx and res_b in res_to_idx:
                dist = dist_matrix[res_to_idx[res_a], res_to_idx[res_b]]
                if np.isinf(dist):
                    continue

                # Get partner sets (aggregated over all time)
                partners_a = set()
                partners_b = set()
                for t in timesteps:
                    partners_a.update(get_interchain_partners(res_a, t))
                    partners_b.update(get_interchain_partners(res_b, t))

                # Jaccard similarity of partner sets
                if len(partners_a | partners_b) > 0:
                    jaccard = len(partners_a & partners_b) / len(
                        partners_a | partners_b
                    )
                    partner_overlap_data.append(
                        {
                            "pair": f"{res_a}-{res_b}",
                            "distance": dist,
                            "jaccard": jaccard,
                        }
                    )

    if partner_overlap_data:
        overlap_df = pd.DataFrame(partner_overlap_data)
        r, p = stats.pearsonr(overlap_df["distance"], overlap_df["jaccard"])

        print(f"\n   Correlation (Intrachain distance vs Partner overlap):")
        print(f"   r = {r:.3f}, p = {p:.3e}")

        # Group by distance bins
        overlap_df["dist_bin"] = pd.cut(
            overlap_df["distance"],
            bins=[0, 2, 4, 6, 100],
            labels=["1-2", "3-4", "5-6", "7+"],
        )
        print(f"\n   Partner overlap (Jaccard) by intrachain distance:")
        for bin_label in ["1-2", "3-4", "5-6", "7+"]:
            bin_data = overlap_df[overlap_df["dist_bin"] == bin_label]
            if len(bin_data) > 0:
                print(
                    f"   {bin_label} hops: Jaccard={bin_data['jaccard'].mean():.3f} (n={len(bin_data)})"
                )

    # ==========================================================================
    # HYPOTHESIS 3: COMPETITION / ANTI-CORRELATION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("HYPOTHESIS 3: COMPETITION (ANTI-CORRELATION)")
    print("=" * 70)
    print("\n   Are intrachain neighbors ANTI-correlated in interchain binding")
    print("   (steric competition for the same partner)?")

    # For each interface neighbor pair that shares at least one partner
    competition_data = []

    for res_a, res_b in interface_neighbor_pairs_A:
        # Get shared partners
        partners_a = set()
        partners_b = set()
        for t in timesteps:
            partners_a.update(get_interchain_partners(res_a, t))
            partners_b.update(get_interchain_partners(res_b, t))

        shared = partners_a & partners_b

        if len(shared) > 0:
            # Check correlation of binding to shared partners
            for partner in shared:
                binding_a = [
                    int(is_pair_interacting(res_a, partner, t)) for t in timesteps
                ]
                binding_b = [
                    int(is_pair_interacting(res_b, partner, t)) for t in timesteps
                ]

                if np.std(binding_a) > 0 and np.std(binding_b) > 0:
                    r, p = stats.pearsonr(binding_a, binding_b)
                    competition_data.append(
                        {
                            "res_pair": f"{res_a}-{res_b}",
                            "shared_partner": partner,
                            "r": r,
                            "p": p,
                        }
                    )

    if competition_data:
        comp_df = pd.DataFrame(competition_data)
        print(f"\n   Correlation of binding to SHARED partners:")
        print(f"   Mean r: {comp_df['r'].mean():.3f}")
        print(
            f"   Negative correlations: {len(comp_df[comp_df['r'] < 0])}/{len(comp_df)}"
        )
        print(
            f"   Significantly negative (r<0, p<0.05): {len(comp_df[(comp_df['r'] < 0) & (comp_df['p'] < 0.05)])}"
        )

        if comp_df["r"].mean() < 0:
            print("\n   → ANTI-CORRELATION detected: neighbors compete for partners")
        else:
            print("\n   → POSITIVE correlation: neighbors engage partners TOGETHER")

    # ==========================================================================
    # HYPOTHESIS 4: CORE VS SURFACE (RIGOROUS)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("HYPOTHESIS 4: CORE VS SURFACE (RIGOROUS)")
    print("=" * 70)
    print("\n   Do residues with MORE intrachain contacts have FEWER interchain?")

    # Compute average intrachain and interchain counts for ALL chain A residues
    all_residues_A = set(intrachain_df[intrachain_df["chain_a"] == "A"]["res_a"]) | set(
        intrachain_df[intrachain_df["chain_a"] == "A"]["res_b"]
    )

    residue_stats = []
    for residue in all_residues_A:
        # Average intrachain contacts
        intra_counts = []
        inter_counts = []
        for t in timesteps:
            t_intra = intrachain_df[intrachain_df["timestep"] == t]
            intra_count = len(
                t_intra[(t_intra["res_a"] == residue) | (t_intra["res_b"] == residue)]
            )
            intra_counts.append(intra_count)

            inter_count = len(get_interchain_partners(residue, t))
            inter_counts.append(inter_count)

        residue_stats.append(
            {
                "residue": residue,
                "avg_intra": np.mean(intra_counts),
                "avg_inter": np.mean(inter_counts),
                "is_interface": residue in interface_A,
            }
        )

    stats_df = pd.DataFrame(residue_stats)

    r, p = stats.pearsonr(stats_df["avg_intra"], stats_df["avg_inter"])
    print(f"\n   Correlation (intrachain count vs interchain count):")
    print(f"   r = {r:.3f}, p = {p:.3e}")

    # Split by intrachain tertiles
    stats_df["intra_tertile"] = pd.qcut(
        stats_df["avg_intra"], 3, labels=["Low", "Med", "High"]
    )
    print(f"\n   Interchain activity by intrachain contact level:")
    for tertile in ["Low", "Med", "High"]:
        t_data = stats_df[stats_df["intra_tertile"] == tertile]
        pct_interface = 100 * t_data["is_interface"].mean()
        avg_inter = t_data["avg_inter"].mean()
        print(
            f"   {tertile} intrachain: {pct_interface:.1f}% interface, avg interchain={avg_inter:.2f}"
        )

    # ==========================================================================
    # HYPOTHESIS 5: ALLOSTERIC PROPAGATION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("HYPOTHESIS 5: ALLOSTERIC PROPAGATION FROM KEY RESIDUES")
    print("=" * 70)
    print("\n   Do changes in A1032's intrachain state precede changes in")
    print("   other interface residues' interchain activity?")

    key_residue = "A1032"  # From our earlier analysis

    if key_residue in intra_neighbors:
        # Build time series for key residue's intrachain count
        key_intra = []
        for t in timesteps:
            t_intra = intrachain_df[intrachain_df["timestep"] == t]
            count = len(
                t_intra[
                    (t_intra["res_a"] == key_residue)
                    | (t_intra["res_b"] == key_residue)
                ]
            )
            key_intra.append(count)

        # Test lagged correlation with other interface residues' interchain activity
        lag_results = []
        for other in interface_A:
            if other == key_residue:
                continue

            other_inter = [
                int(len(get_interchain_partners(other, t)) > 0) for t in timesteps
            ]

            # Lag 1: key intra[t] → other inter[t+1]
            if np.std(key_intra[:-1]) > 0 and np.std(other_inter[1:]) > 0:
                r, p = stats.pearsonr(key_intra[:-1], other_inter[1:])
                lag_results.append({"other": other, "r": r, "p": p})

        if lag_results:
            lag_res_df = pd.DataFrame(lag_results)
            print(f"\n   A1032 intrachain[t] → Other interchain[t+1]:")
            print(f"   Mean r: {lag_res_df['r'].mean():.3f}")
            print(
                f"   Significant (p<0.05): {len(lag_res_df[lag_res_df['p'] < 0.05])}/{len(lag_res_df)}"
            )

            top = lag_res_df.loc[lag_res_df["r"].abs().nlargest(3).index]
            print(f"\n   Strongest effects:")
            for _, row in top.iterrows():
                sig = (
                    "***"
                    if row["p"] < 0.001
                    else "**"
                    if row["p"] < 0.01
                    else "*"
                    if row["p"] < 0.05
                    else ""
                )
                print(f"   A1032 → {row['other']}: r={row['r']:.3f} {sig}")

    # ==========================================================================
    # SUMMARY TABLE
    # ==========================================================================
    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTING SUMMARY")
    print("=" * 70)

    print("""
    | Hypothesis                              | Result               | Learnable? |
    |-----------------------------------------|----------------------|------------|
    | 1. Temporal lag through neighbors       | See above            |            |
    | 2. Graph distance → partner overlap     | See above            |            |
    | 3. Competition for shared partners      | See above            |            |
    | 4. Core vs Surface (intra → inter)      | See above            |            |
    | 5. Allosteric from key residue          | See above            |            |
    """)


if __name__ == "__main__":
    run_hypothesis_tests()
