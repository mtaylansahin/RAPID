#!/usr/bin/env python
"""
Follow-up analysis: Investigating the negative correlation finding.

The initial analysis found a negative correlation (r = -0.20) between
intrachain contact count and interchain activity. This suggests that
residues with MORE intrachain contacts are LESS likely to have interchain
contacts active at any given timestep.

Hypotheses to investigate:
1. This could be a structural effect: interface residues have fewer intrachain
   contacts because they're at the surface
2. This could be a temporal effect: when residues engage in interchain contacts,
   they temporarily lose some intrachain contacts (competition)
3. This could be spurious: other confounding factors

Additional analyses:
- Check if this is a static vs dynamic effect
- Analyze whether interchain contact formation correlates with intrachain contact loss
- Check spatial patterns (interface vs core residues)
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats

FULL_DATA_DIR = Path("data/raw/replica1-full/rep1-interfacea")


def parse_interfacea_file(filepath: Path) -> pd.DataFrame:
    """Parse a single .interfacea file."""
    try:
        df = pd.read_table(filepath, header=0, sep=r"\s+")
        matches = re.findall(r"[0-9]+", filepath.name)
        if matches:
            df["timestep"] = int(matches[0]) - 1
        return df
    except Exception:
        return pd.DataFrame()


def load_all_data(data_dir: Path) -> pd.DataFrame:
    """Load all timesteps from a directory."""
    all_frames = []
    for filepath in sorted(data_dir.glob("*.interfacea")):
        df = parse_interfacea_file(filepath)
        if not df.empty:
            all_frames.append(df)
    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()


def create_residue_id(chain: str, resid: int) -> str:
    return f"{chain}{resid}"


def run_followup_analysis():
    print("=" * 70)
    print("FOLLOW-UP ANALYSIS: NEGATIVE CORRELATION INVESTIGATION")
    print("=" * 70)

    # Load data
    df = load_all_data(FULL_DATA_DIR)
    df["res_a"] = df.apply(
        lambda r: create_residue_id(r["chain_a"], r["resid_a"]), axis=1
    )
    df["res_b"] = df.apply(
        lambda r: create_residue_id(r["chain_b"], r["resid_b"]), axis=1
    )
    df["is_interchain"] = df["chain_a"] != df["chain_b"]

    timesteps = sorted(df["timestep"].unique())

    # ==========================================================================
    # ANALYSIS 1: Static vs Dynamic effect
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: STATIC vs DYNAMIC EFFECT")
    print("=" * 70)

    # Get all residues that ever participate in interchain
    interchain_df = df[df["is_interchain"]]
    all_residues = set(df["res_a"]) | set(df["res_b"])
    interchain_residues = set(interchain_df["res_a"]) | set(interchain_df["res_b"])

    # Calculate AVERAGE intrachain contact count per residue (over all time)
    intrachain_df = df[~df["is_interchain"]]

    def avg_intrachain_count(residue: str) -> float:
        """Average intrachain contacts for a residue across all timesteps."""
        counts = []
        for t in timesteps:
            t_df = intrachain_df[intrachain_df["timestep"] == t]
            count = len(t_df[(t_df["res_a"] == residue) | (t_df["res_b"] == residue)])
            counts.append(count)
        return np.mean(counts)

    # Compare interface residues vs non-interface residues
    interface_A = [r for r in interchain_residues if r.startswith("A")]
    interface_C = [r for r in interchain_residues if r.startswith("C")]

    all_A = [r for r in all_residues if r.startswith("A")]
    all_C = [r for r in all_residues if r.startswith("C")]

    core_A = [r for r in all_A if r not in interface_A]
    core_C = [r for r in all_C if r not in interface_C]

    print(
        f"\n   Chain A: {len(interface_A)} interface residues, {len(core_A)} core residues"
    )
    print(
        f"   Chain C: {len(interface_C)} interface residues, {len(core_C)} core residues"
    )

    # Calculate average intrachain contacts
    print("\n   Computing average intrachain contacts per residue...")

    interface_A_intra = [avg_intrachain_count(r) for r in interface_A[:20]]  # Sample
    core_A_intra = [avg_intrachain_count(r) for r in core_A[:20]]  # Sample

    print(f"\n   Average intrachain contacts (Chain A):")
    print(
        f"   Interface residues: {np.mean(interface_A_intra):.2f} ± {np.std(interface_A_intra):.2f}"
    )
    print(
        f"   Core residues:      {np.mean(core_A_intra):.2f} ± {np.std(core_A_intra):.2f}"
    )

    t_stat, p_val = stats.ttest_ind(interface_A_intra, core_A_intra)
    print(f"   t-test: t={t_stat:.3f}, p={p_val:.3e}")

    print(
        "\n   → This tells us if interface residues INHERENTLY have fewer intrachain contacts"
    )

    # ==========================================================================
    # ANALYSIS 2: Temporal dynamics within interface residues
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: TEMPORAL DYNAMICS (Within interface residues)")
    print("=" * 70)

    print("\n   For interface residues only, comparing timesteps when they're")
    print("   engaged in interchain vs not engaged...")

    # For each interface residue, collect intrachain counts when inter-ON vs inter-OFF
    on_counts = []
    off_counts = []

    for residue in interface_A[:20]:  # Sample for speed
        chain = residue[0]

        for t in timesteps:
            # Count intrachain contacts at this timestep
            t_intra = intrachain_df[intrachain_df["timestep"] == t]
            intra_count = len(
                t_intra[(t_intra["res_a"] == residue) | (t_intra["res_b"] == residue)]
            )

            # Check if engaged in interchain at this timestep
            t_inter = interchain_df[interchain_df["timestep"] == t]
            is_engaged = (
                len(
                    t_inter[
                        (t_inter["res_a"] == residue) | (t_inter["res_b"] == residue)
                    ]
                )
                > 0
            )

            if is_engaged:
                on_counts.append(intra_count)
            else:
                off_counts.append(intra_count)

    print(f"\n   Intrachain contacts of interface residues:")
    print(
        f"   When interchain ON:  {np.mean(on_counts):.2f} ± {np.std(on_counts):.2f} (n={len(on_counts)})"
    )
    print(
        f"   When interchain OFF: {np.mean(off_counts):.2f} ± {np.std(off_counts):.2f} (n={len(off_counts)})"
    )

    t_stat, p_val = stats.ttest_ind(on_counts, off_counts)
    print(f"   t-test: t={t_stat:.3f}, p={p_val:.3e}")

    diff = np.mean(off_counts) - np.mean(on_counts)
    pct_diff = 100 * diff / np.mean(off_counts) if np.mean(off_counts) > 0 else 0
    print(f"\n   → Interface residues have {pct_diff:.1f}% fewer intrachain contacts")
    print(f"     when engaged in interchain interactions")

    # ==========================================================================
    # ANALYSIS 3: Transition analysis (do intrachain contacts change at transitions?)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 3: TRANSITION ANALYSIS")
    print("=" * 70)

    print("\n   Analyzing what happens to intrachain contacts when interchain")
    print("   transitions occur (ON→OFF or OFF→ON)...")

    # Track transitions
    on_to_off_delta = []
    off_to_on_delta = []
    no_change_delta = []

    for residue in interface_A[:20]:
        # Build time series for this residue
        intra_series = []
        inter_series = []

        for t in timesteps:
            t_intra = intrachain_df[intrachain_df["timestep"] == t]
            intra_count = len(
                t_intra[(t_intra["res_a"] == residue) | (t_intra["res_b"] == residue)]
            )
            intra_series.append(intra_count)

            t_inter = interchain_df[interchain_df["timestep"] == t]
            is_engaged = (
                len(
                    t_inter[
                        (t_inter["res_a"] == residue) | (t_inter["res_b"] == residue)
                    ]
                )
                > 0
            )
            inter_series.append(int(is_engaged))

        # Analyze transitions
        for i in range(len(timesteps) - 1):
            intra_change = intra_series[i + 1] - intra_series[i]

            if inter_series[i] == 1 and inter_series[i + 1] == 0:
                # ON → OFF transition
                on_to_off_delta.append(intra_change)
            elif inter_series[i] == 0 and inter_series[i + 1] == 1:
                # OFF → ON transition
                off_to_on_delta.append(intra_change)
            else:
                no_change_delta.append(intra_change)

    print(f"\n   Intrachain contact change at interchain transitions:")
    print(
        f"   ON → OFF:  Δintra = {np.mean(on_to_off_delta):+.2f} ± {np.std(on_to_off_delta):.2f} (n={len(on_to_off_delta)})"
    )
    print(
        f"   OFF → ON:  Δintra = {np.mean(off_to_on_delta):+.2f} ± {np.std(off_to_on_delta):.2f} (n={len(off_to_on_delta)})"
    )
    print(
        f"   No change: Δintra = {np.mean(no_change_delta):+.2f} ± {np.std(no_change_delta):.2f} (n={len(no_change_delta)})"
    )

    # Test if transitions are different from no-change
    if len(off_to_on_delta) > 0 and len(no_change_delta) > 0:
        t_on, p_on = stats.ttest_ind(off_to_on_delta, no_change_delta)
        print(f"\n   OFF→ON vs baseline: t={t_on:.3f}, p={p_on:.3e}")

    if len(on_to_off_delta) > 0 and len(no_change_delta) > 0:
        t_off, p_off = stats.ttest_ind(on_to_off_delta, no_change_delta)
        print(f"   ON→OFF vs baseline: t={t_off:.3f}, p={p_off:.3e}")

    # ==========================================================================
    # ANALYSIS 4: Predictive value - can intrachain change predict interchain?
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 4: PREDICTIVE VALUE")
    print("=" * 70)

    print(
        "\n   Can a DECREASE in intrachain contacts predict an INCREASE in interchain?"
    )

    # For each residue, check if intrachain decrease at t predicts interchain activation at t+1
    true_positives = 0  # Intra decreased AND inter turned ON
    false_positives = 0  # Intra decreased but inter didn't turn ON
    false_negatives = 0  # Inter turned ON but intra didn't decrease
    true_negatives = 0  # Neither happened

    for residue in interface_A[:20]:
        # Build time series
        intra_series = []
        inter_series = []

        for t in timesteps:
            t_intra = intrachain_df[intrachain_df["timestep"] == t]
            intra_count = len(
                t_intra[(t_intra["res_a"] == residue) | (t_intra["res_b"] == residue)]
            )
            intra_series.append(intra_count)

            t_inter = interchain_df[interchain_df["timestep"] == t]
            is_engaged = (
                len(
                    t_inter[
                        (t_inter["res_a"] == residue) | (t_inter["res_b"] == residue)
                    ]
                )
                > 0
            )
            inter_series.append(int(is_engaged))

        # Check prediction
        for i in range(len(timesteps) - 1):
            intra_decreased = intra_series[i + 1] < intra_series[i]
            inter_activated = inter_series[i] == 0 and inter_series[i + 1] == 1

            if intra_decreased and inter_activated:
                true_positives += 1
            elif intra_decreased and not inter_activated:
                false_positives += 1
            elif not intra_decreased and inter_activated:
                false_negatives += 1
            else:
                true_negatives += 1

    total = true_positives + false_positives + false_negatives + true_negatives

    print(f"\n   Confusion matrix (Intra decrease → Inter activation):")
    print(f"   TP: {true_positives}, FP: {false_positives}")
    print(f"   FN: {false_negatives}, TN: {true_negatives}")

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )

    print(f"\n   Precision: {precision:.3f}")
    print(f"   Recall: {recall:.3f}")
    print(
        f"   Base rate (inter activation): {(true_positives + false_negatives) / total:.3f}"
    )

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print("""
    Based on the analyses:
    
    1. STATIC EFFECT: If interface residues have significantly fewer intrachain
       contacts than core residues on average, the negative correlation is
       primarily a STRUCTURAL effect (interface = surface = fewer internal contacts).
       
    2. DYNAMIC EFFECT: If the same interface residue has fewer intrachain contacts
       specifically when engaged in interchain, there's a TEMPORAL competition
       between intra and interchain contacts.
       
    3. PREDICTIVE VALUE: If precision/recall are low (close to base rate),
       intrachain dynamics don't provide meaningful predictive signal beyond
       what you'd get from knowing which residues are at the interface.
    
    RECOMMENDATION:
    - If primarily STATIC: Including intrachain contacts would mainly help
      identify interface residues (which you already know from training data).
    - If primarily DYNAMIC with high predictive value: Worth incorporating.
    - If DYNAMIC but low precision: The relationship exists but is too noisy
      to be useful for prediction.
    """)


if __name__ == "__main__":
    run_followup_analysis()
