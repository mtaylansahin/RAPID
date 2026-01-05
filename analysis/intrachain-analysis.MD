# Intrachain Interaction Analysis for RAPID

## Problem Statement

RAPID predicts protein-protein interaction dynamics from MD simulation data. The current architecture uses only **interchain interactions** — contacts between residues on different protein chains (A and C).

**Key Question**: Does excluding **intrachain interactions** (contacts between residues on the same chain) cause us to miss valuable predictive signal?

Intrachain contacts could theoretically encode:
1. **Spatial proximity**: Residues that are close in the protein structure
2. **Interface geometry**: Which residues form coherent interface patches
3. **Conformational coupling**: Allosteric effects propagating through the chain

---

## Data Overview

```
Dataset: replica1
Timesteps: 201
Interactions per timestep:
  - Intra-A:     887.5 ± 17.0
  - Intra-C:     811.7 ± 16.4
  - Interchain:   46.0 ± 8.4  (what we currently use)

Interface residues: 27 per chain
Unique interchain pairs: 73
```

The intrachain interactions outnumber interchain by ~35:1.

---

## Analyses Performed

### 1. Temporal Dynamics

**Question**: How stable are different interaction types over time?

| Interaction Type | Jaccard Similarity (consecutive timesteps) |
|------------------|-------------------------------------------|
| Intra-A | 0.779 ± 0.019 |
| Intra-C | 0.783 ± 0.019 |
| Interchain | 0.671 ± 0.085 |

**Finding**: Interchain contacts are more dynamic. Intrachain provides stable structural context.

---

### 2. Count-Level Correlation

**Question**: Do aggregate intrachain and interchain counts correlate?

| Comparison | Pearson r | p-value |
|------------|-----------|---------|
| Intra-A ↔ Interchain (lag 0) | -0.008 | 0.915 |
| Intra-C ↔ Interchain (lag 0) | -0.106 | 0.133 |
| Intra-A[t] → Interchain[t+1] | 0.016 | 0.82 |

**Finding**: No aggregate relationship between intra and interchain dynamics.

---

### 3. Residue-Level Correlation

**Question**: At the residue level, does intrachain contact count predict interchain activity?

| Analysis | r | p-value |
|----------|---|---------|
| Point-biserial (intra count ↔ inter active) | -0.199 | 4.5e-37 |
| Lagged (intra[t] → inter[t+1]) | -0.198 | 1.6e-36 |

**Finding**: Weak negative correlation exists — residues with more intrachain contacts are slightly less likely to have interchain contacts active. But effect size is too small to be useful (r ≈ 0.2).

---

### 4. Neighborhood Effects

**Question**: When an interface residue engages in interchain contact, does its intrachain neighborhood state change?

| State | Neighborhood Contact Count |
|-------|---------------------------|
| Interchain ON | 58.78 ± 26.88 |
| Interchain OFF | 65.53 ± 25.89 |
| r | -0.124 |

**Finding**: ~10% fewer neighborhood contacts when engaged. Significant but weak.

---

### 5. Graph Distance and Partner Overlap ⭐

**Question**: Do residues that are close in the intrachain graph share interchain partners?

| Intrachain Distance | Partner Overlap (Jaccard) | n |
|--------------------|--------------------------|---|
| 1-2 hops | 0.186 | 111 |
| 3-4 hops | 0.013 | 177 |
| 5-6 hops | 0.000 | 37 |

**Correlation**: r = **-0.566**, p < 10⁻²⁸

**Finding**: **Strong signal**. Residues close in the intrachain graph share interchain partners; distant ones don't. This encodes interface geometry.

---

### 6. Competition vs Cooperation

**Question**: Do intrachain neighbors compete for the same interchain partner (anti-correlation) or engage together (positive correlation)?

| Metric | Value |
|--------|-------|
| Mean correlation for shared partners | r = +0.091 |
| Negative correlations | 23/53 |

**Finding**: Slight positive correlation — neighbors tend to engage partners together, not compete.

---

### 7. Key Residue Analysis

**Question**: Are there specific residues where intrachain state is highly predictive?

| Residue | Correlation | p-value |
|---------|-------------|---------|
| A1032 | -0.655 | 5.6e-26 |
| A1055 | +0.510 | 1.0e-14 |
| A96 | -0.293 | 2.5e-05 |

**Finding**: 8 of 25 interface residues have significant correlations. Two have |r| > 0.5, suggesting residue-specific effects (possibly allosteric).

---

### 8. Coordinated Dynamics of Intrachain Neighbors

**Question**: Do intrachain neighbors have correlated interchain activity?

| Pair Type | Mean Correlation |
|-----------|-----------------|
| Intrachain neighbors | r = 0.090 |
| Non-neighbors | r = -0.016 |
| t-test p-value | 1.9e-04 |

**Finding**: Significant difference — neighbors have more correlated interchain dynamics.

---

### 9. Allosteric Propagation

**Question**: Does A1032's (key residue) intrachain state predict other residues' interchain activity?

| Residue | Correlation | Significance |
|---------|-------------|--------------|
| A1030 | -0.387 | *** |
| A50 | -0.304 | *** |
| A1 | +0.283 | *** |

**Finding**: Specific effects exist for nearby residues, but mean effect is near zero.

---

## Summary of Findings

| Signal Type | Strength | Learnable? |
|-------------|----------|------------|
| **Graph distance → partner overlap** | Strong (r=-0.57) | ✅ Yes |
| Interface clustering | Strong (p<1e-72) | ✅ Yes |
| Neighbor coordination | Moderate (r=0.09) | ⚠️ Marginal |
| Temporal dynamics | Weak (r<0.1) | ❌ No |
| Allosteric propagation | Specific residues only | ⚠️ Limited |

**Key insight**: Intrachain topology encodes **spatial structure** (interface geometry), not **temporal dynamics**.

---

## Implementation Options

### Option A: Static Node Features (Recommended First)

Add pre-computed per-residue features from the aggregated intrachain graph.

**Features**:
1. `mean_dist_to_interface`: Average shortest path to interface residues
2. `intrachain_degree`: Number of unique intrachain neighbors
3. `interface_neighbor_fraction`: Fraction of neighbors that are interface residues

**Implementation sketch**:

```python
# Preprocessing: compute once from full-interaction data
intra_graph = build_intrachain_graph(full_interactions)
interface_residues = get_interface_residues(interchain_interactions)

features = {}
for entity in all_entities:
    features[entity] = [
        avg_shortest_path(intra_graph, entity, interface_residues),
        intra_graph.degree(entity),
        count_interface_neighbors(intra_graph, entity, interface_residues) / max(1, intra_graph.degree(entity))
    ]

torch.save(features, "data/processed/intra_features.pt")
```

```python
# Model: add to entity embeddings
class RAPIDModel:
    def __init__(self, ..., intra_features: torch.Tensor):
        self.intra_features = nn.Parameter(intra_features, requires_grad=False)
        self.intra_proj = nn.Linear(3, hidden_dim)
    
    def get_entity_embed(self, entity_id):
        base = self.entity_embeds[entity_id]
        intra = self.intra_proj(self.intra_features[entity_id])
        return base + intra
```

**Pros**: Simple, low risk, directly addresses strongest signal  
**Cons**: Static (doesn't change over time)

---

### Option B: Intrachain Edges in RGCN

Include intrachain contacts as edges in the per-timestep graph with a separate relation type.

**Key considerations**:

1. **Edge Count Explosion**: Intrachain edges outnumber interchain ~35:1. This will dominate RGCN message passing.

2. **Separate Relation Type**: Use a new relation type for intrachain edges so the model can learn different weights.

3. **Static vs Dynamic**: Two sub-options:
   - **B1**: Use aggregated intrachain graph (same edges every timestep) — encodes structure
   - **B2**: Use per-timestep intrachain graph — encodes dynamics, but we found dynamics are weak

**Training flow considerations**:

```python
# Option B1: Static intrachain structure
intra_edges = load_aggregated_intrachain_graph()  # Computed once

def build_graph_for_timestep(t, interchain_data):
    inter_edges = get_interchain_edges(t)
    inter_rels = get_interchain_relations(t)
    
    # Combine with static intrachain
    all_edges = torch.cat([inter_edges, intra_edges])
    all_rels = torch.cat([
        inter_rels,  # 0, 1, 2 for hydrophobic/hbond/ionic
        torch.full((len(intra_edges),), 3)  # 3 for intrachain
    ])
    
    return build_undirected_graph(all_edges, all_rels, num_nodes)
```

```python
# Option B2: Dynamic intrachain (per-timestep)
def build_graph_for_timestep(t, full_data):
    inter_edges = get_interchain_edges(t)
    inter_rels = get_interchain_relations(t)
    intra_edges = get_intrachain_edges(t)  # Different each timestep
    
    all_edges = torch.cat([inter_edges, intra_edges])
    all_rels = torch.cat([inter_rels, torch.full((len(intra_edges),), 3)])
    
    return build_undirected_graph(all_edges, all_rels, num_nodes)
```

**Evaluation flow considerations**:

For Option B1 (static intrachain), no change needed — same graph structure in train/valid/test.

For Option B2 (dynamic intrachain), the RGCN would process different graphs at each timestep. The autoregressive prediction loop remains the same, but:
- The model sees richer graph context
- History encoding would include RGCN outputs that aggregate both intra and interchain neighborhoods

**Potential issues**:
- Intrachain edges may overshadow sparse interchain signal
- May need to weight relation types differently (learnable per-relation scaling)
- Increased memory/compute: ~35x more edges

**Mitigation strategies**:
1. **Subsample intrachain edges**: Use only edges above a frequency threshold
2. **Separate aggregation**: Two RGCN passes (one for inter, one for intra), then combine
3. **Attention over edge types**: Learn to weight inter vs intra importance

---

## Recommendation

1. **Start with Option A** — low risk, addresses the strongest signal
2. **Evaluate impact** — does adding static structural features improve metrics?
3. **If beneficial, try Option B1** — static intrachain in RGCN
4. **Skip Option B2** — dynamic intrachain signal is too weak to justify complexity

---

## Analysis Scripts

| Script | Purpose |
|--------|---------|
| `scripts/analyze_intrachain_signal.py` | Basic 5-analysis suite |
| `scripts/analyze_intrachain_followup.py` | Static vs dynamic effect |
| `scripts/analyze_intrachain_extended.py` | Neighborhood, key residues, structural proxy |
| `scripts/analyze_intrachain_hypotheses.py` | Targeted hypothesis tests |
