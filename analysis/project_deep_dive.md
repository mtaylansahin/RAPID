# RAPID Project Deep Dive: Expert Q&A

This document provides comprehensive answers to key questions an expert in computational biology and machine learning would ask to fully understand the RAPID (Recurrent Architecture for Predicting Protein Interaction Dynamics) project. It covers design decisions, architecture choices, and implementation details.

---

## 1. What is the biological problem being solved, and how is it framed as a machine learning task?

**Problem:** RAPID predicts the **dynamic formation and breaking of protein-protein interactions (PPIs)** over time from molecular dynamics (MD) simulations. Unlike static PPI prediction, this is a **temporal binary classification task** where the model must forecast whether a given residue pair will be interacting (edge = 1) or not interacting (edge = 0) at each future timestep.

**Data Source:** The input comes from `.interfacea` files produced by MD simulation analysis tools. Each file represents a frame/snapshot listing every residue-residue contact at that moment. The preprocessing pipeline ([preprocessing.py](file:///Users/taylansahin/Projects/RAPID/src/data/preprocessing.py)) converts these into quadruples `(entity1, relation, entity2, timestep)` in a format adapted from the knowledge graph embedding literature.

**Key Constraint:** Only **interchain** (cross-chain) interactions are predicted. Intrachain contacts are used solely as structural features ([node_features.py](file:///Users/taylansahin/Projects/RAPID/src/data/node_features.py)) to inform the model about local residue environment.

**ML Framing:**
- **Entities:** Individual residues (e.g., "A123" = chain A, residue 123)
- **Relations:** Currently single relation type (contact), but architecture supports multiple
- **Timesteps:** Discrete frames from MD trajectory
- **Train/Val/Test Split:** Chronological split (60/20/20 by default) to prevent temporal leakage

---

## 2. How are candidate pairs selected for training, and why does this matter?

**Selection Strategy:** The project implements a **hard/easy negative mining strategy** ([sampler.py](file:///Users/taylansahin/Projects/RAPID/src/data/sampler.py)):

```
Positive: Pairs actively interacting at timestep t
Negative (Hard): Pairs that have interacted BEFORE but are OFF at t (50% of negatives)
Negative (Easy): Pairs that have NEVER interacted (50% of negatives)
```

**Why Hard Negatives?** Hard negatives represent the most realistic challenge: distinguishing between edges that *could* form (because they've formed before) versus random non-interacting pairs. Without hard negatives, the model could learn superficial features like "certain residues never interact" rather than understanding actual dynamics.

**`hard_ratio` Parameter:** Controls the mix (default 0.5). Higher values create harder training but may lead to lower F1 scores if the model becomes too conservative.

**Evaluation Strategy:** Uses **history-constrained** candidate selection ([dataset.py](file:///Users/taylansahin/Projects/RAPID/src/data/dataset.py#L412-L451)):
- Only evaluates on pairs that have **ever interacted** in the dataset
- This focuses metrics on "dynamics prediction" (will this known pair be on/off?) rather than "discovery" (are there new pairs?)

---

## 3. What is the core model architecture, and how does it handle temporal modeling?

**Architecture Overview:** RAPID is based on **RE-Net** (Recurrent Event Network) adapted for undirected binary classification. The core architecture in [rapid.py](file:///Users/taylansahin/Projects/RAPID/src/models/rapid.py):

```
For each entity pair (u, v) at time t:
1. Build history sequences: [t-seq_len, ..., t-1]
2. For each timestep τ in history:
   a. Run RGCN on graph G_τ → structural embeddings
   b. Concatenate: [RGCN(u)_τ || E_u || R̄ || c_τ]
3. Encode sequence with temporal encoder → z_u, z_v
4. Classifier predicts edge existence from (z_u, z_v, E_u, E_v)
```

**Components:**
1. **UndirectedRGCN** ([rgcn.py](file:///Users/taylansahin/Projects/RAPID/src/models/rgcn.py)): 2-layer RGCN with basis decomposition for weight sharing. Automatically adds reverse edges with symmetric normalization.

2. **Temporal Encoder** ([encoder.py](file:///Users/taylansahin/Projects/RAPID/src/models/encoder.py)): Two options controlled by `--use_attention_encoder`:
   - **GRU (Default):** Single-layer unidirectional GRU on packed sequences
   - **Attention:** Transformer encoder with a learnable CLS token; final CLS embedding is the output

3. **Classifier** ([classifier.py](file:///Users/taylansahin/Projects/RAPID/src/models/classifier.py)): MLP with symmetry enforcement—averages score(u,v) and score(v,u) for undirected prediction.

---

## 4. What is the Transition Prediction head, and why was it introduced?

**Motivation:** The standard formulation predicts P(edge at t) directly, but this leads to a **persistence shortcut**—the model learns to predict "same as t-1" because most edges are stable. This results in low transition recall.

**Solution:** The `--use_transition_prediction` flag activates `SymmetricTransitionClassifier`:

```
Instead of: P(edge at t)
Predict:    P(state changes from t-1 to t)
```

**Inference Conversion:**
```python
# XOR-like conversion in logit space:
edge_logit = transition_logit * (1 - 2 * was_on_t_minus_1)
# If was_on=0: edge_logit = transition_logit (predicting turning ON)
# If was_on=1: edge_logit = -transition_logit (predicting staying ON if no transition)
```

**Dual-Task Training Loss** ([train.py](file:///Users/taylansahin/Projects/RAPID/src/train.py#L123-L155)):
```
L_total = 0.4 * L_edge + 0.6 * L_transition
```
- **L_transition:** BCE on predicted transition vs actual state change (5x upweighting for transitions)
- **L_edge:** BCE on XOR-converted edge prediction vs true edge label

This encourages the model to be **persistence-aware** while still learning to detect transitions.

---

## 5. How does the Global Context Module work?

**Purpose:** Capture the "macroscopic" state of the entire protein interface to inform local predictions. For example, if the overall binding is weakening, individual contacts are more likely to break.

**Architecture** ([global_model.py](file:///Users/taylansahin/Projects/RAPID/src/models/global_model.py)):

```
For timesteps [t-seq_len, ..., t-1]:
1. GlobalRGCNAggregator:
   - Run RGCN on full graph G_τ
   - Max-pool over all nodes → g_τ (fixed-size graph embedding)
2. Sequence Encoder (GRU or Transformer):
   - Process [g_{t-seq_len}, ..., g_{t-1}]
   - Output: global context vector c_t
3. c_t is concatenated to each entity's sequence representation
```

**Encoder Options (`--global_encoder_type`):**
- **GRU (Default):** Standard recurrent encoding
- **Transformer:** Multi-head self-attention with learned positional embeddings

**Pretraining:** Global model is pretrained separately via `main.py pretrain` using soft cross-entropy on next-timestep entity activity, then frozen during main training.

**Temporal Offset:** `global_emb[t]` uses graphs from times < t+1, matching RE-Net's original design to prevent data leakage.

---

## 6. What node-level features are used, and how is data leakage prevented?

**Feature Types** ([node_features.py](file:///Users/taylansahin/Projects/RAPID/src/data/node_features.py)):

| Feature | Description |
|---------|-------------|
| **Hydrophobicity** | Kyte-Doolittle scale, normalized [-1, 1] |
| **Charge** | Side chain charge at pH 7 (D/E=-1, K/R=+1, H=0.5) |
| **Size** | Molecular weight normalized [0, 1] |
| **Polarity** | Binary: polar vs nonpolar |
| **Aromaticity** | Binary: contains aromatic ring (F/W/Y/H) |
| **Mean Distance to Interface** | Graph distance in intrachain contact network |
| **Intrachain Degree** | Number of intrachain neighbors |
| **Interface Neighbor Fraction** | % of neighbors that are interface residues |

**Total:** 8 features per residue (5 physicochemical + 3 structural)

**Leakage Prevention:**
- Intrachain features are computed using **only training timesteps** (`train_cutoff` parameter)
- Interface entities are identified from **training interchain interactions only**
- Features are then frozen and used across all splits

---

## 7. What loss function is used, and why?

**Primary:** Focal Loss ([losses/\_\_init\_\_.py](file:///Users/taylansahin/Projects/RAPID/src/losses/__init__.py)):

```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

**Parameters:**
- `γ` (gamma, default=2.0): Focusing parameter. Higher values down-weight easy examples more aggressively.
- `α` (alpha, optional): Class balancing weight

**Why Focal Loss?**
- PPI dynamics are **highly imbalanced**—most pairs are inactive most of the time
- Focal loss automatically down-weights easy negatives (confident 0 predictions)
- Forces the model to focus learning on hard-to-classify samples (transitions, ambiguous pairs)

**Transition Mode:** When enabled, uses BCE with explicit sample weighting instead:
- Transitions get 5x weight
- Combined with edge reconstruction loss (see Q4)

---

## 8. What metrics are used for evaluation, and how is the threshold determined?

**Primary Metrics** ([metrics/\_\_init\_\_.py](file:///Users/taylansahin/Projects/RAPID/src/metrics/__init__.py)):

| Metric | Description | Threshold-Dependent? |
|--------|-------------|---------------------|
| **AUPRC** | Area Under Precision-Recall Curve | No |
| **AUROC** | Area Under ROC Curve | No |
| **F1** | Harmonic mean of precision/recall | Yes |
| **MCC** | Matthews Correlation Coefficient | Yes |

**Threshold Selection:** The optimal threshold is found on the **validation set** by maximizing F1 score over the precision-recall curve. This threshold is then applied during test evaluation.

**Transition-Specific Metrics** (`TransitionMetrics`):
- **Forming Recall:** Of OFF→ON transitions, how many did we catch?
- **Breaking Recall:** Of ON→OFF transitions, how many did we catch?
- **Transition F1:** F1 for detecting any state change
- **Persistence Accuracy:** How often we correctly predict "no change"

**Why AUPRC is Primary?** AUPRC is more informative than AUROC for imbalanced classification. It measures ranking quality specifically among positive examples, which is what matters for rare events like edge transitions.

---

## 9. How does training differ from validation/test evaluation?

**Training** ([train.py](file:///Users/taylansahin/Projects/RAPID/src/train.py#L95-L174)):
- Uses **oracle (ground-truth) history** via `model.forward()`
- Sees the true graph at each past timestep
- Negative samples regenerated each epoch

**Validation/Test** ([train.py](file:///Users/taylansahin/Projects/RAPID/src/train.py#L176-L255) and [evaluate.py](file:///Users/taylansahin/Projects/RAPID/src/evaluate.py)):
- Uses **autoregressive inference** via `model.predict_batch()`
- Model's own predictions from t-1 update history for predicting t
- Simulates realistic deployment where ground truth isn't available

**History Initialization:**
```python
model.reset_inference_state()
model.init_from_train_history(
    graph_dict=...,          # Graphs from training timesteps
    entity_history=...,      # Per-entity interaction history
    entity_history_t=...,    # Timestamps for each history entry
    global_emb=...,          # Precomputed global embeddings
)
```

This ensures the model has correct context from training data before making predictions on validation/test timesteps.

---

## 10. What is the Trajectory Prediction model, and how does it differ from the main RAPID model?

**Fundamental Difference:** Instead of predicting one timestep at a time autoregressively, the Trajectory model predicts the **entire future trajectory** in one shot.

**Architecture** ([trajectory.py](file:///Users/taylansahin/Projects/RAPID/src/models/trajectory.py)):

```
Input: Target edge history [t-H, ..., t-1] + Neighbor edge histories
Output: Future trajectory [t, t+1, ..., t+K] predictions

1. EdgeHistoryEncoder (Transformer):
   - Project binary sequence to d_model
   - Add sinusoidal positional encodings
   - Pass through Transformer encoder
   - Mean-pool to single vector

2. NeighborCrossAttention:
   - Query: encoded target edge
   - Key/Value: encoded neighbor edges
   - Target edge attends to its spatial neighbors

3. TrajectoryDecoder (Transformer Decoder):
   - Learnable positional queries for each future timestep
   - Memory: refined edge embedding
   - Outputs K logits for future trajectory
```

**Neighbors:** All edges sharing a node with the target edge (u,v)—i.e., edges involving u or v.

**Training** ([train_trajectory.py](file:///Users/taylansahin/Projects/RAPID/src/train_trajectory.py)):
- BCE loss over full trajectory
- **Transition weighting:** Timesteps where state changes get higher weight
- `transition_weight` parameter controls upweighting (default 10x)

---

## 11. How is data split, and what temporal constraints are enforced?

**Split Strategy** ([preprocessing.py](file:///Users/taylansahin/Projects/RAPID/src/data/preprocessing.py#L244-L268)):
- **Chronological split** based on timestep index
- Default: 60% train, 20% validation, 20% test
- All samples at time t are in exactly one split

**Why Chronological?**
- Temporal data violates i.i.d. assumptions
- Random splits would cause leakage (model sees future to predict past)
- Realistic: in deployment, we predict future from past

**Leakage Prevention:**
1. Graphs built from training data only
2. Entity histories computed from training data only
3. Node features use `train_cutoff` to exclude val/test info
4. Global embeddings indexed to prevent accessing future graphs

---

## 12. How are the entity embeddings and relation embeddings initialized and used?

**Entity Embeddings** ([rapid.py](file:///Users/taylansahin/Projects/RAPID/src/models/rapid.py#L85-L90)):
```python
self.entity_embed = nn.Embedding(num_entities, hidden_dim)
```
- Learned from scratch during training
- Each residue gets a unique embedding

**Node Feature Enhancement** ([rapid.py](file:///Users/taylansahin/Projects/RAPID/src/models/rapid.py#L134-L150)):
```python
# If node features provided:
entity_emb = self.entity_embed(entity_ids)
node_feat = self.node_feat_proj(self.node_features[entity_ids])
entity_emb = entity_emb + node_feat  # Additive combination
```
The 8-dimensional node features are projected to `hidden_dim` and **added** to the learned embeddings.

**Relation Embeddings:**
```python
self.relation_embed = nn.Embedding(num_rels, hidden_dim)
```
Currently single relation (contact), but architecture supports multiple relation types for future extensions (e.g., different contact types, distances).

The mean relation embedding `R̄` is concatenated to each timestep's feature vector in the temporal sequence.

---

## 13. What are the key hyperparameters and their defaults?

**Model Configuration** ([config.py](file:///Users/taylansahin/Projects/RAPID/src/config.py)):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 200 | Embedding and hidden layer dimension |
| `num_rgcn_layers` | 2 | RGCN layers per timestep |
| `num_bases` | 100 | Basis functions for RGCN weight decomposition |
| `seq_len` | 10 | History sequence length |
| `classifier_hidden_dim` | 128 | MLP hidden dimension |
| `dropout` | 0.2 | Dropout rate |

**Training Configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-3 | Adam learning rate |
| `weight_decay` | 1e-5 | L2 regularization |
| `grad_clip_norm` | 1.0 | Gradient clipping |
| `max_epochs` | 100 | Maximum training epochs |
| `patience` | 10 | Early stopping patience |
| `focal_gamma` | 2.0 | Focal loss focusing parameter |

**Data Configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `neg_ratio` | 1.0 | Negative samples per positive |
| `hard_ratio` | 0.5 | Fraction of negatives that are hard |
| `batch_size` | 128 | Training batch size |

---

## 14. How does the RGCN handle undirected graphs?

**The Challenge:** Standard RGCN assumes directed graphs with separate edge types for forward and reverse relations. PPI contacts are fundamentally undirected.

**Solution** ([rgcn.py](file:///Users/taylansahin/Projects/RAPID/src/models/rgcn.py#L246-L295)):

```python
def build_undirected_graph(edges, rel_types, num_nodes):
    # Add forward edges
    g.add_edges(src, dst)
    
    # Add reverse edges with SAME relation type
    g.add_edges(dst, src)
    
    # Symmetric normalization: degree^(-1) for each node
    # Applied to both directions equally
```

**Key Design Decisions:**
1. No separate inverse relation types—same relation used in both directions
2. Degree normalization applied symmetrically (`A + A^T` style)
3. Canonical edge ordering `(min_id, max_id)` throughout the codebase ensures consistency

---

## 15. What experiments have been run to validate design choices?

Based on the conversation history and analysis documentation, key experiments include:

1. **Temporal Ablation:** Verified temporal embeddings provide useful signal vs. static predictions

2. **Persistence Correlation:** Diagnosed that baseline model primarily predicts persistence (same as t-1)

3. **Transition Weighting:** Tested various `transition_weight` values (1x, 5x, 10x, 20x) to balance edge accuracy with transition detection

4. **2-Hop Neighbors:** Explored including neighbors-of-neighbors in trajectory model for richer context

5. **Attention vs. GRU Encoder:** Compared `AttentionTemporalEncoder` against standard GRU for temporal modeling

6. **Hard Ratio Effect:** Evaluated impact of `hard_ratio` parameter on model precision/recall trade-off

**Tracking:** Results are logged to `checkpoints/` and `analysis_outputs/` directories with per-experiment subdirectories.

---

## Summary Table: Key Design Decisions

| Aspect | RAPID Choice | Rationale |
|--------|--------------|-----------|
| **Graph Type** | Undirected | PPI contacts are symmetric |
| **Negative Sampling** | 50% hard + 50% easy | Balance challenge with diversity |
| **Temporal Encoding** | GRU (default) / Transformer | Capture sequential patterns |
| **Loss Function** | Focal Loss | Handle class imbalance |
| **Primary Metric** | AUPRC | Best for imbalanced classification |
| **Prediction Target** | Edge existence (default) / Transition | Can focus on dynamics explicitly |
| **Evaluation Candidates** | History-constrained | Focus on dynamics, not discovery |
| **Node Features** | Physicochemical + structural | Biologically meaningful priors |
| **Global Context** | Optional hierarchical model | Capture interface-level trends |

---

*This document covers the default configuration of RAPID. Feature flags (`--use_transition_prediction`, `--use_attention_encoder`, `--use_global_model`) and alternative architectures (Trajectory model) modify various aspects as described above.*
