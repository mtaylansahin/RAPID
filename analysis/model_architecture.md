# RAPID Model Architecture Documentation

This document details the model architectures used in the RAPID (Recurrent Architecture for Predicting Protein Interaction Dynamics) project. It covers two distinct configurations:
1.  **RAPID Combined Model**: The architecture initialized by the command `python main.py all ... --use_global_model --global_encoder_type transformer --use_transition_prediction --use_attention_encoder`.
2.  **Trajectory Prediction Model**: The architecture defined in `src/train_trajectory.py`.

---

## 1. RAPID Combined Model
**Command Context:** `combined_optimized` experiment.

This configuration represents the most advanced version of RAPID, integrating a hierarchical global context, attention-based temporal encoding, and a transition-based prediction objective.

### 1.1 High-Level Data Flow

```mermaid
graph TD
    subgraph Inputs
    E[Entity Pairs (u,v)]
    H[History Graphs G_{t-k}...G_{t-1}]
    end

    subgraph "Global Context Module"
    H -- "Per Timestep" --> AGG[Global Aggregator (RGCN + MaxPool)]
    AGG --> GS[Graph Sequence g_{t-k}...g_{t-1}]
    GS --> GENC[Global Transformer Encoder]
    GENC --> C_T[Global Context Vector c_t]
    end

    subgraph "Local Context Module"
    H -- "Per Timestep" --> L_RGCN[Undirected RGCN]
    L_RGCN --> L_H[Entity Features h_u(t), h_v(t)]
    end

    subgraph "Temporal Encoding"
    L_H --> SEQ[Sequence Construction]
    C_T --> SEQ
    SEQ -- "[h_local, h_static, h_rel, c_t]" --> ATT_ENC[Attention Temporal Encoder]
    ATT_ENC --> TEMP_U[Temporal Emb z_u]
    ATT_ENC --> TEMP_V[Temporal Emb z_v]
    end

    subgraph "Prediction Head (Transition)"
    TEMP_U & TEMP_V --> CLF[Symmetric Transition Classifier]
    CLF --> T_LOGITS[Transition Logits]
    T_LOGITS --> |Sigmoid| P_TRANS[P(Transition)]
    
    subgraph "Dual Objectve"
    P_TRANS -- "Loss 1" --> L_TRANS[Transition Loss]
    P_TRANS & PREV_STATE[State_{t-1}] -- "XOR" --> P_EDGE[P(Edge)]
    P_EDGE -- "Loss 2" --> L_EDGE[Edge Reconstruction Loss]
    end
    end
```

### 1.2 Component Details

#### A. Global Context Module (`src/models/global_model.py`)
This module captures the "macroscopic" state of the protein interaction network to inform local predictions.
*   **Input:** Sequence of full PPI graphs from previous timesteps.
*   **Aggregator (`GlobalRGCNAggregator`):**
    *   Applies a 2-layer **Undirected RGCN** to each historical graph snapshot.
    *   **Pooling:** Uses `max` pooling over all nodes to derive a fixed-size graph embedding $g_t \in \mathbb{R}^{d_{model}}$ for each timestep.
*   **Global Encoder:**
    *   **Type:** `TransformerEncoder` (via `--global_encoder_type transformer`).
    *   **Operation:** Processes the sequence of graph embeddings $\{g_{t-seq\_len}, \dots, g_{t-1}\}$.
    *   **Positional Encoding:** Learned positional embeddings are added.
    *   **Output:** The final hidden state serves as the global context vector $c_t$ for the current prediction step.

#### B. Local & Temporal Encoding (`src/models/rapid.py`, `src/models/encoder.py`)
*   **Input Sequence Construction:** For each entity $u$ and history step $\tau$:
    $$x_u^{(\tau)} = [RGCN(G_\tau)_u ~~||~~ E_u ~~||~~ \bar{R} ~~||~~ c_\tau]$$
    *   $RGCN(G_\tau)_u$: Local structural embedding from RGCN at time $\tau$.
    *   $E_u$: Static learnable entity embedding.
    *   $\bar{R}$: Mean relation embedding (context constant).
    *   $c_\tau$: Global context vector at time $\tau$.
    *   **Total Dimension:** $4 \times d_{hidden}$.

*   **Attention Temporal Encoder:**
    *   **Type:** `AttentionTemporalEncoder` (via `--use_attention_encoder`).
    *   **Mechanism:** Transformer Encoder with a **CLS token**.
    *   The sequence of feature vectors $x_u^{(\tau)}$ is fed into the transformer.
    *   **Output:** The final representation corresponding to the CLS token is used as the entity's temporal embedding $z_u$. This replaces the standard GRU hidden state.

#### C. Transition Prediction Head (`src/models/classifier.py`)
*   **Module:** `SymmetricTransitionClassifier` (via `--use_transition_prediction`).
*   **Input:** Pair of temporal embeddings $(z_u, z_v)$ and static embeddings $(E_u, E_v)$.
*   **Structure:** MLP that predicts a scalar logit.
*   **Symmetry:** $\text{Score}(u,v) = \frac{1}{2}(\text{MLP}(u,v) + \text{MLP}(v,u))$.
*   **Semantic:** The output represents the log-odds of a **state transition** (change from 0 to 1, or 1 to 0) occurring at time $t$, *not* the absolute state.

#### D. Training Objective (`src/train.py`)
The model is trained using a **Dual-Task Loss**:
1.  **Transition Loss ($L_{trans}$):** BCE loss on the predicted transition probability vs. actual state change magnitude. Includes upweighting (5.0x) for positive transitions.
2.  **Edge Reconstruction Loss ($L_{edge}$):**
    *   The transition prediction is differentiated through an XOR-like operation with the previous ground-truth state ($y_{t-1}$):
        $$ \hat{y}_t = \hat{p}_{trans} \cdot (1 - 2y_{t-1}) \quad (\text{Logit Space Logic}) $$
    *   If $y_{t-1}=0$ (was off), $\hat{y}_t = \hat{p}_{trans}$ (predicts turning on).
    *   If $y_{t-1}=1$ (was on), $\hat{y}_t = 1 - \hat{p}_{trans}$ (predicts staying on / not turning off).
    *   Standard BCE is applied to this reconstructed edge prediction against the true target $y_t$.
*   **Total Loss:** $0.4 \cdot L_{edge} + 0.6 \cdot L_{trans}$.

---

## 2. Trajectory Prediction Architecture
**Source:** `src/train_trajectory.py` & `src/models/trajectory.py`.

This model radically differs from the main RAPID model. Instead of predicting the next timestep autoregressively, it predicts the **entire future trajectory** $(t+1, \dots, t+K)$ in one shot, conditioned on the history of the target edge and its spatial neighbors.

### 2.1 High-Level Data Flow

```mermaid
graph TD
    subgraph Inputs
    Target[Target Edge Hist h_{uv}]
    Neighbors[Neighbor Edge Hists {h_{nb}}]
    end

    subgraph "Encoders"
    Target --> TE[Edge History Transformer] --> Z_target
    Neighbors --> NE[Edge History Transformer] --> Z_neighbors
    end

    subgraph "Interaction"
    Z_target & Z_neighbors --> ATT[Neighbor Cross-Attention]
    ATT -- "Target attends to Neighbors" --> Z_refined
    end

    subgraph "Decoder"
    Q[Learnable Position Queries t+1...t+K]
    Z_refined --> DEC[Trajectory Transformer Decoder]
    Q --> DEC
    DEC --> TRAJ[Future Trajectory Logits]
    end
```

### 2.2 Component Details

#### A. Edge History Encoder (`EdgeHistoryEncoder`)
*   **Input:** Binary sequence of edge existence $\{0, 1\}^T$.
*   **Processing:**
    1.  Projects binary input to $d_{model}$.
    2.  Adds sinusoidal positional encodings.
    3.  Passes through a **Transformer Encoder**.
    4.  **Pooling:** Mean pooling over the sequence dimension to get a single vector representation of the edge's history.

#### B. Neighbor Interaction (`NeighborCrossAttention`)
*   **Context:** The model retrieves the histories of "neighboring edges" (edges that share a node with $u$ or $v$).
*   **Cross-Attention:**
    *   **Query:** The encoded history of the target edge $(u,v)$.
    *   **Key/Value:** The encoded histories of all neighbor edges.
    *   **Mechanism:** `MultiheadAttention` allows the target edge to "look at" dynamic patterns in its spatial vicinity to refine its own representation.

#### C. Trajectory Decoder (`TrajectoryDecoder`)
*   **Type:** Transformer Decoder.
*   **Queries:** A set of **Learnable Positional Embeddings**, one for each future timestep to be predicted $(1 \dots K)$.
*   **Memory:** The refined edge embedding (from Cross-Attention) serves as the Key/Value for the decoder.
*   **Function:** The decoder learns to map the static "edge context" into a dynamic future sequence.
*   **Output:** A sequence of logits of length $K$, predicting the probability of the edge being active at each future step.

#### D. Training Logic (`src/train_trajectory.py`)
*   **Loss:** Binary Cross Entropy (BCE) over the full predicted sequence.
*   **Transition Weighting:**
    *   The model explicitly upweights timesteps in the target sequence where a state change occurs.
    *   `weights = 1.0 + (transition_weight - 1.0) * is_transition`
    *   This forces the model to focus on correctly predicting the *moments of change* rather than just minimizing error on static periods (persistence).
