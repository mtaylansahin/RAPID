# RAPID Architecture Improvements: Implementation and Experimental Analysis

**Date:** 2026-01-11  
**Authors:** Automated Implementation Session  
**Purpose:** Document problem, solution approach, implementation details, and experimental results for external verification.

---

## 1. Original Problem Statement

### 1.1 Context
RAPID (Recurrent Architecture for Predicting Protein Interaction Dynamics) is a temporal graph neural network for predicting protein-protein interactions over time. The model uses:
- RGCN for graph structure encoding
- GRU-based temporal encoding for interaction history
- Binary classification for edge prediction

### 1.2 Problem Identified
Analysis of the baseline model revealed a critical issue:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Persistence Accuracy | 99.4% | Model predicts "same as yesterday" almost always |
| Transition MCC | 0.0433 | Barely better than random at detecting state changes |
| Forming Recall | 0.64% | Catches almost no OFF→ON transitions |
| Breaking Recall | 2.93% | Catches almost no ON→OFF transitions |
| Standard AUPRC | 0.69 | Reasonable overall edge prediction |
| Standard F1 | 0.61 | Reasonable overall edge prediction |

**Core Issue:** The model achieves decent standard metrics by exploiting temporal autocorrelation (predicting persistence) rather than learning to predict actual dynamic changes. This is problematic because:
1. The primary research goal is predicting *dynamics*, not static states
2. High standard metrics mask the model's failure to capture what matters

### 1.3 Root Cause Analysis
1. **Class imbalance**: Transitions are rare (~5-10% of samples), so predicting persistence is often "correct"
2. **Lack of explicit transition signal**: Model trained on edge existence, not edge changes
3. **Limited temporal context**: GRU may not capture long-range dependencies needed for transition patterns

---

## 2. Solution Approach

We implemented 5 architectural changes to address the dynamics prediction problem:

| Change | Hypothesis |
|--------|-----------|
| 1. Transition Metrics | Establish baseline by measuring transition-specific performance |
| 2. Transition Classifier | Reformulate prediction from P(edge) to P(transition) |
| 3. Transformer Global Model | Better long-range temporal dependencies via attention |
| 4. Graph Reconstruction Pretraining | Edge prediction pretraining may improve representations |
| 5. Attention Temporal Encoder | Replace GRU with attention for entity history encoding |

---

## 3. Implementation Details

### 3.1 Change 1: Transition Metrics

**Files Modified:**
- `src/metrics/__init__.py` - Added `compute_transition_metrics()` function
- `src/evaluate.py` - Integrated transition metrics into evaluation pipeline

**Key Implementation:**
```python
# Transition detection: compare consecutive timestep predictions
is_transition = (was_on_t_minus_1 != labels)  # True if state changed
predicted_transition = (was_on_t_minus_1 != predictions)

# Matthews Correlation Coefficient for transitions
mcc = matthews_corrcoef(is_transition, predicted_transition)
```

**Metrics Added:**
- Transition MCC (Matthews Correlation Coefficient)
- Transition F1, Precision, Recall
- Forming Recall (OFF→ON detection rate)
- Breaking Recall (ON→OFF detection rate)
- Persistence Accuracy (how often model predicts "same as before")

---

### 3.2 Change 2: Transition Prediction Reformulation

**Files Modified:**
- `src/models/classifier.py` - Added `TransitionEdgeClassifier`, `SymmetricTransitionClassifier`
- `src/data/dataset.py` - Added `was_on_t_minus_1`, `is_transition` computation
- `src/models/rapid.py` - Conditional classifier selection in `__init__`
- `src/train.py` - Updated training loop for transition mode
- `src/config.py` - Added `use_transition_prediction` flag

**Key Design Decisions:**

1. **Classifier Architecture:**
```python
class TransitionEdgeClassifier(nn.Module):
    """
    Predicts P(transition) and converts to P(edge) for evaluation.
    
    Input: entity embeddings + was_on_t_minus_1 indicator
    Output: edge existence logits (not transition logits)
    """
    def get_edge_logits(self, e1_embed, e2_embed, was_on):
        transition_logits = self.forward(e1_embed, e2_embed, was_on, ...)
        # Convert: P(edge) = was_on XOR P(transition)
        # Implemented via logit manipulation
```

2. **Data Pipeline Changes:**
```python
# In _collate_fn:
# Track which edges existed at t-1
was_on_t_minus_1 = []
for (e1, e2) in pairs:
    key = (min(e1, e2), max(e1, e2))
    was_on = 1.0 if key in self.edges_at_t.get(t-1, set()) else 0.0
    was_on_t_minus_1.append(was_on)
```

3. **Autoregressive Inference:**
```python
# In predict_batch (evaluation):
# Use predicted states from t-1, not ground truth
was_on = 1.0 if edge in self._prediction_cache[t-1] else 0.0
```

**CLI Flag:** `--use_transition_prediction`

---

### 3.3 Change 3: Transformer Global Model

**Files Modified:**
- `src/models/global_model.py` - Added transformer encoder option to `PPIGlobalModel`
- `src/pretrain.py` - Updated checkpoint to save encoder_type
- `main.py` - Added `--global_encoder_type` CLI argument

**Key Implementation:**
```python
# In PPIGlobalModel.__init__:
if encoder_type == "transformer":
    self.pos_encoding = nn.Parameter(torch.randn(seq_len, hidden_dim) * 0.1)
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 4,
        dropout=dropout, activation="gelu", batch_first=True
    )
    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
else:
    self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
```

**CLI Flag:** `--global_encoder_type [gru|transformer]`

---

### 3.4 Change 4: Graph Reconstruction Pretraining

**Files Modified:**
- `src/models/global_model.py` - Added `edge_predictor` MLP, `edge_prediction_forward()` method
- `src/pretrain.py` - Added `train_global_model_edges()` function
- `main.py` - Added `--pretrain_mode` CLI argument

**Key Implementation:**
```python
# Edge prediction pretraining loss
def edge_prediction_forward(self, timestep, pos_edges, neg_edges, graph_dict):
    global_emb = self.predict(timestep, graph_dict)  # Uses history < t
    
    # Score positive and negative edges
    pos_logits = self.edge_predictor(cat([e1_embed, e2_embed, global_emb]))
    neg_logits = self.edge_predictor(cat([neg_e1_embed, neg_e2_embed, global_emb]))
    
    return F.binary_cross_entropy_with_logits(logits, labels)
```

**CLI Flag:** `--pretrain_mode [entity|edges]`

---

### 3.5 Change 5: Attention Temporal Encoder

**Files Modified:**
- `src/models/encoder.py` - Added `AttentionTemporalEncoder` class
- `src/models/rapid.py` - Conditional encoder selection
- `src/config.py` - Added `use_attention_encoder` flag
- `src/train.py` - Added to checkpoint save

**Key Implementation:**
```python
class AttentionTemporalEncoder(nn.Module):
    """Attention-based encoder with CLS token for sequence aggregation."""
    
    def __init__(self, input_dim, hidden_dim, num_layers=2, num_heads=4, ...):
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len+1, hidden_dim) * 0.02)
        self.transformer = nn.TransformerEncoder(...)
    
    def forward(self, packed_sequence):
        # Unpack, prepend CLS, add positional encoding
        # Return CLS token representation
        return x[:, 0]  # CLS token output
```

**CLI Flag:** `--use_attention_encoder`

---

## 4. Experimental Results

### 4.1 Baseline Establishment (Change 1)

**Command:**
```bash
python main.py all --dataset 1JPS --epochs 5 --patience 3 --experiment_name baseline
```

**Results:**
| Metric | Value |
|--------|-------|
| AUPRC | 0.69 |
| F1 | 0.61 |
| AUROC | 0.82 |
| Transition MCC | 0.04 |
| Forming Recall | 0.6% |
| Breaking Recall | 2.9% |
| Persistence Accuracy | 99.4% |

---

### 4.2 Transition Classifier (Change 2)

**Command:**
```bash
python main.py all --dataset 1JPS --epochs 5 --patience 3 --use_transition_prediction \
    --experiment_name transition_classifier_v3
```

**Results:**
| Metric | Baseline | Transition Classifier | Delta |
|--------|----------|----------------------|-------|
| AUPRC | 0.69 | **0.77** | +12% ⬆️ |
| F1 | 0.61 | **0.69** | +13% ⬆️ |
| AUROC | 0.82 | **0.86** | +5% ⬆️ |
| MCC | 0.04 | 0.00 | -100% ⬇️ |
| Forming R | 0.6% | 0.0% | -100% |
| Breaking R | 2.9% | 0.0% | -100% |

**Interpretation:** Standard metrics improved significantly, but transition detection dropped to zero. The model learned to use `was_on_t_minus_1` as a strong feature for edge prediction without actually learning to predict transitions.

---

### 4.3 Transformer Global Model (Change 3)

**Command:**
```bash
python main.py all --dataset 1JPS --epochs 5 --patience 3 --use_global_model \
    --global_encoder_type transformer --experiment_name transformer_global_test
```

**Results:**
| Metric | Baseline | Transformer Global | Delta |
|--------|----------|-------------------|-------|
| AUPRC | 0.69 | 0.70 | +1% |
| F1 | 0.61 | 0.66 | +8% |
| AUROC | 0.82 | 0.82 | 0% |
| MCC | 0.04 | **0.07** | +75% ⬆️ |
| Forming R | 0.6% | **1.6%** | +167% ⬆️ |
| Breaking R | 2.9% | **4.3%** | +48% ⬆️ |

**Interpretation:** Only change that improved transition metrics. Transformer attention in the global model captures temporal patterns better than GRU.

---

### 4.4 Edge Prediction Pretraining (Change 4)

**Command:**
```bash
python main.py all --dataset 1JPS --epochs 5 --patience 3 --use_global_model \
    --pretrain_mode edges --experiment_name edge_pretrain_test
```

**Results:**
| Metric | Baseline | Edge Pretrain | Delta |
|--------|----------|--------------|-------|
| AUPRC | 0.69 | 0.71 | +3% |
| MCC | 0.04 | 0.02 | -50% |
| Forming R | 0.6% | 1.2% | +100% |
| Breaking R | 2.9% | 0.4% | -86% |

**Interpretation:** Slight AUPRC improvement but transition metrics degraded. Edge prediction pretraining doesn't help dynamics.

---

### 4.5 Attention Encoder (Change 5)

**Command:**
```bash
python main.py all --dataset 1JPS --epochs 3 --patience 2 --use_attention_encoder \
    --experiment_name attention_encoder_test
```

**Results:**
| Metric | Baseline | Attention Encoder | Delta |
|--------|----------|------------------|-------|
| AUPRC | 0.69 | 0.66 | -4% |
| MCC | 0.04 | 0.04 | 0% |
| Model Params | 1.12M | 1.65M | +47% |

**Interpretation:** Comparable to GRU with 50% more parameters. No benefit observed.

---

### 4.6 Combined All Changes

**Command:**
```bash
python main.py all --dataset 1JPS --epochs 20 --patience 5 --seq_len 15 \
    --use_global_model --global_encoder_type transformer \
    --use_transition_prediction --use_attention_encoder \
    --experiment_name combined_all_changes
```

**Results:**
| Metric | Baseline | Combined | Delta |
|--------|----------|----------|-------|
| AUPRC | 0.69 | **0.77** | +12% ⬆️ |
| F1 | 0.61 | **0.74** | +21% ⬆️ |
| AUROC | 0.82 | **0.86** | +5% |
| Accuracy | - | 0.84 | - |
| MCC | 0.04 | **-0.01** | ❌ |
| Forming R | 0.6% | 0.0% | -100% |
| Breaking R | 2.9% | 0.0% | -100% |
| Persistence Acc | 99.4% | 99.97% | +0.5% |

**Interpretation:** Excellent standard metrics (best AUPRC/F1), but transition detection completely failed. The transition classifier's `was_on_t_minus_1` input dominates, causing extreme persistence behavior.

---

## 5. Summary and Conclusions

### 5.1 What Worked

| Change | Best Improvement |
|--------|-----------------|
| Transition Classifier (Change 2) | **AUPRC +12%** (0.69 → 0.77) |
| Transformer Global (Change 3) | **MCC +75%** (0.04 → 0.07) |

### 5.2 What Didn't Work

- **Edge Pretraining (Change 4):** No benefit for transitions
- **Attention Encoder (Change 5):** No improvement, 50% more parameters
- **Combined Approach:** Transition classifier dominates and kills transition detection

### 5.3 Key Insights

1. **Transition Classifier Paradox:** Adding `was_on_t_minus_1` as input improves standard metrics dramatically but causes the model to simply predict persistence. The conversion from P(transition) → P(edge) works mathematically but the model learns the shortcut.

2. **Best for Dynamics:** The Transformer Global Model (Change 3) alone is the only configuration that improved transition detection (MCC 0.04 → 0.07, +75%).

3. **Metric Discrepancy:** High AUPRC/F1 can mask complete failure at the actual task (dynamics prediction). Transition-specific metrics are essential.

4. **Persistence Shortcut:** The model consistently finds that predicting "same as yesterday" is the path of least resistance. Breaking this pattern may require:
   - Transition-weighted loss functions
   - Curriculum learning
   - Explicit transition supervision

### 5.4 Recommended Configuration

For **dynamics prediction** (the research goal):
```bash
python main.py all --dataset <dataset> --use_global_model --global_encoder_type transformer
```

For **standard edge prediction** (if dynamics don't matter):
```bash
python main.py all --dataset <dataset> --use_transition_prediction
```

---

## 6. Verification Checklist

For external verification, reviewers should:

1. **Verify Baseline:**
   - Run baseline training and confirm ~99% persistence accuracy
   - Confirm low transition MCC (~0.04)

2. **Verify Transition Classifier:**
   - Check that `was_on_t_minus_1` flows correctly through the pipeline
   - Verify `get_edge_logits()` conversion logic
   - Confirm autoregressive inference uses predictions, not ground truth

3. **Verify Transformer Global:**
   - Check positional encoding is applied
   - Verify attention mask creation for variable-length sequences

4. **Check for Data Leakage:**
   - Evaluation must be autoregressive (no ground truth at t-1)
   - Global embeddings for timestep t use only graphs < t
   - Transition labels use previous timestep state

5. **Reproduce Key Results:**
   - Transformer Global should show MCC improvement
   - Transition Classifier should show AUPRC improvement but MCC=0

---

## 7. Files Modified Summary

| File | Changes |
|------|---------|
| `src/config.py` | Added `use_transition_prediction`, `use_attention_encoder` |
| `src/models/classifier.py` | Added `TransitionEdgeClassifier`, `SymmetricTransitionClassifier` |
| `src/models/encoder.py` | Added `AttentionTemporalEncoder` |
| `src/models/global_model.py` | Added transformer encoder, edge prediction |
| `src/models/rapid.py` | Conditional classifier/encoder selection |
| `src/data/dataset.py` | Added `was_on_t_minus_1`, `is_transition`, `edges_at_t` |
| `src/train.py` | Transition mode, checkpoint updates |
| `src/pretrain.py` | Edge prediction training, encoder_type saving |
| `src/evaluate.py` | Transition metrics integration |
| `main.py` | All CLI flags, load_global_model update |

---

*Document generated: 2026-01-11*
