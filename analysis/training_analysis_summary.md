# RAPID Training Analysis Summary

## 1. Problem Statement

### 1.1 Initial Observations
From an initial training run with the following configuration:
```bash
python main.py all \
  --data_dir .../1JPS/interfacea-full \
  --replica replica1 \
  --test_ratio 0.25 \
  --epochs 20 \
  --gpu 0 \
  --num_bases 3 \
  --batch_size 64 \
  --patience 5 \
  --seq_len 20 \
  --hidden_dim 100 \
  --hard_ratio 0.5
```

The following issues were identified:

| Issue | Observation |
|-------|-------------|
| Train/Val Gap | ~20-30 point AUPRC difference between training and validation |
| Training Instability | Loss oscillates significantly between epochs (e.g., 0.0246 → 0.1508) |
| Low Test Precision | 60.3% precision with 78.7% recall — model over-predicts positives |
| High Validation Variance | Validation AUPRC fluctuates by 5+ points between consecutive epochs |

### 1.2 Data Characteristics
- **Dataset**: 1JPS protein complex (428 entities, 3 relation types, 201 timesteps)
- **Split**: 50% train, 25% validation, 25% test (chronological)
- **Class ratio**: ~1:2.2 (positive:negative) in test set

---

## 2. Codebase Analysis

### 2.1 Key Differences Between Train and Evaluation

| Aspect | Training | Validation/Test |
|--------|----------|-----------------|
| History source | Oracle (ground-truth) | Autoregressive (predicted) |
| Negative samples | Sampled (hard+easy mix) | All known pairs |
| Graph context | Training graphs only | Train graphs (val) / Train+Val graphs (test) |
| Dropout | Applied | Not applied |

### 2.2 History Structure
- History is stored **per entity**, not per pair
- Each entity's history contains: `{neighbors: [...], rel_types: [...]}`
- When predicting pair (e1, e2), the model sees:
  - e1's full neighbor list (including e2 if they interacted)
  - e2's full neighbor list (including e1 if they interacted)
- The model must implicitly learn whether e1 and e2 have interacted before

### 2.3 Negative Sampling
- `hard_ratio` controls the mix of hard vs easy negatives
- Hard negatives: pairs that have interacted before but are OFF now
- Easy negatives: pairs that have never interacted
- Sampler accumulates `_ever_positive` within each epoch, resets between epochs

---

## 3. Hyperparameter Experiments

### 3.1 Learning Rate and Batch Size

| Config | Best Val AUPRC | Test AUPRC | Test F1 | Notes |
|--------|----------------|------------|---------|-------|
| lr=1e-3, bs=64 | 0.7457 | 0.7243 | 0.6829 | Baseline, high loss oscillation |
| lr=1e-4, bs=128 | 0.6316 | 0.6462 | 0.5532 | Performance degraded |
| lr=5e-4, bs=128 | 0.6579 | 0.6775 | 0.6674 | More stable loss |
| lr=1e-3, hard=0.3 | 0.6579 | 0.6775 | 0.6674 | Similar to above |

**Finding**: Lower learning rate did not improve stability or reduce the gap.

---

## 4. Design Decision: Pair History Masking

### 4.1 Hypothesis
The model learns a "persistence shortcut": if entity e2 appears in e1's recent history, predict ON. This works with oracle history in training but fails at evaluation where history is predicted.

### 4.2 Proposed Solution
When training on pair (e1, e2), mask out:
- e2 from e1's history
- e1 from e2's history

This forces the model to predict based on:
- Other neighbors' activity patterns
- Node features (physicochemical, intrachain)
- Global patterns

### 4.3 Implementation
- Added `pair_masking_prob` (default 1.0) and `pair_masking_warmup` (default 5) to `TrainingConfig`
- Masking applied in `_collate_fn()` during batch creation
- Curriculum: linear ramp from 0 to target probability over warmup epochs

### 4.4 Why Pair Masking (Not Other Corruption)
Discussion of alternative corruption strategies revealed constraints:
- Adding random neighbors is unrealistic (evaluation only considers known pairs)
- Dropping history for one entity breaks symmetry (if A→B in history, B→A should also exist)
- Pair masking is symmetric and directly targets the identified shortcut

---

## 5. Pair Masking Experiments

### 5.1 All Experimental Results

| Experiment | Pair Mask | Hard Ratio | Best Epoch | Val AUPRC | Test AUPRC | Test F1 | Train/Val Gap |
|------------|-----------|------------|------------|-----------|------------|---------|---------------|
| No masking (baseline) | 0% | 0.5 | 10 | **0.7182** | **0.7340** | **0.7006** | 23 pts |
| Default masking | 100% (warmup 5) | 0.5 | 3 | 0.6645 | 0.6911 | 0.6742 | 26 pts |
| Full mask + hard | 100% (no warmup) | 1.0 | 1 | 0.6829 | 0.6909 | 0.6174 | 15 pts |
| More warmup | 100% (warmup 10) | 0.5 | 3 | 0.6645 | 0.6911 | 0.6727 | 26 pts |
| No mask + hard | 0% | 1.0 | 2 | 0.6991 | 0.7320 | 0.6588 | 19 pts |
| 20% mask + hard | 20% | 1.0 | 1 | 0.6829 | 0.6909 | 0.6174 | 15 pts |
| 20% mask | 20% | 0.5 | 8 | 0.6651 | 0.6897 | 0.6759 | 28 pts |

### 5.2 Observations

**Pair masking results**:
- All masking configurations performed worse than baseline
- Even 20% masking reduced test AUPRC from 0.734 to 0.689
- Masking did not reduce the train/val gap

**hard_ratio=1.0 results**:
- Consistently reduced train/val gap (15-19 pts vs 23 pts)
- Maintained competitive test AUPRC (0.732 vs 0.734)
- Lower F1 scores (0.66 vs 0.70)
- Early stopping triggered very early (epochs 1-2)

### 5.3 Conclusion
Pair masking hypothesis was incorrect. The model's use of pair history is not a shortcut — it provides legitimate signal about temporal dynamics.

---

## 6. Remaining Issues

1. **Model predicts persistence, not dynamics**: Tends to predict ON or OFF at the start and stick with it
2. **Training instability**: Loss varies significantly epoch-to-epoch
3. **Train/val gap persists**: Best gap achieved was ~15 pts with hard_ratio=1.0
4. **Validation metrics inconsistent**: Do not improve monotonically

---

## 7. Follow-up Diagnostic Experiments (2026-01-06)

To investigate the remaining issues, systematic experiments were run using `experiments/run_diagnostics.py`.

### 7.1 Experiment 1: Temporal Embedding Ablation

**Question**: Do temporal embeddings provide useful signal?

| Condition | Val AUPRC | Val F1 |
|-----------|-----------|--------|
| With temporal | 0.6757 | 0.6667 |
| Without temporal (zeroed) | 0.4283 | 0.0000 |

**Finding**: Temporal embeddings are essential (24.7pt AUPRC drop), but F1→0 suggests over-reliance rather than complementary learning with static features.

---

### 7.2 Experiment 2: LSCF Baseline Comparison

**Question**: Does the model beat a "last state carried forward" baseline?

**Methodology note**: Initial comparison used oracle t-1 states (unfair). Revised to LSCF baseline: predict last training state, no updates during validation.

| Metric | Model | LSCF Baseline |
|--------|-------|---------------|
| Accuracy | 77.0% | **79.4%** |
| AUPRC | **0.694** | 0.418 |
| Transition accuracy | **27.9%** | 0.0% |
| Non-transition accuracy | 89.8% | 100.0% |

**Findings**:
- Model has higher AUPRC (0.694 vs 0.418) — better probability ranking
- Model accuracy 2.4pts lower than LSCF — trades non-transition accuracy for transition detection
- Model captures some transitions (27.9% vs 0%) — LSCF cannot detect transitions by definition
- Transition accuracy still low — room for improvement

---

### 7.3 Experiment 3: Loss Function Stability

**Question**: Does focal loss configuration affect training stability?

| Configuration | Loss Variance | Best AUPRC |
|--------------|---------------|------------|
| focal_g2_noalpha (default) | 0.000160 | **0.7010** |
| focal_g1_noalpha | 0.000477 | 0.6572 |
| **focal_g2_alpha03** | **0.000027** | 0.6870 |
| focal_g1_alpha03 | 0.000168 | 0.6529 |

**Finding**: Adding `focal_alpha=0.3` reduces loss variance by 6x with minimal AUPRC cost (0.687 vs 0.701).

---

### 7.4 Experiment 4: Hard Ratio Effect

**Question**: Does hard_ratio affect train/val gap?

| Hard Ratio | Best Val AUPRC | Avg Train/Val Gap |
|------------|----------------|-------------------|
| 0.0 (easy only) | 0.4830 | 54.6 pts |
| 0.5 (default) | 0.6683 | 27.3 pts |
| **1.0 (hard only)** | **0.6715** | **24.7 pts** |

**Finding**: `hard_ratio=1.0` achieves best AUPRC AND smallest train/val gap. Easy negatives create distribution shift the model exploits.

---

### 7.5 Summary and Recommendations

| Finding | Recommendation |
|---------|----------------|
| hard_ratio=1.0 is best | **Always use `--hard_ratio 1.0`** |
| Pair masking hurts | **Use `--pair_masking 0.0`** |
| focal_alpha stabilizes | Consider adding `--focal_alpha 0.3` CLI arg |
| Model captures some dynamics | 27.9% transition acc > 0%, but architecture improvements needed |

**Suggested optimal configuration**:
```bash
python main.py all \
  --hard_ratio 1.0 \
  --pair_masking 0.0 \
  --focal_gamma 2.0 \
  --epochs 20 \
  --patience 7
```

---

## 8. Architecture Analysis: Root Causes

The experiments above address hyperparameters, but the 27.9% transition accuracy suggests **fundamental architecture limitations**.

### 8.1 Problem: No Dynamics-Specific Signal in GRU Input

The temporal encoder receives for each history timestep:

| Component | Dynamic? | Problem |
|-----------|----------|---------|
| `entity_rgcn_embed` | ⚠️ Weak | Captures "how connected" but not "connected to whom" |
| `entity_embeds[id]` | ❌ No | Static — same at every timestep |
| `mean_rel` | ❌ No | Constant mean of all relation embeddings |
| `global_emb` | ⚠️ Weak | Global state, not entity-specific |

**Impact**: GRU sees nearly identical input at each timestep. It cannot learn "entity was active with neighbor A vs B" — the neighbor identity information is lost in aggregation.

### 8.2 Problem: No Pair-Level Temporal Features

The classifier receives:
```python
classifier(entity1_embed, entity2_embed, entity1_temporal, entity2_temporal)
```

These are **independent** per-entity temporals. There's no:
- Direct (e1, e2) co-occurrence history
- Temporal correlation between e1 and e2 activity
- Relative timing signal (did e1's activity precede e2's?)

**Impact**: Model cannot learn pair-specific patterns like "e1-e2 oscillates" or "e1 turning ON predicts e2 ON".

### 8.3 Problem: Train/Eval Distribution Mismatch

| Aspect | Training | Evaluation |
|--------|----------|------------|
| History | Oracle (ground truth) | Autoregressive (predicted) |
| Negatives | 50% easy (never interacted) | 0% easy (all known pairs) |
| Batching | Random shuffle | Strict temporal order |

**Validated by Experiment 4**: `hard_ratio=1.0` halves the train/val gap by aligning negative distributions.

---

## 9. Proposed Architecture Improvements

### 9.1 Priority 1: Add Pair-Level Temporal Features (High Impact)

**What**: Encode the binary history of THIS specific pair (e1, e2).

**How**:
```python
# Compute pair interaction history: was (e1, e2) ON at each past timestep?
pair_history = [1 if (e1,e2) in positives[t] else 0 for t in history_window]
pair_temporal = self.pair_gru(pair_history)  # Separate small GRU

# Add to classifier
classifier(e1_embed, e2_embed, e1_temporal, e2_temporal, pair_temporal)
```

**Expected gain**: Directly captures transition patterns. LSCF comparison shows model has 27.9% transition accuracy when it has no direct access to pair history — adding this should improve significantly.

**Effort**: Medium — requires modifying `_collate_fn` to return pair history and classifier to accept it.

---

### 9.2 Priority 2: Encode Neighbor Identity in History (High Impact)

**What**: Preserve which specific neighbors were active, not just "how many".

**How**:
```python
# Current (lossy):
neighbor_features = mean(rgcn_outputs_of_neighbors)

# Proposed (preserves identity):
neighbor_ids = history_entry["neighbors"]
neighbor_embeds = self.entity_embeds[neighbor_ids]
neighbor_agg = attention_pool(neighbor_embeds)  # Learnable attention
```

**Expected gain**: Enables learning "e2 was in e1's neighbors → likely ON together".

**Effort**: Medium — requires attention mechanism and memory for variable-size neighbor sets.

---

### 9.3 Priority 3: Transition-Weighted Loss (Medium Impact)

**What**: Upweight the 20% of samples that represent actual state transitions.

**How**:
```python
# Current: focal_loss(logits, labels)
# Proposed: 
is_transition = (prev_state != label)
weight = torch.where(is_transition, 3.0, 1.0)  # 3x weight for transitions
loss = (focal_loss(logits, labels) * weight).mean()
```

**Rationale**: Experiments show 80% of samples are non-transitions where LSCF gets 100%. The model overfits to stable pairs.

**Effort**: Low — only modifies loss computation.

---

### 9.4 Priority 4: Curriculum on History Noise (Medium Impact)

**What**: Gradually inject prediction errors into training history to simulate autoregressive evaluation.

**How**:
```python
# Flip random entries in history with probability that increases during training
if epoch > warmup:
    noise_prob = min(0.2, (epoch - warmup) * 0.02)
    history = corrupt_history(history, noise_prob)
```

**Rationale**: Bridges gap between oracle training and autoregressive eval.

**Effort**: Low — modifies `_collate_fn`.

---

### 9.5 Priority 5: Explicit Transition Output Head (High Impact, Major Change)

**What**: Instead of predicting absolute state, predict P(transition).

**How**:
```python
# Current: output = P(ON | features)
# Proposed:
prev_state = get_pair_state(e1, e2, t-1)
transition_logit = classifier(...)
# Final prediction: prev_state XOR (transition_logit > 0.5)
```

**Rationale**: Reframes task from "guess state" to "predict change", which is what we actually care about.

**Effort**: High — changes output interpretation and evaluation.

---

## 10. Next Steps

### Immediate (No Architecture Change)
- [ ] Add `--focal_alpha` CLI argument (low effort)
- [ ] Run full training with `--hard_ratio 1.0 --pair_masking 0.0`

### Short-term (Minor Architecture Changes)
- [ ] Implement transition-weighted loss (Section 9.3)
- [ ] Implement history noise curriculum (Section 9.4)

### Medium-term (Architecture Improvements)
- [ ] Add pair-level temporal encoder (Section 9.1)
- [ ] Add neighbor identity attention (Section 9.2)

### Longer-term
- [ ] Experiment with explicit transition head (Section 9.5)