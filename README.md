# RAPID: A Recurrent Architecture for Predicting Protein Interaction Dynamics

RAPID adapts the RE-Net architecture for binary classification of protein-protein interaction dynamics from molecular dynamics simulations.

## Quick Start

### Using main.py (Recommended)

```bash
# Full pipeline with global model
uv run python main.py all --dataset RAPID --use_global_model --epochs 100

# Train without global model
uv run python main.py train --dataset RAPID --epochs 100

# Pretrain global model only
uv run python main.py pretrain --dataset RAPID --epochs 30

# Evaluate trained model
uv run python main.py evaluate --checkpoint ./checkpoints/RAPID_*/best.pth
```

### Direct Script Usage

```bash
# Step 1: Pretrain global model (optional but recommended)
uv run python src/pretrain.py --dataset RAPID --epochs 30 --pooling max

# Step 2: Train main model
uv run python src/train.py --dataset RAPID --use_global_model --epochs 100

# Step 3: Evaluate
uv run python src/evaluate.py \
    --checkpoint checkpoints/RAPID_*/best.pth \
    --use_global_model --mode all
```

## Global Model

The global model captures graph-level temporal context:

1. **Pretraining**: Runs RGCN on each timestep's full graph, pools to vector, encodes sequence with GRU
2. **Integration**: Global embeddings enrich per-entity temporal encoding during main model training
3. **No Leakage**: `global_emb[t]` uses only graphs from times `< t+1` (same as original RE-Net)

Enable with `--use_global_model` flag. The global model is pretrained separately and frozen during main training.

## Key Differences from Original RE-Net

| Aspect | Original RE-Net | RAPID |
|--------|----------------|-----------------|
| Graph type | Directed | Undirected |
| Task | Entity ranking | Binary classification |
| Loss | Cross-entropy (multi-class) | Focal loss (binary) |
| Metrics | MRR, Hits@K | AUROC, AUPRC, F1 |
| History | Separate s_hist/o_hist | Unified per entity |
| Global model | Required | Optional (behind flag) |

## Architecture

```
src/
├── config.py              # Configuration dataclasses
├── pretrain.py            # Global model pretraining
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── data/
│   ├── dataset.py         # PPIDataset, PPIDataModule
│   └── sampler.py         # NegativeSampler (mixed random + temporal)
├── models/
│   ├── rapid.py           # Main RAPIDModel
│   ├── global_model.py    # Global RGCN model
│   ├── rgcn.py            # Undirected RGCN layers
│   ├── encoder.py         # GRU temporal encoder
│   ├── classifier.py      # Binary edge classifier
│   └── aggregator.py      # History aggregation
├── losses/
│   └── __init__.py        # FocalLoss
└── metrics/
    └── __init__.py        # ClassificationMetrics, MetricsComputer
```

## Configuration

All hyperparameters are configurable via command line or config files:

```python
from src.config import Config, ModelConfig, TrainingConfig

config = Config(
    model=ModelConfig(
        hidden_dim=200,
        seq_len=10,
        num_rgcn_layers=2,
        dropout=0.2,
    ),
    training=TrainingConfig(
        learning_rate=1e-3,
        max_epochs=100,
        focal_gamma=2.0,
    ),
)
```

## Hyperparameters to Experiment With

| Parameter | Default | Suggested Range |
|-----------|---------|-----------------|
| `seq_len` | 10 | 5, 10, 20, 50 |
| `hidden_dim` | 200 | 64, 128, 200, 256 |
| `num_rgcn_layers` | 2 | 1, 2, 3 |
| `neg_ratio` | 1.0 | 0.5, 1.0, 2.0, 5.0 |
| `focal_gamma` | 2.0 | 0, 1, 2, 3 |
| `learning_rate` | 1e-3 | 1e-4, 5e-4, 1e-3 |

## Data Format

Expects data in RE-Net format:
- `train.txt`, `valid.txt`, `test.txt`: Quadruples `entity1 relation entity2 timestep`
- `stat.txt`: `num_entities num_relations num_timesteps`

The model automatically:
- Converts to undirected (canonical edge ordering)
- Generates negative samples (50% random + 50% temporal)
- Builds history per entity from training data
