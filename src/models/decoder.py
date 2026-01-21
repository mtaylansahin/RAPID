"""Transformer decoder for seq2seq edge prediction."""

import math
from typing import Optional

import torch
import torch.nn as nn

from src.models.subgraph_encoder import EdgeCentricSubgraphEncoder


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TemporalEdgeDecoder(nn.Module):
    """
    Transformer decoder for predicting edge states across timesteps.

    Uses EdgeCentricSubgraphEncoder as the PRIMARY temporal signal with
    gated fusion to combine static pair embeddings with edge history context.

    Args:
        hidden_dim: Hidden dimension (must match encoder)
        num_layers: Number of transformer decoder layers
        num_heads: Number of attention heads
        max_timesteps: Maximum number of timesteps to predict
        dropout: Dropout rate
    """

    def __init__(
        self,
        hidden_dim: int = 200,
        num_layers: int = 4,
        num_heads: int = 8,
        max_timesteps: int = 200,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_timesteps = max_timesteps

        # Symmetric pair projection: sum + product + abs_diff
        # Output dim: 3 * hidden_dim -> hidden_dim
        self.pair_proj = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Relative timestep embedding
        self.timestep_embed = nn.Embedding(max_timesteps, hidden_dim)

        # Positional encoding for temporal ordering
        self.pos_encoding = PositionalEncoding(hidden_dim, max_timesteps, dropout)

        # Edge history encoder: PRIMARY temporal signal (always enabled)
        self.edge_history_encoder = EdgeCentricSubgraphEncoder(
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=2,
            max_neighbors=50,
            dropout=dropout,
        )

        # Gated fusion: learn when to use edge history vs pair embedding
        # Input: [pair_emb, edge_hist_emb] -> gate value
        self.gate_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection (hidden -> logit)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.dropout = nn.Dropout(dropout)

    def _symmetric_pair_features(
        self,
        e1_embed: torch.Tensor,
        e2_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Create symmetric pair features from entity embeddings."""
        pair_sum = e1_embed + e2_embed
        pair_prod = e1_embed * e2_embed
        pair_diff = torch.abs(e1_embed - e2_embed)
        return torch.cat([pair_sum, pair_prod, pair_diff], dim=-1)

    def forward(
        self,
        entity_context: torch.Tensor,
        pair_indices: torch.Tensor,
        relative_timesteps: torch.Tensor,
        subgraph_context: dict,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Predict edge states for all pairs across all timesteps.

        Args:
            entity_context: Entity embeddings from encoder (num_entities, hidden_dim)
            pair_indices: Entity pair indices (num_pairs, 2)
            relative_timesteps: Timesteps relative to train boundary (num_timesteps,)
            subgraph_context: Dict with subgraph tensors for EdgeCentricSubgraphEncoder:
                - target_histories: (num_pairs, num_timesteps) target edge histories
                - neighbor_histories: (num_pairs, max_neighbors, num_timesteps)
                - hop_distances: (num_pairs, max_neighbors)
                - neighbor_mask: (num_pairs, max_neighbors) True for padding
            causal: Whether to use causal masking for temporal autoregression

        Returns:
            logits: Prediction logits (num_pairs, num_timesteps)
        """
        num_pairs = pair_indices.size(0)
        num_timesteps = relative_timesteps.size(0)
        device = entity_context.device

        # Build symmetric pair embeddings from entity context
        e1_ctx = entity_context[pair_indices[:, 0]]  # (num_pairs, hidden)
        e2_ctx = entity_context[pair_indices[:, 1]]  # (num_pairs, hidden)
        pair_features = self._symmetric_pair_features(e1_ctx, e2_ctx)
        pair_emb = self.pair_proj(pair_features)  # (num_pairs, hidden)

        # Get edge temporal context from subgraph encoder (PRIMARY signal)
        edge_hist_emb = self.edge_history_encoder(
            target_histories=subgraph_context["target_histories"],
            neighbor_histories=subgraph_context["neighbor_histories"],
            hop_distances=subgraph_context["hop_distances"],
            neighbor_mask=subgraph_context.get("neighbor_mask"),
        )  # (num_pairs, hidden)

        # Gated fusion: learn balance between static and temporal features
        gate = self.gate_proj(torch.cat([pair_emb, edge_hist_emb], dim=-1))
        fused_emb = gate * pair_emb + (1 - gate) * edge_hist_emb

        fused_emb = self.dropout(fused_emb)

        # Build temporal queries
        clamped_t = relative_timesteps.clamp(0, self.max_timesteps - 1)
        t_emb = self.timestep_embed(clamped_t)  # (num_timesteps, hidden)

        # Combine fused pair + timestep embeddings
        queries = fused_emb.unsqueeze(1) + t_emb.unsqueeze(0)
        queries = self.pos_encoding(queries)

        # Memory: entity context for cross-attention
        memory = entity_context.unsqueeze(0).expand(num_pairs, -1, -1)

        # Causal mask for temporal autoregression
        tgt_mask = None
        if causal and num_timesteps > 1:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                num_timesteps, device=device
            )

        # Decode
        decoded = self.decoder(queries, memory, tgt_mask=tgt_mask)

        # Project to logits
        logits = self.output_proj(decoded).squeeze(-1)

        return logits

    def predict(
        self,
        entity_context: torch.Tensor,
        pair_indices: torch.Tensor,
        relative_timesteps: torch.Tensor,
        subgraph_context: dict,
        threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inference mode: returns probabilities and binary predictions.

        Args:
            entity_context: Entity embeddings from encoder
            pair_indices: Entity pair indices
            relative_timesteps: Timesteps relative to train boundary
            subgraph_context: Subgraph context for EdgeCentricSubgraphEncoder
            threshold: Classification threshold

        Returns:
            Tuple of (probabilities, binary predictions, logits)
        """
        with torch.no_grad():
            logits = self.forward(
                entity_context,
                pair_indices,
                relative_timesteps,
                subgraph_context=subgraph_context,
                causal=False,
            )
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()
            return probs, preds, logits


def create_decoder(
    hidden_dim: int = 200,
    num_layers: int = 4,
    num_heads: int = 8,
    max_timesteps: int = 200,
    dropout: float = 0.1,
) -> TemporalEdgeDecoder:
    """Factory function to create decoder."""
    return TemporalEdgeDecoder(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_timesteps=max_timesteps,
        dropout=dropout,
    )
