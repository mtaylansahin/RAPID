"""
Edge-Centric Temporal Subgraph Encoder.

Replaces the simple EdgeHistoryEncoder with a full subgraph-aware encoder
that attends to N-hop neighborhood edge histories.
"""

import math

import torch
import torch.nn as nn


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
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class EdgeCentricSubgraphEncoder(nn.Module):
    """
    Encode local subgraph temporal context for edge prediction.

    For each target edge, this encoder:
    1. Encodes the target edge's temporal history
    2. Encodes all neighbor edges' temporal histories
    3. Uses cross-attention from target to neighbors
    4. Produces a subgraph-aware context embedding

    Args:
        hidden_dim: Hidden dimension for embeddings
        num_heads: Number of attention heads
        num_layers: Number of transformer encoder layers
        max_history_len: Maximum history length for positional encoding
        max_neighbors: Maximum number of neighbor edges
        dropout: Dropout rate
    """

    def __init__(
        self,
        hidden_dim: int = 200,
        num_heads: int = 4,
        num_layers: int = 2,
        max_history_len: int = 500,
        max_neighbors: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_neighbors = max_neighbors

        # Edge state embedding (ON=1, OFF=0)
        self.edge_state_embed = nn.Embedding(2, hidden_dim)

        # Hop distance embedding (0=target, 1=1-hop, 2=2-hop, 3+=distant)
        self.hop_embed = nn.Embedding(4, hidden_dim)

        # Positional encoding for temporal ordering
        self.temporal_pos = PositionalEncoding(hidden_dim, max_history_len, dropout)

        # Self-attention over each edge's temporal history
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Cross-attention: target edge attends to neighborhood
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Output projection: [target_emb, neighborhood_context] -> hidden
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        target_histories: torch.Tensor,
        neighbor_histories: torch.Tensor,
        hop_distances: torch.Tensor,
        neighbor_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Encode subgraph context for a batch of target edges.

        Args:
            target_histories: Binary history of target edges (batch, num_timesteps)
            neighbor_histories: Histories of neighbor edges (batch, max_neighbors, num_timesteps)
            hop_distances: Hop distance for each neighbor (batch, max_neighbors)
            neighbor_mask: Padding mask, True for padding positions (batch, max_neighbors)

        Returns:
            Subgraph context embedding (batch, hidden_dim)
        """
        batch_size = target_histories.size(0)

        # 1. Encode target edge history
        target_emb = self._encode_history(target_histories)  # (batch, hidden)

        # 2. Encode neighbor histories
        if neighbor_histories.size(1) > 0:
            num_neighbors = neighbor_histories.size(1)
            num_timesteps = neighbor_histories.size(2)

            # Reshape for batch encoding: (batch * num_neighbors, num_timesteps)
            neighbor_flat = neighbor_histories.view(-1, num_timesteps)
            neighbor_embs = self._encode_history(neighbor_flat)
            neighbor_embs = neighbor_embs.view(batch_size, num_neighbors, -1)

            # Add hop distance embeddings
            hop_clamped = hop_distances.clamp(max=3)  # Cap at 3+
            hop_embs = self.hop_embed(hop_clamped)
            neighbor_embs = neighbor_embs + hop_embs

            # 3. Cross-attention: target attends to neighbors
            target_query = target_emb.unsqueeze(1)  # (batch, 1, hidden)

            # Check for fully masked rows (edges with no valid neighbors)
            # neighbor_mask: True = ignore, so all-True rows mean no valid neighbors
            if neighbor_mask is not None:
                fully_masked = neighbor_mask.all(dim=1)  # (batch,)

                # Create a safe mask where fully-masked rows have at least one valid position
                # This prevents NaN in cross-attention
                safe_mask = neighbor_mask.clone()
                safe_mask[fully_masked, 0] = False  # Allow at least one position
            else:
                fully_masked = torch.zeros(
                    batch_size, dtype=torch.bool, device=target_emb.device
                )
                safe_mask = None

            # MultiheadAttention expects key_padding_mask where True = ignore
            context, attn_weights = self.cross_attention(
                target_query,
                neighbor_embs,
                neighbor_embs,
                key_padding_mask=safe_mask,
            )
            context = context.squeeze(1)  # (batch, hidden)

            # For fully masked edges, replace context with zeros (no neighborhood info)
            if fully_masked.any():
                context = context.masked_fill(fully_masked.unsqueeze(1), 0.0)
        else:
            # No neighbors - just use target embedding
            context = torch.zeros_like(target_emb)

        # 4. Combine target and context
        combined = torch.cat([target_emb, context], dim=-1)
        output = self.output_proj(combined)

        return output

    def _encode_history(self, history: torch.Tensor) -> torch.Tensor:
        """Encode a single edge's temporal history.

        Args:
            history: Binary history tensor (batch, num_timesteps)

        Returns:
            History embedding (batch, hidden_dim)
        """
        # Embed edge states
        x = self.edge_state_embed(history.long())  # (batch, T, hidden)

        # Add positional encoding
        x = self.temporal_pos(x)

        # Self-attention over history
        x = self.temporal_encoder(x)

        # Take final timestep as summary
        return x[:, -1, :]


def create_subgraph_encoder(
    hidden_dim: int = 200,
    num_heads: int = 4,
    num_layers: int = 2,
    max_neighbors: int = 50,
    dropout: float = 0.1,
) -> EdgeCentricSubgraphEncoder:
    """Factory function to create subgraph encoder."""
    return EdgeCentricSubgraphEncoder(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_neighbors=max_neighbors,
        dropout=dropout,
    )
