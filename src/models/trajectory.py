"""
Trajectory-level prediction model for PPI dynamics.

Predicts full trajectories for each edge, informed by neighbor edge trajectories
via cross-attention. Replaces per-timestep prediction.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model) with positional encoding added
        """
        x = x + self.pe[: x.size(1)]
        return self.dropout(x)


class EdgeHistoryEncoder(nn.Module):
    """
    Encodes an edge's binary state history into a dense representation.

    Uses a Transformer encoder to capture temporal patterns.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project binary state to hidden dim
        self.state_embed = nn.Linear(1, hidden_dim)

        self.pos_encoding = PositionalEncoding(hidden_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(
        self,
        history: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            history: Binary states (batch, seq_len) - 0/1 for edge off/on
            mask: Padding mask (batch, seq_len) - True for padded positions

        Returns:
            Encoded representation (batch, hidden_dim)
        """
        # (batch, seq_len, 1)
        x = history.unsqueeze(-1).float()

        # (batch, seq_len, hidden_dim)
        x = self.state_embed(x)
        x = self.pos_encoding(x)

        # Transformer expects mask where True = ignore
        x = self.transformer(x, src_key_padding_mask=mask)

        # Pool over sequence: mean of non-padded positions
        if mask is not None:
            # Invert mask for multiplication (True = keep)
            keep_mask = ~mask
            x = (x * keep_mask.unsqueeze(-1)).sum(dim=1) / keep_mask.sum(
                dim=1, keepdim=True
            ).clamp(min=1)
        else:
            x = x.mean(dim=1)

        return x


class NeighborCrossAttention(nn.Module):
    """
    Cross-attention from target edge to its neighbor edges.

    Target edge (A,B) attends to all edges involving A or B.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        neighbor_embeds: torch.Tensor,
        neighbor_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: Target edge embedding (batch, hidden_dim)
            neighbor_embeds: Neighbor edge embeddings (batch, n_neighbors, hidden_dim)
            neighbor_mask: Mask for padded neighbors (batch, n_neighbors), True = pad

        Returns:
            Updated query embedding (batch, hidden_dim)
        """
        # Expand query for attention: (batch, 1, hidden_dim)
        q = query.unsqueeze(1)

        # Cross-attention
        attn_out, _ = self.cross_attn(
            query=q,
            key=neighbor_embeds,
            value=neighbor_embeds,
            key_padding_mask=neighbor_mask,
        )

        # Residual connection
        out = query + self.dropout(attn_out.squeeze(1))
        out = self.norm(out)

        return out


class TrajectoryDecoder(nn.Module):
    """
    Decodes edge embedding into full trajectory predictions.

    Uses a Transformer decoder to predict each future timestep,
    attending to the encoded edge representation.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_traj_len: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_traj_len = max_traj_len

        # Learnable position queries for each future timestep
        self.pos_queries = nn.Parameter(torch.randn(max_traj_len, hidden_dim))

        self.pos_encoding = PositionalEncoding(
            hidden_dim, max_len=max_traj_len, dropout=dropout
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output projection to logits
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        edge_embed: torch.Tensor,
        traj_len: int,
    ) -> torch.Tensor:
        """
        Args:
            edge_embed: Encoded edge representation (batch, hidden_dim)
            traj_len: Number of timesteps to predict

        Returns:
            Trajectory logits (batch, traj_len)
        """
        batch_size = edge_embed.size(0)

        # Get position queries for the requested length
        queries = self.pos_queries[:traj_len].unsqueeze(0).expand(batch_size, -1, -1)
        queries = self.pos_encoding(queries)

        # Memory is the edge embedding repeated for cross-attention
        memory = edge_embed.unsqueeze(1)  # (batch, 1, hidden_dim)

        # Decode
        decoded = self.transformer(queries, memory)

        # Project to logits
        logits = self.output_proj(decoded).squeeze(-1)  # (batch, traj_len)

        return logits


class TrajectoryModel(nn.Module):
    """
    Full trajectory prediction model for PPI dynamics.

    Architecture:
    1. Encode target edge's history
    2. Encode neighbor edges' histories
    3. Cross-attend: target attends to neighbors
    4. Decode full future trajectory
    """

    def __init__(
        self,
        num_entities: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 2,
        max_neighbors: int = 50,
        max_traj_len: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.hidden_dim = hidden_dim
        self.max_neighbors = max_neighbors

        # Entity embeddings (optional, can add identity information)
        self.entity_embed = nn.Embedding(num_entities, hidden_dim // 2)

        # History encoder (shared for target and neighbors)
        self.history_encoder = EdgeHistoryEncoder(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            dropout=dropout,
        )

        # Entity projection to combine with history
        self.entity_proj = nn.Linear(hidden_dim, hidden_dim)

        # Cross-attention to neighbors
        self.neighbor_attention = NeighborCrossAttention(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Trajectory decoder
        self.decoder = TrajectoryDecoder(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_decoder_layers,
            max_traj_len=max_traj_len,
            dropout=dropout,
        )

    def _encode_edge(
        self,
        entity1_ids: torch.Tensor,
        entity2_ids: torch.Tensor,
        history: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode an edge with its history."""
        # Get entity embeddings
        e1_embed = self.entity_embed(entity1_ids)
        e2_embed = self.entity_embed(entity2_ids)
        entity_embed = torch.cat([e1_embed, e2_embed], dim=-1)
        entity_embed = self.entity_proj(entity_embed)

        # Encode history
        history_embed = self.history_encoder(history, mask=history_mask)

        # Combine: add entity info to history encoding
        edge_embed = entity_embed + history_embed

        return edge_embed

    def forward(
        self,
        entity1_ids: torch.Tensor,
        entity2_ids: torch.Tensor,
        history: torch.Tensor,
        neighbor_entity1: torch.Tensor,
        neighbor_entity2: torch.Tensor,
        neighbor_history: torch.Tensor,
        traj_len: int,
        history_mask: Optional[torch.Tensor] = None,
        neighbor_mask: Optional[torch.Tensor] = None,
        neighbor_history_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict full trajectory for each edge.

        Args:
            entity1_ids: First entity (batch,)
            entity2_ids: Second entity (batch,)
            history: Edge history (batch, hist_len)
            neighbor_entity1: Neighbor first entities (batch, n_neighbors)
            neighbor_entity2: Neighbor second entities (batch, n_neighbors)
            neighbor_history: Neighbor histories (batch, n_neighbors, hist_len)
            traj_len: Length of trajectory to predict
            history_mask: Padding mask for history (batch, hist_len)
            neighbor_mask: Which neighbors are padding (batch, n_neighbors)
            neighbor_history_mask: Padding mask for neighbor histories (batch, n_neighbors, hist_len)

        Returns:
            Trajectory logits (batch, traj_len)
        """
        batch_size = entity1_ids.size(0)
        n_neighbors = neighbor_entity1.size(1)

        # Encode target edge
        edge_embed = self._encode_edge(entity1_ids, entity2_ids, history, history_mask)

        # Encode neighbor edges
        # Flatten neighbors for batch encoding
        flat_n1 = neighbor_entity1.view(-1)
        flat_n2 = neighbor_entity2.view(-1)
        flat_hist = neighbor_history.view(batch_size * n_neighbors, -1)

        if neighbor_history_mask is not None:
            flat_hist_mask = neighbor_history_mask.view(batch_size * n_neighbors, -1)
        else:
            flat_hist_mask = None

        neighbor_embeds = self._encode_edge(flat_n1, flat_n2, flat_hist, flat_hist_mask)
        neighbor_embeds = neighbor_embeds.view(batch_size, n_neighbors, -1)

        # Cross-attend to neighbors
        edge_embed = self.neighbor_attention(edge_embed, neighbor_embeds, neighbor_mask)

        # Decode trajectory
        logits = self.decoder(edge_embed, traj_len)

        return logits

    def predict(
        self,
        entity1_ids: torch.Tensor,
        entity2_ids: torch.Tensor,
        history: torch.Tensor,
        neighbor_entity1: torch.Tensor,
        neighbor_entity2: torch.Tensor,
        neighbor_history: torch.Tensor,
        traj_len: int,
        threshold: float = 0.5,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict trajectories with probabilities and binary predictions.

        Returns:
            Tuple of (probabilities, binary_predictions)
        """
        logits = self.forward(
            entity1_ids,
            entity2_ids,
            history,
            neighbor_entity1,
            neighbor_entity2,
            neighbor_history,
            traj_len,
            **kwargs,
        )
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).long()

        return probs, preds
