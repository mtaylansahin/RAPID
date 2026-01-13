"""Transformer decoder for seq2seq edge prediction."""

import math
from typing import Tuple

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
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class EdgeHistoryEncoder(nn.Module):
    """
    Transformer-based encoder for edge state history.

    Uses self-attention over the FULL edge history, allowing the model
    to learn which historical timesteps are most important for prediction.

    Args:
        hidden_dim: Output embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer encoder layers
        dropout: Dropout rate
        max_history_len: Maximum history length for positional encoding
    """

    def __init__(
        self,
        hidden_dim: int = 200,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_history_len: int = 500,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project binary state (0/1) to hidden dimension
        self.input_proj = nn.Linear(1, hidden_dim)

        # Positional encoding for temporal ordering
        self.pos_encoding = PositionalEncoding(hidden_dim, max_history_len, dropout)

        # Transformer encoder for self-attention over history
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output: pool temporal dimension and project
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, edge_history: torch.Tensor) -> torch.Tensor:
        """
        Encode full edge history into embedding using self-attention.

        Args:
            edge_history: Binary tensor of shape (num_pairs, num_timesteps)
                          where 1 = edge ON, 0 = edge OFF at that timestep

        Returns:
            Edge history embedding of shape (num_pairs, hidden_dim)
        """
        # edge_history: (num_pairs, T)
        # Add feature dimension: (num_pairs, T, 1)
        x = edge_history.unsqueeze(-1).float()

        # Project to hidden dim: (num_pairs, T, hidden_dim)
        x = self.input_proj(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Self-attention over full history
        # Each position can attend to all other positions
        x = self.transformer(x)  # (num_pairs, T, hidden_dim)

        # Pool: take the last timestep (most recent context)
        # Alternative: mean pooling or learned pooling
        output = x[:, -1, :]  # (num_pairs, hidden_dim)

        return self.output_proj(output)


class TemporalEdgeDecoder(nn.Module):
    """
    Transformer decoder for predicting edge states across timesteps.

    Takes entity context from encoder and predicts all edges × all timesteps.

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
        use_edge_history: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_timesteps = max_timesteps
        self.use_edge_history = use_edge_history

        # Pair embedding projection (combine two entity embeddings)
        self.pair_proj = nn.Linear(2 * hidden_dim, hidden_dim)

        # Relative timestep embedding
        self.timestep_embed = nn.Embedding(max_timesteps, hidden_dim)

        # Positional encoding for temporal ordering
        self.pos_encoding = PositionalEncoding(hidden_dim, max_timesteps, dropout)

        # Edge history encoder (optional) - uses Transformer over full history
        self.edge_history_encoder = None
        if use_edge_history:
            self.edge_history_encoder = EdgeHistoryEncoder(
                hidden_dim=hidden_dim,
                num_heads=4,
                num_layers=2,
                dropout=dropout,
                max_history_len=500,
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

    def forward(
        self,
        entity_context: torch.Tensor,
        pair_indices: torch.Tensor,
        relative_timesteps: torch.Tensor,
        edge_history: torch.Tensor = None,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Predict edge states for all pairs across all timesteps.

        Args:
            entity_context: Entity embeddings from encoder (num_entities, hidden_dim)
            pair_indices: Entity pair indices (num_pairs, 2)
            relative_timesteps: Timesteps relative to train boundary (num_timesteps,)
            edge_history: Optional edge state history (num_pairs, history_len)
            causal: Whether to use causal masking for temporal autoregression

        Returns:
            logits: Prediction logits (num_pairs, num_timesteps)
        """
        num_pairs = pair_indices.size(0)
        num_timesteps = relative_timesteps.size(0)
        device = entity_context.device

        # Build pair embeddings from entity context
        e1_ctx = entity_context[pair_indices[:, 0]]  # (num_pairs, hidden)
        e2_ctx = entity_context[pair_indices[:, 1]]  # (num_pairs, hidden)
        pair_emb = self.pair_proj(torch.cat([e1_ctx, e2_ctx], dim=-1))

        # Add edge history embedding if available
        if self.edge_history_encoder is not None and edge_history is not None:
            edge_hist_emb = self.edge_history_encoder(edge_history)
            pair_emb = pair_emb + edge_hist_emb

        pair_emb = self.dropout(pair_emb)

        # Build temporal queries
        # Clamp timesteps to valid embedding range
        clamped_t = relative_timesteps.clamp(0, self.max_timesteps - 1)
        t_emb = self.timestep_embed(clamped_t)  # (num_timesteps, hidden)

        # Combine pair + timestep embeddings
        # (num_pairs, 1, hidden) + (1, num_timesteps, hidden) -> (num_pairs, num_timesteps, hidden)
        queries = pair_emb.unsqueeze(1) + t_emb.unsqueeze(0)
        queries = self.pos_encoding(queries)

        # Memory: entity context for cross-attention
        # Expand to (num_pairs, num_entities, hidden)
        memory = entity_context.unsqueeze(0).expand(num_pairs, -1, -1)

        # Causal mask for temporal autoregression
        tgt_mask = None
        if causal and num_timesteps > 1:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                num_timesteps, device=device
            )

        # Decode
        decoded = self.decoder(
            queries, memory, tgt_mask=tgt_mask
        )  # (num_pairs, num_timesteps, hidden)

        # Project to logits
        logits = self.output_proj(decoded).squeeze(-1)  # (num_pairs, num_timesteps)

        return logits

    def predict(
        self,
        entity_context: torch.Tensor,
        pair_indices: torch.Tensor,
        relative_timesteps: torch.Tensor,
        edge_history: torch.Tensor = None,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inference mode: returns probabilities and binary predictions.

        Args:
            entity_context: Entity embeddings from encoder
            pair_indices: Entity pair indices
            relative_timesteps: Timesteps relative to train boundary
            edge_history: Optional edge state history (num_pairs, history_len)
            threshold: Classification threshold

        Returns:
            Tuple of (probabilities, binary predictions, logits)
        """
        with torch.no_grad():
            logits = self.forward(
                entity_context,
                pair_indices,
                relative_timesteps,
                edge_history=edge_history,
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
    use_edge_history: bool = False,
) -> TemporalEdgeDecoder:
    """Factory function to create decoder."""
    return TemporalEdgeDecoder(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_timesteps=max_timesteps,
        dropout=dropout,
        use_edge_history=use_edge_history,
    )
