"""GRU-based temporal encoder."""

from typing import Tuple

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """
    GRU-based encoder for temporal sequences.

    Processes packed sequences from the history aggregator
    and outputs temporal embeddings.

    Args:
        input_dim: Input dimension (from aggregator)
        hidden_dim: GRU hidden dimension
        num_layers: Number of GRU layers
        dropout: Dropout rate
        bidirectional: Whether to use bidirectional GRU
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Project bidirectional output if needed
        if bidirectional:
            self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.output_proj = None

    def forward(
        self,
        packed_sequence: torch.nn.utils.rnn.PackedSequence,
    ) -> torch.Tensor:
        """
        Encode packed sequence.

        Args:
            packed_sequence: Packed sequence from aggregator

        Returns:
            Final hidden state of shape (batch_size, hidden_dim)
        """
        output, hidden = self.gru(packed_sequence)

        # hidden shape: (num_layers * num_directions, batch, hidden)
        # Take last layer
        if self.bidirectional:
            # Concatenate forward and backward
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
            hidden = self.output_proj(hidden)
        else:
            hidden = hidden[-1]

        return hidden

    def forward_with_output(
        self,
        packed_sequence: torch.nn.utils.rnn.PackedSequence,
    ) -> Tuple[torch.nn.utils.rnn.PackedSequence, torch.Tensor]:
        """
        Encode and return both output sequence and final hidden state.

        Args:
            packed_sequence: Packed sequence from aggregator

        Returns:
            Tuple of (packed output sequence, final hidden state)
        """
        output, hidden = self.gru(packed_sequence)

        if self.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
            hidden = self.output_proj(hidden)
        else:
            hidden = hidden[-1]

        return output, hidden


class AttentionTemporalEncoder(nn.Module):
    """
    Attention-based encoder for temporal sequences.

    Uses TransformerEncoder with a CLS token to aggregate sequence information.
    Can be used as a drop-in replacement for TemporalEncoder.

    Args:
        input_dim: Input dimension (from aggregator)
        hidden_dim: Hidden dimension for attention
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        max_seq_len: Maximum sequence length for positional encoding
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 20,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Project input to hidden dim if needed
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = None

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Learned positional embeddings (+1 for CLS token)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len + 1, hidden_dim) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        packed_sequence: torch.nn.utils.rnn.PackedSequence,
    ) -> torch.Tensor:
        """
        Encode packed sequence using attention.

        Args:
            packed_sequence: Packed sequence from aggregator

        Returns:
            Final representation from CLS token, shape (batch_size, hidden_dim)
        """
        # Unpack sequence
        padded, lengths = torch.nn.utils.rnn.pad_packed_sequence(
            packed_sequence, batch_first=True
        )
        batch_size, seq_len, _ = padded.shape

        # Project to hidden dim
        if self.input_proj is not None:
            padded = self.input_proj(padded)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, padded], dim=1)  # (batch, seq_len+1, hidden)

        # Add positional embeddings
        pos_len = min(seq_len + 1, self.max_seq_len + 1)
        x[:, :pos_len] = x[:, :pos_len] + self.pos_embedding[:, :pos_len]

        # Create attention mask (True = masked out)
        # CLS token (position 0) should attend to all positions
        # Other positions should only attend to valid sequence positions
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0)
        mask = mask > lengths.unsqueeze(1)  # (batch, seq_len+1)
        mask[:, 0] = False  # CLS token is never masked

        # Apply transformer
        x = self.dropout(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm(x)

        # Return CLS token representation
        return x[:, 0]
