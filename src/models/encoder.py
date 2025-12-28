"""GRU-based temporal encoder."""

import torch
import torch.nn as nn
from typing import Optional, Tuple


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
