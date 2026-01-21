"""Encoder wrapper for RAPID model."""

from typing import Dict, Optional

import dgl
import torch
import torch.nn as nn


class RAPIDEncoder(nn.Module):
    """
    Wrapper around RAPIDModel for use as encoder in encoder-decoder setup.

    Exposes encode_context() for getting static entity representations.
    Can optionally freeze encoder weights for decoder-only training.

    Args:
        rapid_model: The underlying RAPIDModel
        freeze: Whether to freeze encoder weights
    """

    def __init__(self, rapid_model, freeze: bool = False):
        super().__init__()
        self.rapid = rapid_model

        if freeze:
            for param in self.rapid.parameters():
                param.requires_grad = False

    @property
    def num_entities(self) -> int:
        return self.rapid.num_entities

    @property
    def hidden_dim(self) -> int:
        return self.rapid.hidden_dim

    def forward(
        self,
        graph_dict: Dict[int, dgl.DGLGraph],
    ) -> torch.Tensor:
        """
        Encode all entities into context matrix using RGCN on most recent graph.

        Args:
            graph_dict: Dict mapping timestep -> DGLGraph

        Returns:
            Entity context matrix: (num_entities, hidden_dim)
        """
        return self.rapid.encode_context(graph_dict)

    def parameters(self, recurse: bool = True):
        """Return encoder parameters."""
        return self.rapid.parameters(recurse=recurse)
