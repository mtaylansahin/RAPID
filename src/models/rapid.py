"""
RAPID: A Recurrent Architecture for Predicting Protein Interaction Dynamics.

Simplified architecture:
- RGCN for structural entity embeddings
- Entity embeddings (static + optional node features)
- Temporal dynamics handled by EdgeCentricSubgraphEncoder in decoder
"""

from typing import Dict, Optional, Set

import dgl
import torch
import torch.nn as nn

from src.config import ModelConfig
from src.models.rgcn import UndirectedRGCN


class RAPIDModel(nn.Module):
    """
    RAPID: A Model for Predicting Protein Interaction Dynamics.

    Simplified architecture where:
    1. RGCN provides structural context for entity embeddings
    2. Temporal dynamics are handled by EdgeCentricSubgraphEncoder in decoder
    3. This encoder only produces static entity context

    Args:
        num_entities: Number of entities (residues)
        num_rels: Number of relation types (used as edge features)
        config: ModelConfig with architecture hyperparameters
        node_features: Optional precomputed node features
    """

    def __init__(
        self,
        num_entities: int,
        num_rels: int,
        config: ModelConfig,
        node_features: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.num_entities = num_entities
        self.num_rels = num_rels
        self.config = config
        self.hidden_dim = config.hidden_dim

        # Entity embeddings (learnable)
        self.entity_embeds = nn.Parameter(torch.Tensor(num_entities, config.hidden_dim))
        nn.init.xavier_uniform_(self.entity_embeds, gain=nn.init.calculate_gain("relu"))

        # Relation embeddings (for RGCN edge types)
        self.rel_embeds = nn.Parameter(torch.Tensor(num_rels, config.hidden_dim))
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain("relu"))

        # Node features (optional)
        self.use_node_features = node_features is not None
        if self.use_node_features:
            self.register_buffer("node_features", node_features)
            self.node_feature_proj = nn.Linear(
                node_features.shape[1], config.hidden_dim
            )
        else:
            self.register_buffer("node_features", None)
            self.node_feature_proj = None

        # RGCN for structural context
        self.rgcn = UndirectedRGCN(
            hidden_dim=config.hidden_dim,
            num_rels=num_rels,
            num_layers=config.num_rgcn_layers,
            num_bases=config.num_bases,
            dropout=config.dropout,
        )

        self.dropout = nn.Dropout(config.dropout)
        self._dgl_has_cuda = self._check_dgl_cuda_support()

        # Cache for RGCN outputs
        self._rgcn_cache: Dict[int, torch.Tensor] = {}

    def get_entity_embed(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """Get entity embeddings, optionally enhanced with node features."""
        base_embed = self.entity_embeds[entity_ids]

        if self.use_node_features and self.node_features is not None:
            feat = self.node_features[entity_ids]
            feat_proj = self.node_feature_proj(feat)
            return base_embed + feat_proj

        return base_embed

    def _check_dgl_cuda_support(self) -> bool:
        """Check if DGL supports CUDA operations."""
        try:
            test_g = dgl.graph(([0], [1]))
            if torch.cuda.is_available():
                test_g.to("cuda:0")
            return True
        except Exception:
            return False

    def to(self, device, *args, **kwargs):
        """Override to keep RGCN on CPU if DGL doesn't support CUDA."""
        result = super().to(device, *args, **kwargs)
        if not self._dgl_has_cuda and "cuda" in str(device):
            self.rgcn = self.rgcn.to("cpu")
        return result

    def cuda(self, device=None):
        """Override to keep RGCN on CPU if DGL doesn't support CUDA."""
        result = super().cuda(device)
        if not self._dgl_has_cuda:
            self.rgcn = self.rgcn.to("cpu")
        return result

    def _precompute_rgcn(
        self, timesteps: Set[int], graph_dict: Dict[int, dgl.DGLGraph]
    ) -> None:
        """Pre-compute RGCN outputs for all needed timesteps."""
        device = self.entity_embeds.device
        rgcn_device = next(self.rgcn.parameters()).device

        for t in timesteps:
            if t in self._rgcn_cache:
                continue
            if t not in graph_dict:
                continue

            g = graph_dict[t]
            if self._dgl_has_cuda and g.device != rgcn_device:
                g = g.to(rgcn_device)
            node_features = self.entity_embeds[g.ndata["id"].view(-1)].to(rgcn_device)
            node_features = self.rgcn(g, node_features)

            if rgcn_device != device:
                node_features = node_features.to(device)

            self._rgcn_cache[t] = node_features

    def encode_context(
        self,
        graph_dict: Dict[int, dgl.DGLGraph],
    ) -> torch.Tensor:
        """
        Encode all entities into static context matrix.

        Uses RGCN on the most recent graph to get structure-aware embeddings,
        combined with base entity embeddings (+ optional node features).

        Args:
            graph_dict: Graphs per timestep

        Returns:
            Entity context matrix: (num_entities, hidden_dim)
        """
        self._rgcn_cache = {}
        device = self.entity_embeds.device

        # Get most recent timestep with a graph
        if not graph_dict:
            # No graphs - just return base embeddings
            return self.get_entity_embed(torch.arange(self.num_entities, device=device))

        latest_t = max(graph_dict.keys())
        g = graph_dict[latest_t]

        # Run RGCN on latest graph
        rgcn_device = next(self.rgcn.parameters()).device
        if self._dgl_has_cuda and g.device != rgcn_device:
            g = g.to(rgcn_device)

        if "id" in g.ndata:
            node_ids = g.ndata["id"].view(-1)
            node_features = self.entity_embeds[node_ids].to(rgcn_device)
        else:
            node_ids = torch.arange(g.num_nodes(), device=rgcn_device)
            node_features = self.entity_embeds[: g.num_nodes()].to(rgcn_device)

        rgcn_output = self.rgcn(g, node_features)

        if rgcn_device != device:
            rgcn_output = rgcn_output.to(device)

        # Build full entity context
        entity_context = self.get_entity_embed(
            torch.arange(self.num_entities, device=device)
        )

        # Add RGCN output for entities present in the graph
        for local_idx, entity_id in enumerate(node_ids.tolist()):
            entity_context[entity_id] = (
                entity_context[entity_id] + rgcn_output[local_idx]
            )

        return entity_context

    def forward(
        self,
        entity1_ids: torch.Tensor,
        entity2_ids: torch.Tensor,
        graph_dict: Dict[int, dgl.DGLGraph],
    ) -> torch.Tensor:
        """
        Get entity embeddings for a batch of pairs.

        This is a simplified forward pass that only returns entity embeddings.
        The full prediction pipeline uses the decoder with edge history.

        Args:
            entity1_ids: First entity in each pair, shape (batch_size,)
            entity2_ids: Second entity in each pair, shape (batch_size,)
            graph_dict: Graphs per timestep (for RGCN context)

        Returns:
            Tuple of (entity1_embed, entity2_embed), each (batch_size, hidden_dim)
        """
        # Get entity embeddings with node features
        entity1_embed = self.get_entity_embed(entity1_ids)
        entity2_embed = self.get_entity_embed(entity2_ids)

        return self.dropout(entity1_embed), self.dropout(entity2_embed)


def create_model(
    num_entities: int,
    num_rels: int,
    config: Optional[ModelConfig] = None,
    node_features: Optional[torch.Tensor] = None,
) -> RAPIDModel:
    """
    Factory function to create RAPID model.

    Args:
        num_entities: Number of entities
        num_rels: Number of relation types
        config: Model configuration (uses default if None)
        node_features: Pre-computed node features tensor (optional)

    Returns:
        Initialized RAPIDModel
    """
    if config is None:
        config = ModelConfig()

    return RAPIDModel(
        num_entities=num_entities,
        num_rels=num_rels,
        config=config,
        node_features=node_features,
    )
