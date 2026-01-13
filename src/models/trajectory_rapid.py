"""
TrajectoryRAPID: Hybrid model combining RGCN-based entity encoding with trajectory prediction.

Combines strengths of:
- RAPID: RGCN for structure-aware entity embeddings, global context
- Trajectory: Transformer encoder/decoder for full trajectory prediction

Architecture:
1. RGCN encodes entities with structural context at each history timestep
2. Transformer encoder processes edge history with RGCN features
3. Cross-attention to neighbor edges
4. Transformer decoder predicts full future trajectory
"""

import math
from typing import Dict, List, Optional, Tuple

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.rgcn import UndirectedRGCN


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
        """Add positional encoding to input.

        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        x = x + self.pe[: x.size(1)]
        return self.dropout(x)


class EdgeHistoryEncoder(nn.Module):
    """
    Encodes an edge's history into a dense representation.

    Enhanced version that incorporates:
    - Binary edge state
    - RGCN-derived entity features (optional)
    - Global context embedding (optional)

    Uses Transformer encoder for temporal modeling.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_rgcn_features: bool = True,
        use_global_context: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_rgcn_features = use_rgcn_features
        self.use_global_context = use_global_context

        # Input projection
        # Base: binary state (1) + entity pair embeddings (2 * hidden_dim)
        input_dim = 1 + 2 * hidden_dim
        if use_rgcn_features:
            input_dim += 2 * hidden_dim  # RGCN features for both entities
        if use_global_context:
            input_dim += hidden_dim  # Global context

        self.expected_input_dim = input_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.pos_encoding = PositionalEncoding(hidden_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        history: torch.Tensor,
        entity1_embed: torch.Tensor,
        entity2_embed: torch.Tensor,
        entity1_rgcn: Optional[torch.Tensor] = None,
        entity2_rgcn: Optional[torch.Tensor] = None,
        global_context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode edge history.

        Args:
            history: Binary states (batch, seq_len)
            entity1_embed: Entity embeddings (batch, hidden_dim)
            entity2_embed: Entity embeddings (batch, hidden_dim)
            entity1_rgcn: RGCN features (batch, seq_len, hidden_dim) or None
            entity2_rgcn: RGCN features (batch, seq_len, hidden_dim) or None
            global_context: Global embeddings (batch, seq_len, hidden_dim) or None
            mask: Padding mask (batch, seq_len)

        Returns:
            Encoded representation (batch, hidden_dim)
        """
        batch_size, seq_len = history.shape
        device = history.device

        # Expand entity embeddings to sequence
        e1_expand = entity1_embed.unsqueeze(1).expand(-1, seq_len, -1)
        e2_expand = entity2_embed.unsqueeze(1).expand(-1, seq_len, -1)

        # Build input features
        features = [history.unsqueeze(-1), e1_expand, e2_expand]

        if self.use_rgcn_features:
            if entity1_rgcn is not None:
                features.append(entity1_rgcn)
                features.append(entity2_rgcn)
            else:
                # Provide zeros if RGCN expected but not provided
                zeros = torch.zeros(batch_size, seq_len, self.hidden_dim, device=device)
                features.append(zeros)
                features.append(zeros)

        if self.use_global_context:
            if global_context is not None:
                features.append(global_context)
            else:
                # Provide zeros if global context expected but not provided
                zeros = torch.zeros(batch_size, seq_len, self.hidden_dim, device=device)
                features.append(zeros)

        x = torch.cat(features, dim=-1)
        x = self.input_proj(x)
        x = self.pos_encoding(x)

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)

        # Pool: mean over non-padded positions
        if mask is not None:
            keep_mask = ~mask
            x = (x * keep_mask.unsqueeze(-1)).sum(dim=1) / keep_mask.sum(
                dim=1, keepdim=True
            ).clamp(min=1)
        else:
            x = x.mean(dim=1)

        return self.output_proj(x)


class NeighborCrossAttention(nn.Module):
    """
    Cross-attention from target edge to neighbor edges.

    Target edge attends to neighbor edge embeddings to gather context.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "cross_attn": nn.MultiheadAttention(
                            embed_dim=hidden_dim,
                            num_heads=n_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "norm1": nn.LayerNorm(hidden_dim),
                        "ffn": nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim * 4),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(hidden_dim * 4, hidden_dim),
                            nn.Dropout(dropout),
                        ),
                        "norm2": nn.LayerNorm(hidden_dim),
                    }
                )
            )

    def forward(
        self,
        query: torch.Tensor,
        neighbor_embeds: torch.Tensor,
        neighbor_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-attend to neighbors.

        Args:
            query: Target edge embedding (batch, hidden_dim)
            neighbor_embeds: Neighbor embeddings (batch, n_neighbors, hidden_dim)
            neighbor_mask: Padding mask (batch, n_neighbors), True = pad

        Returns:
            Updated query (batch, hidden_dim)
        """
        # Expand query for attention
        q = query.unsqueeze(1)  # (batch, 1, hidden_dim)

        for layer in self.layers:
            # Cross-attention
            attn_out, _ = layer["cross_attn"](
                query=q,
                key=neighbor_embeds,
                value=neighbor_embeds,
                key_padding_mask=neighbor_mask,
            )
            q = layer["norm1"](q + attn_out)

            # FFN
            ffn_out = layer["ffn"](q)
            q = layer["norm2"](q + ffn_out)

        return q.squeeze(1)


class TrajectoryDecoder(nn.Module):
    """
    Transformer decoder for predicting full future trajectory.

    Uses learnable position queries for each timestep.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_traj_len: int = 200,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_traj_len = max_traj_len

        # Learnable position queries
        self.pos_queries = nn.Parameter(torch.randn(max_traj_len, hidden_dim) * 0.02)

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

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        edge_embed: torch.Tensor,
        traj_len: int,
        global_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode future trajectory.

        Args:
            edge_embed: Encoded edge representation (batch, hidden_dim)
            traj_len: Number of timesteps to predict
            global_context: Optional global context (batch, hidden_dim)

        Returns:
            Trajectory logits (batch, traj_len)
        """
        batch_size = edge_embed.size(0)

        # Position queries for requested length
        queries = self.pos_queries[:traj_len].unsqueeze(0).expand(batch_size, -1, -1)
        queries = self.pos_encoding(queries)

        # Memory: edge embedding (+ global context if provided)
        if global_context is not None:
            memory = torch.stack([edge_embed, global_context], dim=1)
        else:
            memory = edge_embed.unsqueeze(1)

        # Decode
        decoded = self.transformer(queries, memory)

        # Project to logits
        logits = self.output_head(decoded).squeeze(-1)

        return logits


class TrajectoryRAPIDModel(nn.Module):
    """
    Hybrid model combining RGCN structure encoding with trajectory prediction.

    Feature Flags:
    - use_rgcn: Enable RGCN for structural context
    - use_global_context: Enable global graph embeddings
    - use_neighbor_attention: Enable cross-attention to neighbors
    - use_node_features: Enable physicochemical node features
    """

    def __init__(
        self,
        num_entities: int,
        num_rels: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 2,
        n_neighbor_layers: int = 1,
        max_neighbors: int = 50,
        max_seq_len: int = 200,
        dropout: float = 0.1,
        # Feature flags for ablation
        use_rgcn: bool = True,
        use_global_context: bool = True,
        use_neighbor_attention: bool = True,
        use_node_features: bool = False,
        node_features: Optional[torch.Tensor] = None,
        # RGCN settings
        num_rgcn_layers: int = 2,
        num_bases: int = 100,
    ):
        super().__init__()

        self.num_entities = num_entities
        self.num_rels = num_rels
        self.hidden_dim = hidden_dim
        self.max_neighbors = max_neighbors

        # Feature flags
        self.use_rgcn = use_rgcn
        self.use_global_context = use_global_context
        self.use_neighbor_attention = use_neighbor_attention
        self.use_node_features = use_node_features and node_features is not None

        # Entity embeddings
        self.entity_embeds = nn.Parameter(torch.Tensor(num_entities, hidden_dim))
        nn.init.xavier_uniform_(self.entity_embeds, gain=nn.init.calculate_gain("relu"))

        # Node features (optional)
        if self.use_node_features:
            self.register_buffer("node_features", node_features)
            self.node_feature_proj = nn.Linear(node_features.shape[1], hidden_dim)
        else:
            self.register_buffer("node_features", None)
            self.node_feature_proj = None

        # RGCN for structural context
        if self.use_rgcn:
            self.rgcn = UndirectedRGCN(
                hidden_dim=hidden_dim,
                num_rels=num_rels,
                num_layers=num_rgcn_layers,
                num_bases=num_bases,
                dropout=dropout,
            )
        else:
            self.rgcn = None

        # Edge history encoder
        self.history_encoder = EdgeHistoryEncoder(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            dropout=dropout,
            use_rgcn_features=use_rgcn,
            use_global_context=use_global_context,
        )

        # Neighbor cross-attention
        if self.use_neighbor_attention:
            self.neighbor_attention = NeighborCrossAttention(
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                n_layers=n_neighbor_layers,
                dropout=dropout,
            )
        else:
            self.neighbor_attention = None

        # Trajectory decoder
        self.decoder = TrajectoryDecoder(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_decoder_layers,
            max_traj_len=max_seq_len,
            dropout=dropout,
        )

        # RGCN output cache (filled during forward)
        self._rgcn_cache: Dict[int, torch.Tensor] = {}

    def get_entity_embed(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """Get entity embeddings with optional node features."""
        base_embed = self.entity_embeds[entity_ids]

        if self.use_node_features and self.node_features is not None:
            feat = self.node_features[entity_ids]
            feat_proj = self.node_feature_proj(feat)
            return base_embed + feat_proj

        return base_embed

    def _compute_rgcn_features(
        self,
        entity_ids: torch.Tensor,
        history_timesteps: torch.Tensor,
        graph_dict: Dict[int, dgl.DGLGraph],
    ) -> torch.Tensor:
        """
        Compute RGCN features for entities at each history timestep.

        Args:
            entity_ids: Entity IDs (batch,)
            history_timesteps: Timesteps (seq_len,)
            graph_dict: Dict of timestep -> DGLGraph

        Returns:
            RGCN features (batch, seq_len, hidden_dim)
        """
        batch_size = entity_ids.size(0)
        seq_len = len(history_timesteps)
        device = self.entity_embeds.device

        rgcn_features = torch.zeros(batch_size, seq_len, self.hidden_dim, device=device)

        for t_idx, t in enumerate(history_timesteps.tolist()):
            if t not in graph_dict:
                # No graph for this timestep, use base embeddings
                rgcn_features[:, t_idx] = self.entity_embeds[entity_ids]
                continue

            # Check cache
            if t not in self._rgcn_cache:
                g = graph_dict[t]
                node_ids = g.ndata.get("id", torch.arange(g.num_nodes()))
                node_features = self.entity_embeds[node_ids]
                updated = self.rgcn(g, node_features)
                self._rgcn_cache[t] = (node_ids, updated)

            node_ids, updated = self._rgcn_cache[t]

            # Map entity_ids to node indices
            for b in range(batch_size):
                eid = entity_ids[b].item()
                # Find if entity is in this graph
                matches = (node_ids == eid).nonzero(as_tuple=True)[0]
                if len(matches) > 0:
                    rgcn_features[b, t_idx] = updated[matches[0]]
                else:
                    rgcn_features[b, t_idx] = self.entity_embeds[eid]

        return rgcn_features

    def _encode_neighbors(
        self,
        neighbor_e1: torch.Tensor,
        neighbor_e2: torch.Tensor,
        neighbor_history: torch.Tensor,
        neighbor_mask: torch.Tensor,
        history_timesteps: torch.Tensor,
        graph_dict: Dict[int, dgl.DGLGraph],
        global_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode neighbor edges.

        Args:
            neighbor_e1: (batch, n_neighbors)
            neighbor_e2: (batch, n_neighbors)
            neighbor_history: (batch, n_neighbors, seq_len)
            neighbor_mask: (batch, n_neighbors)
            history_timesteps: (seq_len,)
            graph_dict: Timestep graphs
            global_context: (batch, seq_len, hidden_dim) or None

        Returns:
            Neighbor embeddings (batch, n_neighbors, hidden_dim)
        """
        batch_size, n_neighbors, seq_len = neighbor_history.shape
        device = self.entity_embeds.device

        # Flatten for batch encoding
        flat_e1 = neighbor_e1.view(-1)
        flat_e2 = neighbor_e2.view(-1)
        flat_history = neighbor_history.view(batch_size * n_neighbors, seq_len)

        # Get entity embeddings
        e1_embed = self.get_entity_embed(flat_e1)
        e2_embed = self.get_entity_embed(flat_e2)

        # Get RGCN features if enabled
        if self.use_rgcn:
            e1_rgcn = self._compute_rgcn_features(flat_e1, history_timesteps, graph_dict)
            e2_rgcn = self._compute_rgcn_features(flat_e2, history_timesteps, graph_dict)
        else:
            e1_rgcn = None
            e2_rgcn = None

        # Expand global context for neighbors
        if global_context is not None:
            flat_global = global_context.unsqueeze(1).expand(
                -1, n_neighbors, -1, -1
            ).reshape(batch_size * n_neighbors, seq_len, -1)
        else:
            flat_global = None

        # Encode
        neighbor_embeds = self.history_encoder(
            history=flat_history,
            entity1_embed=e1_embed,
            entity2_embed=e2_embed,
            entity1_rgcn=e1_rgcn,
            entity2_rgcn=e2_rgcn,
            global_context=flat_global,
        )

        return neighbor_embeds.view(batch_size, n_neighbors, -1)

    def forward(
        self,
        entity1_ids: torch.Tensor,
        entity2_ids: torch.Tensor,
        history: torch.Tensor,
        history_timesteps: torch.Tensor,
        neighbor_entity1: torch.Tensor,
        neighbor_entity2: torch.Tensor,
        neighbor_history: torch.Tensor,
        neighbor_mask: torch.Tensor,
        traj_len: int,
        graph_dict: Optional[Dict[int, dgl.DGLGraph]] = None,
        global_emb: Optional[Dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            entity1_ids: First entity (batch,)
            entity2_ids: Second entity (batch,)
            history: Edge history (batch, hist_len)
            history_timesteps: Timestep indices (hist_len,)
            neighbor_entity1: (batch, n_neighbors)
            neighbor_entity2: (batch, n_neighbors)
            neighbor_history: (batch, n_neighbors, hist_len)
            neighbor_mask: (batch, n_neighbors) True = padding
            traj_len: Number of timesteps to predict
            graph_dict: Optional timestep -> DGLGraph for RGCN
            global_emb: Optional timestep -> global embedding

        Returns:
            Trajectory logits (batch, traj_len)
        """
        # Clear RGCN cache
        self._rgcn_cache = {}

        batch_size, seq_len = history.shape
        device = self.entity_embeds.device

        # Get entity embeddings
        e1_embed = self.get_entity_embed(entity1_ids)
        e2_embed = self.get_entity_embed(entity2_ids)

        # Compute RGCN features if enabled
        if self.use_rgcn and graph_dict is not None:
            e1_rgcn = self._compute_rgcn_features(
                entity1_ids, history_timesteps, graph_dict
            )
            e2_rgcn = self._compute_rgcn_features(
                entity2_ids, history_timesteps, graph_dict
            )
        else:
            e1_rgcn = None
            e2_rgcn = None

        # Build global context sequence if enabled
        if self.use_global_context and global_emb is not None:
            global_context = torch.zeros(batch_size, seq_len, self.hidden_dim, device=device)
            for t_idx, t in enumerate(history_timesteps.tolist()):
                if t in global_emb:
                    global_context[:, t_idx] = global_emb[t]
        else:
            global_context = None

        # Encode target edge history
        edge_embed = self.history_encoder(
            history=history,
            entity1_embed=e1_embed,
            entity2_embed=e2_embed,
            entity1_rgcn=e1_rgcn,
            entity2_rgcn=e2_rgcn,
            global_context=global_context,
        )

        # Neighbor cross-attention if enabled
        if self.use_neighbor_attention and self.neighbor_attention is not None:
            neighbor_embeds = self._encode_neighbors(
                neighbor_e1=neighbor_entity1,
                neighbor_e2=neighbor_entity2,
                neighbor_history=neighbor_history,
                neighbor_mask=neighbor_mask,
                history_timesteps=history_timesteps,
                graph_dict=graph_dict if graph_dict else {},
                global_context=global_context,
            )

            edge_embed = self.neighbor_attention(
                query=edge_embed,
                neighbor_embeds=neighbor_embeds,
                neighbor_mask=neighbor_mask,
            )

        # Decode trajectory
        # Use mean global context for decoder if available
        if global_context is not None:
            decoder_global = global_context.mean(dim=1)
        else:
            decoder_global = None

        logits = self.decoder(
            edge_embed=edge_embed,
            traj_len=traj_len,
            global_context=decoder_global,
        )

        return logits

    def predict(
        self,
        entity1_ids: torch.Tensor,
        entity2_ids: torch.Tensor,
        history: torch.Tensor,
        history_timesteps: torch.Tensor,
        neighbor_entity1: torch.Tensor,
        neighbor_entity2: torch.Tensor,
        neighbor_history: torch.Tensor,
        neighbor_mask: torch.Tensor,
        traj_len: int,
        threshold: float = 0.5,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict trajectories with probabilities and binary predictions.

        Returns:
            (probabilities, binary_predictions) each of shape (batch, traj_len)
        """
        logits = self.forward(
            entity1_ids=entity1_ids,
            entity2_ids=entity2_ids,
            history=history,
            history_timesteps=history_timesteps,
            neighbor_entity1=neighbor_entity1,
            neighbor_entity2=neighbor_entity2,
            neighbor_history=neighbor_history,
            neighbor_mask=neighbor_mask,
            traj_len=traj_len,
            **kwargs,
        )

        probs = torch.sigmoid(logits)
        preds = (probs > threshold).long()

        return probs, preds
