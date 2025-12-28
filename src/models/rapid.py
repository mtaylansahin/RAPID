"""
RAPID: A Recurrent Architecture for Predicting Protein Interaction Dynamics.

Based on the RE-Net architecture, adapted for:
- Undirected graphs (no separate subject/object)
- Binary classification instead of entity ranking
- Fully autoregressive inference
"""

import torch
import torch.nn as nn
import dgl
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import numpy as np

from src.models.encoder import TemporalEncoder
from src.models.classifier import SymmetricEdgeClassifier
from src.models.rgcn import UndirectedRGCN, build_undirected_graph
from src.config import ModelConfig


class RAPIDModel(nn.Module):
    """
    RAPID: A Recurrent Architecture for Predicting Protein Interaction Dynamics.
    
    Based on the RE-Net architecture with key adaptations:
    1. Undirected edges with canonical ordering (min_id, max_id)
    2. Unified history per entity (not separate s_hist/o_hist)
    3. Binary classification head instead of entity ranking
    4. Supports autoregressive inference with predicted history updates
    
    Args:
        num_entities: Number of entities (residues)
        num_rels: Number of relation types (used as edge features)
        config: ModelConfig with architecture hyperparameters
    """
    
    def __init__(
        self,
        num_entities: int,
        num_rels: int,
        config: ModelConfig,
    ):
        super().__init__()
        
        self.num_entities = num_entities
        self.num_rels = num_rels
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.seq_len = config.seq_len
        
        # Entity embeddings
        self.entity_embeds = nn.Parameter(
            torch.Tensor(num_entities, config.hidden_dim)
        )
        nn.init.xavier_uniform_(
            self.entity_embeds,
            gain=nn.init.calculate_gain('relu')
        )
        
        # Relation embeddings (used as edge features in RGCN)
        self.rel_embeds = nn.Parameter(
            torch.Tensor(num_rels, config.hidden_dim)
        )
        nn.init.xavier_uniform_(
            self.rel_embeds,
            gain=nn.init.calculate_gain('relu')
        )
        
        # RGCN for encoding historical graphs
        self.rgcn = UndirectedRGCN(
            hidden_dim=config.hidden_dim,
            num_rels=num_rels,
            num_layers=config.num_rgcn_layers,
            num_bases=config.num_bases,
            dropout=config.dropout,
        )
        
        # Temporal encoder (GRU)
        # Input: entity_embed + neighbor_embed + rel_embed + global_embed = 4 * hidden_dim
        self.temporal_encoder = TemporalEncoder(
            input_dim=4 * config.hidden_dim,
            hidden_dim=config.hidden_dim,
            num_layers=1,
            dropout=config.dropout,
        )
        
        # Binary classifier for edge prediction
        self.classifier = SymmetricEdgeClassifier(
            hidden_dim=config.hidden_dim,
            classifier_hidden_dim=config.classifier_hidden_dim,
            dropout=config.classifier_dropout,
            scoring_fn='concat',
            use_temporal=True,
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
        # === State for autoregressive inference ===
        # History per entity: list of (neighbors, rel_types, timestep)
        self._entity_history: List[List[Dict]] = []
        self._entity_history_t: List[List[int]] = []
        
        # Cache for current timestep predictions (before committing to history)
        self._prediction_cache: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
        
        # Graph dict: timestep -> DGLGraph
        self._graph_dict: Dict[int, dgl.DGLGraph] = {}
        
        # Global embeddings per timestep
        self._global_emb: Dict[int, torch.Tensor] = {}
        
        # Current timestep being processed
        self._latest_time: Optional[int] = None
    
    def _canonical_edge(self, e1: int, e2: int) -> Tuple[int, int]:
        """Return edge in canonical order (smaller id first)."""
        return (min(e1, e2), max(e1, e2))
    
    def reset_inference_state(self) -> None:
        """Reset all inference state. Call before starting new inference run."""
        self._entity_history = [[] for _ in range(self.num_entities)]
        self._entity_history_t = [[] for _ in range(self.num_entities)]
        self._prediction_cache = defaultdict(set)
        self._graph_dict = {}
        self._global_emb = {}
        self._latest_time = None
    
    def init_from_train_history(
        self,
        graph_dict: Dict[int, dgl.DGLGraph],
        entity_history: List[List[Dict]],
        entity_history_t: List[List[int]],
        global_emb: Optional[Dict[int, torch.Tensor]] = None,
    ) -> None:
        """
        Initialize inference state from training data.
        
        Args:
            graph_dict: Pre-computed graphs per timestep
            entity_history: History per entity from training
            entity_history_t: Timestamps for history entries
            global_emb: Global embeddings per timestep
        """
        self._graph_dict = graph_dict.copy()
        self._entity_history = [h.copy() for h in entity_history]
        self._entity_history_t = [t.copy() for t in entity_history_t]
        self._global_emb = global_emb.copy() if global_emb else {}
    
    def _get_entity_temporal_embed(
        self,
        entity_id: int,
    ) -> torch.Tensor:
        """
        Get temporal embedding for an entity based on its history.
        
        Args:
            entity_id: Entity ID
        
        Returns:
            Temporal embedding of shape (hidden_dim,)
        """
        history = self._entity_history[entity_id]
        history_t = self._entity_history_t[entity_id]
        
        if len(history) == 0:
            return torch.zeros(self.hidden_dim, device=self.entity_embeds.device)
        
        # Take last seq_len entries
        history = history[-self.seq_len:]
        history_t = history_t[-self.seq_len:]
        
        # Build sequence tensor
        seq_len = len(history)
        embed_dim = 4 * self.hidden_dim
        seq_tensor = torch.zeros(1, seq_len, embed_dim, device=self.entity_embeds.device)
        
        for i, (hist_entry, t) in enumerate(zip(history, history_t)):
            # Get RGCN-enhanced embedding if graph exists
            if t in self._graph_dict:
                g = self._graph_dict[t]
                # Check if node_ids are stored in ndata
                if 'id' in g.ndata:
                    node_features = self.entity_embeds[g.ndata['id'].view(-1)]
                else:
                    node_features = self.entity_embeds[:g.num_nodes()]
                node_features = self.rgcn(g, node_features)
                
                # Check if entity is in this graph
                # Use ndata['id'] mapping or assume identity mapping
                if 'id' in g.ndata:
                    node_ids = g.ndata['id'].view(-1).tolist()
                    if entity_id in node_ids:
                        local_idx = node_ids.index(entity_id)
                        entity_rgcn_embed = node_features[local_idx]
                    else:
                        entity_rgcn_embed = self.entity_embeds[entity_id]
                elif entity_id < g.num_nodes():
                    entity_rgcn_embed = node_features[entity_id]
                else:
                    entity_rgcn_embed = self.entity_embeds[entity_id]
            else:
                entity_rgcn_embed = self.entity_embeds[entity_id]
            
            # Global embedding
            if t in self._global_emb:
                global_embed = self._global_emb[t]
            else:
                global_embed = torch.zeros(self.hidden_dim, device=self.entity_embeds.device)
            
            # Mean relation embedding
            mean_rel = self.rel_embeds.mean(dim=0)
            
            seq_tensor[0, i] = torch.cat([
                entity_rgcn_embed,
                self.entity_embeds[entity_id],
                mean_rel,
                global_embed,
            ])
        
        # Pack and encode
        lengths = torch.LongTensor([seq_len])
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            seq_tensor, lengths, batch_first=True, enforce_sorted=False
        )
        
        temporal_embed = self.temporal_encoder(packed)
        return temporal_embed.squeeze(0)
    
    def forward(
        self,
        entity1_ids: torch.Tensor,
        entity2_ids: torch.Tensor,
        entity1_history: List[List[Dict]],
        entity2_history: List[List[Dict]],
        entity1_history_t: List[List[int]],
        entity2_history_t: List[List[int]],
        graph_dict: Dict[int, dgl.DGLGraph],
        global_emb: Optional[Dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training.
        
        Uses provided (ground-truth) histories - teacher forcing.
        
        Args:
            entity1_ids: First entity in each pair, shape (batch_size,)
            entity2_ids: Second entity in each pair, shape (batch_size,)
            entity1_history: History for each entity1
            entity2_history: History for each entity2
            entity1_history_t: Timestamps for entity1 history
            entity2_history_t: Timestamps for entity2 history
            graph_dict: Graphs per timestep
            global_emb: Global embeddings per timestep
        
        Returns:
            Logits of shape (batch_size,)
        """
        batch_size = len(entity1_ids)
        device = self.entity_embeds.device
        
        # Get entity embeddings
        entity1_embed = self.entity_embeds[entity1_ids]
        entity2_embed = self.entity_embeds[entity2_ids]
        
        # Get temporal embeddings from history
        entity1_temporal = torch.zeros(batch_size, self.hidden_dim, device=device)
        entity2_temporal = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        for i in range(batch_size):
            # Process entity1 history
            if len(entity1_history[i]) > 0:
                entity1_temporal[i] = self._encode_history(
                    entity1_ids[i].item(),
                    entity1_history[i],
                    entity1_history_t[i],
                    graph_dict,
                    global_emb,
                )
            
            # Process entity2 history
            if len(entity2_history[i]) > 0:
                entity2_temporal[i] = self._encode_history(
                    entity2_ids[i].item(),
                    entity2_history[i],
                    entity2_history_t[i],
                    graph_dict,
                    global_emb,
                )
        
        # Apply dropout
        entity1_embed = self.dropout(entity1_embed)
        entity2_embed = self.dropout(entity2_embed)
        entity1_temporal = self.dropout(entity1_temporal)
        entity2_temporal = self.dropout(entity2_temporal)
        
        # Classify
        logits = self.classifier(
            entity1_embed, entity2_embed,
            entity1_temporal, entity2_temporal,
        )
        
        return logits
    
    def _encode_history(
        self,
        entity_id: int,
        history: List[Dict],
        history_t: List[int],
        graph_dict: Dict[int, dgl.DGLGraph],
        global_emb: Optional[Dict[int, torch.Tensor]],
    ) -> torch.Tensor:
        """Encode history for a single entity."""
        history = history[-self.seq_len:]
        history_t = history_t[-self.seq_len:]
        
        seq_len = len(history)
        embed_dim = 4 * self.hidden_dim
        device = self.entity_embeds.device
        
        seq_tensor = torch.zeros(1, seq_len, embed_dim, device=device)
        
        for i, (hist_entry, t) in enumerate(zip(history, history_t)):
            if t in graph_dict:
                g = graph_dict[t]
                node_features = self.entity_embeds[g.ndata['id'].view(-1)]
                node_features = self.rgcn(g, node_features)
                
                if entity_id in g.ids:
                    entity_rgcn_embed = node_features[g.ids[entity_id]]
                else:
                    entity_rgcn_embed = self.entity_embeds[entity_id]
            else:
                entity_rgcn_embed = self.entity_embeds[entity_id]
            
            if global_emb and t in global_emb:
                g_emb = global_emb[t]
            else:
                g_emb = torch.zeros(self.hidden_dim, device=device)
            
            mean_rel = self.rel_embeds.mean(dim=0)
            
            seq_tensor[0, i] = torch.cat([
                entity_rgcn_embed,
                self.entity_embeds[entity_id],
                mean_rel,
                g_emb,
            ])
        
        lengths = torch.LongTensor([seq_len])
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            seq_tensor, lengths, batch_first=True, enforce_sorted=False
        )
        
        return self.temporal_encoder(packed).squeeze(0)
    
    def predict(
        self,
        entity1_id: int,
        entity2_id: int,
        timestep: int,
    ) -> float:
        """
        Predict edge existence autoregressively.
        
        Uses internal history state which gets updated with predictions.
        
        Args:
            entity1_id: First entity
            entity2_id: Second entity
            timestep: Current timestep
        
        Returns:
            Probability of edge existence
        """
        self.eval()
        
        # Handle timestep transition
        if self._latest_time is not None and self._latest_time != timestep:
            self._commit_predictions(self._latest_time)
        self._latest_time = timestep
        
        # Get embeddings
        device = self.entity_embeds.device
        e1 = torch.LongTensor([entity1_id]).to(device)
        e2 = torch.LongTensor([entity2_id]).to(device)
        
        entity1_embed = self.entity_embeds[e1]
        entity2_embed = self.entity_embeds[e2]
        
        # Get temporal embeddings from stored history
        entity1_temporal = self._get_entity_temporal_embed(entity1_id).unsqueeze(0)
        entity2_temporal = self._get_entity_temporal_embed(entity2_id).unsqueeze(0)
        
        with torch.no_grad():
            logit = self.classifier(
                entity1_embed, entity2_embed,
                entity1_temporal, entity2_temporal,
            )
            prob = torch.sigmoid(logit).item()
        
        return prob
    
    def predict_batch(
        self,
        entity1_ids: torch.Tensor,
        entity2_ids: torch.Tensor,
        timestep: int,
        threshold: float = 0.5,
        update_history: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict batch of edges autoregressively.
        
        Args:
            entity1_ids: First entities, shape (batch_size,)
            entity2_ids: Second entities, shape (batch_size,)
            timestep: Current timestep
            threshold: Classification threshold
            update_history: Whether to update history with predictions
        
        Returns:
            Tuple of (probabilities, predictions)
        """
        self.eval()
        
        # Handle timestep transition
        if self._latest_time is not None and self._latest_time != timestep:
            self._commit_predictions(self._latest_time)
        self._latest_time = timestep
        
        batch_size = len(entity1_ids)
        device = self.entity_embeds.device
        
        entity1_embed = self.entity_embeds[entity1_ids]
        entity2_embed = self.entity_embeds[entity2_ids]
        
        # Get temporal embeddings
        entity1_temporal = torch.zeros(batch_size, self.hidden_dim, device=device)
        entity2_temporal = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        with torch.no_grad():
            for i in range(batch_size):
                entity1_temporal[i] = self._get_entity_temporal_embed(entity1_ids[i].item())
                entity2_temporal[i] = self._get_entity_temporal_embed(entity2_ids[i].item())
            
            logits = self.classifier(
                entity1_embed, entity2_embed,
                entity1_temporal, entity2_temporal,
            )
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()
        
        # Update prediction cache for positive predictions
        if update_history:
            for i in range(batch_size):
                if preds[i] == 1:
                    e1, e2 = entity1_ids[i].item(), entity2_ids[i].item()
                    self._prediction_cache[timestep].add(self._canonical_edge(e1, e2))
        
        return probs, preds
    
    def _commit_predictions(self, timestep: int) -> None:
        """
        Commit predictions from cache to history.
        
        Called when moving to next timestep.
        """
        if timestep not in self._prediction_cache:
            return
        
        predicted_edges = self._prediction_cache[timestep]
        
        if len(predicted_edges) == 0:
            return
        
        # Build graph for this timestep
        edges = torch.LongTensor(list(predicted_edges))
        # Assign relation type 0 for predicted edges (since we predict binary)
        rel_types = torch.zeros(len(edges), dtype=torch.long)
        
        g = build_undirected_graph(
            edges=edges,
            rel_types=rel_types,
            num_nodes=self.num_entities,
            node_ids=torch.arange(self.num_entities),
        )
        
        # Add to graph dict
        self._graph_dict[timestep] = g
        
        # Update entity histories
        for e1, e2 in predicted_edges:
            # Add to entity1's history
            self._entity_history[e1].append({
                'neighbors': [e2],
                'rel_types': [0],
            })
            self._entity_history_t[e1].append(timestep)
            
            # Add to entity2's history
            self._entity_history[e2].append({
                'neighbors': [e1],
                'rel_types': [0],
            })
            self._entity_history_t[e2].append(timestep)
            
            # Trim to seq_len
            if len(self._entity_history[e1]) > self.seq_len:
                self._entity_history[e1] = self._entity_history[e1][-self.seq_len:]
                self._entity_history_t[e1] = self._entity_history_t[e1][-self.seq_len:]
            if len(self._entity_history[e2]) > self.seq_len:
                self._entity_history[e2] = self._entity_history[e2][-self.seq_len:]
                self._entity_history_t[e2] = self._entity_history_t[e2][-self.seq_len:]
        
        # Clear cache for this timestep
        del self._prediction_cache[timestep]


def create_model(
    num_entities: int,
    num_rels: int,
    config: Optional[ModelConfig] = None,
) -> RAPIDModel:
    """
    Factory function to create RAPID model.
    
    Args:
        num_entities: Number of entities
        num_rels: Number of relation types
        config: Model configuration (uses default if None)
    
    Returns:
        Initialized RAPIDModel
    """
    if config is None:
        config = ModelConfig()
    
    return RAPIDModel(
        num_entities=num_entities,
        num_rels=num_rels,
        config=config,
    )
