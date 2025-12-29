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
        
        # Check if DGL supports CUDA
        self._dgl_has_cuda = self._check_dgl_cuda_support()
        
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
        
        # Reference to global model for on-the-fly embedding computation
        self._global_model: Optional[nn.Module] = None
        
        # Current timestep being processed
        self._latest_time: Optional[int] = None
        
        # Cache for RGCN outputs per timestep (cleared each forward pass)
        self._rgcn_cache: Dict[int, torch.Tensor] = {}
    
    def _canonical_edge(self, e1: int, e2: int) -> Tuple[int, int]:
        """Return edge in canonical order (smaller id first)."""
        return (min(e1, e2), max(e1, e2))
    
    def _get_node_idx(self, g: dgl.DGLGraph, entity_id: int) -> Optional[int]:
        """Get the local node index for an entity in a graph, or None if not present."""
        if hasattr(g, 'ids') and g.ids is not None:
            # Legacy custom attribute
            if entity_id in g.ids:
                return g.ids[entity_id]
            return None
        elif 'id' in g.ndata:
            # Standard ndata approach
            node_ids = g.ndata['id'].view(-1).tolist()
            if entity_id in node_ids:
                return node_ids.index(entity_id)
            return None
        else:
            # Assume identity mapping
            if entity_id < g.num_nodes():
                return entity_id
            return None
    
    def _check_dgl_cuda_support(self) -> bool:
        """Check if DGL supports CUDA operations."""
        try:
            # Create a small test graph and try to move it to CUDA
            test_g = dgl.graph(([0], [1]))
            if torch.cuda.is_available():
                test_g.to('cuda:0')
            return True
        except Exception:
            return False
    
    def to(self, device, *args, **kwargs):
        """Override to keep RGCN on CPU if DGL doesn't support CUDA."""
        result = super().to(device, *args, **kwargs)
        # If DGL doesn't support CUDA, keep RGCN on CPU
        if not self._dgl_has_cuda and 'cuda' in str(device):
            self.rgcn = self.rgcn.to('cpu')
        return result
    
    def cuda(self, device=None):
        """Override to keep RGCN on CPU if DGL doesn't support CUDA."""
        result = super().cuda(device)
        if not self._dgl_has_cuda:
            self.rgcn = self.rgcn.to('cpu')
        return result
    
    def reset_inference_state(self) -> None:
        """Reset all inference state. Call before starting new inference run."""
        self._entity_history = [[] for _ in range(self.num_entities)]
        self._entity_history_t = [[] for _ in range(self.num_entities)]
        self._prediction_cache = defaultdict(set)
        self._graph_dict = {}
        self._global_emb = {}
        self._global_model = None
        self._latest_time = None
    
    def init_from_train_history(
        self,
        graph_dict: Dict[int, dgl.DGLGraph],
        entity_history: List[List[Dict]],
        entity_history_t: List[List[int]],
        global_emb: Optional[Dict[int, torch.Tensor]] = None,
        global_model: Optional[nn.Module] = None,
    ) -> None:
        """
        Initialize inference state from training data.
        
        Args:
            graph_dict: Pre-computed graphs per timestep
            entity_history: History per entity from training
            entity_history_t: Timestamps for history entries
            global_emb: Global embeddings per timestep (precomputed)
            global_model: Global model for on-the-fly embedding computation
                          (used for timesteps not in global_emb)
        """
        self._graph_dict = graph_dict.copy()
        self._entity_history = [h.copy() for h in entity_history]
        self._entity_history_t = [t.copy() for t in entity_history_t]
        self._global_emb = global_emb.copy() if global_emb else {}
        self._global_model = global_model
    
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
        device = self.entity_embeds.device
        seq_tensor = torch.zeros(1, seq_len, embed_dim, device=device)
        
        # Pre-compute mean relation embedding
        mean_rel = self.rel_embeds.mean(dim=0)
        
        for i, (hist_entry, t) in enumerate(zip(history, history_t)):
            # Check cache first, then compute if needed
            if t in self._rgcn_cache:
                node_features = self._rgcn_cache[t]
                g = self._graph_dict[t]
                node_idx = self._get_node_idx(g, entity_id)
                if node_idx is not None:
                    entity_rgcn_embed = node_features[node_idx]
                else:
                    entity_rgcn_embed = self.entity_embeds[entity_id]
            elif t in self._graph_dict:
                g = self._graph_dict[t]
                rgcn_device = next(self.rgcn.parameters()).device
                # Move graph to RGCN device if DGL supports CUDA
                if self._dgl_has_cuda and g.device != rgcn_device:
                    g = g.to(rgcn_device)
                if 'id' in g.ndata:
                    node_features = self.entity_embeds[g.ndata['id'].view(-1)].to(rgcn_device)
                else:
                    node_features = self.entity_embeds[:g.num_nodes()].to(rgcn_device)
                node_features = self.rgcn(g, node_features)
                if rgcn_device != device:
                    node_features = node_features.to(device)
                
                # Cache the result
                self._rgcn_cache[t] = node_features
                
                node_idx = self._get_node_idx(g, entity_id)
                if node_idx is not None:
                    entity_rgcn_embed = node_features[node_idx]
                else:
                    entity_rgcn_embed = self.entity_embeds[entity_id]
            else:
                entity_rgcn_embed = self.entity_embeds[entity_id]
            
            # Global embedding - use precomputed or compute on-the-fly
            if t in self._global_emb:
                global_embed = self._global_emb[t]
            elif self._global_model is not None:
                # Compute on-the-fly for new timesteps (e.g., during autoregressive eval)
                with torch.no_grad():
                    global_embed = self._global_model.predict(t, self._graph_dict)
                    # Cache for future use
                    self._global_emb[t] = global_embed.detach()
            else:
                global_embed = torch.zeros(self.hidden_dim, device=device)
            
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
        
        # Clear RGCN cache for this forward pass
        self._rgcn_cache = {}
        
        # Pre-compute RGCN outputs for all timesteps needed in this batch
        all_timesteps = set()
        for hist_t in entity1_history_t + entity2_history_t:
            all_timesteps.update(hist_t[-self.seq_len:])
        
        self._precompute_rgcn(all_timesteps, graph_dict)
        
        # Get entity embeddings
        entity1_embed = self.entity_embeds[entity1_ids]
        entity2_embed = self.entity_embeds[entity2_ids]
        
        # Batch encode temporal embeddings
        entity1_temporal = self._encode_history_batch(
            entity1_ids, entity1_history, entity1_history_t, graph_dict, global_emb
        )
        entity2_temporal = self._encode_history_batch(
            entity2_ids, entity2_history, entity2_history_t, graph_dict, global_emb
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
    
    def _precompute_rgcn(self, timesteps: Set[int], graph_dict: Dict[int, dgl.DGLGraph]) -> None:
        """Pre-compute RGCN outputs for all needed timesteps."""
        device = self.entity_embeds.device
        rgcn_device = next(self.rgcn.parameters()).device
        
        for t in timesteps:
            if t in self._rgcn_cache:
                continue
            if t not in graph_dict:
                continue
            
            g = graph_dict[t]
            # Move graph to RGCN device if DGL supports CUDA
            if self._dgl_has_cuda and g.device != rgcn_device:
                g = g.to(rgcn_device)
            node_features = self.entity_embeds[g.ndata['id'].view(-1)].to(rgcn_device)
            node_features = self.rgcn(g, node_features)
            
            if rgcn_device != device:
                node_features = node_features.to(device)
            
            self._rgcn_cache[t] = node_features
    
    def _encode_history_batch(
        self,
        entity_ids: torch.Tensor,
        histories: List[List[Dict]],
        histories_t: List[List[int]],
        graph_dict: Dict[int, dgl.DGLGraph],
        global_emb: Optional[Dict[int, torch.Tensor]],
    ) -> torch.Tensor:
        """Encode histories for a batch of entities."""
        batch_size = len(entity_ids)
        device = self.entity_embeds.device
        embed_dim = 4 * self.hidden_dim
        
        # Find max sequence length in batch
        seq_lens = [min(len(h), self.seq_len) for h in histories]
        max_seq_len = max(seq_lens) if seq_lens else 0
        
        if max_seq_len == 0:
            return torch.zeros(batch_size, self.hidden_dim, device=device)
        
        # Pre-compute mean relation embedding (same for all)
        mean_rel = self.rel_embeds.mean(dim=0)
        
        # Build padded sequence tensor
        seq_tensor = torch.zeros(batch_size, max_seq_len, embed_dim, device=device)
        actual_lens = []
        
        for b in range(batch_size):
            history = histories[b][-self.seq_len:]
            history_t = histories_t[b][-self.seq_len:]
            seq_len = len(history)
            actual_lens.append(seq_len)
            
            if seq_len == 0:
                continue
            
            entity_id = entity_ids[b].item()
            
            for i, (hist_entry, t) in enumerate(zip(history, history_t)):
                # Get RGCN embedding from cache
                if t in self._rgcn_cache and t in graph_dict:
                    g = graph_dict[t]
                    node_features = self._rgcn_cache[t]
                    node_idx = self._get_node_idx(g, entity_id)
                    if node_idx is not None:
                        entity_rgcn_embed = node_features[node_idx]
                    else:
                        entity_rgcn_embed = self.entity_embeds[entity_id]
                else:
                    entity_rgcn_embed = self.entity_embeds[entity_id]
                
                # Global embedding
                if global_emb and t in global_emb:
                    g_emb = global_emb[t]
                else:
                    g_emb = torch.zeros(self.hidden_dim, device=device)
                
                seq_tensor[b, i] = torch.cat([
                    entity_rgcn_embed,
                    self.entity_embeds[entity_id],
                    mean_rel,
                    g_emb,
                ])
        
        # Handle all-zero sequences
        actual_lens = torch.LongTensor([max(l, 1) for l in actual_lens])
        
        # Pack and encode in batch
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            seq_tensor, actual_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        
        temporal_embeds = self.temporal_encoder(packed)
        
        # Zero out embeddings for entities with no history
        for b in range(batch_size):
            if seq_lens[b] == 0:
                temporal_embeds[b] = 0
        
        return temporal_embeds
    
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
                rgcn_device = next(self.rgcn.parameters()).device
                # Move features to RGCN's device for computation
                node_features = self.entity_embeds[g.ndata['id'].view(-1)].to(rgcn_device)
                node_features = self.rgcn(g, node_features)
                # Move results back to model device
                if rgcn_device != device:
                    node_features = node_features.to(device)
                
                node_idx = self._get_node_idx(g, entity_id)
                if node_idx is not None:
                    entity_rgcn_embed = node_features[node_idx]
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
