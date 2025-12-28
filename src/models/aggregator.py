"""History aggregation for temporal encoding."""

import torch
import torch.nn as nn
import dgl
from typing import List, Dict, Optional, Tuple
from src.models.rgcn import UndirectedRGCN, build_undirected_graph


class HistoryAggregator(nn.Module):
    """
    Aggregates historical graph snapshots using RGCN.
    
    For each entity, processes its neighborhood history through RGCN
    and combines with entity/relation embeddings for GRU input.
    
    This is an undirected version that uses unified history per entity
    instead of separate subject/object histories.
    
    Args:
        hidden_dim: Hidden dimension
        num_nodes: Number of entities
        num_rels: Number of relation types
        num_bases: Number of basis functions for RGCN
        seq_len: Maximum sequence length for history
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_nodes: int,
        num_rels: int,
        num_bases: int = 100,
        seq_len: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.seq_len = seq_len
        
        # RGCN for encoding historical subgraphs
        self.rgcn = UndirectedRGCN(
            hidden_dim=hidden_dim,
            num_rels=num_rels,
            num_layers=2,
            num_bases=num_bases,
            dropout=dropout,
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        entity_ids: torch.Tensor,
        histories: List[List[Dict]],
        history_times: List[List[int]],
        entity_embeds: nn.Parameter,
        rel_embeds: nn.Parameter,
        graph_dict: Dict[int, dgl.DGLGraph],
        global_emb: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Tuple[Optional[torch.nn.utils.rnn.PackedSequence], torch.Tensor]:
        """
        Aggregate historical information for a batch of entities.
        
        Args:
            entity_ids: Entity IDs of shape (batch_size,)
            histories: List of history for each entity, where each history
                      is a list of dicts with 'neighbors' and 'rel_types'
            history_times: List of timestamps for each history entry
            entity_embeds: Entity embedding parameters
            rel_embeds: Relation embedding parameters
            graph_dict: Dictionary mapping timestep to graph
            global_emb: Optional dict mapping timestep to global embedding
        
        Returns:
            Tuple of:
            - Packed sequence for GRU input (or None if no history)
            - Length tensor for unpacking
        """
        batch_size = len(entity_ids)
        
        # Count non-empty histories
        hist_lengths = torch.LongTensor([len(h) for h in histories])
        
        if hist_lengths.sum() == 0:
            # No history for any entity in batch
            return None, hist_lengths
        
        # Sort by length for packing
        sorted_lengths, sort_idx = hist_lengths.sort(descending=True)
        
        # Filter to non-empty
        non_zero_mask = sorted_lengths > 0
        non_zero_lengths = sorted_lengths[non_zero_mask]
        
        if len(non_zero_lengths) == 0:
            return None, hist_lengths
        
        # Prepare sequence tensor
        # Input dim: entity_embed + rel_embed + rgcn_embed + global_embed
        # = hidden_dim * 4
        max_len = min(self.seq_len, int(non_zero_lengths[0].item()))
        embed_dim = 4 * self.hidden_dim
        
        seq_tensor = torch.zeros(
            len(non_zero_lengths),
            max_len,
            embed_dim,
            device=entity_embeds.device,
        )
        
        # Process each entity's history
        for batch_idx, orig_idx in enumerate(sort_idx[:len(non_zero_lengths)]):
            entity_id = entity_ids[orig_idx]
            history = histories[orig_idx]
            history_t = history_times[orig_idx]
            
            # Take last seq_len entries
            history = history[-self.seq_len:]
            history_t = history_t[-self.seq_len:]
            
            for seq_idx, (hist_entry, t) in enumerate(zip(history, history_t)):
                if seq_idx >= max_len:
                    break
                
                # Get graph for this timestep
                if t in graph_dict:
                    g = graph_dict[t]
                    
                    # Get entity's embedding after RGCN on historical graph
                    with torch.no_grad():
                        node_features = entity_embeds[g.ndata['id'].view(-1)]
                    
                    node_features = self.rgcn(g, node_features)
                    
                    # Find entity's position in graph
                    if entity_id.item() in g.ids:
                        node_idx = g.ids[entity_id.item()]
                        entity_rgcn_embed = node_features[node_idx]
                    else:
                        entity_rgcn_embed = entity_embeds[entity_id]
                else:
                    entity_rgcn_embed = entity_embeds[entity_id]
                
                # Get global embedding for timestep
                if global_emb is not None and t in global_emb:
                    global_embed = global_emb[t]
                else:
                    global_embed = torch.zeros(
                        self.hidden_dim,
                        device=entity_embeds.device
                    )
                
                # Combine embeddings
                # Note: We use a mean relation embedding since we're predicting
                # edge existence, not specific relations
                mean_rel_embed = rel_embeds.mean(dim=0)
                
                combined = torch.cat([
                    entity_rgcn_embed,
                    entity_embeds[entity_id],
                    mean_rel_embed,
                    global_embed,
                ])
                
                seq_tensor[batch_idx, seq_idx] = combined
        
        # Apply dropout
        seq_tensor = self.dropout(seq_tensor)
        
        # Pack for GRU
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            seq_tensor,
            non_zero_lengths.cpu(),
            batch_first=True,
        )
        
        return packed, hist_lengths


class SimpleHistoryAggregator(nn.Module):
    """
    Simplified history aggregator using mean pooling.
    
    Faster but less expressive than RGCN-based aggregation.
    Good for debugging or when computational resources are limited.
    
    Args:
        hidden_dim: Hidden dimension
        seq_len: Maximum sequence length
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        hidden_dim: int,
        seq_len: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Project neighbor embeddings
        self.neighbor_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self,
        entity_ids: torch.Tensor,
        histories: List[List[Dict]],
        history_times: List[List[int]],
        entity_embeds: nn.Parameter,
        rel_embeds: nn.Parameter,
        **kwargs,
    ) -> Tuple[Optional[torch.nn.utils.rnn.PackedSequence], torch.Tensor]:
        """
        Aggregate using mean pooling of neighbor embeddings.
        
        Args:
            entity_ids: Entity IDs of shape (batch_size,)
            histories: List of history for each entity
            history_times: List of timestamps for each history entry
            entity_embeds: Entity embedding parameters
            rel_embeds: Relation embedding parameters
            **kwargs: Ignored (for compatibility with RGCN aggregator)
        
        Returns:
            Tuple of packed sequence and lengths
        """
        batch_size = len(entity_ids)
        
        hist_lengths = torch.LongTensor([len(h) for h in histories])
        
        if hist_lengths.sum() == 0:
            return None, hist_lengths
        
        sorted_lengths, sort_idx = hist_lengths.sort(descending=True)
        non_zero_mask = sorted_lengths > 0
        non_zero_lengths = sorted_lengths[non_zero_mask]
        
        if len(non_zero_lengths) == 0:
            return None, hist_lengths
        
        max_len = min(self.seq_len, int(non_zero_lengths[0].item()))
        embed_dim = 3 * self.hidden_dim  # entity + rel + neighbor_mean
        
        seq_tensor = torch.zeros(
            len(non_zero_lengths),
            max_len,
            embed_dim,
            device=entity_embeds.device,
        )
        
        for batch_idx, orig_idx in enumerate(sort_idx[:len(non_zero_lengths)]):
            entity_id = entity_ids[orig_idx]
            history = histories[orig_idx][-self.seq_len:]
            
            for seq_idx, hist_entry in enumerate(history):
                if seq_idx >= max_len:
                    break
                
                neighbors = hist_entry.get('neighbors', [])
                
                if len(neighbors) > 0:
                    neighbor_embeds = entity_embeds[torch.LongTensor(neighbors)]
                    neighbor_mean = self.neighbor_proj(neighbor_embeds.mean(dim=0))
                else:
                    neighbor_mean = torch.zeros(
                        self.hidden_dim,
                        device=entity_embeds.device
                    )
                
                mean_rel_embed = rel_embeds.mean(dim=0)
                
                combined = torch.cat([
                    entity_embeds[entity_id],
                    mean_rel_embed,
                    neighbor_mean,
                ])
                
                seq_tensor[batch_idx, seq_idx] = combined
        
        seq_tensor = self.dropout(seq_tensor)
        
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            seq_tensor,
            non_zero_lengths.cpu(),
            batch_first=True,
        )
        
        return packed, hist_lengths
