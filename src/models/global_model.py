"""
Global RGCN model for computing graph-level temporal embeddings.

Part of RAPID: A Recurrent Architecture for Predicting Protein Interaction Dynamics

This model processes entire timestep graphs to capture global interface state,
which enriches the per-entity temporal encoding.

Key design (following original RE-Net to avoid leakage):
- predict(t) uses only graphs from times < t
- global_emb[t] is computed from graphs < t+1, stored for use when predicting t+1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

from src.models.rgcn import UndirectedRGCN, build_undirected_graph


class GlobalRGCNAggregator(nn.Module):
    """
    Aggregates graph-level embeddings from a sequence of timestep graphs.
    
    For each timestep, runs RGCN on the graph and pools to a single vector.
    
    Args:
        hidden_dim: Hidden dimension
        num_rels: Number of relation types
        num_bases: Number of basis functions for RGCN
        seq_len: Maximum sequence length for history
        pooling: Pooling method ('max' or 'mean')
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_rels: int,
        num_bases: int = 5,
        seq_len: int = 10,
        pooling: str = 'max',
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_rels = num_rels
        self.seq_len = seq_len
        self.pooling = pooling
        
        # Two-layer RGCN for graph encoding
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
        t_list: torch.Tensor,
        ent_embeds: torch.Tensor,
        graph_dict: Dict[int, dgl.DGLGraph],
    ) -> torch.nn.utils.rnn.PackedSequence:
        """
        Process batch of timesteps, returning packed sequence of graph embeddings.
        
        Args:
            t_list: Tensor of timesteps, shape (batch_size,), sorted descending
            ent_embeds: Entity embeddings, shape (num_entities, hidden_dim)
            graph_dict: Dict mapping timestep -> DGLGraph
        
        Returns:
            PackedSequence of graph embeddings for GRU input
        """
        times = sorted(graph_dict.keys())
        if len(times) < 2:
            time_unit = 1
        else:
            time_unit = times[1] - times[0]
        
        # Filter to valid timesteps (>= 0, since t=0 is a valid timestep)
        # The batch may be padded with -1 or other invalid values at the end
        valid_mask = t_list >= 0
        num_valid = valid_mask.sum().item()
        t_list = t_list[:num_valid]
        
        # Collect time sequences for each sample (times < t)
        time_list = []
        len_non_zero = []
        
        for tim in t_list:
            tim = tim.item()
            # Find index where tim would be inserted
            idx = 0
            for tt in times:
                if tt >= tim:
                    break
                idx += 1
            
            # Get up to seq_len timesteps before tim
            if self.seq_len <= idx:
                time_list.append(times[idx - self.seq_len:idx])
                len_non_zero.append(self.seq_len)
            else:
                time_list.append(times[:idx])
                len_non_zero.append(idx)
        
        # Handle empty sequences
        if not any(len_non_zero):
            # Return zero embeddings
            embed_seq = torch.zeros(len(t_list), 1, self.hidden_dim, device=ent_embeds.device)
            return torch.nn.utils.rnn.pack_padded_sequence(
                embed_seq, [1] * len(t_list), batch_first=True, enforce_sorted=False
            )
        
        # Collect unique timesteps and batch graphs
        unique_times = sorted(set(t for times in time_list for t in times))
        time_to_idx = {t: i for i, t in enumerate(unique_times)}
        
        g_list = [graph_dict[t] for t in unique_times]
        
        if not g_list:
            embed_seq = torch.zeros(len(t_list), 1, self.hidden_dim, device=ent_embeds.device)
            return torch.nn.utils.rnn.pack_padded_sequence(
                embed_seq, [1] * len(t_list), batch_first=True, enforce_sorted=False
            )
        
        # Batch process all graphs
        batched_graph = dgl.batch(g_list)
        
        # Get node features from entity embeddings
        node_ids = batched_graph.ndata.get('id', None)
        if node_ids is not None:
            node_features = ent_embeds[node_ids]
        else:
            node_features = ent_embeds[:batched_graph.num_nodes()]
        
        # Handle device - move graph and features to RGCN device
        device = ent_embeds.device
        rgcn_device = next(self.rgcn.parameters()).device
        
        # Move graph to RGCN device (works when DGL has CUDA support)
        if batched_graph.device != rgcn_device:
            try:
                batched_graph = batched_graph.to(rgcn_device)
            except Exception:
                pass  # DGL doesn't support CUDA, keep on CPU
        
        # Move features to RGCN's device
        if rgcn_device != device:
            node_features = node_features.to(rgcn_device)
        
        updated_features = self.rgcn(batched_graph, node_features)
        
        # Move results back to model device
        if rgcn_device != device:
            updated_features = updated_features.to(device)
        
        batched_graph.ndata['h'] = updated_features.to(batched_graph.device)
        
        # Pool each graph to single vector
        if self.pooling == 'max':
            global_info = dgl.max_nodes(batched_graph, 'h')
        else:
            global_info = dgl.mean_nodes(batched_graph, 'h')
        
        # Build sequence tensor
        max_len = max(len_non_zero) if len_non_zero else 1
        embed_seq_tensor = torch.zeros(
            len(len_non_zero), max_len, self.hidden_dim,
            device=ent_embeds.device
        )
        
        for i, times_i in enumerate(time_list):
            for j, t in enumerate(times_i):
                embed_seq_tensor[i, j, :] = global_info[time_to_idx[t]]
        
        embed_seq_tensor = self.dropout(embed_seq_tensor)
        
        # Pack for GRU (handles variable lengths)
        # Ensure lengths are valid
        valid_lengths = [max(1, l) for l in len_non_zero]
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            embed_seq_tensor,
            valid_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        
        return packed_input
    
    def predict(
        self,
        t: int,
        ent_embeds: torch.Tensor,
        graph_dict: Dict[int, dgl.DGLGraph],
    ) -> torch.Tensor:
        """
        Compute graph embeddings for prediction at time t (uses graphs < t).
        
        Args:
            t: Target timestep
            ent_embeds: Entity embeddings
            graph_dict: Dict mapping timestep -> DGLGraph
        
        Returns:
            Sequence of graph embeddings, shape (seq_len, hidden_dim)
        """
        times = sorted(graph_dict.keys())
        
        # Find timesteps < t
        idx = 0
        for tt in times:
            if tt >= t:
                break
            idx += 1
        
        # Get up to seq_len timesteps
        if self.seq_len <= idx:
            selected_times = times[idx - self.seq_len:idx]
        else:
            selected_times = times[:idx]
        
        if not selected_times:
            return torch.zeros(1, self.hidden_dim, device=ent_embeds.device)
        
        # Process graphs
        g_list = [graph_dict[tt] for tt in selected_times]
        batched_graph = dgl.batch(g_list)
        
        node_ids = batched_graph.ndata.get('id', None)
        if node_ids is not None:
            node_features = ent_embeds[node_ids]
        else:
            node_features = ent_embeds[:batched_graph.num_nodes()]
        
        # Handle device - move graph and features to RGCN device
        device = ent_embeds.device
        rgcn_device = next(self.rgcn.parameters()).device
        
        # Move graph to RGCN device (works when DGL has CUDA support)
        if batched_graph.device != rgcn_device:
            try:
                batched_graph = batched_graph.to(rgcn_device)
            except Exception:
                pass  # DGL doesn't support CUDA, keep on CPU
        
        if rgcn_device != device:
            node_features = node_features.to(rgcn_device)
        
        updated_features = self.rgcn(batched_graph, node_features)
        
        if rgcn_device != device:
            updated_features = updated_features.to(device)
        
        batched_graph.ndata['h'] = updated_features.to(batched_graph.device)
        
        if self.pooling == 'max':
            global_info = dgl.max_nodes(batched_graph, 'h')
        else:
            global_info = dgl.mean_nodes(batched_graph, 'h')
        
        return global_info


class PPIGlobalModel(nn.Module):
    """
    Global model for PPI dynamics - captures graph-level temporal context.
    
    Architecture:
    1. GlobalRGCNAggregator: RGCN + pooling on each timestep graph
    2. GRU: Encodes sequence of graph embeddings
    3. Linear: Projects to prediction space (for pretraining)
    
    During inference, the GRU hidden state is used as global context
    to enrich per-entity predictions.
    
    Args:
        num_entities: Number of entities
        hidden_dim: Hidden dimension
        num_rels: Number of relation types
        num_bases: Number of RGCN bases
        seq_len: Maximum history length
        pooling: Graph pooling method
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        num_entities: int,
        hidden_dim: int,
        num_rels: int,
        num_bases: int = 5,
        seq_len: int = 10,
        pooling: str = 'max',
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.num_entities = num_entities
        self.hidden_dim = hidden_dim
        self.num_rels = num_rels
        self.seq_len = seq_len
        
        # Entity embeddings (shared with main model during training)
        self.ent_embeds = nn.Parameter(torch.Tensor(num_entities, hidden_dim))
        nn.init.xavier_uniform_(self.ent_embeds, gain=nn.init.calculate_gain('relu'))
        
        # Graph aggregator
        self.aggregator = GlobalRGCNAggregator(
            hidden_dim=hidden_dim,
            num_rels=num_rels,
            num_bases=num_bases,
            seq_len=seq_len,
            pooling=pooling,
            dropout=dropout,
        )
        
        # Temporal encoder
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # Projection for pretraining (predict entity distribution)
        self.linear = nn.Linear(hidden_dim, num_entities)
        
        self.dropout = nn.Dropout(dropout)
        
        # Precomputed global embeddings
        self.global_emb: Optional[Dict[int, torch.Tensor]] = None
        
        # Check if DGL supports CUDA
        self._dgl_has_cuda = self._check_dgl_cuda_support()
    
    def _check_dgl_cuda_support(self) -> bool:
        """Check if DGL supports CUDA operations."""
        try:
            test_g = dgl.graph(([0], [1]))
            if torch.cuda.is_available():
                test_g.to('cuda:0')
            return True
        except Exception:
            return False
    
    def to(self, device, *args, **kwargs):
        """Override to keep RGCN on CPU if DGL doesn't support CUDA."""
        result = super().to(device, *args, **kwargs)
        if not self._dgl_has_cuda and 'cuda' in str(device):
            self.aggregator.rgcn = self.aggregator.rgcn.to('cpu')
        return result
    
    def cuda(self, device=None):
        """Override to keep RGCN on CPU if DGL doesn't support CUDA."""
        result = super().cuda(device)
        if not self._dgl_has_cuda:
            self.aggregator.rgcn = self.aggregator.rgcn.to('cpu')
        return result
    
    def forward(
        self,
        t_list: torch.Tensor,
        true_prob: torch.Tensor,
        graph_dict: Dict[int, dgl.DGLGraph],
    ) -> torch.Tensor:
        """
        Training forward pass - predicts entity distribution.
        
        Args:
            t_list: Timesteps, shape (batch_size,)
            true_prob: True entity probability distribution for soft cross-entropy
            graph_dict: Dict of timestep -> graph
        
        Returns:
            Loss value
        """
        # Sort by time (descending) for packing
        sorted_t, idx = t_list.sort(0, descending=True)
        
        # Get packed graph embeddings
        packed_input = self.aggregator(sorted_t, self.ent_embeds, graph_dict)
        
        # Encode with GRU
        _, hidden = self.encoder(packed_input)
        hidden = hidden.squeeze(0)  # (batch, hidden_dim)
        
        # Pad to original batch size if needed
        if len(hidden) < len(t_list):
            padding = torch.zeros(
                len(t_list) - len(hidden), self.hidden_dim,
                device=hidden.device
            )
            hidden = torch.cat([hidden, padding], dim=0)
        
        # Predict entity distribution
        pred = self.linear(hidden)
        
        # Soft cross-entropy loss
        loss = self._soft_cross_entropy(pred, true_prob[idx])
        
        return loss
    
    def _soft_cross_entropy(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Soft cross-entropy with probability targets."""
        log_probs = F.log_softmax(pred, dim=-1)
        loss = -(target * log_probs).sum(dim=-1).mean()
        return loss
    
    @torch.no_grad()
    def compute_global_embeddings(
        self,
        timesteps: List[int],
        graph_dict: Dict[int, dgl.DGLGraph],
    ) -> Dict[int, torch.Tensor]:
        """
        Precompute global embeddings for all timesteps.
        
        Important: global_emb[t] uses graphs from times < t+1,
        so it's safe to use when predicting at time t+1.
        
        This matches the original RE-Net's temporal offset to avoid leakage.
        
        Args:
            timesteps: Sorted list of all timesteps
            graph_dict: Dict of timestep -> graph
        
        Returns:
            Dict mapping timestep -> global embedding
        """
        global_emb = OrderedDict()
        
        if len(timesteps) < 2:
            return global_emb
        
        time_unit = timesteps[1] - timesteps[0]
        prev_t = timesteps[0]
        
        for t in timesteps:
            if t == timesteps[0]:
                continue
            
            # Compute embedding using graphs < t
            emb = self.predict(t, graph_dict)
            
            # Store at prev_t (temporal offset for safety)
            global_emb[prev_t] = emb.detach()
            prev_t = t
        
        # Handle last timestep
        if timesteps:
            last_t = timesteps[-1]
            emb = self.predict(last_t + time_unit, graph_dict)
            global_emb[last_t] = emb.detach()
        
        self.global_emb = global_emb
        return global_emb
    
    def predict(
        self,
        t: int,
        graph_dict: Dict[int, dgl.DGLGraph],
    ) -> torch.Tensor:
        """
        Predict global embedding for time t (uses graphs < t).
        
        Args:
            t: Target timestep
            graph_dict: Dict of timestep -> graph
        
        Returns:
            Global embedding, shape (hidden_dim,)
        """
        # Get sequence of graph embeddings
        graph_embs = self.aggregator.predict(t, self.ent_embeds, graph_dict)
        
        # Encode with GRU
        _, hidden = self.encoder(graph_embs.unsqueeze(0))
        
        return hidden.squeeze()
    
    def get_embedding_for_timestep(
        self,
        t: int,
        graph_dict: Optional[Dict[int, dgl.DGLGraph]] = None,
    ) -> Optional[torch.Tensor]:
        """
        Get global embedding for timestep t.
        
        Uses precomputed embeddings if available, otherwise computes on-the-fly.
        
        Args:
            t: Timestep
            graph_dict: Graph dict (required if not precomputed)
        
        Returns:
            Global embedding or None if not available
        """
        if self.global_emb is not None and t in self.global_emb:
            return self.global_emb[t]
        
        if graph_dict is not None:
            return self.predict(t, graph_dict)
        
        return None
    
    @torch.no_grad()
    def extend_embeddings(
        self,
        new_graph_dict: Dict[int, dgl.DGLGraph],
    ) -> Dict[int, torch.Tensor]:
        """
        Extend global embeddings to cover new timesteps.
        
        Computes global embeddings for timesteps in new_graph_dict that
        are not already in self.global_emb. Useful for adding validation
        timestep embeddings before test evaluation.
        
        Args:
            new_graph_dict: Graph dict with additional timesteps
        
        Returns:
            Updated global_emb dict
        """
        if self.global_emb is None:
            self.global_emb = {}
        
        # Find new timesteps not already covered
        existing_times = set(self.global_emb.keys())
        all_times = sorted(new_graph_dict.keys())
        
        for t in all_times:
            if t not in existing_times:
                emb = self.predict(t, new_graph_dict)
                self.global_emb[t] = emb.detach()
        
        return self.global_emb


def create_global_model(
    num_entities: int,
    num_rels: int,
    hidden_dim: int = 200,
    num_bases: int = 5,
    seq_len: int = 10,
    pooling: str = 'max',
    dropout: float = 0.2,
) -> PPIGlobalModel:
    """Factory function for creating global model."""
    return PPIGlobalModel(
        num_entities=num_entities,
        hidden_dim=hidden_dim,
        num_rels=num_rels,
        num_bases=num_bases,
        seq_len=seq_len,
        pooling=pooling,
        dropout=dropout,
    )
