"""
N-hop subgraph extraction for edge-centric temporal encoding.

Extracts local subgraphs around target edges and builds temporal
histories for all edges within the subgraph.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import torch


@dataclass
class SubgraphContext:
    """Context for a single edge's local subgraph.

    Attributes:
        target_edge: The (src, dst) pair being predicted (canonical order)
        local_edges: All edges within N-hop neighborhood
        edge_histories: Binary history tensor (num_edges, num_timesteps)
        hop_distances: Distance from target edge (0=target, 1=1-hop, etc.)
        is_interchain: Whether each edge is interchain
        edge_to_idx: Mapping from edge tuple to index in tensors
    """

    target_edge: Tuple[int, int]
    local_edges: List[Tuple[int, int]]
    edge_histories: torch.Tensor
    hop_distances: torch.Tensor
    is_interchain: torch.Tensor
    edge_to_idx: Dict[Tuple[int, int], int]


class SubgraphExtractor:
    """Extract N-hop subgraphs and temporal histories for edges.

    Uses union adjacency from training timesteps to define neighborhoods.
    """

    def __init__(
        self,
        full_graphs: Dict[int, Dict[str, torch.Tensor]],
        train_timesteps: List[int],
        n_hops: int = 2,
    ):
        """
        Args:
            full_graphs: Dict mapping timestep -> {"src": tensor, "dst": tensor, "is_inter": tensor}
            train_timesteps: Timesteps to use for history (training set)
            n_hops: Number of hops for neighborhood
        """
        self.full_graphs = full_graphs
        self.train_times = sorted(train_timesteps)
        self.n_hops = n_hops
        self.num_timesteps = len(self.train_times)

        # Build union adjacency from train timesteps
        self.union_adj = self._build_union_adjacency()

        # Pre-compute edge histories and interchain status
        self.edge_histories = self._build_edge_histories()
        self.edge_is_interchain = self._build_edge_interchain_map()

    def _build_union_adjacency(self) -> Dict[int, Set[int]]:
        """Build node adjacency from union of all train timesteps."""
        adj: Dict[int, Set[int]] = defaultdict(set)

        for t in self.train_times:
            if t not in self.full_graphs:
                continue
            graph = self.full_graphs[t]
            src_tensor = graph["src"]
            dst_tensor = graph["dst"]
            for i in range(len(src_tensor)):
                src, dst = src_tensor[i].item(), dst_tensor[i].item()
                adj[src].add(dst)
                adj[dst].add(src)

        return dict(adj)

    def _build_edge_histories(self) -> Dict[Tuple[int, int], List[int]]:
        """Build binary history for each edge across train timesteps."""
        histories: Dict[Tuple[int, int], List[int]] = defaultdict(
            lambda: [0] * self.num_timesteps
        )

        for i, t in enumerate(self.train_times):
            if t not in self.full_graphs:
                continue
            graph = self.full_graphs[t]
            src_tensor = graph["src"]
            dst_tensor = graph["dst"]
            for j in range(len(src_tensor)):
                src, dst = src_tensor[j].item(), dst_tensor[j].item()
                edge = tuple(sorted([src, dst]))
                histories[edge][i] = 1

        return dict(histories)

    def _build_edge_interchain_map(self) -> Dict[Tuple[int, int], bool]:
        """Determine if each edge is interchain."""
        is_inter: Dict[Tuple[int, int], bool] = {}

        for t in self.train_times:
            if t not in self.full_graphs:
                continue
            graph = self.full_graphs[t]
            src_tensor = graph["src"]
            dst_tensor = graph["dst"]
            inter_mask = graph["is_inter"]
            for j in range(len(src_tensor)):
                src, dst = src_tensor[j].item(), dst_tensor[j].item()
                edge = tuple(sorted([src, dst]))
                if edge not in is_inter:
                    is_inter[edge] = inter_mask[j].item()

        return is_inter

    def extract(self, entity_a: int, entity_b: int) -> SubgraphContext:
        """Extract N-hop subgraph context for target edge.

        Args:
            entity_a: First entity of target edge
            entity_b: Second entity of target edge

        Returns:
            SubgraphContext with neighborhood edges and histories
        """
        target_edge = tuple(sorted([entity_a, entity_b]))

        # BFS to find N-hop nodes
        visited_nodes: Set[int] = {entity_a, entity_b}
        frontier = visited_nodes.copy()
        node_distances: Dict[int, int] = {entity_a: 0, entity_b: 0}

        for hop in range(1, self.n_hops + 1):
            next_frontier: Set[int] = set()
            for node in frontier:
                for neighbor in self.union_adj.get(node, set()):
                    if neighbor not in visited_nodes:
                        visited_nodes.add(neighbor)
                        next_frontier.add(neighbor)
                        node_distances[neighbor] = hop
            frontier = next_frontier

        # Collect all edges within subgraph (both endpoints in visited)
        local_edges: List[Tuple[int, int]] = []
        edge_to_idx: Dict[Tuple[int, int], int] = {}

        for node in visited_nodes:
            for neighbor in self.union_adj.get(node, set()):
                if neighbor in visited_nodes:
                    edge = tuple(sorted([node, neighbor]))
                    if edge not in edge_to_idx:
                        edge_to_idx[edge] = len(local_edges)
                        local_edges.append(edge)

        # Ensure target edge is first (index 0)
        if target_edge in edge_to_idx and edge_to_idx[target_edge] != 0:
            target_idx = edge_to_idx[target_edge]
            # Swap with first
            local_edges[0], local_edges[target_idx] = (
                local_edges[target_idx],
                local_edges[0],
            )
            edge_to_idx[local_edges[0]] = 0
            edge_to_idx[local_edges[target_idx]] = target_idx
        elif target_edge not in edge_to_idx:
            # Target edge not in union adj (never appeared in train) - add it
            local_edges.insert(0, target_edge)
            edge_to_idx = {e: i for i, e in enumerate(local_edges)}

        # Build history tensor
        num_edges = len(local_edges)
        edge_histories = torch.zeros(num_edges, self.num_timesteps, dtype=torch.long)

        for i, edge in enumerate(local_edges):
            if edge in self.edge_histories:
                edge_histories[i] = torch.tensor(self.edge_histories[edge])

        # Compute hop distances for edges
        hop_distances = torch.zeros(num_edges, dtype=torch.long)
        for i, (e1, e2) in enumerate(local_edges):
            if (e1, e2) == target_edge or (e2, e1) == target_edge:
                hop_distances[i] = 0
            else:
                # Edge distance = min hop distance of either endpoint
                d1 = node_distances.get(e1, self.n_hops + 1)
                d2 = node_distances.get(e2, self.n_hops + 1)
                hop_distances[i] = max(1, min(d1, d2))

        # Build interchain mask
        is_interchain = torch.zeros(num_edges, dtype=torch.bool)
        for i, edge in enumerate(local_edges):
            is_interchain[i] = self.edge_is_interchain.get(edge, False)

        return SubgraphContext(
            target_edge=target_edge,
            local_edges=local_edges,
            edge_histories=edge_histories,
            hop_distances=hop_distances,
            is_interchain=is_interchain,
            edge_to_idx=edge_to_idx,
        )

    def extract_batch(
        self,
        entity_pairs: List[Tuple[int, int]],
        max_neighbors: int = 50,
    ) -> Dict[str, torch.Tensor]:
        """Extract subgraph contexts for a batch of edges.

        Args:
            entity_pairs: List of (entity_a, entity_b) pairs
            max_neighbors: Maximum number of neighbor edges to include

        Returns:
            Dictionary with batched tensors:
                - target_histories: (batch, num_timesteps)
                - neighbor_histories: (batch, max_neighbors, num_timesteps)
                - hop_distances: (batch, max_neighbors)
                - neighbor_mask: (batch, max_neighbors) True for padding
        """
        batch_size = len(entity_pairs)

        target_histories = torch.zeros(batch_size, self.num_timesteps, dtype=torch.long)
        neighbor_histories = torch.zeros(
            batch_size, max_neighbors, self.num_timesteps, dtype=torch.long
        )
        hop_distances = torch.zeros(batch_size, max_neighbors, dtype=torch.long)
        neighbor_mask = torch.ones(batch_size, max_neighbors, dtype=torch.bool)

        for b, (e1, e2) in enumerate(entity_pairs):
            ctx = self.extract(e1, e2)

            # Target history (first edge)
            target_histories[b] = ctx.edge_histories[0]

            # Neighbor histories (remaining edges, up to max_neighbors)
            num_neighbors = min(len(ctx.local_edges) - 1, max_neighbors)
            if num_neighbors > 0:
                neighbor_histories[b, :num_neighbors] = ctx.edge_histories[
                    1 : num_neighbors + 1
                ]
                hop_distances[b, :num_neighbors] = ctx.hop_distances[
                    1 : num_neighbors + 1
                ]
                neighbor_mask[b, :num_neighbors] = False

        return {
            "target_histories": target_histories,
            "neighbor_histories": neighbor_histories,
            "hop_distances": hop_distances,
            "neighbor_mask": neighbor_mask,
        }
