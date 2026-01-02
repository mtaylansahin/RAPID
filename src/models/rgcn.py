"""Undirected Relational Graph Convolutional Network (RGCN) layer."""

from typing import Optional

import dgl
import dgl.function as fn
import torch
import torch.nn as nn


class RGCNLayer(nn.Module):
    """
    Base RGCN layer with optional self-loop and activation.

    Subclasses must implement the `propagate` method.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = False,
        activation: Optional[nn.Module] = None,
        self_loop: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        self.self_loop = self_loop

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)

        if self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
            nn.init.xavier_uniform_(
                self.loop_weight, gain=nn.init.calculate_gain("relu")
            )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def propagate(self, g: dgl.DGLGraph) -> None:
        """Propagate messages through graph. Must be implemented by subclass."""
        raise NotImplementedError

    def forward(self, g: dgl.DGLGraph) -> dgl.DGLGraph:
        """
        Forward pass.

        Args:
            g: DGLGraph with node features in g.ndata['h']

        Returns:
            Graph with updated node features
        """
        if self.self_loop:
            loop_message = torch.mm(g.ndata["h"], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)

        self.propagate(g)

        # Apply bias and activation
        node_repr = g.ndata["h"]

        if self.bias is not None:
            node_repr = node_repr + self.bias

        if self.self_loop:
            node_repr = node_repr + loop_message

        if self.activation is not None:
            node_repr = self.activation(node_repr)

        g.ndata["h"] = node_repr
        return g


class RGCNBlockLayer(RGCNLayer):
    """
    RGCN layer with basis decomposition for weight sharing.

    Adapted for undirected graphs - uses single edge type per relation
    instead of separate forward/reverse types.

    Uses a simpler approach: per-relation weight matrices with basis sharing.

    Args:
        in_dim: Input feature dimension
        out_dim: Output feature dimension
        num_rels: Number of relation types
        num_bases: Number of basis functions for weight decomposition
        bias: Whether to use bias
        activation: Activation function
        self_loop: Whether to include self-loops
        dropout: Dropout rate
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_rels: int,
        num_bases: int,
        bias: bool = False,
        activation: Optional[nn.Module] = None,
        self_loop: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            bias=bias,
            activation=activation,
            self_loop=self_loop,
            dropout=dropout,
        )

        self.num_rels = num_rels
        self.num_bases = min(num_bases, num_rels)  # Can't have more bases than rels

        # Basis weight matrices: (num_bases, in_dim, out_dim)
        self.basis_weights = nn.Parameter(torch.Tensor(self.num_bases, in_dim, out_dim))
        nn.init.xavier_uniform_(self.basis_weights, gain=nn.init.calculate_gain("relu"))

        # Coefficients to combine bases for each relation: (num_rels, num_bases)
        self.coefficients = nn.Parameter(torch.Tensor(num_rels, self.num_bases))
        nn.init.xavier_uniform_(self.coefficients)

    def _get_relation_weights(self) -> torch.Tensor:
        """
        Compute relation-specific weights from basis decomposition.

        Returns:
            Weight tensor of shape (num_rels, in_dim, out_dim)
        """
        # coefficients: (num_rels, num_bases)
        # basis_weights: (num_bases, in_dim, out_dim)
        # result: (num_rels, in_dim, out_dim)
        return torch.einsum("rb,bio->rio", self.coefficients, self.basis_weights)

    def _message_func(self, edges: dgl.udf.EdgeBatch) -> dict:
        """
        Message function for RGCN.

        Uses edge type to select relation-specific weight matrix.
        """
        # Get relation types
        rel_types = edges.data["rel_type"]

        # Get relation-specific weights
        all_weights = self._get_relation_weights()  # (num_rels, in_dim, out_dim)

        # Select weights for each edge's relation type
        weight = all_weights[rel_types]  # (num_edges, in_dim, out_dim)

        # Source node features: (num_edges, in_dim)
        node = edges.src["h"]

        # Compute messages via batch matrix multiply
        # (num_edges, 1, in_dim) @ (num_edges, in_dim, out_dim) -> (num_edges, 1, out_dim)
        msg = torch.bmm(node.unsqueeze(1), weight).squeeze(1)

        return {"msg": msg}

    def _apply_func(self, nodes: dgl.udf.NodeBatch) -> dict:
        """Apply normalization to aggregated messages."""
        return {"h": nodes.data["h"] * nodes.data["norm"]}

    def propagate(self, g: dgl.DGLGraph) -> None:
        """Propagate messages through graph."""
        g.update_all(self._message_func, fn.sum(msg="msg", out="h"), self._apply_func)


class UndirectedRGCN(nn.Module):
    """
    Multi-layer undirected RGCN for PPI dynamics.

    Processes interaction graphs with symmetric message passing
    (A + A^T normalization for undirected edges).

    Args:
        hidden_dim: Hidden dimension
        num_rels: Number of relation types
        num_layers: Number of RGCN layers
        num_bases: Number of basis functions
        dropout: Dropout rate
    """

    def __init__(
        self,
        hidden_dim: int,
        num_rels: int,
        num_layers: int = 2,
        num_bases: int = 100,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_rels = num_rels
        self.num_layers = num_layers

        # Build RGCN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            activation = nn.ReLU() if i < num_layers - 1 else None
            layer = RGCNBlockLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_rels=num_rels,
                num_bases=num_bases,
                activation=activation,
                self_loop=True,
                dropout=dropout,
            )
            self.layers.append(layer)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            g: DGLGraph with edges and edge types
            node_features: Node features of shape (num_nodes, hidden_dim)

        Returns:
            Updated node features of shape (num_nodes, hidden_dim)
        """
        g.ndata["h"] = node_features

        for layer in self.layers:
            layer(g)

        return g.ndata.pop("h")


def build_undirected_graph(
    edges: torch.Tensor,
    rel_types: torch.Tensor,
    num_nodes: int,
    node_ids: Optional[torch.Tensor] = None,
) -> dgl.DGLGraph:
    """
    Build an undirected DGL graph from edge list.

    Automatically adds reverse edges to make graph undirected,
    with symmetric normalization.

    Args:
        edges: Edge tensor of shape (num_edges, 2) with (src, dst) pairs
        rel_types: Relation types of shape (num_edges,)
        num_nodes: Number of nodes in graph
        node_ids: Optional node IDs to store

    Returns:
        DGLGraph with undirected edges and normalization factors
    """
    if len(edges) == 0:
        g = dgl.graph(([], []), num_nodes=num_nodes)
        g.ndata["norm"] = torch.ones(num_nodes, 1)
        if node_ids is not None:
            g.ndata["id"] = node_ids
        return g

    src = edges[:, 0]
    dst = edges[:, 1]

    # Add reverse edges for undirected
    src_bidir = torch.cat([src, dst])
    dst_bidir = torch.cat([dst, src])
    rel_types_bidir = torch.cat([rel_types, rel_types])  # Same type for both directions

    # Create graph
    g = dgl.graph((src_bidir, dst_bidir), num_nodes=num_nodes)
    g.edata["rel_type"] = rel_types_bidir

    # Compute symmetric normalization: 1 / sqrt(d_i * d_j)
    # Simplified: just use 1/degree
    in_deg = g.in_degrees().float().clamp(min=1)
    norm = 1.0 / in_deg
    g.ndata["norm"] = norm.view(-1, 1)

    if node_ids is not None:
        g.ndata["id"] = node_ids

    return g
