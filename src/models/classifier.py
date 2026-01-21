"""Binary classifier head for edge prediction."""

from typing import Optional

import torch
import torch.nn as nn


class SymmetricPairProjection(nn.Module):
    """
    Create symmetric pair embedding from two entity embeddings.

    Uses symmetric operations (sum, product, abs diff) to ensure
    score(e1, e2) = score(e2, e1) by construction.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Output: sum + product + abs_diff = 3 * hidden_dim
        self.output_dim = 3 * hidden_dim

    def forward(
        self,
        entity1_embed: torch.Tensor,
        entity2_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create symmetric pair embedding.

        Args:
            entity1_embed: First entity embedding (batch, hidden_dim)
            entity2_embed: Second entity embedding (batch, hidden_dim)

        Returns:
            Symmetric pair embedding (batch, 3 * hidden_dim)
        """
        pair_sum = entity1_embed + entity2_embed
        pair_prod = entity1_embed * entity2_embed
        pair_diff = torch.abs(entity1_embed - entity2_embed)
        return torch.cat([pair_sum, pair_prod, pair_diff], dim=-1)


class EdgeClassifier(nn.Module):
    """
    Binary classifier for edge prediction.

    Takes a fused pair embedding and outputs a logit for binary classification.

    Args:
        input_dim: Input dimension (from pair projection + temporal fusion)
        hidden_dim: Hidden dimension in classifier MLP
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, pair_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute edge existence logits.

        Args:
            pair_embedding: Fused pair embedding (batch, input_dim)

        Returns:
            Logits of shape (batch,)
        """
        return self.classifier(pair_embedding).squeeze(-1)


class SymmetricEdgeClassifier(nn.Module):
    """
    Symmetric edge classifier combining pair projection and classification.

    For an undirected edge (i, j), the score is the same as (j, i)
    by construction through symmetric operations.

    Args:
        hidden_dim: Entity embedding dimension
        classifier_hidden_dim: Hidden dimension in classifier MLP
        dropout: Dropout rate
        use_temporal: Whether temporal embedding is included in input
    """

    def __init__(
        self,
        hidden_dim: int,
        classifier_hidden_dim: int = 128,
        dropout: float = 0.2,
        use_temporal: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_temporal = use_temporal

        # Symmetric pair projection
        self.pair_proj = SymmetricPairProjection(hidden_dim)

        # Calculate input dimension
        # From pair projection: 3 * hidden_dim
        # If use_temporal, we expect temporal embedding to be added externally
        input_dim = self.pair_proj.output_dim
        if use_temporal:
            # Temporal embedding (from edge history encoder) is same dim as hidden
            input_dim += hidden_dim

        self.classifier = EdgeClassifier(
            input_dim=input_dim,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        entity1_embed: torch.Tensor,
        entity2_embed: torch.Tensor,
        temporal_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute symmetric edge existence logits.

        Args:
            entity1_embed: First entity embedding (batch, hidden_dim)
            entity2_embed: Second entity embedding (batch, hidden_dim)
            temporal_embed: Optional temporal embedding from edge history encoder
                           (batch, hidden_dim). This should already be symmetric
                           since it's computed per-edge, not per-entity.

        Returns:
            Logits of shape (batch,)
        """
        # Create symmetric pair embedding
        pair_emb = self.pair_proj(entity1_embed, entity2_embed)

        # Add temporal if provided
        if self.use_temporal and temporal_embed is not None:
            pair_emb = torch.cat([pair_emb, temporal_embed], dim=-1)
        elif self.use_temporal:
            # Pad with zeros if temporal expected but not provided
            zeros = torch.zeros(
                pair_emb.size(0), self.hidden_dim, device=pair_emb.device
            )
            pair_emb = torch.cat([pair_emb, zeros], dim=-1)

        return self.classifier(pair_emb)
