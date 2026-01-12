"""Binary classifier head for edge prediction."""

from typing import Optional

import torch
import torch.nn as nn


class EdgeClassifier(nn.Module):
    """
    Binary classifier for predicting edge existence.

    Takes embeddings of two entities and optional relation/temporal context,
    and outputs a logit for binary classification.

    Supports multiple scoring functions:
    - 'concat': Concatenate embeddings and pass through MLP
    - 'bilinear': Bilinear scoring function
    - 'dot': Simple dot product (no learnable parameters)

    Args:
        hidden_dim: Entity embedding dimension
        classifier_hidden_dim: Hidden dimension in classifier MLP
        dropout: Dropout rate
        scoring_fn: Scoring function type ('concat', 'bilinear', 'dot')
        use_relation: Whether to include relation embedding
        use_temporal: Whether to include temporal embedding
    """

    def __init__(
        self,
        hidden_dim: int,
        classifier_hidden_dim: int = 128,
        dropout: float = 0.2,
        scoring_fn: str = "concat",
        use_relation: bool = True,
        use_temporal: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.scoring_fn = scoring_fn
        self.use_relation = use_relation
        self.use_temporal = use_temporal

        # Compute input dimension based on what's included
        # Base: entity1_embed + entity2_embed
        input_dim = 2 * hidden_dim

        # Add temporal embeddings (from GRU)
        if use_temporal:
            input_dim += 2 * hidden_dim  # temporal for each entity

        # Note: We don't add relation dim here since we're predicting
        # edge existence regardless of type. Relation info is used as
        # edge features in RGCN, not in the classifier.

        if scoring_fn == "concat":
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, classifier_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_dim // 2, 1),
            )
        elif scoring_fn == "bilinear":
            # W matrix for bilinear: h1^T W h2
            self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)
            if use_temporal:
                self.temporal_linear = nn.Linear(2 * hidden_dim, 1)
        elif scoring_fn == "dot":
            # Project to same space then dot
            self.proj = nn.Linear(hidden_dim, hidden_dim)
            if use_temporal:
                self.temporal_linear = nn.Linear(2 * hidden_dim, 1)
        else:
            raise ValueError(f"Unknown scoring function: {scoring_fn}")

    def forward(
        self,
        entity1_embed: torch.Tensor,
        entity2_embed: torch.Tensor,
        entity1_temporal: Optional[torch.Tensor] = None,
        entity2_temporal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute edge existence logits.

        Args:
            entity1_embed: Embedding of first entity, shape (batch_size, hidden_dim)
            entity2_embed: Embedding of second entity, shape (batch_size, hidden_dim)
            entity1_temporal: Temporal embedding for entity1, shape (batch_size, hidden_dim)
            entity2_temporal: Temporal embedding for entity2, shape (batch_size, hidden_dim)

        Returns:
            Logits of shape (batch_size,) or (batch_size, 1)
        """
        if self.scoring_fn == "concat":
            # Concatenate all embeddings
            inputs = [entity1_embed, entity2_embed]

            if self.use_temporal:
                if entity1_temporal is not None:
                    inputs.append(entity1_temporal)
                else:
                    inputs.append(torch.zeros_like(entity1_embed))

                if entity2_temporal is not None:
                    inputs.append(entity2_temporal)
                else:
                    inputs.append(torch.zeros_like(entity2_embed))

            x = torch.cat(inputs, dim=-1)
            logits = self.classifier(x).squeeze(-1)

        elif self.scoring_fn == "bilinear":
            # Bilinear scoring
            logits = self.bilinear(entity1_embed, entity2_embed).squeeze(-1)

            if self.use_temporal and entity1_temporal is not None:
                temporal = torch.cat([entity1_temporal, entity2_temporal], dim=-1)
                logits = logits + self.temporal_linear(temporal).squeeze(-1)

        elif self.scoring_fn == "dot":
            # Dot product scoring
            proj1 = self.proj(entity1_embed)
            proj2 = self.proj(entity2_embed)
            logits = (proj1 * proj2).sum(dim=-1)

            if self.use_temporal and entity1_temporal is not None:
                temporal = torch.cat([entity1_temporal, entity2_temporal], dim=-1)
                logits = logits + self.temporal_linear(temporal).squeeze(-1)

        return logits


class SymmetricEdgeClassifier(EdgeClassifier):
    """
    Edge classifier that enforces symmetry for undirected graphs.

    For an undirected edge (i, j), the score should be the same as (j, i).
    This is achieved by processing both orderings and averaging.

    Args:
        Same as EdgeClassifier
    """

    def forward(
        self,
        entity1_embed: torch.Tensor,
        entity2_embed: torch.Tensor,
        entity1_temporal: Optional[torch.Tensor] = None,
        entity2_temporal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute symmetric edge existence logits.

        Averages score(e1, e2) and score(e2, e1) to ensure symmetry.
        """
        # Score in both directions
        logits_12 = super().forward(
            entity1_embed, entity2_embed, entity1_temporal, entity2_temporal
        )

        logits_21 = super().forward(
            entity2_embed, entity1_embed, entity2_temporal, entity1_temporal
        )

        # Average for symmetry
        return (logits_12 + logits_21) / 2


class TransitionEdgeClassifier(nn.Module):
    """
    Edge classifier that predicts P(transition) instead of P(edge exists).

    This reformulates the problem to directly predict state changes:
    - P(transition) = P(state changes from t-1 to t)

    The model does NOT receive was_on_t_minus_1 as input - it must learn to
    detect transitions from temporal patterns. The XOR conversion to edge
    predictions happens only at inference time via get_edge_logits().

    Args:
        hidden_dim: Entity embedding dimension
        classifier_hidden_dim: Hidden dimension in classifier MLP
        dropout: Dropout rate
        use_temporal: Whether to include temporal embeddings
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

        # Input: e1 + e2 + (optional: e1_temp + e2_temp)
        # Note: was_on is NOT included as input - model learns from temporal context
        input_dim = 2 * hidden_dim
        if use_temporal:
            input_dim += 2 * hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim // 2, 1),
        )

    def forward(
        self,
        entity1_embed: torch.Tensor,
        entity2_embed: torch.Tensor,
        entity1_temporal: Optional[torch.Tensor] = None,
        entity2_temporal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute P(transition) logits.

        Args:
            entity1_embed: (batch_size, hidden_dim)
            entity2_embed: (batch_size, hidden_dim)
            entity1_temporal: (batch_size, hidden_dim)
            entity2_temporal: (batch_size, hidden_dim)

        Returns:
            transition_logits: (batch_size,) - logits for P(transition)
        """
        # Build input features - no was_on, must learn from temporal patterns
        inputs = [entity1_embed, entity2_embed]

        if self.use_temporal:
            if entity1_temporal is not None:
                inputs.append(entity1_temporal)
            else:
                inputs.append(torch.zeros_like(entity1_embed))

            if entity2_temporal is not None:
                inputs.append(entity2_temporal)
            else:
                inputs.append(torch.zeros_like(entity2_embed))

        x = torch.cat(inputs, dim=-1)
        transition_logits = self.classifier(x).squeeze(-1)

        return transition_logits

    def get_edge_logits(
        self,
        entity1_embed: torch.Tensor,
        entity2_embed: torch.Tensor,
        was_on_t_minus_1: torch.Tensor,
        entity1_temporal: Optional[torch.Tensor] = None,
        entity2_temporal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convert transition logits to edge existence logits.

        This is used at INFERENCE TIME ONLY to convert P(transition) to P(edge).
        The was_on_t_minus_1 is only used here for XOR conversion, not as model input.

        Returns:
            edge_logits: (batch_size,) - logits for P(edge exists at t)
        """
        transition_logits = self.forward(
            entity1_embed, entity2_embed, entity1_temporal, entity2_temporal
        )

        # Convert P(transition) to P(on)
        # If was_on: P(on) = 1 - P(transition) = sigmoid(-transition_logits)
        # If was_off: P(on) = P(transition) = sigmoid(transition_logits)
        # In logit space: negate logits when was_on
        was_on = was_on_t_minus_1.float()
        edge_logits = transition_logits * (1 - 2 * was_on)  # Negate when was_on=1

        return edge_logits


class SymmetricTransitionClassifier(TransitionEdgeClassifier):
    """
    Symmetric version of TransitionEdgeClassifier for undirected graphs.
    """

    def forward(
        self,
        entity1_embed: torch.Tensor,
        entity2_embed: torch.Tensor,
        entity1_temporal: Optional[torch.Tensor] = None,
        entity2_temporal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Average scores in both directions for symmetry."""
        logits_12 = super().forward(
            entity1_embed,
            entity2_embed,
            entity1_temporal,
            entity2_temporal,
        )

        logits_21 = super().forward(
            entity2_embed,
            entity1_embed,
            entity2_temporal,
            entity1_temporal,
        )

        return (logits_12 + logits_21) / 2
