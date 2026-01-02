"""Loss functions for PPI dynamics classification."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self, gamma: float = 2.0, alpha: Optional[float] = None, reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Raw logits of shape (N,) or (N, 1)
            targets: Binary targets of shape (N,) or (N, 1), values in {0, 1}

        Returns:
            Focal loss value
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Compute probabilities
        probs = torch.sigmoid(logits)

        # For numerical stability
        eps = 1e-7
        probs = probs.clamp(eps, 1 - eps)

        # Compute focal weights
        # p_t = p for y=1, (1-p) for y=0
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma

        # Compute BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Apply focal weight
        focal_loss = focal_weight * bce

        # Apply alpha weighting if specified
        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal_loss = alpha_t * focal_loss

        # Reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class BCEWithLogitsLossWrapper(nn.Module):
    """
    Wrapper around BCEWithLogitsLoss to match FocalLoss interface.

    Useful for comparing focal loss vs standard BCE.
    """

    def __init__(self, pos_weight: Optional[float] = None, reduction: str = "mean"):
        super().__init__()
        if pos_weight is not None:
            pos_weight = torch.tensor([pos_weight])
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        return self.loss_fn(logits, targets)


def get_loss_function(
    loss_type: str = "focal",
    gamma: float = 2.0,
    alpha: Optional[float] = None,
    pos_weight: Optional[float] = None,
) -> nn.Module:
    """
    Factory function to create loss function.

    Args:
        loss_type: 'focal' or 'bce'
        gamma: Focal loss gamma parameter
        alpha: Focal loss alpha parameter
        pos_weight: BCE positive class weight

    Returns:
        Loss function module
    """
    if loss_type == "focal":
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif loss_type == "bce":
        return BCEWithLogitsLossWrapper(pos_weight=pos_weight)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
