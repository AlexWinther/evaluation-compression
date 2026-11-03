"""Loss functions used by active testing objectives."""

import torch
from torch import nn


class ZeroOneLoss(nn.Module):
    """Return one zero-or-one classification error for each example."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (logits.argmax(dim=1) != targets).float()
