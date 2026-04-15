import torch.nn as nn
import torch.nn.functional as F


class SimpleL1L2Loss(nn.Module):
    def __init__(self, l1_weight=1.0, l2_weight=1.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def forward(self, prediction, target):
        """Return weighted total loss and individual L1/L2 components."""
        l1 = F.l1_loss(prediction, target)
        l2 = F.mse_loss(prediction, target)
        total = self.l1_weight * l1 + self.l2_weight * l2
        return total, {"l1": l1, "l2": l2}
