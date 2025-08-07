import torch
from torch import nn


class CrossEntropy(nn.Module):
    """
    Just a CrossEntropyLoss
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()


    def forward(self, outputs: torch.Tensor, labels: torch.Tensor, **batch):
        """
        Loss function calculation logic.

        Args:
            outputs (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        return {"loss": self.loss(outputs, labels)}
