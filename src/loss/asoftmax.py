import torch
from torch import nn
import torch.nn.functional as F
import math
'''
class ASoftmax(nn.Module):
    """
    A-Softmax.
    """
    def __init__(self, in_features: int, out_features: int, m: float = 4.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, data_object, labels, outputs) -> dict:
        x_norm = F.normalize(data_object, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)

        cos_theta_all = F.linear(x_norm, w_norm)

        s = data_object.norm(p=2, dim=1)

        cos_theta_target_values = cos_theta_all.gather(1, labels.view(-1, 1)).squeeze(1)

        phi_target = self._calculate_sphereface_phi(cos_theta_target_values, self.m)

        one_hot = torch.zeros_like(cos_theta_all).scatter_(1, labels.view(-1, 1), 1)

        modified_cos_theta = one_hot * phi_target.unsqueeze(1) + (1 - one_hot) * cos_theta_all

        final_logits = modified_cos_theta * s.unsqueeze(1)

        loss_value = F.nll_loss(F.log_softmax(final_logits, dim=1), labels)

        return {"loss": loss_value}

    def _calculate_sphereface_phi(self, cos_theta: torch.Tensor, m: float) -> torch.Tensor:
        cos_theta_clamped = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_theta_clamped)
        k = (m * theta / math.pi).floor()
        phi = (-1.0)**k * torch.cos(m * theta) - 2 * k
        return phi

import torch
from torch import nn
'''

class ASoftmax(nn.Module):
    """
    Example of a loss function to use.
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()


    def forward(self, outputs: torch.Tensor, labels: torch.Tensor, **batch):
        """
        Loss function calculation logic.

        Note that loss function must return dict. It must contain a value for
        the 'loss' key. If several losses are used, accumulate them into one 'loss'.
        Intermediate losses can be returned with other loss names.

        For example, if you have loss = a_loss + 2 * b_loss. You can return dict
        with 3 keys: 'loss', 'a_loss', 'b_loss'. You can log them individually inside
        the writer. See config.writer.loss_names.

        Args:
            outputs (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        return {"loss": self.loss(outputs, labels)}
