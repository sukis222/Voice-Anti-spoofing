import torch
import sys
import os

import numpy as np

from .calculate_eer import compute_eer
from src.metrics.base_metric import BaseMetric


class EERMetric(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.bonafide_scores = np.empty((0, 1))
        self.spoofed_scores = np.empty((0, 1))

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        """
        Metric calculation logic. Accumulates scores for EER calculation.

        Args:
            logits (Tensor): model output predictions, shape (batch_size, num_classes).
            labels (Tensor): ground-truth labels, shape (batch_size,).
        Returns:
            float: Placeholder value (0.0) as EER is computed in result().
        """
        logits_np = logits.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        mask = labels_np == 1
        bonafide_batch = logits_np[mask, 1:2]
        spoofed_batch = logits_np[~mask, 1:2]

        self.bonafide_scores = np.concatenate([self.bonafide_scores, bonafide_batch])
        self.spoofed_scores = np.concatenate([self.spoofed_scores, spoofed_batch])

        return 0.0  # Возвращаем заглушку для MetricTracker

    def result(self):
        if len(self.bonafide_scores) == 0 or len(self.spoofed_scores) == 0:
            eer = 1.0
        else:
            eer, _ = compute_eer(self.bonafide_scores[:, 0], self.spoofed_scores[:, 0])
        self.bonafide_scores = np.empty((0, 1))
        self.spoofed_scores = np.empty((0, 1))
        return eer