import torch

from src.metrics.base_metric import BaseMetric


class ExampleMetric(BaseMetric):
    def __init__(self, metric, device, *args, **kwargs):

        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor, **kwargs):
        """
        Metric calculation logic.

        Args:
            outputs (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """
        classes = outputs.argmax(dim=-1)
        return self.metric(classes, labels)
