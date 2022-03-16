"""Define the Network, loss and metrics"""

from typing import Callable, Dict, Tuple

import torch
import torch.nn as tnn
import torchvision.models as models

from utils.utils import Params


class Net(tnn.Module):
    """Extend the torch.nn.Module class to define a custom neural network
    """
    def __init__(self, params: Params) -> None:
        """Initialize the different layers in the neural network
        Args:
            params: Hyperparameters
        """
        super(Net, self).__init__()

        self.model = models.resnet18(pretrained=True)
        in_feats = self.model.fc.in_features
        self.model.fc = tnn.Linear(
            in_features=in_feats,
            out_features=params.num_classes
        )
        self.dropout_rate = params.dropout

    def forward(self, x: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """Defines the forward propagation through the network
        Args:
            x: Batch of images
        Returns:
            Embeddings and logits
        """
        return self.model(x)


def loss_fn(outputs: torch.tensor, ground_truth: torch.tensor) -> torch.tensor:
    """Compute the loss given outputs and ground_truth.
    Args:
        outputs: Logits of network forward pass
        ground_truth: Batch of ground truth
    Returns:
        loss for all the inputs in the batch
    """
    criterion = tnn.BCEWithLogitsLoss()
    loss = criterion(outputs, ground_truth)
    return loss


def avg_acc_gpu(outputs: torch.tensor, labels: torch.tensor, thr: float = 0.5) -> float:
    """Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: Logits of the network
        labels: Ground truth labels
        thr: Threshold
    Returns:
        average accuracy in [0,1]
    """
    labels = labels.to(torch.int32)
    outputs = (torch.sigmoid(outputs) > thr).to(torch.int32)
    avg_acc = (outputs == labels).sum() / outputs.numel()  # for multi-label else use shape[0]
    return avg_acc.item()


def avg_f1_score_gpu(
    outputs: torch.tensor,
    labels: torch.tensor,
    thr: float = 0.5,
    eps: float = 1e-7
) -> float:
    """Compute the F1 score, given the outputs and labels for all images.
    Args:
        outputs: Logits of the network
        labels: Ground truth labels
        thr: Threshold
        eps: Epsilon
    Returns:
        average f1 score
    """
    labels = labels.to(torch.int32)
    outputs = (torch.sigmoid(outputs) > thr).to(torch.int32)
    tp = (labels * outputs).sum()
    fp = ((1 - labels) * outputs).sum()
    fn = (labels * (1 - outputs)).sum()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)

    return f1.item()


# Maintain all metrics required during training and evaluation.
def get_metrics() -> Dict[str, Callable]:
    """Returns a dictionary of all the metrics to be used
    """
    metrics = {
        "accuracy": avg_acc_gpu,
        "f1-score": avg_f1_score_gpu,
    }
    return metrics
