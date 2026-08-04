"""
Evaluation metrics used during training.

This module contains reusable metric functions.
"""

from __future__ import annotations

import torch
from torch import Tensor


@torch.no_grad()
def accuracy(
    predictions: Tensor,
    targets: Tensor,
) -> float:
    """
    Compute classification accuracy.

    Parameters
    ----------
    predictions : Tensor
        Raw model outputs (logits).

    targets : Tensor
        Ground-truth labels.

    Returns
    -------
    float
        Accuracy in the range [0, 1].
    """

    predicted = predictions.argmax(dim=1)

    correct = (predicted == targets).sum().item()

    return correct / len(targets)