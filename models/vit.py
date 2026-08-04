"""
Vision Transformer model builder.

This module is responsible only for constructing the model.
It contains no training logic, optimizer, scheduler, or loss.
"""

from torchvision.models import (
    ViT_B_16_Weights,
    vit_b_16,
)

from torch import nn


def build_vit(
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build a Vision Transformer model.

    Parameters
    ----------
    num_classes : int
        Number of output classes.

    pretrained : bool
        Whether to load ImageNet pretrained weights.

    Returns
    -------
    nn.Module
        Configured Vision Transformer.
    """

    weights = (
        ViT_B_16_Weights.DEFAULT
        if pretrained
        else None
    )

    model = vit_b_16(weights=weights)

    in_features = model.heads.head.in_features

    model.heads.head = nn.Linear(
        in_features,
        num_classes,
    )

    return model