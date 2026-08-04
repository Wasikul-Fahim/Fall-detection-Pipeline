"""
Configuration for Vision Transformer experiments.

This module centralizes all experiment-related parameters.
In future phases, this file will be replaced by the YAML
configuration system without requiring changes to the training code.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ViTConfig:
    """
    Configuration container for a ViT experiment.
    """

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    data_dir: str = "dataset"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    num_classes: int = 2
    pretrained: bool = True

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 20

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    random_seed: int = 42

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    checkpoint_dir: str = "checkpoints"