"""
Common utility functions used throughout the framework.
"""

from pathlib import Path
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    Seed all random number generators for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Automatically select the best available device.

    Priority
    --------
    CUDA
        NVIDIA GPUs

    MPS
        Apple Silicon

    CPU
        Fallback
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def create_directory(directory: str | Path) -> Path:
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    directory : str | Path

    Returns
    -------
    Path
        Created directory.
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    return directory


def save_checkpoint(
    state: dict,
    filepath: str | Path,
) -> None:
    """
    Save a training checkpoint.

    Parameters
    ----------
    state : dict
        Training state.

    filepath : str | Path
        Output checkpoint path.
    """

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    torch.save(state, filepath)


def load_checkpoint(
    filepath: str | Path,
    device: torch.device,
) -> dict:
    """
    Load a saved checkpoint.

    Parameters
    ----------
    filepath : str | Path

    device : torch.device

    Returns
    -------
    dict
        Loaded checkpoint.
    """

    return torch.load(filepath, map_location=device)