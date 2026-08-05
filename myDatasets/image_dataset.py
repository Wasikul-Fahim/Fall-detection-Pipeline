"""
Generic image dataset for activity classification.

Expected directory structure:

dataset_root/
    ADL/
        Walking/
            Subject01/
                image001.png
            Subject02/
                ...
    Fall/
        ForwardFall/
            Subject03/
                image001.png

The dataset infers labels from the top-level class folders (ADL/Fall).
"""


from __future__ import annotations

from pathlib import Path
from typing import Callable

from cv2 import transform
import pandas as pd
from PIL import Image

from torch import Tensor
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """
    Dataset backed by a metadata CSV.

    Parameters
    ----------
    csv_file : str | Path
        CSV file containing image paths and labels.

    transform : Callable | None
        Image transformation pipeline.
    """

    def __init__(
            self,
            csv_file: str | Path,
            transform: Callable | None = None,
    ) -> None:
        self.data = pd.read_csv(csv_file)

    self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
            self,
            index: int,
    ) -> tuple[Tensor, int]:
        row = self.data.iloc[index]

        image = Image.open(
            row["image_path"]
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, int(row["label"])class ImageDataset(Dataset):

    def __init__(
        self,
        csv_file: str | Path,
        transform: Callable | None = None,
    ) -> None:

        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Tensor, int]:

        row = self.data.iloc[index]

        image = Image.open(
            row["image_path"]
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, int(row["label"])