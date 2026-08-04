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

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """
    Generic image dataset.

    Parameters
    ----------
    root_dir : str | Path
        Root dataset directory.

    transform : Callable | None
        Image transformation pipeline.
    """

    CLASS_TO_INDEX = {
        "ADL": 0,
        "Fall": 1,
    }

    VALID_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
    }

    def __init__(
        self,
        root_dir: str | Path,
        transform: Callable | None = None,
    ) -> None:

        self.root_dir = Path(root_dir)
        self.transform = transform

        self.samples: list[tuple[Path, int]] = []

        self._index_dataset()

    def _index_dataset(self) -> None:
        """
        Index every image in the dataset.
        """

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.root_dir}"
            )

        for class_name, label in self.CLASS_TO_INDEX.items():

            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                continue

            for image_path in class_dir.rglob("*"):

                if image_path.suffix.lower() in self.VALID_EXTENSIONS:
                    self.samples.append((image_path, label))

        if not self.samples:
            raise RuntimeError("No images found in dataset.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Tensor, int]:

        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label