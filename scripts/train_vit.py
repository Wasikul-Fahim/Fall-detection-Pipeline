"""
Entry point for Vision Transformer training.
"""

from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from configs.vit_config import ViTConfig
from datasets import ImageDataset
from models import build_vit
from training import Trainer
from utils.common import (
    get_device,
    seed_everything,
)


def main() -> None:

    config = ViTConfig()

    seed_everything(config.random_seed)

    device = get_device()

    transform = transforms.Compose(
        [
            transforms.Resize(
                (config.image_size, config.image_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = ImageDataset(
        root_dir=config.data_dir,
        transform=transform,
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = build_vit(
        num_classes=config.num_classes,
        pretrained=config.pretrained,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=config.checkpoint_dir,
    )

    trainer.fit(config.epochs)


if __name__ == "__main__":
    main()