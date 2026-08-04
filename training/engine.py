"""
Training engine.

This module contains the Trainer class responsible for
training and validation.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from training.metrics import accuracy
from utils.common import save_checkpoint


class Trainer:
    """
    Generic trainer for image classification.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        checkpoint_dir: str,
    ) -> None:

        self.model = model.to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = criterion
        self.optimizer = optimizer

        self.device = device

        self.checkpoint_dir = Path(checkpoint_dir)

    def train_one_epoch(self) -> tuple[float, float]:
        """
        Train for one epoch.

        Returns
        -------
        tuple
            Average loss and accuracy.
        """

        self.model.train()

        total_loss = 0.0
        total_acc = 0.0

        for images, labels in self.train_loader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()
            total_acc += accuracy(outputs, labels)

        n = len(self.train_loader)

        return total_loss / n, total_acc / n

    @torch.no_grad()
    def validate(self) -> tuple[float, float]:
        """
        Validate the model.
        """

        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0

        for images, labels in self.val_loader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            total_acc += accuracy(outputs, labels)

        n = len(self.val_loader)

        return total_loss / n, total_acc / n

    def fit(
        self,
        epochs: int,
    ) -> None:
        """
        Run the training loop.
        """

        best_accuracy = 0.0

        for epoch in range(1, epochs + 1):

            train_loss, train_acc = self.train_one_epoch()

            val_loss, val_acc = self.validate()

            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss {train_loss:.4f} | "
                f"Train Acc {train_acc:.4f} | "
                f"Val Loss {val_loss:.4f} | "
                f"Val Acc {val_acc:.4f}"
            )

            if val_acc > best_accuracy:

                best_accuracy = val_acc

                save_checkpoint(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "accuracy": val_acc,
                        "epoch": epoch,
                    },
                    self.checkpoint_dir / "best_model.pth",
                )